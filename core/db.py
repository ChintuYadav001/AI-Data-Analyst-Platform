"""
Unified Database Layer supporting SQLite, MySQL, and PostgreSQL.
Provides connection pooling, cursor abstraction (dictionary access and parameter adaptation),
schema initialization, and automated migration.
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from core.config import (
    AUTH_DB_PATH,
    DB_TYPE,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DATABASE,
)


class DatabaseCursorWrapper:
    """Wraps cursor to provide uniform dict access and ? -> %s placeholder translation."""

    def __init__(self, raw_cursor: Any, is_mysql: bool = False, is_pg: bool = False):
        self._cursor = raw_cursor
        self._is_mysql = is_mysql
        self._is_pg = is_pg

    def _convert_query(self, query: str) -> str:
        if self._is_mysql or self._is_pg:
            # Replace ? with %s for MySQL/PostgreSQL
            return query.replace("?", "%s")
        return query

    def execute(self, query: str, params: Union[Tuple, List] = ()) -> Any:
        converted = self._convert_query(query)
        if params:
            return self._cursor.execute(converted, params)
        return self._cursor.execute(converted)

    def executemany(self, query: str, seq_of_params: List[Union[Tuple, List]]) -> Any:
        converted = self._convert_query(query)
        return self._cursor.executemany(converted, seq_of_params)

    def fetchone(self) -> Optional[Dict[str, Any]]:
        row = self._cursor.fetchone()
        if row is None:
            return None
        if hasattr(row, "keys"):
            return dict(row)
        if isinstance(row, dict):
            return row
        if isinstance(row, (tuple, list)):
            if self._cursor.description:
                cols = [col[0] for col in self._cursor.description]
                return dict(zip(cols, row))
        return row

    def fetchall(self) -> List[Dict[str, Any]]:
        rows = self._cursor.fetchall()
        if not rows:
            return []
        result = []
        for r in rows:
            if hasattr(r, "keys"):
                result.append(dict(r))
            elif isinstance(r, dict):
                result.append(r)
            elif isinstance(r, (tuple, list)) and self._cursor.description:
                cols = [col[0] for col in self._cursor.description]
                result.append(dict(zip(cols, r)))
            else:
                result.append(r)
        return result

    @property
    def rowcount(self) -> int:
        return getattr(self._cursor, "rowcount", -1)

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:
            pass


class DatabaseConnectionWrapper:
    """Wraps connection to provide uniform context management and cursor factory."""

    def __init__(self, raw_conn: Any, db_type: str):
        self._conn = raw_conn
        self._db_type = db_type

    def cursor(self) -> DatabaseCursorWrapper:
        is_mysql = self._db_type == "mysql"
        is_pg = self._db_type == "postgres"
        raw_cur = self._conn.cursor()
        return DatabaseCursorWrapper(raw_cur, is_mysql=is_mysql, is_pg=is_pg)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        try:
            self._conn.rollback()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "DatabaseConnectionWrapper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()


def get_db_type() -> str:
    from core import config
    return getattr(config, "DB_TYPE", "sqlite").lower()


def get_connection(target_db_type: Optional[str] = None) -> DatabaseConnectionWrapper:
    """Return an active database connection based on configuration."""
    from core import config
    db_mode = (target_db_type or get_db_type()).lower()

    if db_mode == "mysql":
        try:
            import pymysql
            from pymysql.cursors import DictCursor

            raw_conn = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                cursorclass=DictCursor,
                autocommit=False,
                charset="utf8mb4",
                connect_timeout=5,
            )
            return DatabaseConnectionWrapper(raw_conn, "mysql")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"MySQL connection to {config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE} failed ({e}). Falling back to SQLite."
            )
            return _get_sqlite_connection()

    elif db_mode == "postgres":
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            from core.neon_db import get_neon_url

            neon_url = get_neon_url(getattr(config, "NEON_DATABASE_URL", None))
            if neon_url:
                raw_conn = psycopg2.connect(
                    neon_url,
                    cursor_factory=RealDictCursor,
                    connect_timeout=8,
                )
            else:
                raw_conn = psycopg2.connect(
                    host=config.PG_HOST,
                    port=config.PG_PORT,
                    user=config.PG_USER,
                    password=config.PG_PASSWORD,
                    dbname=config.PG_DATABASE,
                    cursor_factory=RealDictCursor,
                    connect_timeout=5,
                )
            return DatabaseConnectionWrapper(raw_conn, "postgres")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"PostgreSQL connection to {config.PG_HOST}:{config.PG_PORT}/{config.PG_DATABASE} failed ({e}). Falling back to SQLite."
            )
            return _get_sqlite_connection()

    return _get_sqlite_connection()


def _get_sqlite_connection() -> DatabaseConnectionWrapper:
    from core.config import AUTH_DB_PATH
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_conn = sqlite3.connect(str(AUTH_DB_PATH))
    raw_conn.row_factory = sqlite3.Row
    return DatabaseConnectionWrapper(raw_conn, "sqlite")


_initialized_dbs = set()


def init_db(target_db_type: Optional[str] = None) -> None:
    """Initialise database schema (users, login_logs, otps, email_otps)."""
    db_mode = (target_db_type or get_db_type()).lower()
    if db_mode in _initialized_dbs:
        return

    with get_connection(db_mode) as conn:
        cursor = conn.cursor()

        if conn._db_type == "mysql":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(64) PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    salt VARCHAR(64) NOT NULL,
                    full_name VARCHAR(255),
                    phone VARCHAR(50),
                    role VARCHAR(50) DEFAULT 'analyst',
                    created_at VARCHAR(64) NOT NULL,
                    last_login_at VARCHAR(64),
                    login_count INT DEFAULT 0,
                    is_active INT DEFAULT 1,
                    INDEX idx_users_username (username),
                    INDEX idx_users_email (email),
                    INDEX idx_users_phone (phone)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_logs (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64),
                    username VARCHAR(100) NOT NULL,
                    timestamp VARCHAR(64) NOT NULL,
                    ip_address VARCHAR(100),
                    user_agent TEXT,
                    status VARCHAR(50) NOT NULL,
                    failure_reason TEXT,
                    INDEX idx_logs_user (user_id),
                    INDEX idx_logs_timestamp (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otps (
                    id VARCHAR(64) PRIMARY KEY,
                    target VARCHAR(255) NOT NULL,
                    target_type VARCHAR(50) NOT NULL,
                    otp_code VARCHAR(20) NOT NULL,
                    created_at VARCHAR(64) NOT NULL,
                    expires_at VARCHAR(64) NOT NULL,
                    is_verified INT DEFAULT 0,
                    INDEX idx_otps_target (target),
                    INDEX idx_otps_code (otp_code)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_otps (
                    id VARCHAR(64) PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    otp_code VARCHAR(20) NOT NULL,
                    created_at VARCHAR(64) NOT NULL,
                    expires_at VARCHAR(64) NOT NULL,
                    is_verified INT DEFAULT 0,
                    INDEX idx_email_otps (email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
        elif conn._db_type == "postgres":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(64) PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    salt VARCHAR(64) NOT NULL,
                    full_name VARCHAR(255),
                    phone VARCHAR(50),
                    role VARCHAR(50) DEFAULT 'analyst',
                    created_at VARCHAR(64) NOT NULL,
                    last_login_at VARCHAR(64),
                    login_count INT DEFAULT 0,
                    is_active INT DEFAULT 1
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_logs (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64),
                    username VARCHAR(100) NOT NULL,
                    timestamp VARCHAR(64) NOT NULL,
                    ip_address VARCHAR(100),
                    user_agent TEXT,
                    status VARCHAR(50) NOT NULL,
                    failure_reason TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otps (
                    id VARCHAR(64) PRIMARY KEY,
                    target VARCHAR(255) NOT NULL,
                    target_type VARCHAR(50) NOT NULL,
                    otp_code VARCHAR(20) NOT NULL,
                    created_at VARCHAR(64) NOT NULL,
                    expires_at VARCHAR(64) NOT NULL,
                    is_verified INT DEFAULT 0
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_otps (
                    id VARCHAR(64) PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    otp_code VARCHAR(20) NOT NULL,
                    created_at VARCHAR(64) NOT NULL,
                    expires_at VARCHAR(64) NOT NULL,
                    is_verified INT DEFAULT 0
                );
            """)
        else:
            # SQLite
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT,
                    phone TEXT,
                    role TEXT DEFAULT 'analyst',
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    login_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1
                )
            """)
            cursor.execute("PRAGMA table_info(users)")
            cols = [r["name"] if isinstance(r, dict) else r[1] for r in cursor.fetchall()]
            if "phone" not in cols:
                cursor.execute("ALTER TABLE users ADD COLUMN phone TEXT")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    username TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    status TEXT NOT NULL,
                    failure_reason TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otps (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    otp_code TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    is_verified INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_otps (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    otp_code TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    is_verified INTEGER DEFAULT 0
                )
            """)

        _seed_default_users(conn)
        _initialized_dbs.add(db_mode)


def _seed_default_users(conn: DatabaseConnectionWrapper) -> None:
    """Seed initial platform users if not already present."""
    import hashlib
    import os
    import uuid
    from datetime import datetime, timezone

    users_to_seed = [
        {
            "username": "analyst",
            "email": "analyst@quantum.ai",
            "full_name": "Quantum Lead Analyst",
            "phone": "+1 555-0199",
            "role": "Lead Data Scientist",
            "password": "QuantumPassword2026!",
        },
        {
            "username": "chintu",
            "email": "chintukumaredu00@gmail.com",
            "full_name": "Chintu Kumar",
            "phone": "+91 9262000000",
            "role": "Executive BI Director",
            "password": "QuantumPassword2026!",
        }
    ]

    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    for u in users_to_seed:
        try:
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (u["username"], u["email"]))
            row = cursor.fetchone()
            salt = os.urandom(16).hex()
            p_hash = hashlib.pbkdf2_hmac(
                'sha256',
                u["password"].encode('utf-8'),
                bytes.fromhex(salt),
                100_000
            ).hex()
            if not row:
                cursor.execute(
                    """
                    INSERT INTO users (id, username, email, phone, password_hash, salt, full_name, role, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (str(uuid.uuid4()), u["username"], u["email"], u["phone"], p_hash, salt, u["full_name"], u["role"], now)
                )
            else:
                cursor.execute(
                    "UPDATE users SET password_hash = ?, salt = ?, is_active = 1 WHERE id = ?",
                    (p_hash, salt, row["id"])
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error seeding user {u['username']}: {e}")


def sync_dataframe_to_sql(table_name: str, df: Any, project_name: str = "") -> bool:
    """Automatically writes and synchronizes a Pandas DataFrame into MySQL as a real table in analytics_platform."""
    if df is None or getattr(df, "empty", True):
        return False

    # For large datasets (e.g. 2M rows), cap SQL sync sample to 10k rows
    if len(df) > 10_000:
        df = df.head(10_000)

    clean_table = re.sub(r"\W+", "_", table_name).lower().strip("_")
    if not clean_table:
        clean_table = "dataset"

    from core import config
    if getattr(config, "DB_TYPE", "").lower() != "mysql":
        return False

    try:
        import json
        import uuid
        from datetime import datetime, timezone
        import pandas as pd
        import pymysql

        conn = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            charset="utf8mb4",
            autocommit=False,
            connect_timeout=5,
        )

        col_defs = []
        clean_cols = []
        for col in df.columns:
            c_name = re.sub(r"\W+", "_", str(col)).lower().strip("_")
            if not c_name:
                c_name = "col"
            base = c_name
            counter = 1
            while c_name in clean_cols:
                c_name = f"{base}_{counter}"
                counter += 1
            clean_cols.append(c_name)

            dtype = str(df[col].dtype).lower()
            if "int" in dtype:
                sql_type = "BIGINT"
            elif "float" in dtype:
                sql_type = "DOUBLE"
            elif "bool" in dtype:
                sql_type = "TINYINT(1)"
            else:
                sql_type = "TEXT"
            col_defs.append(f"`{c_name}` {sql_type}")

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS datasets_catalog (
                    id VARCHAR(64) PRIMARY KEY,
                    table_name VARCHAR(128) UNIQUE NOT NULL,
                    project_name VARCHAR(255),
                    row_count INT DEFAULT 0,
                    column_count INT DEFAULT 0,
                    columns_json TEXT,
                    updated_at VARCHAR(64) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            cur.execute(f"DROP TABLE IF EXISTS `{clean_table}`;")
            cur.execute(f"CREATE TABLE `{clean_table}` ({', '.join(col_defs)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;")

            placeholders = ", ".join(["%s"] * len(clean_cols))
            insert_sql = f"INSERT INTO `{clean_table}` ({', '.join([f'`{c}`' for c in clean_cols])}) VALUES ({placeholders})"

            rows = []
            for row in df.itertuples(index=False):
                cleaned_row = []
                for val in row:
                    if pd.isna(val):
                        cleaned_row.append(None)
                    elif isinstance(val, (int, float, str, bool)):
                        cleaned_row.append(val)
                    else:
                        cleaned_row.append(str(val))
                rows.append(tuple(cleaned_row))

            batch_size = 500
            for i in range(0, len(rows), batch_size):
                cur.executemany(insert_sql, rows[i:i + batch_size])

            catalog_id = str(uuid.uuid4())
            now_str = datetime.now(timezone.utc).isoformat()
            cur.execute("""
                INSERT INTO datasets_catalog (id, table_name, project_name, row_count, column_count, columns_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    project_name = VALUES(project_name),
                    row_count = VALUES(row_count),
                    column_count = VALUES(column_count),
                    columns_json = VALUES(columns_json),
                    updated_at = VALUES(updated_at);
            """, (
                catalog_id,
                clean_table,
                project_name or clean_table.replace("_", " ").title(),
                len(df),
                len(clean_cols),
                json.dumps(clean_cols),
                now_str
            ))

            conn.commit()
            conn.close()
            return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error auto-syncing dataframe to MySQL: {e}")
        return False
