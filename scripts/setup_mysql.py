"""
MySQL Setup & SQLite Migration Script for Quantum Analytics Platform.

This utility:
1. Connects to your running MySQL server (default: localhost:3306).
2. Creates the database schema 'analytics_platform' if it does not exist.
3. Initializes the tables: `users`, `login_logs`, `otps`, `email_otps`.
4. Migrates all user accounts and audit records from SQLite (data/auth.db) into MySQL.
5. Updates the local .env configuration so the app seamlessly routes all future registrations to MySQL.
6. Displays the exact MySQL Workbench connection instructions and query samples.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pymysql
from pymysql.cursors import DictCursor
from core.config import AUTH_DB_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Configure MySQL and migrate SQLite records.")
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", "localhost"), help="MySQL Host (default: localhost)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", 3306)), help="MySQL Port (default: 3306)")
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", "root"), help="MySQL User (default: root)")
    parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD", ""), help="MySQL Password")
    parser.add_argument("--database", default=os.getenv("MYSQL_DATABASE", "analytics_platform"), help="Database Name (default: analytics_platform)")
    parser.add_argument("--no-update-env", action="store_true", help="Do not modify .env file")
    return parser.parse_args()


def update_env_file(host: str, port: int, user: str, password: str, database: str):
    env_file = BASE_DIR / ".env"
    lines = []
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

    keys_to_set = {
        "DB_TYPE": "mysql",
        "MYSQL_HOST": host,
        "MYSQL_PORT": str(port),
        "MYSQL_USER": user,
        "MYSQL_PASSWORD": password,
        "MYSQL_DATABASE": database,
    }

    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        matched = False
        for k in keys_to_set:
            if stripped.startswith(f"{k}=") or stripped.startswith(f"{k} ="):
                new_lines.append(f"{k}={keys_to_set[k]}\n")
                updated_keys.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for k, v in keys_to_set.items():
        if k not in updated_keys:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f"{k}={v}\n")

    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"  [+] Updated .env configuration with DB_TYPE=mysql and credentials.")


def main():
    args = parse_args()
    pwd = args.password
    if not pwd:
        # Prompt if interactive
        if sys.stdin.isatty():
            import getpass
            pwd = getpass.getpass(f"Enter MySQL password for '{args.user}@{args.host}': ")
        else:
            print(f"[!] Warning: No password provided via --password. Attempting connection with empty password...")

    print("\n========================================================")
    print("      QUANTUM ANALYTICS - MYSQL WORKBENCH SETUP         ")
    print("========================================================")
    print(f"Connecting to MySQL server at {args.host}:{args.port} as user '{args.user}'...")

    try:
        root_conn = pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=pwd,
            charset="utf8mb4",
            connect_timeout=10,
        )
    except Exception as e:
        print(f"\n[X] Connection Failed: {e}")
        print(f"\nPlease verify your MySQL password and run again:")
        print(f"   python scripts/setup_mysql.py --password \"YOUR_PASSWORD\"\n")
        sys.exit(1)

    print(f"  [+] Connection established to MySQL server.")

    # 1. Create database
    with root_conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{args.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        root_conn.commit()
    root_conn.close()
    print(f"  [+] Database `{args.database}` verified/created.")

    # 2. Connect to specific database
    db_conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=pwd,
        database=args.database,
        charset="utf8mb4",
        cursorclass=DictCursor,
    )

    # 3. Create tables
    with db_conn.cursor() as cur:
        cur.execute("""
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

        cur.execute("""
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

        cur.execute("""
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

        cur.execute("""
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
        db_conn.commit()
    print(f"  [+] Tables initialized (`users`, `login_logs`, `otps`, `email_otps`).")

    # 4. Migrate data from SQLite
    if AUTH_DB_PATH.exists():
        print(f"\nScanning SQLite database at {AUTH_DB_PATH} for migration...")
        s_conn = sqlite3.connect(str(AUTH_DB_PATH))
        s_conn.row_factory = sqlite3.Row
        s_cur = s_conn.cursor()

        # Migrate users
        try:
            s_cur.execute("PRAGMA table_info(users)")
            cols = [r["name"] for r in s_cur.fetchall()]
            has_phone = "phone" in cols

            if has_phone:
                s_cur.execute("SELECT id, username, email, password_hash, salt, full_name, phone, role, created_at, last_login_at, login_count, is_active FROM users")
            else:
                s_cur.execute("SELECT id, username, email, password_hash, salt, full_name, role, created_at, last_login_at, login_count, is_active FROM users")

            sqlite_users = s_cur.fetchall()
            migrated_users = 0
            with db_conn.cursor() as cur:
                for u in sqlite_users:
                    u_dict = dict(u)
                    cur.execute("""
                        INSERT INTO users (id, username, email, password_hash, salt, full_name, phone, role, created_at, last_login_at, login_count, is_active)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            full_name = VALUES(full_name),
                            phone = VALUES(phone),
                            last_login_at = VALUES(last_login_at),
                            login_count = VALUES(login_count);
                    """, (
                        u_dict.get("id"),
                        u_dict.get("username"),
                        u_dict.get("email"),
                        u_dict.get("password_hash"),
                        u_dict.get("salt"),
                        u_dict.get("full_name"),
                        u_dict.get("phone"),
                        u_dict.get("role", "analyst"),
                        u_dict.get("created_at"),
                        u_dict.get("last_login_at"),
                        u_dict.get("login_count", 0),
                        u_dict.get("is_active", 1),
                    ))
                    migrated_users += 1
                db_conn.commit()
            print(f"  [+] Migrated {migrated_users} user account(s) into MySQL `users` table.")
        except Exception as e:
            print(f"  [!] Error migrating users: {e}")

        # Migrate login_logs
        try:
            s_cur.execute("SELECT id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason FROM login_logs")
            sqlite_logs = s_cur.fetchall()
            migrated_logs = 0
            with db_conn.cursor() as cur:
                for l in sqlite_logs:
                    l_dict = dict(l)
                    cur.execute("""
                        INSERT IGNORE INTO login_logs (id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        l_dict.get("id"),
                        l_dict.get("user_id"),
                        l_dict.get("username"),
                        l_dict.get("timestamp"),
                        l_dict.get("ip_address"),
                        l_dict.get("user_agent"),
                        l_dict.get("status"),
                        l_dict.get("failure_reason"),
                    ))
                    migrated_logs += 1
                db_conn.commit()
            print(f"  [+] Migrated {migrated_logs} audit log(s) into MySQL `login_logs` table.")
        except Exception as e:
            print(f"  [!] Error migrating logs: {e}")

        # Migrate otps
        try:
            s_cur.execute("SELECT id, target, target_type, otp_code, created_at, expires_at, is_verified FROM otps")
            sqlite_otps = s_cur.fetchall()
            migrated_otps = 0
            with db_conn.cursor() as cur:
                for o in sqlite_otps:
                    o_dict = dict(o)
                    cur.execute("""
                        INSERT IGNORE INTO otps (id, target, target_type, otp_code, created_at, expires_at, is_verified)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (
                        o_dict.get("id"),
                        o_dict.get("target"),
                        o_dict.get("target_type"),
                        o_dict.get("otp_code"),
                        o_dict.get("created_at"),
                        o_dict.get("expires_at"),
                        o_dict.get("is_verified", 0),
                    ))
                    migrated_otps += 1
                db_conn.commit()
            print(f"  [+] Migrated {migrated_otps} OTP verification record(s) into MySQL `otps` table.")
        except Exception as e:
            print(f"  [!] Error migrating otps: {e}")

        s_conn.close()

    # 5. Update .env
    if not args.no_update_env:
        update_env_file(args.host, args.port, args.user, pwd, args.database)

    # 6. Verify row count in MySQL
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM users;")
        user_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM login_logs;")
        log_count = cur.fetchone()["c"]

    db_conn.close()

    print("\n========================================================")
    print("               MYSQL SETUP SUCCESSFUL!                  ")
    print("========================================================")
    print(f"Database Name: {args.database}")
    print(f"Total Users in MySQL:     {user_count}")
    print(f"Total Audit Logs in MySQL: {log_count}")
    print("\n--- HOW TO VIEW IN MYSQL WORKBENCH ---")
    print("1. Open MySQL Workbench on your desktop.")
    print("2. Click '+' or click your existing local connection (localhost:3306).")
    print(f"3. In the left 'Schemas' pane, refresh and double-click `{args.database}`.")
    print("4. Expand 'Tables' to see `users`, `login_logs`, `otps`.")
    print("5. Right-click `users` and select 'Select Rows - Limit 1000' or run:")
    print(f"   USE {args.database};")
    print("   SELECT id, username, email, phone, role, created_at, last_login_at FROM users;\n")


if __name__ == "__main__":
    main()
