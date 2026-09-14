import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from core.config import AUTH_DB_PATH

_cache_data: Dict[str, Any] = {}

def _invalidate_cache():
    _cache_data.clear()

def _get_connection():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_notifications_table():
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_notifications (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            notification_type TEXT DEFAULT 'info',
            icon TEXT DEFAULT '🔔',
            link_route TEXT DEFAULT 'Dashboard',
            created_at TEXT NOT NULL,
            is_read INTEGER DEFAULT 0
        )
        """)
        conn.commit()

init_notifications_table()

class NotificationModel:
    @classmethod
    def create_notification(
        cls,
        title: str,
        message: str,
        user_id: Optional[str] = None,
        notification_type: str = "info",
        icon: str = "🔔",
        link_route: str = "Dashboard"
    ) -> str:
        _invalidate_cache()
        notif_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO user_notifications (id, user_id, title, message, notification_type, icon, link_route, created_at, is_read)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (notif_id, user_id, title, message, notification_type, icon, link_route, now))
            conn.commit()
        return notif_id

    @classmethod
    def get_notifications(cls, user_id: Optional[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
        now = time.time()
        cache_key = f"notifs_{user_id}_{limit}"
        if cache_key in _cache_data:
            val, exp = _cache_data[cache_key]
            if now < exp:
                return val

        with _get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("""
                SELECT id, user_id, title, message, notification_type, icon, link_route, created_at, is_read
                FROM user_notifications
                WHERE user_id = ? OR user_id IS NULL
                ORDER BY created_at DESC LIMIT ?
                """, (user_id, limit))
            else:
                cursor.execute("""
                SELECT id, user_id, title, message, notification_type, icon, link_route, created_at, is_read
                FROM user_notifications
                ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            if not rows:
                cls.seed_default_notifications(user_id)
                return cls.get_notifications(user_id=user_id, limit=limit)
            res = [dict(r) for r in rows]
            _cache_data[cache_key] = (res, now + 5.0)
            return res

    @classmethod
    def get_unread_count(cls, user_id: Optional[str] = None) -> int:
        now = time.time()
        cache_key = f"unread_{user_id}"
        if cache_key in _cache_data:
            val, exp = _cache_data[cache_key]
            if now < exp:
                return val

        with _get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("""
                SELECT COUNT(*) FROM user_notifications
                WHERE (user_id = ? OR user_id IS NULL) AND is_read = 0
                """, (user_id,))
            else:
                cursor.execute("""
                SELECT COUNT(*) FROM user_notifications WHERE is_read = 0
                """)
            res = cursor.fetchone()
            val = res[0] if res else 0
            _cache_data[cache_key] = (val, now + 5.0)
            return val

    @classmethod
    def mark_all_as_read(cls, user_id: Optional[str] = None):
        _invalidate_cache()
        with _get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("""
                UPDATE user_notifications SET is_read = 1
                WHERE user_id = ? OR user_id IS NULL
                """, (user_id,))
            else:
                cursor.execute("UPDATE user_notifications SET is_read = 1")
            conn.commit()

    @classmethod
    def mark_as_read(cls, notif_id: str):
        _invalidate_cache()
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_notifications SET is_read = 1 WHERE id = ?", (notif_id,))
            conn.commit()

    @classmethod
    def clear_all(cls, user_id: Optional[str] = None):
        _invalidate_cache()
        with _get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("DELETE FROM user_notifications WHERE user_id = ? OR user_id IS NULL", (user_id,))
            else:
                cursor.execute("DELETE FROM user_notifications")
            conn.commit()

    @classmethod
    def seed_default_notifications(cls, user_id: Optional[str] = None):
        samples = [
            ("⚡ High-Speed DuckDB Initialized", "In-memory query acceleration active with sub-millisecond cache latency.", "success", "⚡", "SQL Analytics"),
            ("🛡️ Identity Clearance Verified", "Your secure PBKDF2 salt and session clearance are active on this machine.", "security", "🔒", "Auth"),
            ("📊 Automated EDA Pipeline Ready", "Ingest any CSV, Excel, Parquet, or JSON to trigger automatic statistical profiling.", "info", "📈", "Dashboard"),
            ("✨ AI Insight Engine Standby", "Deterministic synthesis and Gemini/OpenAI endpoints are calibrated.", "ai", "✨", "AI Insights")
        ]
        for title, msg, n_type, icon, route in samples:
            cls.create_notification(
                title=title,
                message=msg,
                user_id=user_id,
                notification_type=n_type,
                icon=icon,
                link_route=route
            )

