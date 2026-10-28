"""
Authentication and Audit Logging Model using SQLite with OTP Verification.
Stores user accounts with salted PBKDF2 hashes, an audit ledger of all login events,
and cryptographically verified One-Time Passwords (OTPs).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.config import AUTH_DB_PATH
from core.db import get_connection as _get_connection, init_db
from core.services.notification_service import NotificationService, mask_destination


def _is_expired(expires_at: Any) -> bool:
    """Robust expiration check for timestamps across SQLite and Postgres."""
    if not expires_at:
        return True
    try:
        if isinstance(expires_at, datetime):
            exp = expires_at
        else:
            exp = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return exp < datetime.now(timezone.utc)
    except Exception:
        return str(expires_at) < datetime.now(timezone.utc).isoformat()


def _get_target_variations(target: str) -> List[str]:
    """Generate variations of email/phone so formatting differences (+91, spaces) match."""
    raw = target.strip().lower()
    variations = [raw]
    digits = re.sub(r'[^0-9]', '', raw)
    if digits and len(digits) >= 6:
        if digits not in variations:
            variations.append(digits)
        if len(digits) >= 10 and digits[-10:] not in variations:
            variations.append(digits[-10:])
        with_plus = "+" + digits
        if with_plus not in variations:
            variations.append(with_plus)
    return variations


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password with salt using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        bytes.fromhex(salt),
        100_000
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify password against hash."""
    calculated, _ = hash_password(password, salt)
    return calculated == expected_hash


class AuthModel:
    _get_connection = staticmethod(_get_connection)

    def __init__(self) -> None:
        init_db()

    def generate_otp(self, target: str, target_type: str = "email", expiry_minutes: int = 10, purpose: str = "verification") -> str:
        """Generate a 6-digit numeric OTP for email or phone number and record in SQLite."""
        target = target.strip().lower()
        if not target:
            raise ValueError("Target destination (email or phone) cannot be empty for OTP generation")

        otp_code = f"{random.randint(100000, 999999)}"
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=expiry_minutes)
        otp_id = str(uuid.uuid4())

        with _get_connection() as conn:
            cursor = conn.cursor()
            # Invalidate older unverified OTPs for this target in otps table
            cursor.execute(
                "UPDATE otps SET is_verified = -1 WHERE target = ? AND is_verified = 0",
                (target,)
            )
            cursor.execute(
                """
                INSERT INTO otps (id, target, target_type, otp_code, created_at, expires_at, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (otp_id, target, target_type, otp_code, now.isoformat(), expires.isoformat())
            )
            # Maintain backward compatibility with legacy email_otps table
            if target_type == "email" or "@" in target:
                cursor.execute(
                    "UPDATE email_otps SET is_verified = -1 WHERE email = ? AND is_verified = 0",
                    (target,)
                )
                cursor.execute(
                    """
                    INSERT INTO email_otps (id, email, otp_code, created_at, expires_at, is_verified)
                    VALUES (?, ?, ?, ?, ?, 0)
                    """,
                    (otp_id, target, otp_code, now.isoformat(), expires.isoformat())
                )
            conn.commit()

        # Dispatch OTP and verification link directly to recipient email or phone
        try:
            self.last_dispatch_info = NotificationService.dispatch_otp(target=target, target_type=target_type, otp_code=otp_code, purpose=purpose)
        except Exception as e:
            self.last_dispatch_info = {"success": False, "dispatched_live": False, "error": str(e), "otp_code": otp_code, "target": target}

        return otp_code

    def get_last_dispatch_info(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "last_dispatch_info", None)

    def verify_otp(self, target: str, otp_code: str) -> bool:
        """Verify the 6-digit OTP from SQLite."""
        target = target.strip().lower()
        otp_code = str(otp_code).strip()
        now_str = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cursor = conn.cursor()
            # Check unified otps table first
            cursor.execute(
                """
                SELECT id, expires_at, is_verified FROM otps
                WHERE target = ? AND otp_code = ? AND is_verified = 0
                ORDER BY created_at DESC LIMIT 1
                """,
                (target, otp_code)
            )
            row = cursor.fetchone()
            if row:
                if row["expires_at"] < now_str:
                    return False
                cursor.execute("UPDATE otps SET is_verified = 1 WHERE id = ?", (row["id"],))
                # Also update legacy email_otps if present
                cursor.execute("UPDATE email_otps SET is_verified = 1 WHERE email = ? AND otp_code = ?", (target, otp_code))
                conn.commit()
                return True

            # Fallback to legacy email_otps table
            cursor.execute(
                """
                SELECT id, expires_at, is_verified FROM email_otps
                WHERE email = ? AND otp_code = ? AND is_verified = 0
                ORDER BY created_at DESC LIMIT 1
                """,
                (target, otp_code)
            )
            legacy_row = cursor.fetchone()
            if not legacy_row:
                return False

            if legacy_row["expires_at"] < now_str:
                return False

            cursor.execute("UPDATE email_otps SET is_verified = 1 WHERE id = ?", (legacy_row["id"],))
            conn.commit()
            return True

    def is_email_verified(self, email: str) -> bool:
        return self.is_target_verified(email)

    def is_target_verified(self, target: str) -> bool:
        """Check if target (email or phone) completed OTP verification within last 30 minutes."""
        target = target.strip().lower()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM otps
                WHERE target = ? AND is_verified = 1 AND created_at >= ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (target, cutoff)
            )
            if cursor.fetchone():
                return True

            # Legacy check
            cursor.execute(
                """
                SELECT id FROM email_otps
                WHERE email = ? AND is_verified = 1 AND created_at >= ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (target, cutoff)
            )
            return cursor.fetchone() is not None

    def verify_and_consume_otp(self, target: str, otp_code: str) -> bool:
        """
        Verify the OTP for a target (email or phone) and immediately consume it (is_verified = 2)
        so it cannot be reused. Accepts active OTPs in either is_verified = 0 (unverified)
        or is_verified = 1 (pre-verified via verify_otp endpoint).
        """
        target = target.strip().lower()
        otp_code = str(otp_code).strip()
        if not otp_code:
            return False
        now_str = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cursor = conn.cursor()
            # 1. Check in unified otps table
            cursor.execute(
                """
                SELECT id, expires_at FROM otps
                WHERE target = ? AND otp_code = ? AND is_verified IN (0, 1)
                ORDER BY created_at DESC LIMIT 1
                """,
                (target, otp_code)
            )
            row = cursor.fetchone()
            if row:
                if row["expires_at"] < now_str:
                    return False
                cursor.execute("UPDATE otps SET is_verified = 2 WHERE id = ?", (row["id"],))
                # Also mark legacy if duplicate exists
                cursor.execute("UPDATE email_otps SET is_verified = 2 WHERE email = ? AND otp_code = ?", (target, otp_code))
                conn.commit()
                return True

            # 2. Check legacy email_otps table
            cursor.execute(
                """
                SELECT id, expires_at FROM email_otps
                WHERE email = ? AND otp_code = ? AND is_verified IN (0, 1)
                ORDER BY created_at DESC LIMIT 1
                """,
                (target, otp_code)
            )
            legacy_row = cursor.fetchone()
            if not legacy_row:
                return False

            if legacy_row["expires_at"] < now_str:
                return False

            cursor.execute("UPDATE email_otps SET is_verified = 2 WHERE id = ?", (legacy_row["id"],))
            conn.commit()
            return True

    def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        phone: Optional[str] = None,
        role: str = 'analyst',
        otp_code: Optional[str] = None,
        require_otp: bool = True
    ) -> Dict[str, Any]:
        username = username.strip().lower()
        email = email.strip().lower()
        clean_phone = phone.strip() if phone else None

        if not username or not email or not password:
            raise ValueError('Username, email, and password cannot be empty')

        # Check existing username or email before OTP checking
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                raise ValueError(f'Username "{username}" is already registered.')
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                raise ValueError(f'Email "{email}" is already registered.')
            if clean_phone:
                cursor.execute("SELECT id FROM users WHERE phone = ?", (clean_phone,))
                if cursor.fetchone():
                    raise ValueError(f'Phone number "{clean_phone}" is already registered.')

        if require_otp:
            if not otp_code or not str(otp_code).strip():
                raise ValueError("One-Time Password (OTP) verification code is required.")
            clean_otp = str(otp_code).strip()
            # Verify and consume the OTP directly for either target: email or phone
            verified = self.verify_and_consume_otp(email, clean_otp) or (bool(clean_phone) and self.verify_and_consume_otp(clean_phone, clean_otp))
            if not verified:
                raise ValueError("Invalid or expired One-Time Password (OTP). Please request a new code.")

        pwd_hash, salt = hash_password(password)
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with _get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO users (id, username, email, phone, password_hash, salt, full_name, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, username, email, clean_phone, pwd_hash, salt, full_name or username.title(), role, now)
                )
                conn.commit()
            except (sqlite3.IntegrityError, Exception) as e:
                err_msg = str(e).lower()
                if 'username' in err_msg:
                    raise ValueError(f'Username "{username}" is already registered.')
                elif 'email' in err_msg:
                    raise ValueError(f'Email "{email}" is already registered.')
                elif 'phone' in err_msg:
                    raise ValueError(f'Phone "{clean_phone}" is already registered.')
                raise ValueError('User registration failed: ' + str(e))

        return self.get_user_by_id(user_id)  # type: ignore

    def authenticate(
        self,
        username_or_email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        target = username_or_email.strip().lower()
        now = datetime.now(timezone.utc).isoformat()
        log_id = str(uuid.uuid4())

        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, email, phone, password_hash, salt, full_name, role, created_at, last_login_at, login_count, is_active
                FROM users
                WHERE username = ? OR email = ? OR phone = ?
                """,
                (target, target, target)
            )
            row = cursor.fetchone()

            if not row:
                cursor.execute(
                    """
                    INSERT INTO login_logs (id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (log_id, None, target, now, ip_address, user_agent, 'FAILED', 'User not found')
                )
                conn.commit()
                return None

            user_dict = dict(row)
            if not user_dict['is_active']:
                cursor.execute(
                    """
                    INSERT INTO login_logs (id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (log_id, user_dict['id'], user_dict['username'], now, ip_address, user_agent, 'FAILED', 'Account disabled')
                )
                conn.commit()
                return None

            if not verify_password(password, user_dict['salt'], user_dict['password_hash']):
                cursor.execute(
                    """
                    INSERT INTO login_logs (id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (log_id, user_dict['id'], user_dict['username'], now, ip_address, user_agent, 'FAILED', 'Invalid password')
                )
                conn.commit()
                return None

            # Success
            new_count = (user_dict.get('login_count') or 0) + 1
            cursor.execute(
                """
                UPDATE users
                SET last_login_at = ?, login_count = ?
                WHERE id = ?
                """,
                (now, new_count, user_dict['id'])
            )
            cursor.execute(
                """
                INSERT INTO login_logs (id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, user_dict['id'], user_dict['username'], now, ip_address, user_agent, 'SUCCESS', None)
            )
            conn.commit()

            # Return user object without sensitive hash/salt
            del user_dict['password_hash']
            del user_dict['salt']
            user_dict['last_login_at'] = now
            user_dict['login_count'] = new_count
            return user_dict

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, email, phone, full_name, role, created_at, last_login_at, login_count, is_active
                FROM users WHERE id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_login_logs(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with _get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    """
                    SELECT id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason
                    FROM login_logs WHERE user_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (user_id, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason
                    FROM login_logs
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (limit,)
                )
            return [dict(r) for r in cursor.fetchall()]

    def get_user_count(self) -> int:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as cnt FROM users')
            res = cursor.fetchone()
            if res:
                if isinstance(res, dict):
                    return res.get('cnt') or list(res.values())[0]
                return res[0]
            return 0

    def find_user_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Look up user by email, username, or phone number."""
        target = identifier.strip().lower()
        if not target:
            return None
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, email, phone, full_name, role, created_at, is_active
                FROM users
                WHERE username = ? OR email = ? OR phone = ?
                LIMIT 1
                """,
                (target, target, target)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def initiate_password_reset(self, identifier: str, channel: str = "email") -> Dict[str, Any]:
        """
        Initiate password recovery by verifying user exists and dispatching a 6-digit OTP.
        Supports delivery via email or phone/SMS.
        """
        user = self.find_user_by_identifier(identifier)
        if not user:
            raise ValueError(f"No account found matching '{identifier}'. Please check your spelling or register.")

        if not user.get("is_active", True):
            raise ValueError("This account has been disabled. Please contact your administrator.")

        target_type = "phone" if (channel == "phone" and user.get("phone")) else "email"
        target_dest = (user.get("phone") if target_type == "phone" else user.get("email") or "").strip()

        if not target_dest:
            raise ValueError(f"No valid {target_type} destination found on record for this account.")

        # Generate OTP with purpose='password_reset'
        otp_code = self.generate_otp(target=target_dest, target_type=target_type, purpose="password_reset")
        dispatch_info = self.get_last_dispatch_info() or {}

        return {
            "success": True,
            "user_id": user["id"],
            "username": user["username"],
            "email": user.get("email"),
            "phone": user.get("phone"),
            "target": target_dest,
            "target_type": target_type,
            "masked_target": mask_destination(target_dest, target_type),
            "otp_code": otp_code,
            "dispatched_live": dispatch_info.get("dispatched_live", False),
            "dispatch_info": dispatch_info,
        }

    def reset_password_with_otp(self, identifier: str, otp_code: str, new_password: str) -> bool:
        """
        Verify the OTP and securely update the user's password with PBKDF2 hashing.
        Consumes the OTP so it cannot be reused.
        """
        user = self.find_user_by_identifier(identifier)
        if not user:
            raise ValueError("No account found matching this identifier.")

        if not user.get("is_active", True):
            raise ValueError("This account has been disabled. Please contact your administrator.")

        if not new_password or len(new_password) < 4:
            raise ValueError("New password must be at least 4 characters long.")

        clean_otp = str(otp_code).strip()
        if not clean_otp:
            raise ValueError("One-Time Password (OTP) verification code is required.")

        user_email = (user.get("email") or "").strip().lower()
        user_phone = (user.get("phone") or "").strip()
        user_ident = identifier.strip().lower()

        # Verify and consume OTP against email, phone, or raw identifier
        verified = (
            (bool(user_email) and self.verify_and_consume_otp(user_email, clean_otp))
            or (bool(user_phone) and self.verify_and_consume_otp(user_phone, clean_otp))
            or self.verify_and_consume_otp(user_ident, clean_otp)
        )

        if not verified:
            raise ValueError("Invalid or expired One-Time Password (OTP). Please check the code or request a new one.")

        pwd_hash, salt = hash_password(new_password)
        now = datetime.now(timezone.utc).isoformat()
        log_id = str(uuid.uuid4())

        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users
                SET password_hash = ?, salt = ?
                WHERE id = ?
                """,
                (pwd_hash, salt, user["id"])
            )
            cursor.execute(
                """
                INSERT INTO login_logs (id, user_id, username, timestamp, ip_address, user_agent, status, failure_reason)
                VALUES (?, ?, ?, ?, ?, ?, 'PASSWORD_RESET', 'User reset password via verified OTP')
                """,
                (log_id, user["id"], user["username"], now, "127.0.0.1", "Web-Password-Recovery")
            )
            conn.commit()

        return True

