"""
Notification & Dispatch Service for Multi-Channel Security Clearances.
Sends OTP codes and one-click verification links directly to User Email or Mobile/SMS only.
"""
import os
import json
import logging
import smtplib
import urllib.parse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from core.config import (
    SMTP_SERVER,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SMTP_FROM,
    SMTP_USE_TLS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    APP_BASE_URL,
    DATA_DIR
)

logger = logging.getLogger("notification_service")

OUTBOX_LOG_PATH = DATA_DIR / "outbox.log"


def mask_destination(target: str, target_type: str = "email") -> str:
    """Mask target for secure display (e.g., a***e@domain.com or +1555***4567)."""
    target = target.strip()
    if target_type == "email" or "@" in target:
        try:
            user_part, domain = target.split("@", 1)
            if len(user_part) <= 2:
                masked_user = user_part[0] + "***"
            else:
                masked_user = user_part[0] + "***" + user_part[-1]
            return f"{masked_user}@{domain}"
        except Exception:
            return target[:3] + "***"
    else:
        # Phone
        if len(target) > 6:
            return target[:4] + "***" + target[-3:]
        return target[:2] + "***"


class NotificationService:
    @staticmethod
    def build_verification_link(target: str, otp_code: str) -> str:
        """Constructs a direct 1-click authentication verification link."""
        encoded_target = urllib.parse.quote(target.strip())
        return f"{APP_BASE_URL}/?verify_target={encoded_target}&verify_code={otp_code}"

    @classmethod
    def send_email_otp(cls, to_email: str, otp_code: str) -> Dict[str, Any]:
        """Dispatches an email containing the 6-digit OTP and 1-click verification link."""
        to_email = to_email.strip().lower()
        verification_link = cls.build_verification_link(to_email, otp_code)
        subject = f"Your Security Verification Code: {otp_code} | Quantum Analyst Platform"

        text_content = (
            f"Institutional Security Clearance & Verification\n\n"
            f"Your 6-Digit Verification Code: {otp_code}\n\n"
            f"Alternatively, click the verification link below to verify instantly:\n"
            f"{verification_link}\n\n"
            f"This code will expire in 10 minutes.\n"
            f"If you did not request this verification, please contact your security administrator."
        )

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0B0F19; color: #F8FAFC; margin: 0; padding: 24px; }}
            .container {{ max-width: 540px; margin: 0 auto; background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            .badge {{ display: inline-block; font-family: monospace; font-size: 11px; letter-spacing: 0.1em; color: #00F0FF; background: rgba(0,240,255,0.1); border: 1px solid rgba(0,240,255,0.3); padding: 4px 8px; border-radius: 4px; }}
            h1 {{ font-size: 20px; font-weight: 700; margin: 16px 0 8px 0; color: #FFFFFF; }}
            p {{ font-size: 14px; line-height: 1.6; color: #94A3B8; margin: 8px 0; }}
            .code-box {{ background: #030712; border: 1px solid #00F0FF; border-radius: 8px; padding: 16px; text-align: center; margin: 24px 0; }}
            .otp-code {{ font-family: 'Courier New', monospace; font-size: 32px; font-weight: 800; letter-spacing: 0.25em; color: #00F0FF; }}
            .btn {{ display: block; text-align: center; background: linear-gradient(135deg, #00F0FF, #0077FE); color: #000000; font-weight: 700; text-decoration: none; padding: 12px 24px; border-radius: 6px; margin: 20px 0; font-size: 14px; }}
            .footer {{ font-size: 11px; color: #64748B; border-top: 1px solid #1E293B; padding-top: 16px; margin-top: 24px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <span class="badge">// INSTITUTIONAL CLEARANCE SYSTEM</span>
            <h1>One-Time Password (OTP) Verification</h1>
            <p>Use the following 6-digit code to complete your user account registration:</p>
            
            <div class="code-box">
              <div class="otp-code">{otp_code}</div>
              <div style="font-size: 12px; color: #64748B; margin-top: 6px;">Expires in 10 minutes</div>
            </div>

            <p>Or click the button below to verify your destination directly:</p>
            <a href="{verification_link}" class="btn" target="_blank">🔗 Click to Verify & Complete Registration</a>

            <div class="footer">
              This message was sent directly to <b>{to_email}</b>. If you did not initiate this request, no action is needed.
            </div>
          </div>
        </body>
        </html>
        """

        dispatched_live = False
        dispatch_error = None

        from dotenv import load_dotenv
        load_dotenv(override=True)
        smtp_server = os.getenv("SMTP_SERVER", "")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USERNAME", "")
        smtp_pass = os.getenv("SMTP_PASSWORD", "")
        smtp_from = os.getenv("SMTP_FROM", smtp_user or "no-reply@analytics.platform")
        smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

        if smtp_server and smtp_user and smtp_pass:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = smtp_from
                msg["To"] = to_email
                msg.attach(MIMEText(text_content, "plain"))
                msg.attach(MIMEText(html_content, "html"))

                with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                    if smtp_use_tls:
                        server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                dispatched_live = True
                logger.info(f"Email OTP successfully delivered via SMTP to {to_email}")
            except Exception as e:
                dispatch_error = str(e)
                logger.error(f"SMTP dispatch error to {to_email}: {e}")

        # Record in secure server outbox
        cls._record_outbox(
            target=to_email,
            target_type="email",
            otp_code=otp_code,
            verification_link=verification_link,
            dispatched_live=dispatched_live,
            error=dispatch_error
        )

        return {
            "success": True,
            "target": to_email,
            "target_type": "email",
            "masked_target": mask_destination(to_email, "email"),
            "dispatched_live": dispatched_live,
            "verification_link": verification_link,
            "otp_code": otp_code,
            "error": dispatch_error
        }

    @classmethod
    def send_sms_otp(cls, to_phone: str, otp_code: str) -> Dict[str, Any]:
        """Dispatches an SMS containing the 6-digit OTP and 1-click verification link."""
        to_phone = to_phone.strip()
        verification_link = cls.build_verification_link(to_phone, otp_code)
        sms_text = f"Your Quantum Analyst verification code is: {otp_code}. Valid for 10 min. Verify link: {verification_link}"

        dispatched_live = False
        dispatch_error = None

        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
            try:
                import urllib.request
                import base64

                twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
                auth_str = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode("utf-8")
                b64_auth = base64.b64encode(auth_str).decode("ascii")

                payload_data = urllib.parse.urlencode({
                    "From": TWILIO_PHONE_NUMBER,
                    "To": to_phone,
                    "Body": sms_text
                }).encode("utf-8")

                req = urllib.request.Request(twilio_url, data=payload_data, method="POST")
                req.add_header("Authorization", f"Basic {b64_auth}")
                req.add_header("Content-Type", "application/x-www-form-urlencoded")

                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 201):
                        dispatched_live = True
                        logger.info(f"SMS successfully dispatched via Twilio to {to_phone}")
            except Exception as e:
                dispatch_error = str(e)
                logger.error(f"Twilio SMS dispatch error to {to_phone}: {e}")

        # Record in secure server outbox
        cls._record_outbox(
            target=to_phone,
            target_type="phone",
            otp_code=otp_code,
            verification_link=verification_link,
            dispatched_live=dispatched_live,
            error=dispatch_error
        )

        return {
            "success": True,
            "target": to_phone,
            "target_type": "phone",
            "masked_target": mask_destination(to_phone, "phone"),
            "dispatched_live": dispatched_live,
            "verification_link": verification_link,
            "otp_code": otp_code,
            "error": dispatch_error
        }

    @classmethod
    def dispatch_otp(cls, target: str, target_type: str, otp_code: str) -> Dict[str, Any]:
        """Unified entry point to dispatch OTP directly to phone or email."""
        if target_type == "phone" or (target.startswith("+") or target.replace("-", "").isdigit()):
            return cls.send_sms_otp(target, otp_code)
        return cls.send_email_otp(target, otp_code)

    @classmethod
    def is_smtp_configured(cls) -> bool:
        """Checks whether live SMTP credentials are set."""
        from core import config
        return bool(config.SMTP_SERVER and config.SMTP_USERNAME and config.SMTP_PASSWORD)

    @classmethod
    def is_sms_configured(cls) -> bool:
        """Checks whether Twilio credentials are set."""
        from core import config
        return bool(config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN and config.TWILIO_PHONE_NUMBER)

    @classmethod
    def configure_smtp(
        cls,
        server: str,
        port: int,
        username: str,
        password: str,
        from_email: Optional[str] = None,
        use_tls: bool = True
    ) -> Dict[str, Any]:
        """Test and save SMTP configuration into runtime config and local .env."""
        from core import config
        import smtplib

        # 1. Test live connection
        try:
            with smtplib.SMTP(server, port, timeout=10) as s:
                if use_tls:
                    s.starttls()
                s.login(username, password)
        except Exception as e:
            return {"success": False, "error": f"SMTP Connection Failed: {str(e)}"}

        # 2. Update runtime config
        config.SMTP_SERVER = server
        config.SMTP_PORT = port
        config.SMTP_USERNAME = username
        config.SMTP_PASSWORD = password
        config.SMTP_FROM = from_email or username
        config.SMTP_USE_TLS = use_tls

        # 3. Persist to .env
        env_path = config.BASE_DIR / ".env"
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        keys_to_update = {
            "SMTP_SERVER": server,
            "SMTP_PORT": str(port),
            "SMTP_USERNAME": username,
            "SMTP_PASSWORD": password,
            "SMTP_FROM": from_email or username,
            "SMTP_USE_TLS": "true" if use_tls else "false"
        }

        updated_keys = set()
        new_lines = []
        for line in lines:
            if "=" in line:
                k = line.split("=", 1)[0].strip()
                if k in keys_to_update:
                    new_lines.append(f"{k}={keys_to_update[k]}")
                    updated_keys.add(k)
                    continue
            new_lines.append(line)

        for k, v in keys_to_update.items():
            if k not in updated_keys:
                new_lines.append(f"{k}={v}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return {"success": True, "message": "SMTP credentials verified and saved successfully."}

    @staticmethod
    def _record_outbox(
        target: str,
        target_type: str,
        otp_code: str,
        verification_link: str,
        dispatched_live: bool,
        error: Optional[str] = None
    ) -> None:
        """Persists dispatch log strictly to server storage."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "target_type": target_type,
            "masked_target": mask_destination(target, target_type),
            "otp_code": otp_code,
            "verification_link": verification_link,
            "dispatched_live": dispatched_live,
            "error": error
        }
        try:
            with open(OUTBOX_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Could not append to outbox log: {e}")
