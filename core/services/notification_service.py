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
    RESEND_API_KEY,
    BREVO_API_KEY,
    EMAIL_WEBHOOK_URL,
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
    def build_verification_link(target: str, otp_code: str, mode: str = "verify") -> str:
        """Constructs a direct 1-click authentication verification link."""
        encoded_target = urllib.parse.quote(target.strip())
        mode_param = f"&mode={mode}" if mode != "verify" else ""
        return f"{APP_BASE_URL}/?verify_target={encoded_target}&verify_code={otp_code}{mode_param}"

    @classmethod
    def send_email_otp(cls, to_email: str, otp_code: str, purpose: str = "verification") -> Dict[str, Any]:
        """Dispatches an email containing the 6-digit OTP and 1-click verification link."""
        to_email = to_email.strip().lower()
        is_reset = (purpose == "password_reset")
        verification_link = cls.build_verification_link(to_email, otp_code, mode="reset" if is_reset else "verify")
        
        if is_reset:
            subject = f"Password Reset Verification Code: {otp_code} | Quantum Analyst Platform"
            badge_text = "// PASSWORD RECOVERY CLEARANCE"
            action_title = "Password Reset Verification"
            action_desc = "We received a request to reset your password. Use the following 6-digit code to complete verification and set your new password:"
            button_label = "🔗 Click to Verify & Reset Password"
            footer_note = f"This password recovery request was initiated for <b>{to_email}</b>. If you did not initiate this, please secure your account immediately."
        else:
            subject = f"Your Security Verification Code: {otp_code} | Quantum Analyst Platform"
            badge_text = "// INSTITUTIONAL CLEARANCE SYSTEM"
            action_title = "One-Time Password (OTP) Verification"
            action_desc = "Use the following 6-digit code to complete your user account registration:"
            button_label = "🔗 Click to Verify & Complete Registration"
            footer_note = f"This message was sent directly to <b>{to_email}</b>. If you did not initiate this request, no action is needed."

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
            <span class="badge">{badge_text}</span>
            <h1>{action_title}</h1>
            <p>{action_desc}</p>
            
            <div class="code-box">
              <div class="otp-code">{otp_code}</div>
              <div style="font-size: 12px; color: #64748B; margin-top: 6px;">Expires in 10 minutes</div>
            </div>

            <p>Or click the button below to verify your destination directly:</p>
            <a href="{verification_link}" class="btn" target="_blank">{button_label}</a>

            <div class="footer">
              {footer_note}
            </div>
          </div>
        </body>
        </html>
        """

        dispatched_live = False
        dispatch_error = None

        from dotenv import load_dotenv
        load_dotenv(override=True)
        from core.config import (
            SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM, SMTP_USE_TLS,
            RESEND_API_KEY, BREVO_API_KEY, EMAIL_WEBHOOK_URL, DEFAULT_EMAIL_RELAY
        )

        resend_key = os.getenv("RESEND_API_KEY") or RESEND_API_KEY
        brevo_key = os.getenv("BREVO_API_KEY") or BREVO_API_KEY
        webhook_url = os.getenv("EMAIL_WEBHOOK_URL") or DEFAULT_EMAIL_RELAY or EMAIL_WEBHOOK_URL

        # -------------------------------------------------------------
        # 1. ATTEMPT HTTPS EMAIL APIs (Bypasses Render Free Tier SMTP block)
        # -------------------------------------------------------------
        if resend_key and not dispatched_live:
            try:
                import urllib.request
                payload = json.dumps({
                    "from": "Quantum Security <onboarding@resend.dev>",
                    "to": [to_email],
                    "subject": subject,
                    "html": html_content
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.resend.com/emails",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {resend_key.strip()}",
                        "Content-Type": "application/json"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=4) as resp:
                    if resp.status in (200, 201):
                        dispatched_live = True
                        dispatch_error = None
                        logger.info(f"Email delivered via Resend HTTPS API to {to_email}")
            except Exception as e:
                dispatch_error = f"Resend API error: {str(e)}"
                logger.warning(dispatch_error)

        if brevo_key and not dispatched_live:
            try:
                import urllib.request
                payload = json.dumps({
                    "sender": {"name": "Quantum Analyst", "email": "chintukumaredu00@gmail.com"},
                    "to": [{"email": to_email}],
                    "subject": subject,
                    "htmlContent": html_content
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.brevo.com/v3/smtp/email",
                    data=payload,
                    headers={
                        "api-key": brevo_key.strip(),
                        "Content-Type": "application/json"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=4) as resp:
                    if resp.status in (200, 201):
                        dispatched_live = True
                        dispatch_error = None
                        logger.info(f"Email delivered via Brevo HTTPS API to {to_email}")
            except Exception as e:
                dispatch_error = f"Brevo API error: {str(e)}"
                logger.warning(dispatch_error)

        if webhook_url and not dispatched_live:
            try:
                import urllib.request
                payload = json.dumps({
                    "to": to_email,
                    "subject": subject,
                    "text": text_content,
                    "html": html_content,
                    "otp_code": otp_code,
                    "verification_link": verification_link
                }).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url.strip(),
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    if resp.status in (200, 201, 204):
                        dispatched_live = True
                        dispatch_error = None
                        logger.info(f"Email delivered via Webhook to {to_email}")
            except Exception as e:
                dispatch_error = f"Webhook error: {str(e)}"
                logger.warning(dispatch_error)

        # -------------------------------------------------------------
        # 2. ATTEMPT DIRECT SMTP (Port 465 SSL & Port 587 TLS)
        # -------------------------------------------------------------
        smtp_server = os.getenv("SMTP_SERVER") or SMTP_SERVER
        smtp_port = int(os.getenv("SMTP_PORT") or SMTP_PORT or 587)
        smtp_user = os.getenv("SMTP_USERNAME") or SMTP_USERNAME
        smtp_pass = os.getenv("SMTP_PASSWORD") or SMTP_PASSWORD
        smtp_from = os.getenv("SMTP_FROM") or SMTP_FROM or smtp_user
        smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

        if not dispatched_live and smtp_server and smtp_user and smtp_pass:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_from
            msg["To"] = to_email
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Order ports: try 465 SSL, then 587 TLS with rapid 3.5s timeout
            ports_to_try = [smtp_port]
            for p in (465, 587):
                if p not in ports_to_try:
                    ports_to_try.append(p)

            for port in ports_to_try:
                try:
                    if port == 465:
                        with smtplib.SMTP_SSL(smtp_server, 465, timeout=3.5) as server:
                            server.login(smtp_user, smtp_pass)
                            server.send_message(msg)
                    else:
                        with smtplib.SMTP(smtp_server, port, timeout=3.5) as server:
                            if smtp_use_tls:
                                server.starttls()
                            server.login(smtp_user, smtp_pass)
                            server.send_message(msg)
                    dispatched_live = True
                    dispatch_error = None
                    logger.info(f"Email OTP successfully delivered via SMTP port {port} to {to_email}")
                    break
                except Exception as e:
                    dispatch_error = f"SMTP (port {port}): {str(e)}"
                    logger.warning(f"SMTP port {port} failed: {e}")

        # If SMTP failed due to timeout/firewall, annotate clearly for cloud deployments
        if not dispatched_live and dispatch_error and ("timed out" in dispatch_error.lower() or "110" in dispatch_error or "refused" in dispatch_error.lower()):
            dispatch_error = "Render Free Tier firewall blocked raw outbound SMTP ports (587/465). Use on-screen clearance code or HTTPS Email API."

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
    def send_sms_otp(cls, to_phone: str, otp_code: str, purpose: str = "verification") -> Dict[str, Any]:
        """Dispatches an SMS containing the 6-digit OTP and 1-click verification link."""
        to_phone = to_phone.strip()
        is_reset = (purpose == "password_reset")
        verification_link = cls.build_verification_link(to_phone, otp_code, mode="reset" if is_reset else "verify")
        if is_reset:
            sms_text = f"Quantum Security: Your password reset verification code is: {otp_code}. Valid for 10 min. Reset link: {verification_link}"
        else:
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
    def dispatch_otp(cls, target: str, target_type: str, otp_code: str, purpose: str = "verification") -> Dict[str, Any]:
        """Unified entry point to dispatch OTP directly to phone or email."""
        if target_type == "phone" or (target.startswith("+") or target.replace("-", "").isdigit()):
            return cls.send_sms_otp(target, otp_code, purpose=purpose)
        return cls.send_email_otp(target, otp_code, purpose=purpose)

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

    @classmethod
    def configure_https_api(cls, provider: str, key_or_url: str) -> Dict[str, Any]:
        """Test and save HTTPS Email API credentials (Resend, Brevo, Webhook)."""
        from core import config
        clean_val = key_or_url.strip()
        if not clean_val:
            return {"success": False, "error": "API Key or Webhook URL cannot be empty."}

        env_key = "RESEND_API_KEY" if provider.lower() == "resend" else ("BREVO_API_KEY" if provider.lower() == "brevo" else "EMAIL_WEBHOOK_URL")

        if provider.lower() == "resend":
            config.RESEND_API_KEY = clean_val
        elif provider.lower() == "brevo":
            config.BREVO_API_KEY = clean_val
        else:
            config.EMAIL_WEBHOOK_URL = clean_val

        os.environ[env_key] = clean_val

        env_path = config.BASE_DIR / ".env"
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        updated = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{env_key}="):
                new_lines.append(f"{env_key}={clean_val}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{env_key}={clean_val}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return {"success": True, "message": f"{provider} HTTPS Email integration configured and active!"}

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
