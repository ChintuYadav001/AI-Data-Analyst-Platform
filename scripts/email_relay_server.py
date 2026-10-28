import os
import sys
import json
import logging
import smtplib
from http.server import HTTPServer, BaseHTTPRequestHandler
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("email_relay")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "chintukumaredu00@gmail.com"
SMTP_PASS = "bcrskjjmnkgfyqol"

class EmailRelayHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/dispatch_email":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)
                to_email = data.get("to")
                subject = data.get("subject", "Quantum Security Verification Code")
                html_content = data.get("html", "")
                text_content = data.get("text", "")

                if not to_email:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "missing recipient email"}')
                    return

                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = SMTP_USER
                msg["To"] = to_email
                if text_content:
                    msg.attach(MIMEText(text_content, "plain"))
                if html_content:
                    msg.attach(MIMEText(html_content, "html"))

                with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
                    s.login(SMTP_USER, SMTP_PASS)
                    s.send_message(msg)

                logger.info(f"Relayed email OTP directly via Gmail SMTP to {to_email}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"success": true, "message": "Email delivered to inbox"}')
            except Exception as e:
                logger.error(f"Failed to relay email: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "online", "service": "Quantum Email Relay"}')

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8002), EmailRelayHandler)
    logger.info("Quantum Email Relay listening on http://0.0.0.0:8002")
    server.serve_forever()
