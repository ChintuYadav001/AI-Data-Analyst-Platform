"""
scripts/render_keep_alive.py
===========================
Automated Keep-Alive Ping Service for Render Cloud Deployment.
Pings the Render web service URL every 10 minutes to prevent the free-tier container
from spinning down to sleep, ensuring instantaneous sub-second load times for users.
"""

import time
import sys
import logging
from datetime import datetime
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DEFAULT_URL = "https://ai-data-analyst-platform-7rhs.onrender.com"
PING_INTERVAL_SECONDS = 600  # 10 minutes (Render spins down at 15 minutes)


def ping_service(url: str = DEFAULT_URL) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "QuantumAnalyst-KeepAlive/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            logging.info(f"Ping successful -> {url} (HTTP {status})")
            return status < 400
    except urllib.error.HTTPError as e:
        # HTTP 404 or other HTTP code still proves the server or routing layer answered!
        logging.info(f"Ping reached host -> {url} (HTTP {e.code})")
        return True
    except Exception as e:
        logging.warning(f"Ping failed -> {url}: {e}")
        return False


def run_loop(url: str = DEFAULT_URL, interval: int = PING_INTERVAL_SECONDS):
    logging.info(f"🚀 Render Keep-Alive service started. Target: {url} | Interval: {interval}s")
    while True:
        ping_service(url)
        time.sleep(interval)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    run_loop(target)
