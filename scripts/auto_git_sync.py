import os
import sys
import time
import subprocess
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(REPO_DIR, "data", "git_sync.log")

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def run_git(args):
    result = subprocess.run(
        ["git"] + args,
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def has_changes():
    code, stdout, stderr = run_git(["status", "--porcelain"])
    if code != 0:
        return False
    lines = [l for l in stdout.splitlines() if l.strip()]
    return len(lines) > 0

def sync_once(commit_msg=None):
    if not has_changes():
        return False, "Working directory clean; nothing to sync."
    
    log("Detected modified or new files. Starting auto-sync...")
    
    # 1. Stage all changes
    code, stdout, stderr = run_git(["add", "-A"])
    if code != 0:
        log(f"Error staging changes: {stderr}")
        return False, f"Git add failed: {stderr}"
    
    code, stdout, stderr = run_git(["diff", "--cached", "--name-only"])
    if not stdout.strip():
        log("No staged changes found.")
        return False, "No staged changes to commit."
    
    staged_files = stdout.splitlines()
    file_summary = ", ".join([os.path.basename(f) for f in staged_files[:4]])
    if len(staged_files) > 4:
        file_summary += f" and {len(staged_files)-4} more"

    if not commit_msg:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"auto: sync updates ({file_summary}) at {ts}"

    # 2. Commit
    code, stdout, stderr = run_git(["commit", "-m", commit_msg])
    if code != 0:
        log(f"Error committing: {stderr}")
        return False, f"Git commit failed: {stderr}"
    
    log(f"Committed changes: {commit_msg}")

    # 3. Push to origin main
    code, stdout, stderr = run_git(["push", "origin", "main"])
    if code != 0:
        log(f"Push to main failed ({stderr}), attempting push to origin master...")
        code_master, _, stderr_master = run_git(["push", "origin", "master"])
        if code_master != 0:
            log(f"Git push failed: {stderr_master}")
            return False, f"Git push failed: {stderr}"
    
    log("✓ Successfully pushed updates to GitHub (ChintuYadav001/AI-Data-Analyst-Platform)!")
    return True, "Successfully synced changes to GitHub."

def run_daemon(poll_interval_seconds=15):
    log(f"Starting GitHub Auto-Sync daemon (polling every {poll_interval_seconds}s)...")
    log("Monitoring workspace: " + REPO_DIR)
    
    while True:
        try:
            if has_changes():
                time.sleep(5)
                success, msg = sync_once()
                if success:
                    log(msg)
            time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            log("Auto-Sync daemon stopped by user.")
            break
        except Exception as e:
            log(f"Unexpected error in sync daemon: {e}")
            time.sleep(poll_interval_seconds)

if __name__ == "__main__":
    if "--once" in sys.argv:
        success, msg = sync_once()
        print(msg)
        sys.exit(0 if success else 1)
    else:
        interval = 15
        for arg in sys.argv[1:]:
            if arg.isdigit():
                interval = int(arg)
        run_daemon(poll_interval_seconds=interval)
