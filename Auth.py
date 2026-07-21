"""
auth.py
-------
Real per-user signup/login for RepoMind: username + password accounts,
securely hashed (never stored in plaintext), persisted to a local JSON file.

SECURITY NOTES (being upfront about what this is and isn't):
  - Passwords are hashed with PBKDF2-HMAC-SHA256 (100,000 iterations) + a
    unique random salt per user, using only Python's built-in `hashlib` --
    no new dependency needed, and no plaintext password ever touches disk.
  - This is a genuine, correctly-implemented hashing scheme -- not a toy.
  - Login attempts are rate-limited per username (see below): after too
    many wrong passwords in a row, that username is locked out for a
    cooldown period. This blocks simple brute-force password guessing
    against a known username.
  - What this is NOT: production-grade infrastructure. There's no email
    verification or password reset flow, and the user store is a local
    JSON file (fine for a portfolio demo / single small deployment; a real
    production system would use a proper database and additional
    hardening, e.g. IP-based limiting in front of the app too).
  - Account scope: accounts control WHO can use the app, and each user's
    indexed repo data lives in its own ChromaDB collection (see
    ingest.py's get_user_collection_name) -- genuine per-user isolation,
    not just a login screen in front of shared data.

INPUT:  username + password (signup, login, or change_password)
OUTPUT: (success: bool, message: str) for signup, login, and change_password
"""

import os
import json
import time
import hashlib
import secrets

USERS_FILE = "./sessions/users.json"
ATTEMPTS_FILE = "./sessions/login_attempts.json"
PBKDF2_ITERATIONS = 100_000

# Rate limiting: after MAX_FAILED_ATTEMPTS wrong passwords in a row for a
# given username, that username is locked out for LOCKOUT_SECONDS. The
# counter resets to 0 on any successful login.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 5 * 60


def _load_users() -> dict:
    """Returns the user store dict, or {} on any error (missing file,
    corrupted JSON, etc.) -- never crashes the app over a storage issue."""
    try:
        if not os.path.exists(USERS_FILE):
            return {}
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f)
        return True
    except Exception:
        return False


def _load_attempts() -> dict:
    """Returns the login-attempts tracking dict, or {} on any error. Keyed
    by lowercased username -> {"count": int, "locked_until": float epoch}."""
    try:
        if not os.path.exists(ATTEMPTS_FILE):
            return {}
        with open(ATTEMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_attempts(attempts: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(ATTEMPTS_FILE), exist_ok=True)
        with open(ATTEMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(attempts, f)
        return True
    except Exception:
        return False


def _seconds_locked_remaining(username: str) -> float:
    """Returns how many seconds are left on this username's lockout, or 0
    if it isn't currently locked. Keyed by lowercased username so lockouts
    can't be dodged by varying letter case."""
    attempts = _load_attempts()
    record = attempts.get(username.lower())
    if not record:
        return 0
    remaining = record.get("locked_until", 0) - time.time()
    return max(0, remaining)


def _register_failed_attempt(username: str) -> None:
    """Increments the failed-attempt counter for a username and locks it
    out once MAX_FAILED_ATTEMPTS is reached. Tracked even for usernames
    that don't exist, so an attacker can't use "does it lock out?" as a
    signal for whether an account exists."""
    key = username.lower()
    attempts = _load_attempts()
    record = attempts.get(key, {"count": 0, "locked_until": 0})
    record["count"] = record.get("count", 0) + 1
    if record["count"] >= MAX_FAILED_ATTEMPTS:
        record["locked_until"] = time.time() + LOCKOUT_SECONDS
        record["count"] = 0
    attempts[key] = record
    _save_attempts(attempts)


def _clear_failed_attempts(username: str) -> None:
    """Resets a username's failed-attempt counter after a successful login."""
    key = username.lower()
    attempts = _load_attempts()
    if key in attempts:
        del attempts[key]
        _save_attempts(attempts)


def _hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Returns (hash_hex, salt_hex). Generates a new random salt if none is
    given (signup); pass the stored salt back in to verify a login attempt."""
    if salt is None:
        salt = secrets.token_hex(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return hash_bytes.hex(), salt


def _valid_username(username: str) -> bool:
    username = username.strip()
    return 3 <= len(username) <= 30 and username.replace("_", "").replace("-", "").isalnum()


def signup(username: str, password: str) -> tuple[bool, str]:
    """Creates a new account. Returns (success, message) -- message is
    always safe to show directly to the user (no internal detail leaks)."""
    username = username.strip()
    if not _valid_username(username):
        return False, "Username must be 3-30 characters (letters, numbers, _ or - only)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username.lower() in {u.lower() for u in users}:
        return False, "That username is already taken."

    password_hash, salt = _hash_password(password)
    users[username] = {"password_hash": password_hash, "salt": salt}
    if not _save_users(users):
        return False, "Couldn't create the account right now — please try again."
    return True, "Account created! You can log in now."


def login(username: str, password: str) -> tuple[bool, str]:
    """Verifies credentials. Returns (success, message). Deliberately uses
    the same generic error for 'no such user' and 'wrong password' -- never
    reveal which one it was, since that helps an attacker enumerate valid
    usernames.

    Rate-limited: a username that's racked up MAX_FAILED_ATTEMPTS wrong
    passwords in a row is locked out for LOCKOUT_SECONDS, checked BEFORE
    the password is even verified -- so a locked-out attacker can't keep
    guessing during the cooldown."""
    username = username.strip()

    locked_seconds = _seconds_locked_remaining(username)
    if locked_seconds > 0:
        wait_min = max(1, round(locked_seconds / 60))
        return False, f"Too many failed attempts. Try again in about {wait_min} minute(s)."

    users = _load_users()

    user_record = None
    for stored_username, record in users.items():
        if stored_username.lower() == username.lower():
            user_record = record
            username = stored_username  # use the stored casing
            break

    if not user_record:
        _register_failed_attempt(username)
        return False, "Incorrect username or password."

    check_hash, _ = _hash_password(password, salt=user_record["salt"])
    if not secrets.compare_digest(check_hash, user_record["password_hash"]):
        _register_failed_attempt(username)
        return False, "Incorrect username or password."

    _clear_failed_attempts(username)
    return True, f"Welcome back, {username}!"


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """Changes a logged-in user's password. Requires the current password
    (not just the fact that they're logged in) so someone who walks up to
    an unlocked, still-logged-in session can't silently lock the real
    owner out by resetting it. Returns (success, message)."""
    username = username.strip()
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    users = _load_users()
    stored_key = None
    for stored_username in users:
        if stored_username.lower() == username.lower():
            stored_key = stored_username
            break

    if not stored_key:
        return False, "Account not found."

    user_record = users[stored_key]
    check_hash, _ = _hash_password(old_password, salt=user_record["salt"])
    if not secrets.compare_digest(check_hash, user_record["password_hash"]):
        return False, "Current password is incorrect."

    if secrets.compare_digest(old_password, new_password):
        return False, "New password must be different from the current one."

    new_hash, new_salt = _hash_password(new_password)
    users[stored_key] = {"password_hash": new_hash, "salt": new_salt}
    if not _save_users(users):
        return False, "Couldn't update the password right now — please try again."
    return True, "Password updated."