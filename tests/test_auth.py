"""
Unit tests for Authentication, SQL Database Persistence, and Audit Logging.
"""
import os
import random
import uuid
import pytest
from core.models.auth_model import AuthModel
from core.config import AUTH_DB_PATH

@pytest.fixture
def auth_model():
    return AuthModel()

def test_register_and_authenticate(auth_model):
    uid = str(uuid.uuid4())[:8]
    username = f"test_analyst_{uid}"
    email = f"analyst_{uid}@test.com"
    pwd = "Secr3tPassword!"

    # Generate OTP
    otp = auth_model.generate_otp(email, target_type="email")

    # Registration with OTP
    user = auth_model.register(
        username=username,
        email=email,
        password=pwd,
        full_name="Alex Test",
        role="Lead Analyst",
        otp_code=otp,
        require_otp=True
    )
    assert user is not None
    assert user["username"] == username
    assert user["email"] == email
    assert "password_hash" not in user
    assert "salt" not in user

    # Successful authentication
    auth_res = auth_model.authenticate(
        username_or_email=username,
        password=pwd,
        ip_address="127.0.0.1",
        user_agent="PyTest"
    )
    assert auth_res is not None
    assert auth_res["id"] == user["id"]
    assert auth_res["login_count"] >= 1
    assert auth_res["last_login_at"] is not None

    # Failed authentication with wrong password
    failed_res = auth_model.authenticate(
        username_or_email=username,
        password="WrongPassword123"
    )
    assert failed_res is None

    # Failed authentication with non-existent user
    non_existent = auth_model.authenticate(
        username_or_email="ghost_user",
        password="any"
    )
    assert non_existent is None

    # Verify audit logs in SQLite database
    logs = auth_model.list_login_logs(limit=10)
    assert len(logs) >= 2
    # Check that failed login was logged
    failed_logs = [l for l in logs if l["status"] == "FAILED"]
    assert len(failed_logs) >= 1
    success_logs = [l for l in logs if l["status"] == "SUCCESS"]
    assert len(success_logs) >= 1

def test_duplicate_registration_rejected(auth_model):
    uid = str(uuid.uuid4())[:8]
    username = f"unique_analyst_{uid}"
    email = f"analyst_{uid}@test.com"
    pwd = "Secr3tPassword!"

    otp1 = auth_model.generate_otp(email, target_type="email")
    auth_model.register(username=username, email=email, password=pwd, otp_code=otp1)

    otp2 = auth_model.generate_otp(f"another_{uid}@test.com", target_type="email")
    with pytest.raises(ValueError, match="already registered"):
        auth_model.register(username=username, email=f"another_{uid}@test.com", password=pwd, otp_code=otp2)

    otp3 = auth_model.generate_otp(email, target_type="email")
    with pytest.raises(ValueError, match="already registered"):
        auth_model.register(username=f"another_user_{uid}", email=email, password=pwd, otp_code=otp3)


def test_otp_generation_and_verification(auth_model):
    email = f"otp_test_{uuid.uuid4().hex[:8]}@example.com"
    otp = auth_model.generate_otp(email)
    assert len(otp) == 6
    assert otp.isdigit()

    # Wrong OTP code
    assert auth_model.verify_otp(email, "000000") is False

    # Correct OTP code
    assert auth_model.verify_otp(email, otp) is True

    # Re-use should fail (single-use token)
    assert auth_model.verify_otp(email, otp) is False


def test_register_with_otp_gated(auth_model):
    uid = uuid.uuid4().hex[:8]
    email = f"otp_user_{uid}@example.com"
    username = f"otp_user_{uid}"
    pwd = "StrongPassword99!"

    # Generating OTP
    otp = auth_model.generate_otp(email)

    # Attempting registration with bad OTP
    with pytest.raises(ValueError, match="Invalid or expired One-Time Password"):
        auth_model.register(
            username=username,
            email=email,
            password=pwd,
            otp_code="999999",
            require_otp=True
        )

    # Registration with valid OTP
    user = auth_model.register(
        username=username,
        email=email,
        password=pwd,
        otp_code=otp,
        require_otp=True
    )
    assert user is not None
    assert user["username"] == username
    assert user["email"] == email


def test_phone_otp_and_login(auth_model):
    uid = uuid.uuid4().hex[:8]
    phone = f"+1555{random.randint(100000, 999999)}"
    username = f"phone_user_{uid}"
    email = f"phone_{uid}@test.com"
    pwd = "Secr3tPhonePassword!"

    # 1. Generate OTP for phone
    otp = auth_model.generate_otp(phone, target_type="phone")
    assert len(otp) == 6
    assert otp.isdigit()

    # 2. Register user with phone and OTP
    user = auth_model.register(
        username=username,
        email=email,
        password=pwd,
        phone=phone,
        full_name="Phone Analyst",
        otp_code=otp,
        require_otp=True
    )
    assert user is not None
    assert user["phone"] == phone
    assert user["username"] == username

    # 3. Authenticate using phone number
    auth_user = auth_model.authenticate(
        username_or_email=phone,
        password=pwd,
        ip_address="127.0.0.1",
        user_agent="PyTest"
    )
    assert auth_user is not None
    assert auth_user["id"] == user["id"]
    assert auth_user["phone"] == phone


def test_registration_strictly_requires_valid_otp(auth_model):
    uid = uuid.uuid4().hex[:8]
    email = f"strict_otp_{uid}@test.com"
    username = f"strict_otp_{uid}"
    pwd = "Secr3tPassword!"

    # 1. Registration with no OTP code should fail
    with pytest.raises(ValueError, match="One-Time Password \\(OTP\\) verification code is required"):
        auth_model.register(username=username, email=email, password=pwd, otp_code=None)

    # 2. Registration with empty string OTP should fail
    with pytest.raises(ValueError, match="One-Time Password \\(OTP\\) verification code is required"):
        auth_model.register(username=username, email=email, password=pwd, otp_code="   ")

    # 3. Registration with dummy/wrong OTP should fail
    with pytest.raises(ValueError, match="Invalid or expired One-Time Password"):
        auth_model.register(username=username, email=email, password=pwd, otp_code="000000")

    # 4. Generate valid OTP and register successfully
    otp = auth_model.generate_otp(email, target_type="email")
    user = auth_model.register(username=username, email=email, password=pwd, otp_code=otp)
    assert user is not None
    assert user["email"] == email


def test_otp_cannot_be_reused(auth_model):
    uid = uuid.uuid4().hex[:8]
    email = f"reuse_{uid}@test.com"
    username = f"reuse_{uid}"
    pwd = "Secr3tPassword!"

    otp = auth_model.generate_otp(email, target_type="email")
    # First registration consumes the OTP
    user1 = auth_model.register(username=username, email=email, password=pwd, otp_code=otp)
    assert user1 is not None

    # Attempting to reuse the exact same OTP for another account must fail
    another_email = f"another_reuse_{uid}@test.com"
    another_username = f"another_reuse_{uid}"
    with pytest.raises(ValueError, match="Invalid or expired One-Time Password"):
        auth_model.register(username=another_username, email=another_email, password=pwd, otp_code=otp)


def test_notification_service_dispatch_and_masking():
    from core.services.notification_service import NotificationService, mask_destination

    # 1. Test masking
    assert mask_destination("alex.vance@company.com", "email") == "a***e@company.com"
    assert mask_destination("+15551234567", "phone") == "+155***567"

    # 2. Test email dispatch
    email_res = NotificationService.dispatch_otp("analyst@example.com", "email", "654321")
    assert email_res["success"] is True
    assert "verify_code=654321" in email_res["verification_link"]
    assert email_res["target"] == "analyst@example.com"

    # 3. Test phone dispatch
    phone_res = NotificationService.dispatch_otp("+15559876543", "phone", "123456")
    assert phone_res["success"] is True
    assert "verify_code=123456" in phone_res["verification_link"]
    assert phone_res["target"] == "+15559876543"




