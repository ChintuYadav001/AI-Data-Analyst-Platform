"""
Unit tests for Password Reset via OTP Verification.
"""
import uuid
import pytest
from core.models.auth_model import AuthModel

@pytest.fixture
def auth_model():
    return AuthModel()

def test_password_reset_flow_with_otp(auth_model):
    uid = str(uuid.uuid4())[:8]
    username = f"reset_user_{uid}"
    email = f"reset_{uid}@test.com"
    old_pwd = "OriginalPassword123!"
    new_pwd = "NewSecurePassword2026!"

    # 1. Register user
    reg_otp = auth_model.generate_otp(email, target_type="email")
    user = auth_model.register(
        username=username,
        email=email,
        password=old_pwd,
        otp_code=reg_otp,
        require_otp=True
    )
    assert user is not None

    # 2. Verify login works with old password
    auth1 = auth_model.authenticate(username, old_pwd)
    assert auth1 is not None

    # 3. Non-existent user cannot initiate reset
    with pytest.raises(ValueError, match="No account found"):
        auth_model.initiate_password_reset("non_existent_user_999@test.com")

    # 4. Initiate password reset for valid user
    reset_info = auth_model.initiate_password_reset(username)
    assert reset_info["success"] is True
    assert reset_info["email"] == email
    otp = reset_info["otp_code"]
    assert len(otp) == 6

    # 5. Reset fails with wrong OTP
    with pytest.raises(ValueError, match="Invalid or expired One-Time Password"):
        auth_model.reset_password_with_otp(username, "000000", new_pwd)

    # 6. Reset fails with too short password
    with pytest.raises(ValueError, match="at least 4 characters"):
        auth_model.reset_password_with_otp(username, otp, "12")

    # 7. Successful reset with valid OTP
    assert auth_model.reset_password_with_otp(username, otp, new_pwd) is True

    # 8. Old password must now fail
    failed_auth = auth_model.authenticate(username, old_pwd)
    assert failed_auth is None

    # 9. New password succeeds
    success_auth = auth_model.authenticate(username, new_pwd)
    assert success_auth is not None
    assert success_auth["id"] == user["id"]

    # 10. OTP cannot be reused to change password again
    with pytest.raises(ValueError, match="Invalid or expired One-Time Password"):
        auth_model.reset_password_with_otp(username, otp, "YetAnotherPassword999!")
