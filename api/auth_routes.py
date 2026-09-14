"""
Authentication API Router.
Handles user registration, login, profile info, and audit ledger logs.
"""
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from core.models.auth_model import AuthModel
from core.services.notification_service import mask_destination
from api.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    LoginLogResponse,
    SendOTPRequest,
    VerifyOTPRequest
)

router = APIRouter(prefix="/api/auth", tags=["auth"])
auth_model = AuthModel()


@router.post("/otp/send")
def send_otp(payload: SendOTPRequest):
    target = payload.target or payload.email or payload.phone
    if not target:
        raise HTTPException(status_code=400, detail="Either email, phone, or target destination is required.")
    
    target_type = "phone" if (payload.phone or (not "@" in target and any(c.isdigit() for c in target))) else "email"
    try:
        otp_code = auth_model.generate_otp(target, target_type=target_type)
        resp = {
            "success": True,
            "target": target,
            "target_type": target_type,
            "message": f"Verification OTP & link dispatched directly to your {target_type} ({mask_destination(target, target_type)}) only.",
            "expires_in_minutes": 10
        }
        if os.getenv("ENABLE_TEST_OTP_PREVIEW", "").lower() in ("true", "1", "yes"):
            resp["demo_otp_preview"] = otp_code
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/otp/verify")
def verify_otp(payload: VerifyOTPRequest):
    target = payload.target or payload.email or payload.phone
    if not target:
        raise HTTPException(status_code=400, detail="Target destination (email or phone) is required.")

    valid = auth_model.verify_otp(target, payload.otp_code)
    if not valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP code.")
    return {
        "success": True,
        "target": target,
        "message": f"Successfully verified {target}."
    }


@router.post("/signup", response_model=UserResponse)
def signup(payload: UserRegisterRequest):
    if not payload.otp_code or not payload.otp_code.strip():
        raise HTTPException(status_code=400, detail="One-Time Password (OTP) verification code is required.")
    try:
        user = auth_model.register(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            phone=payload.phone,
            role=payload.role or "analyst",
            otp_code=payload.otp_code.strip(),
            require_otp=True
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")



@router.post("/login", response_model=UserResponse)
def login(payload: UserLoginRequest, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    user_agent = request.headers.get("user-agent", "Unknown")
    
    user = auth_model.authenticate(
        username_or_email=payload.username_or_email,
        password=payload.password,
        ip_address=client_ip,
        user_agent=user_agent
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username/email or password")
    return user


@router.get("/me/{user_id}", response_model=UserResponse)
def get_user(user_id: str):
    user = auth_model.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/logs", response_model=List[LoginLogResponse])
def get_login_logs(limit: int = 50, user_id: Optional[str] = None):
    logs = auth_model.list_login_logs(limit=limit, user_id=user_id)
    return logs


@router.get("/stats")
def get_auth_stats():
    return {
        "total_users": auth_model.get_user_count()
    }
