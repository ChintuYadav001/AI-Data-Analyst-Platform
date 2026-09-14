from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class CleaningAction(BaseModel):
    action: str
    column: Optional[str] = "All"
    value: Optional[Any] = None

class CleanRequest(BaseModel):
    actions: List[CleaningAction]

class FeatureRequest(BaseModel):
    type: str
    target_col: Optional[str] = None
    col1: Optional[str] = None
    col2: Optional[str] = None
    feature_name: Optional[str] = None
    customer_col: Optional[str] = None
    metric_col: Optional[str] = None

class SQLQueryRequest(BaseModel):
    query: str

class ChatQueryRequest(BaseModel):
    message: str

class DashboardFilterRequest(BaseModel):
    filters: Dict[str, List[str]] = Field(default_factory=dict)


class SendOTPRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    target: Optional[str] = None


class VerifyOTPRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    target: Optional[str] = None
    otp_code: str


class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    otp_code: str = Field(..., min_length=4, max_length=10, description="Mandatory OTP verification code")
    phone: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = "analyst"



class UserLoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    phone: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    created_at: str
    last_login_at: Optional[str] = None
    login_count: int
    is_active: int


class LoginLogResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    username: str
    timestamp: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: str
    failure_reason: Optional[str] = None

