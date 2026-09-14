import os
os.environ["ENABLE_TEST_OTP_PREVIEW"] = "true"

import pytest
from fastapi.testclient import TestClient
from main import app
from core.sample_data import generate_ecommerce_dataset
import io

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_upload_and_pipeline():
    # 1. Test CSV upload
    df = generate_ecommerce_dataset(50)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    files = {"file": ("test_sales.csv", csv_buf, "text/csv")}
    upload_resp = client.post("/api/upload", files=files)
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    project_id = data["project_id"]
    assert project_id is not None
    
    # 2. Test Profile
    prof_resp = client.get(f"/api/dataset/{project_id}/profile")
    assert prof_resp.status_code == 200
    assert "profile" in prof_resp.json()
    assert "quality" in prof_resp.json()
    
    # 3. Test EDA
    eda_resp = client.get(f"/api/dataset/{project_id}/eda")
    assert eda_resp.status_code == 200
    assert "numerical" in eda_resp.json()
    
    # 4. Test SQL
    sql_resp = client.post(
        f"/api/dataset/{project_id}/sql",
        json={"query": f'SELECT region, count(*) as cnt FROM "ds_{project_id}" GROUP BY region'}
    )
    assert sql_resp.status_code == 200
    assert sql_resp.json()["success"] is True
    
    # 5. Test SQL Security Guardrail
    bad_sql = client.post(
        f"/api/dataset/{project_id}/sql",
        json={"query": f'DROP TABLE "ds_{project_id}"'}
    )
    assert bad_sql.json()["success"] is False
    
    # 6. Test Insights
    ins_resp = client.get(f"/api/dataset/{project_id}/insights")
    assert ins_resp.status_code == 200
    assert "key_insights" in ins_resp.json()["insights"]
    
    # 7. Test Export
    exp_resp = client.get(f"/api/dataset/{project_id}/export?format=csv")
    assert exp_resp.status_code == 200


def test_api_auth_phone_otp():
    import random
    phone = f"+1999{random.randint(1000000, 9999999)}"
    # Send OTP to phone
    otp_resp = client.post("/api/auth/otp/send", json={"phone": phone})
    assert otp_resp.status_code == 200
    otp_data = otp_resp.json()
    assert otp_data["success"] is True
    assert otp_data["target_type"] == "phone"
    code = otp_data["demo_otp_preview"]

    # Verify OTP
    verify_resp = client.post("/api/auth/otp/verify", json={"phone": phone, "otp_code": code})
    assert verify_resp.status_code == 200
    assert verify_resp.json()["success"] is True

    # Register with phone
    import uuid
    uid = uuid.uuid4().hex[:6]
    reg_resp = client.post("/api/auth/signup", json={
        "username": f"phone_api_{uid}",
        "email": f"phone_api_{uid}@example.com",
        "phone": phone,
        "password": "Password123!",
        "otp_code": code
    })
    assert reg_resp.status_code == 200
    user = reg_resp.json()
    assert user["phone"] == phone
    assert user["username"] == f"phone_api_{uid}"


def test_api_signup_strictly_requires_otp():
    import uuid
    uid = uuid.uuid4().hex[:6]
    email = f"strict_api_{uid}@example.com"
    username = f"strict_api_{uid}"
    pwd = "StrictPassword123!"

    # 1. Missing otp_code in payload must be rejected by FastAPI / Pydantic (422)
    resp_missing_otp = client.post("/api/auth/signup", json={
        "username": username,
        "email": email,
        "password": pwd
    })
    assert resp_missing_otp.status_code == 422

    # 2. Fake / wrong otp_code must be rejected (400)
    resp_wrong_otp = client.post("/api/auth/signup", json={
        "username": username,
        "email": email,
        "password": pwd,
        "otp_code": "000000"
    })
    assert resp_wrong_otp.status_code == 400
    assert "Invalid or expired" in resp_wrong_otp.json()["detail"]

    # 3. Generate valid OTP and signup
    otp_resp = client.post("/api/auth/otp/send", json={"email": email})
    assert otp_resp.status_code == 200
    valid_otp = otp_resp.json()["demo_otp_preview"]

    resp_valid = client.post("/api/auth/signup", json={
        "username": username,
        "email": email,
        "password": pwd,
        "otp_code": valid_otp
    })
    assert resp_valid.status_code == 200
    assert resp_valid.json()["username"] == username

    # 4. Attempting to reuse the same OTP for another user must be rejected (400)
    resp_reuse = client.post("/api/auth/signup", json={
        "username": f"another_{uid}",
        "email": f"another_{uid}@example.com",
        "password": pwd,
        "otp_code": valid_otp
    })
    assert resp_reuse.status_code == 400
    assert "Invalid or expired" in resp_reuse.json()["detail"]

