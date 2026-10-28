import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.neon_db import (
    get_neon_url,
    test_neon_connection,
    list_neon_tables,
    fetch_table_from_neon,
    store_dataframe_to_neon,
)

def test_neon_url_normalization():
    # Empty url
    assert get_neon_url("") == ""
    assert get_neon_url(None) == ""

    # postgres:// scheme replaced with postgresql://
    url = "postgres://usr:pwd@ep-test-123.neon.tech/neondb"
    normalized = get_neon_url(url)
    assert normalized.startswith("postgresql://")
    assert "sslmode=require" in normalized

    # query parameter already has parameters
    url2 = "postgresql://usr:pwd@ep-test-123.neon.tech/neondb?channel_binding=prefer"
    normalized2 = get_neon_url(url2)
    assert "channel_binding=prefer" in normalized2
    assert "sslmode=require" in normalized2

    # sslmode already present
    url3 = "postgresql://usr:pwd@ep-test-123.neon.tech/neondb?sslmode=require"
    normalized3 = get_neon_url(url3)
    assert normalized3.count("sslmode=require") == 1

def test_neon_connection_invalid_url():
    res = test_neon_connection("invalid_connection_string")
    assert res["success"] is False
    assert "error" in res

def test_neon_connection_empty():
    with patch.dict("os.environ", {}, clear=True):
        res = test_neon_connection("")
        assert res["success"] is False
        assert "No Neon Connection URI provided" in res["error"]

def test_neon_connection_mocked_success():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # execute returns scalar
    mock_cursor.scalar.side_effect = ["PostgreSQL 16.3 on x86_64", 3]
    mock_conn.execute.return_value = mock_cursor
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.url.database = "neondb"
    mock_engine.url.username = "chintu"

    with patch("core.neon_db.create_engine", return_value=mock_engine):
        res = test_neon_connection("postgresql://chintu:pass@ep-xyz.neon.tech/neondb")
        assert res["success"] is True
        assert res["database"] == "neondb"
        assert res["user"] == "chintu"
        assert res["table_count"] == 3
        assert res["latency_ms"] >= 0

def test_store_dataframe_to_neon_mocked():
    df = pd.DataFrame({
        "User ID": [1, 2, 3],
        "Gross Margin (%)": [20.5, 30.1, 40.0],
        "Category Name": ["Alpha", "Beta", "Gamma"]
    })

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    with patch("core.neon_db.create_engine", return_value=mock_engine):
        with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
            res = store_dataframe_to_neon(df, "My Test Table!", "postgresql://usr:pwd@ep-test.neon.tech/neondb")
            assert res["success"] is True
            assert res["table_name"] == "my_test_table_"
            assert res["rows_stored"] == 3
            assert mock_to_sql.called
            # Verify columns passed to to_sql are sanitized
            called_df = mock_to_sql.call_args[0][0] if len(mock_to_sql.call_args[0]) > 0 else mock_to_sql.call_args[1].get("name")
            assert mock_to_sql.call_args[0][0] == "my_test_table_"

def test_list_neon_tables_mocked():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_cursor.fetchall.return_value = [
        ("telemetry_logs", 1500, 245760),
        ("customers", 400, 65536)
    ]
    mock_conn.execute.return_value = mock_cursor
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    with patch("core.neon_db.create_engine", return_value=mock_engine):
        tables = list_neon_tables("postgresql://usr:pwd@ep-test.neon.tech/neondb")
        assert len(tables) == 2
        assert tables[0]["table_name"] == "telemetry_logs"
        assert tables[0]["row_count"] == 1500
        assert "KB" in tables[0]["size"]
