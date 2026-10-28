import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.neon_db import (
    get_neon_url,
    sanitize_table_name,
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

def test_sanitize_table_name():
    assert sanitize_table_name("My Test Table!") == "my_test_table"
    assert sanitize_table_name("123_numbers") == "ds_123_numbers"
    assert sanitize_table_name("___ecom___sales___") == "ecom_sales"

def test_neon_connection_invalid_url():
    res = test_neon_connection("invalid_connection_string")
    assert res["success"] is False
    assert "error" in res

def test_neon_connection_empty():
    with patch.dict("os.environ", {}, clear=True):
        res = test_neon_connection("")
        assert res["success"] is False
        assert "No Neon connection URL provided" in res["error"]

def test_neon_connection_mocked_success():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.side_effect = [
        ("PostgreSQL 16.3 on x86_64", "neondb", "chintu"),
        [5]
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("psycopg2.connect", return_value=mock_conn):
        res = test_neon_connection("postgresql://chintu:pass@ep-xyz.neon.tech/neondb")
        assert res["success"] is True
        assert res["database_name"] == "neondb"
        assert res["current_user"] == "chintu"
        assert res["public_tables_count"] == 5
        assert res["latency_ms"] >= 0

def test_store_dataframe_to_neon_mocked():
    df = pd.DataFrame({
        "User ID": [1, 2, 3],
        "Gross Margin (%)": [20.5, 30.1, 40.0],
        "Category Name": ["Alpha", "Beta", "Gamma"]
    })

    mock_engine = MagicMock()
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
            res = store_dataframe_to_neon(df, "My Test Table!", "postgresql://usr:pwd@ep-test.neon.tech/neondb")
            assert res["success"] is True
            assert res["table_name"] == "my_test_table"
            assert res["rows_stored"] == 3
            assert mock_to_sql.called
            assert mock_to_sql.call_args[0][0] == "my_test_table"

def test_list_neon_tables_mocked():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        {"table_name": "telemetry_logs", "estimated_rows": 1500, "size_formatted": "240 kB", "size_bytes": 245760},
        {"table_name": "customers", "estimated_rows": 400, "size_formatted": "64 kB", "size_bytes": 65536}
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("psycopg2.connect", return_value=mock_conn):
        tables = list_neon_tables("postgresql://usr:pwd@ep-test.neon.tech/neondb")
        assert len(tables) == 2
        assert tables[0]["table_name"] == "telemetry_logs"
        assert tables[0]["estimated_rows"] == 1500
        assert "240 kB" in tables[0]["size_formatted"]

def test_fetch_table_from_neon_mocked():
    mock_engine = MagicMock()
    fake_df = pd.DataFrame({"id": [1, 2], "value": [10.5, 20.2]})
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        with patch("pandas.read_sql", return_value=fake_df):
            df, meta = fetch_table_from_neon("customer_data", "postgresql://usr:pwd@ep-test.neon.tech/neondb", limit=100)
            assert len(df) == 2
            assert meta["table_name"] == "customer_data"
            assert meta["rows"] == 2
            assert meta["columns"] == 2
