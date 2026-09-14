import duckdb
import pandas as pd
import time
import re
from typing import Dict, Any, Tuple, Optional

class SQLAnalyticsModel:
    """DuckDB SQL analytics engine providing safe execution of analytical queries

    with strict AST-like security validation blocking DDL/DML destructive statements.
    """

    FORBIDDEN_KEYWORDS = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
        "REPLACE", "CREATE", "GRANT", "REVOKE", "ATTACH", "DETACH",
        "EXEC", "EXECUTE", "SYSTEM", "COPY"
    ]

    ALLOWED_START_KEYWORDS = ["SELECT", "WITH", "DESCRIBE", "EXPLAIN", "SHOW", "SUMMARIZE"]

    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(database=db_path)
        self.registered_tables: Dict[str, str] = {}

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        clean_name = re.sub(r"\W+", "_", table_name).lower().strip("_")
        if not clean_name:
            clean_name = "dataset"
        self.con.register(clean_name, df)
        self.registered_tables[clean_name] = f"Table with {len(df)} rows, {len(df.columns)} columns"
        try:
            from core.db import sync_dataframe_to_sql
            sync_dataframe_to_sql(clean_name, df)
        except Exception:
            pass

    def get_table_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        if not table_name and self.registered_tables:
            table_name = list(self.registered_tables.keys())[0]
        if not table_name:
            return {"tables": {}, "columns": []}

        try:
            res = self.con.execute(f"DESCRIBE {table_name}").df()
            return {
                "table_name": table_name,
                "all_tables": list(self.registered_tables.keys()),
                "columns": res.to_dict(orient="records")
            }
        except Exception as e:
            return {"table_name": table_name, "error": str(e), "columns": []}

    @classmethod
    def validate_sql_security(cls, query: str) -> Tuple[bool, Optional[str]]:
        cleaned = re.sub(r"--.*?$", "", query, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            return False, "Query is empty."

        first_word = cleaned.split()[0].upper()
        if first_word not in cls.ALLOWED_START_KEYWORDS:
            return False, f"Unauthorized statement type '{first_word}'. Only analytical queries ({', '.join(cls.ALLOWED_START_KEYWORDS)}) are permitted."

        tokens = re.findall(r"\b[A-Za-z_]+\b", cleaned)
        for token in tokens:
            upper_tok = token.upper()
            if upper_tok in cls.FORBIDDEN_KEYWORDS:
                return False, f"Security Violation: '{upper_tok}' statement is strictly forbidden. Database is read-only."

        # Block multiple statements delimited by semicolon
        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        if len(statements) > 1:
            return False, "Multi-statement execution via semicolon is prohibited for security."

        return True, None

    def execute_query(self, query: str) -> Dict[str, Any]:
        is_safe, error_msg = self.validate_sql_security(query)
        if not is_safe:
            return {
                "success": False,
                "error": error_msg,
                "data": None,
                "execution_time_ms": 0.0,
                "row_count": 0
            }

        start_time = time.perf_counter()
        try:
            result_df = self.con.execute(query).df()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            # Auto-recommend visualization for query result
            recommended_chart = None
            if len(result_df) > 0 and len(result_df.columns) >= 2:
                col_types = {col: str(result_df[col].dtype) for col in result_df.columns}
                first_col = result_df.columns[0]
                second_col = result_df.columns[1]
                
                if pd.api.types.is_numeric_dtype(result_df[second_col]):
                    recommended_chart = {
                        "type": "bar",
                        "x": first_col,
                        "y": second_col,
                        "title": f"{second_col} by {first_col}"
                    }

            return {
                "success": True,
                "dataframe": result_df,
                "row_count": len(result_df),
                "columns": list(result_df.columns),
                "execution_time_ms": elapsed_ms,
                "error": None,
                "recommended_chart": recommended_chart
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "success": False,
                "error": f"SQL Execution Error: {str(e)}",
                "dataframe": None,
                "execution_time_ms": elapsed_ms,
                "row_count": 0
            }
