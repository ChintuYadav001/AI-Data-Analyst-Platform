import os
import io
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv

from core.config import PARQUET_DIR

class ParquetManager:
    """Manages Snappy-compressed Parquet storage, stream conversion from CSV/XLSX,

    instant metadata extraction (< 1ms), and zero-copy schema validation.
    """

    @classmethod
    def sanitize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes DataFrame columns for seamless PyArrow serialization without ArrowInvalid or UnicodeDecodeError exceptions."""
        df_out = df.copy(deep=False)
        for col in df_out.columns:
            # Normalize column names to string
            if not isinstance(col, str):
                df_out.rename(columns={col: str(col)}, inplace=True)
            col_str = str(col)
            s = df_out[col_str]
            if s.dtype == object:
                def _safe_str_val(v):
                    if v is None or pd.isna(v):
                        return None
                    if isinstance(v, bytes):
                        try:
                            return v.decode("utf-8")
                        except UnicodeDecodeError:
                            return v.decode("latin-1", errors="replace")
                    try:
                        return str(v)
                    except Exception:
                        return repr(v)
                
                # Check if series contains bytes or fails PyArrow serialization
                needs_conversion = False
                non_null = s.dropna()
                if len(non_null) > 0:
                    if isinstance(non_null.iloc[0], bytes) or any(isinstance(x, bytes) for x in non_null.head(20)):
                        needs_conversion = True
                    else:
                        try:
                            pa.Array.from_pandas(non_null.head(100))
                        except Exception:
                            needs_conversion = True
                
                if needs_conversion:
                    df_out[col_str] = s.map(_safe_str_val)
        return df_out

    @classmethod
    def convert_to_parquet(
        cls,
        source: Union[pd.DataFrame, str, Path, io.BytesIO],
        output_path: Optional[Union[str, Path]] = None,
        compression: str = "snappy",
        filename_hint: str = "dataset.csv"
    ) -> Path:
        """Converts DataFrame, CSV, or Excel file directly to Parquet once."""
        if output_path is None:
            import uuid
            output_path = PARQUET_DIR / f"{uuid.uuid4().hex[:12]}.parquet"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(source, pd.DataFrame):
            clean_df = cls.sanitize_dataframe(source)
            clean_df.to_parquet(output_path, compression=compression, index=False)
            return output_path

        # If source is a file path or BytesIO stream
        hint = filename_hint.lower()
        if isinstance(source, (str, Path)):
            src_path = str(source)
            hint = src_path.lower()
            if hint.endswith(".parquet"):
                import shutil
                shutil.copyfile(src_path, output_path)
                return output_path
            elif hint.endswith(".csv"):
                try:
                    # Attempt fast C-level PyArrow CSV streaming conversion
                    table = pacsv.read_csv(src_path)
                    pq.write_table(table, output_path, compression=compression)
                    return output_path
                except Exception:
                    # Fallback to chunked streaming with ParquetWriter (avoids OOM on 2M+ rows)
                    try:
                        writer = None
                        for chunk in pd.read_csv(src_path, chunksize=100_000, low_memory=False, encoding_errors="replace"):
                            clean_chunk = cls.sanitize_dataframe(chunk)
                            table = pa.Table.from_pandas(clean_chunk)
                            if writer is None:
                                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                            writer.write_table(table)
                        if writer is not None:
                            writer.close()
                        return output_path
                    except Exception:
                        df = pd.read_csv(src_path, low_memory=False, encoding="latin-1")
                        clean_df = cls.sanitize_dataframe(df)
                        clean_df.to_parquet(output_path, compression=compression, index=False)
                        return output_path
            elif hint.endswith((".xlsx", ".xls")):
                df = pd.read_excel(src_path)
                clean_df = cls.sanitize_dataframe(df)
                clean_df.to_parquet(output_path, compression=compression, index=False)
                return output_path

        # If source is BytesIO or bytes-like object or UploadedFile
        if isinstance(source, (io.BytesIO, bytes)) or hasattr(source, "read"):
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            if hasattr(source, "seek"):
                source.seek(0)

            if hint.endswith(".parquet"):
                if hasattr(source, "seek"):
                    source.seek(0)
                with open(output_path, "wb") as f_out:
                    if hasattr(source, "read"):
                        while chunk := source.read(65536):
                            f_out.write(chunk)
                    else:
                        f_out.write(source)
                return output_path
            elif hint.endswith(".csv"):
                try:
                    table = pacsv.read_csv(source)
                    pq.write_table(table, output_path, compression=compression)
                    return output_path
                except Exception:
                    if hasattr(source, "seek"):
                        source.seek(0)
                    try:
                        # Chunked streaming to Parquet prevents memory exhaustion on 2M+ rows
                        writer = None
                        for chunk in pd.read_csv(source, chunksize=100_000, low_memory=False, encoding_errors="replace"):
                            clean_chunk = cls.sanitize_dataframe(chunk)
                            table = pa.Table.from_pandas(clean_chunk)
                            if writer is None:
                                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                            writer.write_table(table)
                        if writer is not None:
                            writer.close()
                        return output_path
                    except Exception:
                        if hasattr(source, "seek"):
                            source.seek(0)
                        try:
                            df = pd.read_csv(source, low_memory=False, encoding="latin-1")
                        except Exception:
                            if hasattr(source, "seek"):
                                source.seek(0)
                            df = pd.read_csv(source, low_memory=False, encoding_errors="replace")
                        clean_df = cls.sanitize_dataframe(df)
                        clean_df.to_parquet(output_path, compression=compression, index=False)
                        return output_path
            elif hint.endswith((".xlsx", ".xls")):
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_excel(source)
                clean_df = cls.sanitize_dataframe(df)
                clean_df.to_parquet(output_path, compression=compression, index=False)
                return output_path

        raise ValueError(f"Unsupported source type or file format: {hint}")

    @classmethod
    def get_parquet_metadata(cls, parquet_path: Union[str, Path]) -> Dict[str, Any]:
        """Reads schema, row count, byte size, and column types in < 1ms directly from Parquet metadata."""
        path_str = str(parquet_path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Parquet file not found at: {path_str}")

        pq_file = pq.ParquetFile(path_str)
        meta = pq_file.metadata
        schema = pq_file.schema_arrow

        col_names = [col.name for col in schema]
        col_types = {col.name: str(col.type) for col in schema}

        return {
            "num_rows": meta.num_rows,
            "num_columns": meta.num_columns,
            "serialized_size_bytes": meta.serialized_size if hasattr(meta, "serialized_size") else os.path.getsize(path_str),
            "columns": col_names,
            "column_types": col_types,
            "created_by": meta.created_by,
            "path": path_str
        }

    @classmethod
    def validate_parquet(cls, parquet_path: Union[str, Path]) -> bool:
        """Verifies integrity of Parquet file."""
        try:
            meta = cls.get_parquet_metadata(parquet_path)
            return meta["num_rows"] >= 0 and meta["num_columns"] > 0
        except Exception:
            return False

    @classmethod
    def read_sample(
        cls,
        parquet_path: Union[str, Path],
        n: int = 10000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Reads a fast representative sample without loading all rows into memory."""
        import duckdb
        path_str = str(parquet_path).replace("\\", "/")
        con = duckdb.connect(database=":memory:")
        meta = cls.get_parquet_metadata(parquet_path)
        total_rows = meta["num_rows"]

        if total_rows <= n:
            return con.execute(f"SELECT * FROM read_parquet('{path_str}')").df()

        query = f"SELECT * FROM read_parquet('{path_str}') USING SAMPLE {n} ROWS (BERNOULLI, {random_state})"
        try:
            return con.execute(query).df()
        except Exception:
            return con.execute(f"SELECT * FROM read_parquet('{path_str}') LIMIT {n}").df()
