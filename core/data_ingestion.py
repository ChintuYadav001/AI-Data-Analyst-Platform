import os
import io
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

from core.config import PARQUET_DIR, MAX_UPLOAD_SIZE_MB
from core.parquet_manager import ParquetManager
from core.dataset_registry import DatasetRegistry
from core.cache_manager import CacheManager

class DataIngestion:
    """Enterprise file ingestion gateway supporting CSV/XLSX stream parsing,

    SHA-256 duplicate collision detection, zero-loss Parquet conversion, and version lineage.
    """

    @classmethod
    def compute_stream_hash(cls, stream: Union[io.BytesIO, bytes, str, Path]) -> str:
        """Computes deterministic SHA-256 hash for dataset identification."""
        hasher = hashlib.sha256()
        if isinstance(stream, (str, Path)):
            with open(stream, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()

        if isinstance(stream, bytes):
            hasher.update(stream)
            return hasher.hexdigest()

        if hasattr(stream, "seek") and hasattr(stream, "read"):
            stream.seek(0)
            while chunk := stream.read(65536):
                hasher.update(chunk)
            stream.seek(0)
            return hasher.hexdigest()

        raise ValueError("Unsupported stream type for hashing.")

    @classmethod
    def ingest_file(
        cls,
        source: Union[io.BytesIO, bytes, str, Path, pd.DataFrame],
        filename: str = "dataset.csv",
        project_id: Optional[str] = None,
        workspace_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingests file once, converts to Snappy Parquet, registers version, and returns metadata."""
        t0 = time.perf_counter()

        # 1. Validation & Hash Extraction
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
        if ext not in ["csv", "xlsx", "xls", "parquet"] and not isinstance(source, pd.DataFrame):
            raise ValueError(f"Unsupported file format '.{ext}'. Supported: CSV, XLSX, XLS.")

        if isinstance(source, pd.DataFrame):
            # Compute hash from shape and sample values
            h_str = f"{len(source)}x{len(source.columns)}_{tuple(source.columns)}_{source.iloc[0].to_dict() if len(source) else ''}"
            dataset_hash = hashlib.sha256(h_str.encode("utf-8")).hexdigest()[:16]
        else:
            dataset_hash = cls.compute_stream_hash(source)[:16]

        parquet_path = PARQUET_DIR / f"{dataset_hash}.parquet"
        name = workspace_name or filename.rsplit(".", 1)[0].replace("_", " ").title()

        # 2. Check for Duplicate Upload
        existing = DatasetRegistry.find_by_hash(dataset_hash)
        if existing and parquet_path.exists():
            meta = ParquetManager.get_parquet_metadata(parquet_path)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "dataset_hash": dataset_hash,
                "project_id": existing.get("dataset_id", project_id),
                "workspace_name": existing.get("name", name),
                "parquet_path": str(parquet_path),
                "rows": meta["num_rows"],
                "columns": meta["num_columns"],
                "column_names": meta["columns"],
                "column_types": meta["column_types"],
                "is_duplicate": True,
                "ingestion_time_ms": elapsed_ms,
                "message": "Existing dataset match detected. Reused cached Parquet representation."
            }

        # 3. Convert once to Parquet
        ParquetManager.convert_to_parquet(
            source=source,
            output_path=parquet_path,
            filename_hint=filename
        )

        # 4. Extract Parquet Metadata in < 1ms
        meta = ParquetManager.get_parquet_metadata(parquet_path)

        # 5. Register in DatasetRegistry
        reg_entry = DatasetRegistry.register(
            dataset_hash=dataset_hash,
            name=name,
            parquet_path=parquet_path,
            rows=meta["num_rows"],
            cols=meta["num_columns"],
            version="dataset_v1",
            project_id=project_id,
            lineage={"source_file": filename, "source_type": ext}
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "dataset_hash": dataset_hash,
            "project_id": reg_entry.get("dataset_id"),
            "workspace_name": name,
            "parquet_path": str(parquet_path),
            "rows": meta["num_rows"],
            "columns": meta["num_columns"],
            "column_names": meta["columns"],
            "column_types": meta["column_types"],
            "is_duplicate": False,
            "ingestion_time_ms": elapsed_ms,
            "message": f"Successfully ingested {meta['num_rows']:,} records into Parquet storage."
        }
