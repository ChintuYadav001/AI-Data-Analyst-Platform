import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from core.config import DATA_DIR, PARQUET_DIR

REGISTRY_FILE = DATA_DIR / "dataset_registry.json"

class DatasetRegistry:
    """Maintains a persistent registry of all ingested datasets, versions

    (raw, clean, feature-expanded), lineage graphs, and physical Parquet references.
    """

    _cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _load_registry(cls) -> Dict[str, Any]:
        if cls._cache is not None:
            return cls._cache
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    cls._cache = json.load(f)
                    return cls._cache
            except Exception:
                pass
        cls._cache = {"datasets": {}, "hashes": {}}
        return cls._cache

    @classmethod
    def _save_registry(cls) -> None:
        if cls._cache is None:
            return
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(cls._cache, f, indent=2)

    @classmethod
    def register(
        cls,
        dataset_hash: str,
        name: str,
        parquet_path: Union[str, Path],
        rows: int,
        cols: int,
        version: str = "dataset_v1",
        project_id: Optional[str] = None,
        lineage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registers a new dataset or registers an update version."""
        reg = cls._load_registry()
        import uuid
        ds_id = project_id or f"ds_{uuid.uuid4().hex[:8]}"

        entry = reg["datasets"].get(ds_id, {
            "dataset_id": ds_id,
            "name": name,
            "dataset_hash": dataset_hash,
            "created_at": time.time(),
            "versions": {},
            "active_version": version
        })

        entry["versions"][version] = {
            "parquet_path": str(parquet_path),
            "rows": rows,
            "columns": cols,
            "registered_at": time.time(),
            "lineage": lineage or {}
        }
        entry["active_version"] = version
        entry["dataset_hash"] = dataset_hash

        reg["datasets"][ds_id] = entry
        reg["hashes"][dataset_hash] = ds_id
        cls._save_registry()
        return entry

    @classmethod
    def find_by_hash(cls, dataset_hash: str) -> Optional[Dict[str, Any]]:
        reg = cls._load_registry()
        ds_id = reg.get("hashes", {}).get(dataset_hash)
        if ds_id and ds_id in reg.get("datasets", {}):
            return reg["datasets"][ds_id]
        return None

    @classmethod
    def get(cls, dataset_id_or_hash: str) -> Optional[Dict[str, Any]]:
        reg = cls._load_registry()
        if dataset_id_or_hash in reg["datasets"]:
            return reg["datasets"][dataset_id_or_hash]
        # Check by hash
        return cls.find_by_hash(dataset_id_or_hash)

    @classmethod
    def get_active_parquet_path(cls, dataset_id_or_hash: str) -> Optional[Path]:
        entry = cls.get(dataset_id_or_hash)
        if not entry:
            return None
        active_ver = entry.get("active_version", "dataset_v1")
        ver_info = entry.get("versions", {}).get(active_ver)
        if ver_info and "parquet_path" in ver_info:
            p = Path(ver_info["parquet_path"])
            if p.exists():
                return p
        return None

    @classmethod
    def update_version(
        cls,
        dataset_id: str,
        new_version: str,
        parquet_path: Union[str, Path],
        rows: int,
        cols: int,
        lineage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        reg = cls._load_registry()
        entry = reg["datasets"].get(dataset_id)
        if not entry:
            raise ValueError(f"Dataset {dataset_id} not found in registry.")

        entry["versions"][new_version] = {
            "parquet_path": str(parquet_path),
            "rows": rows,
            "columns": cols,
            "registered_at": time.time(),
            "lineage": lineage or {}
        }
        entry["active_version"] = new_version
        cls._save_registry()
        return entry

    @classmethod
    def list_datasets(cls) -> List[Dict[str, Any]]:
        reg = cls._load_registry()
        return list(reg.get("datasets", {}).values())
