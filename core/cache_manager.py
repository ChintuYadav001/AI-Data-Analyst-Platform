import os
import json
import pickle
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union

from core.config import CACHE_DIR, ANALYTICS_RESULTS_DIR

class CacheManager:
    """Multi-tiered caching architecture for enterprise scalability:

    L1: Sub-millisecond bounded in-memory LRU cache.
    L2: Local disk / Redis cache with serialization.
    L3: Persistent analytical result storage partitioned by dataset hash.
    """

    # L1: In-Memory Bounded Store
    _l1_cache: Dict[str, Any] = {}
    _l1_access_time: Dict[str, float] = {}
    _L1_MAX_ITEMS = 2000

    @classmethod
    def _make_key(
        cls,
        dataset_hash: str,
        version: str,
        analysis_type: str,
        params: Optional[Any] = None
    ) -> str:
        param_str = ""
        if params is not None:
            if isinstance(params, (dict, list, tuple)):
                try:
                    param_str = json.dumps(params, sort_keys=True, default=str)
                except Exception:
                    param_str = str(params)
            else:
                param_str = str(params)
        param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:8] if param_str else "base"
        return f"{dataset_hash}_{version}_{analysis_type}_{param_hash}"

    @classmethod
    def get(
        cls,
        dataset_hash: str,
        version: str,
        analysis_type: str,
        params: Optional[Any] = None
    ) -> Optional[Any]:
        """Retrieves result checking L1 -> L2 -> L3 in sequence."""
        key = cls._make_key(dataset_hash, version, analysis_type, params)

        # 1. Check L1 Cache
        if key in cls._l1_cache:
            cls._l1_access_time[key] = time.time()
            return cls._l1_cache[key]

        # 2. Check L2 Disk Cache
        l2_path = CACHE_DIR / f"{key}.pkl"
        if l2_path.exists():
            try:
                with open(l2_path, "rb") as f:
                    val = pickle.load(f)
                    # Promote to L1
                    cls._set_l1(key, val)
                    return val
            except Exception:
                pass

        # 3. Check L3 Persistent Store
        l3_path = ANALYTICS_RESULTS_DIR / dataset_hash / f"{version}_{analysis_type}.json"
        if l3_path.exists():
            try:
                with open(l3_path, "r", encoding="utf-8") as f:
                    val = json.load(f)
                    cls._set_l1(key, val)
                    return val
            except Exception:
                pass

        return None

    @classmethod
    def _set_l1(cls, key: str, value: Any) -> None:
        if len(cls._l1_cache) >= cls._L1_MAX_ITEMS:
            # Evict oldest 20%
            sorted_keys = sorted(cls._l1_access_time.items(), key=lambda x: x[1])
            for k, _ in sorted_keys[: cls._L1_MAX_ITEMS // 5]:
                cls._l1_cache.pop(k, None)
                cls._l1_access_time.pop(k, None)

        cls._l1_cache[key] = value
        cls._l1_access_time[key] = time.time()

    @classmethod
    def set(
        cls,
        dataset_hash: str,
        version: str,
        analysis_type: str,
        value: Any,
        params: Optional[Any] = None,
        persist_l3: bool = False
    ) -> None:
        """Stores result into L1, L2, and optionally L3 persistent storage."""
        key = cls._make_key(dataset_hash, version, analysis_type, params)

        # 1. Store in L1
        cls._set_l1(key, value)

        # 2. Store in L2 (Disk cache)
        l2_path = CACHE_DIR / f"{key}.pkl"
        try:
            with open(l2_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

        # 3. Optionally store in L3 (JSON-friendly analytical artifacts)
        if persist_l3:
            l3_dir = ANALYTICS_RESULTS_DIR / dataset_hash
            l3_dir.mkdir(parents=True, exist_ok=True)
            l3_path = l3_dir / f"{version}_{analysis_type}.json"
            try:
                with open(l3_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2, default=str)
            except Exception:
                pass

    @classmethod
    def invalidate(cls, dataset_hash: str, version: Optional[str] = None) -> None:
        """Invalidates all cached data for a dataset or specific version."""
        prefix = f"{dataset_hash}_{version}_" if version else f"{dataset_hash}_"

        # Evict L1
        keys_to_del = [k for k in cls._l1_cache if k.startswith(prefix)]
        for k in keys_to_del:
            cls._l1_cache.pop(k, None)
            cls._l1_access_time.pop(k, None)

        # Evict L2 files
        for p in CACHE_DIR.glob(f"{prefix}*.pkl"):
            try:
                p.unlink()
            except Exception:
                pass

    @classmethod
    def clear_all(cls) -> None:
        """Clears in-memory and disk caches."""
        cls._l1_cache.clear()
        cls._l1_access_time.clear()
        for p in CACHE_DIR.glob("*.pkl"):
            try:
                p.unlink()
            except Exception:
                pass
