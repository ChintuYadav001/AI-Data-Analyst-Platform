"""
core.cache_cleaner
==================
Institutional cache and temporary storage cleaner for Quantum Analyst.
Purges stale project files, temporary parquet streams, experiment artifacts,
and memory buffers to accelerate cold starts and conserve RAM/disk on cloud deployments (e.g. Render).
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from core.config import DATA_DIR, CACHE_DIR, PARQUET_DIR, ANALYTICS_RESULTS_DIR, PROJECTS_DIR, UPLOADS_DIR

logger = logging.getLogger(__name__)


def get_cache_size_mb() -> Dict[str, float]:
    """Calculate the current disk footprint of all cache and temporary directories."""
    categories = {
        "projects": PROJECTS_DIR,
        "parquet": PARQUET_DIR,
        "cache": CACHE_DIR,
        "analytics_results": ANALYTICS_RESULTS_DIR,
        "uploads": UPLOADS_DIR,
        "experiments": DATA_DIR / "experiments",
    }
    
    sizes = {}
    total = 0.0
    for name, path in categories.items():
        if path.exists():
            size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / (1024 * 1024)
            sizes[name] = round(size, 2)
            total += size
        else:
            sizes[name] = 0.0
            
    sizes["total"] = round(total, 2)
    return sizes


def clean_all_cache(keep_active_project_id: str = None) -> Dict[str, Any]:
    """
    Purge stale temporary files across the data directory.
    Safely preserves active project, user database, and core registries.
    """
    total_freed_bytes = 0
    files_deleted = 0
    dirs_cleaned = []

    # 1. Clean data/projects/ (preserve active project if specified)
    if PROJECTS_DIR.exists():
        for item in PROJECTS_DIR.iterdir():
            if item.is_dir():
                if keep_active_project_id and item.name == str(keep_active_project_id):
                    continue
                try:
                    for f in item.glob("**/*"):
                        if f.is_file():
                            total_freed_bytes += f.stat().st_size
                            files_deleted += 1
                    shutil.rmtree(item, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Could not remove project dir {item}: {e}")
        dirs_cleaned.append("data/projects/")

    # 2. Clean data/parquet/
    if PARQUET_DIR.exists():
        for f in PARQUET_DIR.glob("**/*"):
            if f.is_file():
                try:
                    total_freed_bytes += f.stat().st_size
                    f.unlink(missing_ok=True)
                    files_deleted += 1
                except Exception as e:
                    logger.warning(f"Could not delete parquet file {f}: {e}")
        dirs_cleaned.append("data/parquet/")

    # 3. Clean data/cache/
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("**/*"):
            if f.is_file():
                try:
                    total_freed_bytes += f.stat().st_size
                    f.unlink(missing_ok=True)
                    files_deleted += 1
                except Exception as e:
                    logger.warning(f"Could not delete cache file {f}: {e}")
        dirs_cleaned.append("data/cache/")

    # 4. Clean data/analytics_results/
    if ANALYTICS_RESULTS_DIR.exists():
        for f in ANALYTICS_RESULTS_DIR.glob("**/*"):
            if f.is_file():
                try:
                    total_freed_bytes += f.stat().st_size
                    f.unlink(missing_ok=True)
                    files_deleted += 1
                except Exception as e:
                    logger.warning(f"Could not delete analytics result file {f}: {e}")
        dirs_cleaned.append("data/analytics_results/")

    # 5. Clean data/uploads/
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.glob("**/*"):
            if f.is_file():
                try:
                    total_freed_bytes += f.stat().st_size
                    f.unlink(missing_ok=True)
                    files_deleted += 1
                except Exception as e:
                    logger.warning(f"Could not delete upload file {f}: {e}")
        dirs_cleaned.append("data/uploads/")

    # 5b. Clean temporary exports in data/exports/
    exports_dir = DATA_DIR / "exports"
    if exports_dir.exists():
        for f in exports_dir.glob("Project *.csv"):
            try:
                total_freed_bytes += f.stat().st_size
                f.unlink(missing_ok=True)
                files_deleted += 1
            except Exception:
                pass
        for f in exports_dir.glob("test_*.pdf"):
            try:
                total_freed_bytes += f.stat().st_size
                f.unlink(missing_ok=True)
                files_deleted += 1
            except Exception:
                pass
        dirs_cleaned.append("data/exports/")

    # 6. Clean stale exp_* directories inside data/experiments/ (keep active_champion.json)
    exp_dir = DATA_DIR / "experiments"
    if exp_dir.exists():
        for item in exp_dir.iterdir():
            if item.is_dir() and item.name.startswith("exp_"):
                try:
                    for f in item.glob("**/*"):
                        if f.is_file():
                            total_freed_bytes += f.stat().st_size
                            files_deleted += 1
                    shutil.rmtree(item, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Could not delete experiment dir {item}: {e}")
        dirs_cleaned.append("data/experiments/exp_*/")

    # 7. Clean Python __pycache__ and .pytest_cache in workspace
    base_dir = DATA_DIR.parent
    for pycache in base_dir.glob("**/__pycache__"):
        try:
            shutil.rmtree(pycache, ignore_errors=True)
        except Exception:
            pass
            
    pytest_cache = base_dir / ".pytest_cache"
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache, ignore_errors=True)
        except Exception:
            pass

    # Ensure required empty directories still exist
    for d in [PROJECTS_DIR, PARQUET_DIR, CACHE_DIR, ANALYTICS_RESULTS_DIR, UPLOADS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    freed_mb = round(total_freed_bytes / (1024 * 1024), 2)
    logger.info(f"Cleaned {files_deleted} files, freed {freed_mb} MB")

    return {
        "success": True,
        "files_deleted": files_deleted,
        "freed_mb": freed_mb,
        "dirs_cleaned": dirs_cleaned,
    }
