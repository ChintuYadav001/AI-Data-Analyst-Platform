import pytest
from pathlib import Path
from core.cache_cleaner import get_cache_size_mb, clean_all_cache
from core.config import DATA_DIR, CACHE_DIR, PROJECTS_DIR

def test_get_cache_size_mb():
    sizes = get_cache_size_mb()
    assert isinstance(sizes, dict)
    assert "total" in sizes
    assert "projects" in sizes
    assert sizes["total"] >= 0.0

def test_clean_all_cache():
    # Create a temporary dummy file in cache_dir
    test_file = CACHE_DIR / "dummy_test_cache.txt"
    test_file.write_text("temporary test data")

    res = clean_all_cache()
    assert res["success"] is True
    assert not test_file.exists()
    assert "data/cache/" in res["dirs_cleaned"]
