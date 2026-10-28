import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from core.config import PROJECTS_DIR

class ProjectModel:
    """Manages project repositories, preserving raw & cleaned datasets,

    audit histories, queries, dashboard settings, and reports across sessions.
    """

    def __init__(self, project_id: str, name: str, goal: str = "General Business Intelligence", goal_explanation: str = ""):
        self.project_id = project_id
        self.name = name
        self.goal = goal
        self.goal_explanation = goal_explanation
        self.project_path = PROJECTS_DIR / project_id
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        self.raw_df_path = self.project_path / "raw.parquet"
        self.cleaned_df_path = self.project_path / "cleaned.parquet"
        self.meta_path = self.project_path / "metadata.json"
        
        if not self.meta_path.exists():
            self._save_metadata({
                "id": self.project_id,
                "name": self.name,
                "goal": self.goal,
                "goal_explanation": self.goal_explanation,
                "cleaning_history": [],
                "saved_queries": [],
                "created_at": pd.Timestamp.now().isoformat()
            })
        else:
            current_meta = self.get_metadata()
            self.goal = current_meta.get("goal", self.goal)
            self.goal_explanation = current_meta.get("goal_explanation", self.goal_explanation)

    def _save_metadata(self, meta: Dict[str, Any]) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.get_metadata()

    def get_metadata(self) -> Dict[str, Any]:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"id": self.project_id, "name": self.name}

    @staticmethod
    def _prepare_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """Ensures DataFrame columns are strictly typed and free of mixed object types that cause PyArrow ArrowInvalid."""
        from core.parquet_manager import ParquetManager
        return ParquetManager.sanitize_dataframe(df)

    def save_raw_dataset(self, df: pd.DataFrame) -> None:
        clean_df = self._prepare_df_for_parquet(df)
        clean_df.to_parquet(self.raw_df_path, index=False)
        try:
            from core.db import sync_dataframe_to_sql
            sync_df = clean_df.head(10000) if len(clean_df) > 10000 else clean_df
            sync_dataframe_to_sql(f"proj_{self.project_id[:8]}", sync_df, project_name=self.name)
        except Exception:
            pass

    def save_cleaned_dataset(self, df: pd.DataFrame, cleaning_log: List[Dict[str, Any]]) -> None:
        clean_df = self._prepare_df_for_parquet(df)
        clean_df.to_parquet(self.cleaned_df_path, index=False)
        meta = self.get_metadata()
        meta["cleaning_history"] = cleaning_log
        meta["updated_at"] = pd.Timestamp.now().isoformat()
        self._save_metadata(meta)
        try:
            from core.db import sync_dataframe_to_sql
            sync_df = clean_df.head(10000) if len(clean_df) > 10000 else clean_df
            sync_dataframe_to_sql(f"proj_{self.project_id[:8]}_cleaned", sync_df, project_name=f"{self.name} (Cleaned)")
        except Exception:
            pass

    def get_raw_dataset(self) -> Optional[pd.DataFrame]:
        if self.raw_df_path.exists():
            return pd.read_parquet(self.raw_df_path)
        return None

    def get_cleaned_dataset(self) -> Optional[pd.DataFrame]:
        if self.cleaned_df_path.exists():
            return pd.read_parquet(self.cleaned_df_path)
        return self.get_raw_dataset()

    def get_active_parquet_path(self) -> Path:
        if self.cleaned_df_path.exists():
            return self.cleaned_df_path
        return self.raw_df_path

    def get_dataset_sample(self, n: int = 10000, random_state: int = 42) -> pd.DataFrame:
        from core.parquet_manager import ParquetManager
        active_p = self.get_active_parquet_path()
        if active_p.exists():
            return ParquetManager.read_sample(active_p, n=n, random_state=random_state)
        raw = self.get_cleaned_dataset()
        if raw is not None:
            return raw.sample(n, random_state=random_state) if len(raw) > n else raw
        return pd.DataFrame()

    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        return self.get_metadata().get("cleaning_history", [])

    def save_query(self, query: str) -> None:
        meta = self.get_metadata()
        queries = meta.get("saved_queries", [])
        if query not in queries:
            queries.append(query)
            meta["saved_queries"] = queries
            self._save_metadata(meta)

    def update_project_goal(self, goal: str, goal_explanation: str = "") -> None:
        self.goal = goal
        self.goal_explanation = goal_explanation
        meta = self.get_metadata()
        meta["goal"] = goal
        meta["goal_explanation"] = goal_explanation
        meta["updated_at"] = pd.Timestamp.now().isoformat()
        self._save_metadata(meta)

    @classmethod
    def list_all_projects(cls) -> List[Dict[str, Any]]:
        projects = []
        for p in PROJECTS_DIR.iterdir():
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    with open(p / "metadata.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        projects.append(meta)
                except Exception:
                    pass
        return sorted(projects, key=lambda x: x.get("created_at", ""), reverse=True)

    @classmethod
    def get_project(cls, project_id: str) -> Optional["ProjectModel"]:
        proj_dir = PROJECTS_DIR / project_id
        if proj_dir.is_dir() and (proj_dir / "metadata.json").exists():
            with open(proj_dir / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            return cls(project_id, meta.get("name", "Project"), meta.get("goal", "General Business Intelligence"), meta.get("goal_explanation", ""))
        return None

    @classmethod
    def create_project(cls, name: str, df: pd.DataFrame, goal: str = "General Business Intelligence", goal_explanation: str = "", parquet_source: Optional[Union[str, Path]] = None) -> "ProjectModel":
        p_id = str(uuid.uuid4())[:8]
        proj = cls(p_id, name, goal, goal_explanation)
        if parquet_source and Path(parquet_source).exists():
            import shutil
            shutil.copyfile(str(parquet_source), str(proj.raw_df_path))
            shutil.copyfile(str(parquet_source), str(proj.cleaned_df_path))
        else:
            proj.save_raw_dataset(df)
            proj.save_cleaned_dataset(df, [])
        return proj
