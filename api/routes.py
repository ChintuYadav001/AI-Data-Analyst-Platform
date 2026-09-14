import os
import io
import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any, List

from core.models import (
    DataProfilerModel,
    DataQualityModel,
    DataCleanerModel,
    EDAModel,
    ChartModel,
    FeatureEngineeringModel,
    SQLAnalyticsModel,
    AIInsightsModel,
    AIChatModel,
    DashboardModel,
    ReportModel,
    ProjectModel
)
from core.config import EXPORTS_DIR
from api.schemas import CleanRequest, FeatureRequest, SQLQueryRequest, ChatQueryRequest, DashboardFilterRequest

from core.job_manager import JobManager
from core.data_ingestion import DataIngestion
from core.profile_engine import ProfileEngine
from core.quality_engine import QualityEngine
from core.eda_planner import EDAPlanner
from core.eda_engine import EDAEngine
from core.duckdb_engine import DuckDBEngine

router = APIRouter()
sql_engine = SQLAnalyticsModel()

def _get_project_and_df(project_id: str):
    proj = ProjectModel(project_id, f"Project {project_id}")
    df = proj.get_cleaned_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found for this project ID.")
    return proj, df

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    
    try:
        ingest_res = DataIngestion.ingest_file(contents, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest file: {str(e)}")

    name = ingest_res["workspace_name"]
    project_id = ingest_res["project_id"]
    parquet_path = ingest_res["parquet_path"]
    
    # Initialize ProjectModel directory, metadata, and ensure raw/cleaned parquet files are copied
    import shutil
    proj = ProjectModel(project_id, name)
    shutil.copy2(parquet_path, proj.raw_df_path)
    shutil.copy2(parquet_path, proj.cleaned_df_path)

    # Register in SQL engine for analytical queries
    clean_p = str(parquet_path).replace("\\", "/")
    sql_engine.con.execute(f"CREATE OR REPLACE VIEW \"ds_{project_id}\" AS SELECT * FROM read_parquet('{clean_p}');")
    sql_engine.registered_tables[f"ds_{project_id}"] = f"Table with {ingest_res['rows']} rows, {ingest_res['columns']} columns"
    
    return {
        "message": ingest_res["message"],
        "project_id": project_id,
        "dataset_hash": ingest_res["dataset_hash"],
        "name": name,
        "rows": ingest_res["rows"],
        "columns": ingest_res["columns"],
        "is_duplicate": ingest_res.get("is_duplicate", False),
        "ingestion_time_ms": ingest_res.get("ingestion_time_ms", 0.0)
    }

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@router.get("/jobs")
def list_jobs(limit: int = 20):
    return JobManager.list_jobs(limit=limit)

@router.get("/dataset/{project_id}/quick-eda")
def get_quick_eda(project_id: str, goal: str = "General Business Intelligence"):
    proj = ProjectModel.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    parquet_path = proj.get_active_parquet_path()
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Parquet dataset not found.")
        
    ds_hash = project_id
    profile = ProfileEngine.profile_dataset_fast(parquet_path, dataset_hash=ds_hash)
    plan = EDAPlanner.create_eda_plan(profile, goal=goal)
    quick_res = EDAEngine.run_quick_eda(parquet_path, profile, plan)
    return quick_res

@router.post("/dataset/{project_id}/deep-eda")
def launch_deep_eda(project_id: str, goal: str = "General Business Intelligence"):
    proj = ProjectModel.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    parquet_path = proj.get_active_parquet_path()
    
    def runner(job_id):
        JobManager.update_progress(job_id, 20, "STAGE 6: EDA Planning", "Formulating priority analysis matrix...")
        profile = ProfileEngine.profile_dataset_fast(parquet_path, dataset_hash=project_id)
        plan = EDAPlanner.create_eda_plan(profile, goal=goal)
        JobManager.update_progress(job_id, 60, "STAGE 7: Deep Analytics", "Fitting Isolation Forest and ANOVA driver rankings...")
        deep_res = EDAEngine.run_deep_eda(parquet_path, profile, plan)
        JobManager.update_progress(job_id, 95, "STAGE 8: AI Insights", "Persisting Deep EDA analytical cache...")
        return deep_res

    job_id = JobManager.submit("deep_eda", runner, stages=[
        "STAGE 6: EDA Planning",
        "STAGE 7: Deep Analytics",
        "STAGE 8: AI Insights",
        "STAGE 9: Ready"
    ])
    return {"job_id": job_id, "status": "submitted", "message": "Deep EDA job launched in background."}

@router.get("/dataset/{project_id}/profile")
def get_dataset_profile(project_id: str):
    _, df = _get_project_and_df(project_id)
    profile = DataProfilerModel.profile_dataset(df)
    quality = DataQualityModel.calculate_quality_score(df)
    return {
        "profile": profile,
        "quality": quality
    }

@router.post("/dataset/{project_id}/clean")
def clean_dataset(project_id: str, request: CleanRequest):
    proj, df = _get_project_and_df(project_id)
    actions = [a.model_dump() for a in request.actions]
    cleaned_df, log = DataCleanerModel.apply_cleaning_actions(df, actions)
    
    proj.save_cleaned_dataset(cleaned_df, log)
    sql_engine.register_dataframe(f"ds_{project_id}", cleaned_df)

    return {
        "message": "Dataset cleaned successfully.",
        "remaining_rows": len(cleaned_df),
        "cleaning_log": log
    }

@router.get("/dataset/{project_id}/eda")
def get_eda_results(project_id: str):
    _, df = _get_project_and_df(project_id)
    return {
        "numerical": EDAModel.analyze_numerical(df),
        "categorical": EDAModel.analyze_categorical(df),
        "relationships": EDAModel.analyze_relationships(df),
        "time_trends": EDAModel.analyze_time_trends(df)
    }

@router.post("/dataset/{project_id}/features")
def generate_features(project_id: str, request: FeatureRequest):
    proj, df = _get_project_and_df(project_id)
    spec = request.model_dump()
    new_df, message = FeatureEngineeringModel.apply_feature(df, spec)
    
    proj.save_cleaned_dataset(new_df, proj.get_cleaning_history())
    sql_engine.register_dataframe(f"ds_{project_id}", new_df)

    return {
        "message": message,
        "columns": list(new_df.columns),
        "rows": len(new_df)
    }

@router.post("/dataset/{project_id}/sql")
def execute_sql(project_id: str, request: SQLQueryRequest):
    proj, df = _get_project_and_df(project_id)
    table_name = f"ds_{project_id}"
    sql_engine.register_dataframe(table_name, df)
    
    res = sql_engine.execute_query(request.query)
    if res.get("success"):
        proj.save_query(request.query)
        # Convert df to dict for JSON response, replacing NaN with None for JSON compliance
        df_res = res["dataframe"]
        if df_res is not None:
            clean_res = df_res.replace({np.nan: None}).where(pd.notnull(df_res), None)
            res["data"] = clean_res.to_dict(orient="records")
        else:
            res["data"] = []
        del res["dataframe"]
    return res

@router.post("/dataset/{project_id}/chat")
def chat_with_data(project_id: str, request: ChatQueryRequest):
    _, df = _get_project_and_df(project_id)
    table_name = f"ds_{project_id}"
    sql_engine.register_dataframe(table_name, df)
    
    chat_model = AIChatModel(sql_engine)
    result = chat_model.answer_query(request.message, df, table_name)
    if "data" in result and isinstance(result["data"], pd.DataFrame):
        clean_df = result["data"].head(20).replace({np.nan: None}).where(pd.notnull(result["data"].head(20)), None)
        result["data"] = clean_df.to_dict(orient="records")
    if "chart_figure" in result:
        del result["chart_figure"] # JSON serialization
    return result

@router.get("/dataset/{project_id}/insights")
def get_insights(project_id: str):
    _, df = _get_project_and_df(project_id)
    profile = DataProfilerModel.profile_dataset(df)
    quality = DataQualityModel.calculate_quality_score(df)
    eda = {
        "numerical": EDAModel.analyze_numerical(df),
        "categorical": EDAModel.analyze_categorical(df),
        "relationships": EDAModel.analyze_relationships(df),
        "time_trends": EDAModel.analyze_time_trends(df)
    }
    insights = AIInsightsModel.generate_insights(profile, eda, quality)
    return {
        "insights": insights
    }

@router.post("/dataset/{project_id}/dashboard")
def get_dashboard(project_id: str, request: DashboardFilterRequest):
    _, df = _get_project_and_df(project_id)
    filtered_df = DashboardModel.apply_filters(df, request.filters)
    return DashboardModel.build_dashboard_config(filtered_df)

@router.get("/dataset/{project_id}/export")
def export_dataset(project_id: str, format: str = "csv"):
    proj, df = _get_project_and_df(project_id)
    export_file = EXPORTS_DIR / f"{proj.name}_{project_id}.{format}"
    
    if format == "csv":
        df.to_csv(export_file, index=False)
        return FileResponse(export_file, filename=f"{proj.name}_cleaned.csv", media_type="text/csv")
    elif format in ["xlsx", "excel"]:
        export_file = EXPORTS_DIR / f"{proj.name}_{project_id}.xlsx"
        df.to_excel(export_file, index=False)
        return FileResponse(export_file, filename=f"{proj.name}_cleaned.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: csv, xlsx")
