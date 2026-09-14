import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Callable

class JobManager:
    """Thread-safe background job manager enabling non-blocking execution,

    live progress polling (0-100%), stage tracking, and failure resilience.
    """

    _jobs: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()
    _executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="quantum_worker")

    STANDARD_STAGES = [
        "STAGE 1: Ingestion & Validation",
        "STAGE 2: Parquet Conversion",
        "STAGE 3: Dataset Registration",
        "STAGE 4: Fast Profiling",
        "STAGE 5: Quality Analysis",
        "STAGE 6: EDA Planning",
        "STAGE 7: Parallel Analytics",
        "STAGE 8: AI Context & Insights",
        "STAGE 9: Ready"
    ]

    @classmethod
    def create_job(
        cls,
        job_type: str,
        stages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        st_list = stages or cls.STANDARD_STAGES

        with cls._lock:
            cls._jobs[job_id] = {
                "job_id": job_id,
                "job_type": job_type,
                "status": "pending",
                "progress": 0,
                "current_stage": st_list[0] if st_list else "Initializing",
                "message": "Job registered in processing queue.",
                "stages": [{"name": s, "status": "pending"} for s in st_list],
                "result": None,
                "error": None,
                "metadata": metadata or {},
                "created_at": time.time(),
                "updated_at": time.time(),
                "completed_at": None
            }
        return job_id

    @classmethod
    def update_progress(
        cls,
        job_id: str,
        progress: int,
        current_stage: str,
        message: str = ""
    ) -> None:
        with cls._lock:
            job = cls._jobs.get(job_id)
            if not job:
                return

            job["status"] = "running"
            job["progress"] = max(0, min(100, progress))
            job["current_stage"] = current_stage
            job["message"] = message or f"Processing {current_stage}..."
            job["updated_at"] = time.time()

            # Mark stages completed
            found_current = False
            for st in job["stages"]:
                if st["name"] == current_stage:
                    st["status"] = "running"
                    found_current = True
                elif not found_current:
                    st["status"] = "completed"

    @classmethod
    def complete_job(
        cls,
        job_id: str,
        result: Any,
        message: str = "Job completed successfully."
    ) -> None:
        with cls._lock:
            job = cls._jobs.get(job_id)
            if not job:
                return

            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = message
            job["result"] = result
            job["completed_at"] = time.time()
            job["updated_at"] = time.time()
            for st in job["stages"]:
                st["status"] = "completed"

    @classmethod
    def fail_job(cls, job_id: str, error: str) -> None:
        with cls._lock:
            job = cls._jobs.get(job_id)
            if not job:
                return

            job["status"] = "failed"
            job["error"] = error
            job["message"] = f"Failed: {error}"
            job["completed_at"] = time.time()
            job["updated_at"] = time.time()

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict[str, Any]]:
        with cls._lock:
            job = cls._jobs.get(job_id)
            if not job:
                return None
            return dict(job)

    @classmethod
    def submit(
        cls,
        job_type: str,
        fn: Callable,
        *args,
        stages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Creates a background job and submits fn(job_id, *args, **kwargs) to the thread pool."""
        job_id = cls.create_job(job_type, stages=stages, metadata=metadata)

        def runner():
            try:
                cls.update_progress(job_id, 5, stages[0] if stages else "STAGE 1", "Starting task execution...")
                result = fn(job_id, *args, **kwargs)
                cls.complete_job(job_id, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                cls.fail_job(job_id, str(e))

        cls._executor.submit(runner)
        return job_id

    @classmethod
    def list_jobs(cls, limit: int = 50) -> List[Dict[str, Any]]:
        with cls._lock:
            sorted_jobs = sorted(cls._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            return sorted_jobs[:limit]
