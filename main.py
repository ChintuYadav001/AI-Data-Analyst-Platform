from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.auth_routes import router as auth_router

app = FastAPI(
    title="AI Data Analyst API",
    description="Production-grade API for automated end-to-end data analytics workflow",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(auth_router)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Data Analyst API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
