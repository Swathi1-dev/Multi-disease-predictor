from fastapi import FastAPI

from src.backend.api.routes import router

app = FastAPI(
    title="Dr. ML Prediction APP", description="Multi diseases prediction system"
)

app.include_router(router, prefix="/api")
