# fastapi_app.py
from fastapi import FastAPI
from crm.api import notifications

app = FastAPI(title="BoWiseCRM API", version="1.0")

app.include_router(notifications.router, prefix="/api")

@app.get("/")
def home():
    return {"message": "BoWiseCRM API running"}

# Run using:
# uvicorn fastapi_app:app --reload --port 8001