# app/api/notifications.py

from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/notifications")
def get_notifications():
    df = pd.read_csv("data/notifications.csv")
    return df.to_dict(orient="records")