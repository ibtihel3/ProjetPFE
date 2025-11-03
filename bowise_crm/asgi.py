# bowise_crm/asgi.py
import os
from django.core.asgi import get_asgi_application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_app import app as fastapi_app

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bowise_crm.settings")

# Initialize Django ASGI
django_asgi_app = get_asgi_application()

# FastAPI middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main ASGI app cpmbining both
main_app = FastAPI(title="BoWise Unified App")

# Mount FastAPI under /api
main_app.mount("/api", fastapi_app)

# Mount Django for all other routes (root)
main_app.mount("/", django_asgi_app)

# Final entrypoint
app = main_app  # Uvicorn entrypoint
