from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Data subfolders
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_DIR = DATA_DIR / "validation"
SCHEMA_DIR = DATA_DIR / "schemas"
SEED_DIR = DATA_DIR / "seed"
# Ensure all folders exist
for folder in [SYNTHETIC_DIR, PROCESSED_DIR, VALIDATION_DIR, SCHEMA_DIR, SEED_DIR]:
    folder.mkdir(parents=True,
                 exist_ok=True)
