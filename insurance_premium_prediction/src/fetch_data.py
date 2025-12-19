import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from common_utils.db_utils import DatabaseConnector


def fetch_and_store_data():
    db = DatabaseConnector(
        db_path=PROJECT_ROOT / "database" / "Regression.db"
    )

    df = db.fetch_table("Insurance_Prediction")

    output_path = (
        PROJECT_ROOT /
        "insurance_premium_prediction" /
        "data" /
        "raw" /
        "insurance_raw.parquet"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    fetch_and_store_data()
