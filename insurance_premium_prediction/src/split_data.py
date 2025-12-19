import pandas as pd
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parents[1] /
    "data"
)

RAW_PATH = DATA_DIR / "raw" / "insurance_raw.parquet"


def split_and_store():
    df = pd.read_parquet(RAW_PATH)

    train_df = df.iloc[:700_000]
    val_df = df.iloc[700_000:900_000]
    live_df = df.iloc[900_000:]

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    live_df.to_parquet(processed_dir / "live.parquet", index=False)

    print("Data split completed:")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Live:", live_df.shape)


if __name__ == "__main__":
    split_and_store()
