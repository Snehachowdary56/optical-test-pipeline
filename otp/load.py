from pathlib import Path

import pandas as pd

from otp import REQUIRED_COLUMNS


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Reads every CSV in the folder into one DataFrame and checks the columns are present."""
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
