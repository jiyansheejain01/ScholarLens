import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config['paths']['raw_data'])
        self.processed_path = Path(config['paths']['processed_data'])
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_openreview(self, split: str = "train") -> pd.DataFrame:
        """
        Loads the OpenReview dataset from HuggingFace.
        Contains real ICLR paper + review pairs.
        """
        print(f"Loading OpenReview dataset ({split} split)...")
        dataset = load_dataset("aclanthology/openreview", split=split, trust_remote_code=True)
        df = pd.DataFrame(dataset)
        print(f"Loaded {len(df)} paper-review pairs.")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and extracts the fields we need:
        - paper title
        - paper abstract
        - review text
        - human-suggested citations (ground truth)
        """
        print("Preprocessing dataset...")

        # Keep only relevant columns
        columns_needed = ['title', 'abstract', 'review', 'references']
        available = [col for col in columns_needed if col in df.columns]
        df = df[available].copy()

        # Drop rows with missing abstracts (core input for encoder)
        df.dropna(subset=['abstract'], inplace=True)

        # Clean whitespace
        df['abstract'] = df['abstract'].str.strip()
        if 'title' in df.columns:
            df['title'] = df['title'].str.strip()

        # Combine title + abstract into one text field for encoding
        df['full_text'] = df.apply(
            lambda row: f"{row.get('title', '')} {row.get('abstract', '')}".strip(),
            axis=1
        )

        print(f"Preprocessed {len(df)} valid records.")
        return df

    def save_processed(self, df: pd.DataFrame, filename: str = "processed_papers.csv"):
        """Saves cleaned data to the processed folder."""
        out_path = self.processed_path / filename
        df.to_csv(out_path, index=False)
        print(f"Saved processed data to {out_path}")

    def load_processed(self, filename: str = "processed_papers.csv") -> pd.DataFrame:
        """Loads already-processed data (avoids re-downloading)."""
        file_path = self.processed_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No processed file found at {file_path}. Run preprocess first.")
        return pd.read_csv(file_path)

    def run(self) -> pd.DataFrame:
        """Full pipeline: download → preprocess → save → return."""
        df_raw = self.load_openreview()
        df_clean = self.preprocess(df_raw)
        self.save_processed(df_clean)
        return df_clean
