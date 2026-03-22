import os
import torch
import numpy as np
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class T5Encoder:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['model']['name']
        self.max_length = config['model']['max_length']
        self.batch_size = config['model']['batch_size']
        self.device = torch.device(config['model']['device'])
        self.embeddings_path = Path(config['paths']['embeddings'])
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """
        Averages token embeddings into one vector per paper.
        This is the standard way to get a sentence-level embedding.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        Processes in batches to avoid memory issues.
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch = texts[i: i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pooling(output, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def save_embeddings(self, embeddings: np.ndarray, filename: str = "paper_embeddings.pkl"):
        """Caches embeddings to disk so you don't re-encode every run."""
        out_path = self.embeddings_path / filename
        with open(out_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {out_path}")

    def load_embeddings(self, filename: str = "paper_embeddings.pkl") -> np.ndarray:
        """Loads cached embeddings from disk."""
        file_path = self.embeddings_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No embeddings found at {file_path}. Run encode() first.")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
