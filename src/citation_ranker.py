import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CitationRanker:
    def __init__(self, config: dict):
        self.top_k = config['ranking']['top_k']
        self.threshold = config['ranking']['similarity_threshold']

    def rank(self, paper_embedding: np.ndarray, candidate_papers: list[dict],
             candidate_embeddings: np.ndarray) -> list[dict]:
        """
        Compares the submitted paper embedding against all arXiv candidate embeddings.
        Returns top-k most similar papers above the similarity threshold.
        """
        # paper_embedding shape: (1, dim) — reshape if needed
        if paper_embedding.ndim == 1:
            paper_embedding = paper_embedding.reshape(1, -1)

        similarities = cosine_similarity(paper_embedding, candidate_embeddings)[0]

        # Attach scores to each candidate
        scored = []
        for i, paper in enumerate(candidate_papers):
            score = float(similarities[i])
            if score >= self.threshold:
                scored.append({**paper, 'similarity_score': round(score, 4)})

        # Sort by score descending, return top-k
        scored.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_citations = scored[:self.top_k]

        print(f"Found {len(top_citations)} suggested missing citations.")
        return top_citations