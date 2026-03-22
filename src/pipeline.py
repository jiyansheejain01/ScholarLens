import yaml
import numpy as np
from pathlib import Path
from src.data_loader import DataLoader
from src.encoder import T5Encoder
from src.arxiv_search import ArxivSearcher
from src.citation_ranker import CitationRanker
from src.quality_assessor import QualityAssessor

class ReviewerPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.encoder   = T5Encoder(self.config)
        self.searcher  = ArxivSearcher(self.config)
        self.ranker    = CitationRanker(self.config)
        self.assessor  = QualityAssessor(self.config)

    def review(self, title: str, abstract: str) -> dict:
        """
        Full review pipeline for a single paper.
        Input: title + abstract
        Output: quality scores + missing citation suggestions
        """
        print("\n=== Reviewer-T5 Pipeline Started ===")
        full_text = f"{title} {abstract}"

        # Step 1: Quality assessment
        print("\n[1/4] Assessing paper quality...")
        quality = self.assessor.assess(full_text)
        print(f"Quality: {quality}")

        # Step 2: Encode the submitted paper
        print("\n[2/4] Encoding paper...")
        paper_embedding = self.encoder.encode([full_text])

        # Step 3: Search arXiv for recent related papers
        print("\n[3/4] Searching arXiv for recent work...")
        keywords = self._extract_keywords(abstract)
        candidates = self.searcher.search(keywords)

        if not candidates:
            print("No recent arXiv papers found.")
            return {'quality': quality, 'missing_citations': []}

        # Step 4: Encode candidates and rank by similarity
        print("\n[4/4] Ranking missing citations...")
        candidate_texts = [c['full_text'] for c in candidates]
        candidate_embeddings = self.encoder.encode(candidate_texts)
        missing_citations = self.ranker.rank(paper_embedding, candidates, candidate_embeddings)

        print("\n=== Review Complete ===")
        return {
            'title': title,
            'quality': quality,
            'missing_citations': missing_citations
        }

    def _extract_keywords(self, abstract: str, top_n: int = 5) -> list[str]:
        """
        Simple keyword extraction: removes stopwords, returns most frequent content words.
        You can replace this with KeyBERT later for better results.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'in', 'of', 'to', 'we',
            'is', 'are', 'for', 'with', 'on', 'that', 'this', 'our',
            'by', 'from', 'be', 'as', 'at', 'it', 'which', 'have'
        }
        words = abstract.lower().split()
        words = [w.strip('.,()[]') for w in words if w not in stopwords and len(w) > 4]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq, key=freq.get, reverse=True)
        return sorted_words[:top_n]