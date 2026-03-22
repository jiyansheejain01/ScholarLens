import arxiv
import time
from datetime import datetime, timedelta

class ArxivSearcher:
    def __init__(self, config: dict):
        self.config = config
        self.max_results = config['arxiv']['max_results']
        self.months_back = config['arxiv']['months_back']
        self.categories = config['arxiv']['categories']
        self.client = arxiv.Client()

    def _build_query(self, keywords: list[str]) -> str:
        """
        Builds an arXiv query string from keywords + category filters.
        Example: (ti:attention OR abs:attention) AND cat:cs.LG
        """
        keyword_query = " OR ".join([f"abs:{kw}" for kw in keywords])
        category_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        return f"({keyword_query}) AND ({category_query})"

    def _is_recent(self, paper) -> bool:
        """Checks if a paper was published within the configured months_back window."""
        cutoff = datetime.now(paper.published.tzinfo) - timedelta(days=30 * self.months_back)
        return paper.published >= cutoff

    def search(self, keywords: list[str]) -> list[dict]:
        """
        Searches arXiv for recent papers matching the given keywords.
        Returns a list of paper dicts with title, abstract, authors, url.
        """
        query = self._build_query(keywords)
        print(f"Searching arXiv: {query}")

        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        results = []
        for paper in self.client.results(search):
            if not self._is_recent(paper):
                continue
            results.append({
                'arxiv_id': paper.entry_id,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [a.name for a in paper.authors],
                'published': str(paper.published.date()),
                'url': paper.pdf_url,
                'full_text': f"{paper.title} {paper.summary}"
            })
            time.sleep(0.5)  # be polite to arXiv servers

        print(f"Found {len(results)} recent papers.")
        return results
