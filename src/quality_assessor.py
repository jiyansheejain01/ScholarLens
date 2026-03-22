import numpy as np

class QualityAssessor:
    def __init__(self, config: dict):
        self.weights = config['quality']['weights']
        self.min_score = config['quality']['min_score']
        self.max_score = config['quality']['max_score']

    def _score_coherence(self, text: str) -> float:
        """
        Estimates coherence by checking structural completeness.
        A well-structured abstract has: problem, method, result keywords.
        """
        indicators = [
            'propose', 'present', 'introduce',   # contribution signals
            'method', 'approach', 'model',         # methodology signals
            'result', 'show', 'achieve', 'outperform',  # result signals
            'experiment', 'evaluate', 'dataset'    # validation signals
        ]
        text_lower = text.lower()
        hits = sum(1 for word in indicators if word in text_lower)
        return min(hits / len(indicators), 1.0)

    def _score_completeness(self, text: str) -> float:
        """
        Checks if the paper covers expected sections/topics.
        Longer, denser abstracts tend to be more complete.
        """
        word_count = len(text.split())
        # Abstracts under 50 words are likely incomplete
        # Abstracts around 150-250 words are ideal
        if word_count < 50:
            return 0.2
        elif word_count < 100:
            return 0.5
        elif word_count <= 300:
            return 1.0
        else:
            return 0.8  # very long abstracts can be unfocused

    def _score_novelty(self, text: str) -> float:
        """
        Estimates novelty by looking for claim-making language.
        """
        novelty_words = [
            'novel', 'new', 'first', 'state-of-the-art', 'sota',
            'outperform', 'superior', 'improve', 'advance', 'beyond'
        ]
        text_lower = text.lower()
        hits = sum(1 for word in novelty_words if word in text_lower)
        return min(hits / len(novelty_words), 1.0)

    def assess(self, text: str) -> dict:
        """
        Runs all three scoring functions and returns a weighted final score.
        """
        coherence   = self._score_coherence(text)
        completeness = self._score_completeness(text)
        novelty     = self._score_novelty(text)

        final_score = (
            self.weights['coherence']    * coherence +
            self.weights['completeness'] * completeness +
            self.weights['novelty']      * novelty
        )

        return {
            'coherence':    round(coherence, 3),
            'completeness': round(completeness, 3),
            'novelty':      round(novelty, 3),
            'final_score':  round(final_score, 3),
            'grade': self._grade(final_score)
        }

    def _grade(self, score: float) -> str:
        if score >= 0.75:   return "Strong"
        elif score >= 0.5:  return "Moderate"
        elif score >= 0.25: return "Weak"
        else:               return "Insufficient"