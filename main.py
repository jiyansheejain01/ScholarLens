from src.pipeline import ReviewerPipeline

if __name__ == "__main__":
    pipeline = ReviewerPipeline(config_path="config/config.yaml")

    # Example paper — replace with any real paper
    title = "Attention Is All You Need"
    abstract = """
    We propose a new simple network architecture, the Transformer, based solely on 
    attention mechanisms, dispensing with recurrence and convolutions entirely. 
    Experiments on two machine translation tasks show these models to be superior 
    in quality while being more parallelizable and requiring significantly less time 
    to train. We achieve state-of-the-art results on English-to-German and 
    English-to-French translation tasks.
    """

    result = pipeline.review(title=title, abstract=abstract)

    print("\n===== REVIEW REPORT =====")
    print(f"Title: {result['title']}")
    print(f"\nQuality Scores:")
    for k, v in result['quality'].items():
        print(f"  {k}: {v}")

    print(f"\nSuggested Missing Citations ({len(result['missing_citations'])}):")
    for i, cite in enumerate(result['missing_citations'], 1):
        print(f"\n  {i}. {cite['title']}")
        print(f"     Published : {cite['published']}")
        print(f"     Similarity: {cite['similarity_score']}")
        print(f"     URL       : {cite['url']}")