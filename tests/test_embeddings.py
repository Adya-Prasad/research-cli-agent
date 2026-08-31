def test_sentence_transformer_adaptter_is_importable() -> None:
    from research_agent.retrieval.embeddings import SentenceTransformerEmbedder

    assert SentenceTransformerEmbedder.__name__ == "SentenceTransformerEmbedder"
