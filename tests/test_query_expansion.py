def test_query_expansion():
    from lokum_engine.rag.engine import RAGEngine
    engine = RAGEngine()
    expanded = engine._expand_query("Apple MacBook Pro 2024!")
    assert isinstance(expanded, list)
    assert len(expanded) >= 1
    assert "Apple MacBook Pro 2024!" in expanded
    assert "apple macbook pro 2024 " in expanded
