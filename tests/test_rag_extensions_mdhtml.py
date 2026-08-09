def test_parse_markdown():
    from lokum_engine.rag.engine import RAGEngine
    import os
    with open("test.md", "w") as f:
        f.write("# Title\n\nSome **bold** text.")
    
    engine = RAGEngine()
    chunks = engine.process_file("test.md")
    assert len(chunks) > 0
    assert "bold text" in chunks[0].lower() or "title" in chunks[0].lower()
    os.remove("test.md")

def test_parse_html():
    from lokum_engine.rag.engine import RAGEngine
    import os
    with open("test.html", "w") as f:
        f.write("<html><body><h1>Hello</h1><p>World</p></body></html>")
    
    engine = RAGEngine()
    chunks = engine.process_file("test.html")
    assert len(chunks) > 0
    assert "hello" in chunks[0].lower()
    assert "world" in chunks[0].lower()
    os.remove("test.html")
