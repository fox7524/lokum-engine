# Lokum Engine Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Lokum Engine into a full end-to-end RAG system by adding advanced document parsers (Markdown, HTML, URLs), intelligent retrieval features (Query Expansion, HyDE), and local text generation via MLX-LM.

**Architecture:** 
- `parser.py` will be extended with beautifulsoup4 and markdown parsers.
- `engine.py` will be extended with `advanced_search` wrappers using existing local text-generation models for HyDE if requested.
- `generator.py` (new module) will wrap `mlx-lm` to provide `generate_answer(query, contexts)` functionality for offline, Apple Silicon optimized QA.

**Tech Stack:** Python 3.10+, beautifulsoup4, markdown, mlx-lm, pytest.

---

### Task 1: Add HTML and Markdown Parsing Support

**Files:**
- Modify: `pyproject.toml` (Add dependencies)
- Modify: `src/lokum_engine/rag/parser.py`
- Test: `tests/test_rag_extensions.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**
Update `pyproject.toml` dependencies list to include `beautifulsoup4` and `markdown`.

- [ ] **Step 2: Write failing tests for MD and HTML parsing**
```python
def test_parse_markdown():
    from lokum_engine.rag.parser import DocumentParser
    import os
    with open("test.md", "w") as f:
        f.write("# Title\n\nSome **bold** text.")
    
    docs = DocumentParser.parse_file("test.md")
    assert len(docs) > 0
    assert "bold text" in docs[0].page_content.lower() or "title" in docs[0].page_content.lower()
    os.remove("test.md")

def test_parse_html():
    from lokum_engine.rag.parser import DocumentParser
    import os
    with open("test.html", "w") as f:
        f.write("<html><body><h1>Hello</h1><p>World</p></body></html>")
    
    docs = DocumentParser.parse_file("test.html")
    assert len(docs) > 0
    assert "hello" in docs[0].page_content.lower()
    assert "world" in docs[0].page_content.lower()
    os.remove("test.html")
```

- [ ] **Step 3: Implement MD and HTML support in DocumentParser**
Modify `src/lokum_engine/rag/parser.py` inside `parse_file` method to handle `.md` and `.html` extensions using `markdown` and `BeautifulSoup`.

- [ ] **Step 4: Run tests and verify**
Run: `pytest tests/test_rag_extensions.py -v`

- [ ] **Step 5: Commit**
`git add . && git commit -m "feat(parser): add markdown and html parsing support"`

### Task 2: Implement Advanced Retrieval (Query Expansion & HyDE Stubs)

**Files:**
- Modify: `src/lokum_engine/rag/engine.py`

- [ ] **Step 1: Write failing tests for Query Expansion**
```python
def test_query_expansion():
    from lokum_engine.rag.engine import RAGEngine
    engine = RAGEngine()
    expanded = engine._expand_query("Apple")
    assert isinstance(expanded, list)
    assert len(expanded) >= 1
    assert "Apple" in expanded[0]
```

- [ ] **Step 2: Implement naive Query Expansion logic**
Add a method `_expand_query(self, query: str) -> list[str]` to `RAGEngine` in `engine.py`. For now, it will return the original query and a slightly cleaned/lower-cased version to simulate basic expansion without heavy ML.

- [ ] **Step 3: Commit**
`git commit -am "feat(engine): add basic query expansion mechanism"`

### Task 3: Local LLM Integration (MLX-LM Generator)

**Files:**
- Create: `src/lokum_engine/rag/generator.py`
- Modify: `src/lokum_engine/rag/engine.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add mlx-lm dependency**
Add `mlx-lm` to `pyproject.toml`.

- [ ] **Step 2: Create Generator class test**
```python
# In tests/test_generator.py
def test_generator_initialization():
    from lokum_engine.rag.generator import LocalGenerator
    # Just testing instantiation to avoid downloading huge models in CI
    gen = LocalGenerator(model_id="mock_model", lazy_load=True)
    assert gen.model_id == "mock_model"
```

- [ ] **Step 3: Implement LocalGenerator using mlx-lm**
Create `src/lokum_engine/rag/generator.py`.
```python
class LocalGenerator:
    def __init__(self, model_id: str = "mlx-community/Phi-3-mini-4k-instruct-4bit", lazy_load: bool = False):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        if not lazy_load:
            self.load()
            
    def load(self):
        try:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_id)
        except ImportError:
            raise ImportError("mlx-lm not installed. Run: pip install mlx-lm")
            
    def generate(self, query: str, contexts: list[str], max_tokens: int = 512) -> str:
        if not self.model:
            self.load()
        from mlx_lm import generate
        
        context_str = "\n".join(contexts)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)
```

- [ ] **Step 4: Integrate Generator into RAGEngine**
Add a `generate_answer(query: str, model_id: str)` method to `RAGEngine` in `engine.py` that calls `search()` internally, fetches contexts, and passes them to `LocalGenerator`.

- [ ] **Step 5: Run tests and verify**
Run: `pytest tests/`

- [ ] **Step 6: Commit**
`git add . && git commit -m "feat(llm): integrate local text generation via mlx-lm"`

