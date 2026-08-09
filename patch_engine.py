import sys
import os
import re

file_path = "src/lokum_engine/rag/engine.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

rich_imports = """
try:
    from rich.console import Console
    from rich.prompt import Prompt, IntPrompt
    console = Console()
    RICH_ENABLED = True
except ImportError:
    console = None
    RICH_ENABLED = False

def _print_info(msg: str):
    if console:
        console.print(f"[bold cyan]ℹ [RAG][/bold cyan] {msg}")
    else:
        print(f"[RAG] {msg}")

def _print_success(msg: str):
    if console:
        console.print(f"[bold green]✓ [RAG][/bold green] {msg}")
    else:
        print(f"[RAG] {msg}")

def _print_warning(msg: str):
    if console:
        console.print(f"[bold yellow]⚠ [RAG] Warning:[/bold yellow] {msg}")
    else:
        print(f"[RAG] Warning: {msg}")

def _print_error(msg: str):
    if console:
        console.print(f"[bold red]❌ [RAG] Error:[/bold red] {msg}")
    else:
        print(f"[RAG] Error: {msg}")

logger = logging.getLogger(__name__)
"""

content = content.replace("logger = logging.getLogger(__name__)", rich_imports)

content = content.replace('print("Warning: faiss not installed', '_print_warning("faiss not installed')
content = content.replace('print("Warning: PyMuPDF not installed', '_print_warning("PyMuPDF not installed')
content = content.replace('print("Warning: python-docx not installed', '_print_warning("python-docx not installed')
content = content.replace('print("Warning: pillow not installed', '_print_warning("pillow not installed')
content = content.replace('print(\n        "Warning: pytesseract not installed', '_print_warning(\n        "pytesseract not installed')

custom_profile = """    "fab": RAGQualityProfile(
        name="fab",
        chunk_size=1000,
        overlap=150,
        embedding_model="all-mpnet-base-v2",
        fetch_multiplier=20,
        fetch_min=80,
        fetch_cap=1000,
        rerank_model_name="BAAI/bge-reranker-base",
        rerank_multiplier=4,
    ),
    "custom": RAGQualityProfile(
        name="custom",
        chunk_size=800,
        overlap=100,
        embedding_model="all-MiniLM-L6-v2",
        fetch_multiplier=10,
        fetch_min=50,
        fetch_cap=500,
        rerank_model_name=None,
        rerank_multiplier=1,
    ),"""
content = content.replace('    "fab": RAGQualityProfile(\n        name="fab",\n        chunk_size=1000,\n        overlap=150,\n        embedding_model="all-mpnet-base-v2",\n        fetch_multiplier=20,\n        fetch_min=80,\n        fetch_cap=1000,\n        rerank_model_name="BAAI/bge-reranker-base",\n        rerank_multiplier=4,\n    ),', custom_profile)

content = content.replace('    if v in ("fab", "fabulous", "faboulous", "fabolous", "fabulus", "high", "hq", "best"):\n        return "fab"', '    if v in ("fab", "fabulous", "faboulous", "fabolous", "fabulus", "high", "hq", "best"):\n        return "fab"\n    if v in ("custom", "user", "manual"):\n        return "custom"')

interactive_method = """    def _configure_custom_profile(self):
        \"\"\"Interactively configure the custom quality profile.\"\"\"
        import sys
        if not sys.stdout.isatty():
            _print_warning("Not running in an interactive terminal. Skipping custom profile setup.")
            return
            
        if not console:
            _print_warning("Rich library not installed, skipping interactive setup. Using default custom values.")
            return
            
        _print_info("Configuring [bold magenta]CUSTOM[/bold magenta] RAG Quality Profile")
        try:
            chunk_size = IntPrompt.ask("Enter chunk size (words/tokens)", default=int(self.quality_profile.chunk_size))
            overlap = IntPrompt.ask("Enter chunk overlap", default=int(self.quality_profile.overlap))
            embedding_model = Prompt.ask("Enter embedding model name", default=str(self.quality_profile.embedding_model))
            fetch_multiplier = IntPrompt.ask("Enter fetch multiplier", default=int(self.quality_profile.fetch_multiplier))
            fetch_min = IntPrompt.ask("Enter minimum chunks to fetch", default=int(self.quality_profile.fetch_min))
            fetch_cap = IntPrompt.ask("Enter maximum chunks to fetch", default=int(self.quality_profile.fetch_cap))
            
            use_rerank = Prompt.ask("Use reranker model? (y/n)", choices=["y", "n"], default="n")
            rerank_model_name = None
            rerank_multiplier = 1
            if use_rerank == "y":
                rerank_model_name = Prompt.ask("Enter reranker model name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
                rerank_multiplier = IntPrompt.ask("Enter rerank multiplier", default=3)

            self.quality_profile = RAGQualityProfile(
                name="custom",
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_model=embedding_model,
                fetch_multiplier=fetch_multiplier,
                fetch_min=fetch_min,
                fetch_cap=fetch_cap,
                rerank_model_name=rerank_model_name,
                rerank_multiplier=rerank_multiplier
            )
            _print_success("Custom profile configured successfully!")
        except Exception as e:
            _print_error(f"Interactive setup aborted or failed: {e}. Using defaults.")

"""
content = content.replace('    def __init__(self, storage_dir: str | None = None, quality: str | None = None):', interactive_method + '    def __init__(self, storage_dir: str | None = None, quality: str | None = None):')

init_patch = """        # ---- Quality profile (chunking + retrieval + embedding model selection)
        env_quality = (os.environ.get("LOKUMAI_RAG_QUALITY") or "").strip()
        self.quality_profile = get_rag_quality_profile(quality or env_quality)
        
        if self.quality_profile.name == "custom":
            self._configure_custom_profile()"""
content = content.replace('        # ---- Quality profile (chunking + retrieval + embedding model selection)\n        env_quality = (os.environ.get("LOKUMAI_RAG_QUALITY") or "").strip()\n        self.quality_profile = get_rag_quality_profile(quality or env_quality)', init_patch)

# Simple replacements
content = re.sub(r'print\(\s*f?"\[RAG\] Error (.*?)"\s*\)', r'_print_error(f"\1")', content)
content = re.sub(r'print\(\s*f?"\[RAG\] (.*?error.*?)"\s*\)', r'_print_error(f"\1")', content, flags=re.IGNORECASE)
content = re.sub(r'print\(\s*f?"Warning: (.*?)"\s*\)', r'_print_warning(f"\1")', content)
content = re.sub(r'print\(\s*f?"\[RAG\] Loaded (.*?)"\s*\)', r'_print_success(f"Loaded \1")', content)
content = re.sub(r'print\(\s*f?"\[RAG\] Saved (.*?)"\s*\)', r'_print_success(f"Saved \1")', content)
content = re.sub(r'print\(\s*f?"\[RAG\] Index reset(.*?)"\s*\)', r'_print_success(f"Index reset\1")', content)
content = re.sub(r'print\(\s*f?"\[RAG\] (.*?)"\s*\)', r'_print_info(f"\1")', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patch applied.")
