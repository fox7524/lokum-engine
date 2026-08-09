import logging
import os

logger = logging.getLogger(__name__)

class LocalGenerator:
    """
    Local Text Generator using mlx-lm, highly optimized for Apple Silicon.
    """
    def __init__(self, model_id: str = "mlx-community/Phi-3-mini-4k-instruct-4bit", lazy_load: bool = False):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        if not lazy_load:
            self.load()
            
    def load(self):
        try:
            from mlx_lm import load
            logger.info(f"Loading mlx-lm model: {self.model_id}...")
            self.model, self.tokenizer = load(self.model_id)
            logger.info("mlx-lm model loaded successfully.")
        except ImportError:
            raise ImportError("mlx-lm is not installed. Please install it using `pip install mlx-lm`")
        except Exception as e:
            logger.error(f"Failed to load mlx-lm model: {e}")
            raise e
            
    def generate(self, query: str, contexts: list[str], max_tokens: int = 512) -> str:
        if not self.model or not self.tokenizer:
            self.load()
            
        try:
            from mlx_lm import generate
        except ImportError:
            return "Error: mlx-lm is not installed."

        context_str = "\n\n---\n\n".join(contexts)
        prompt = (
            f"You are a helpful assistant. Use the following pieces of context to answer the question at the end.\n"
            f"If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        try:
            return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation error: {e}"
