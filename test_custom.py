import os
import sys

os.environ["LOKUMAI_RAG_QUALITY"] = "custom"
from lokum_engine.rag.engine import RAGEngine

print("Initializing RAG engine...")
engine = RAGEngine(quality="custom")
print(f"Profile: {engine.quality_profile.name}")
print(f"Chunk size: {engine.quality_profile.chunk_size}")
print(f"Overlap: {engine.quality_profile.overlap}")
