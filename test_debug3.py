from sentence_transformers import SentenceTransformer
import os
print("Loading model")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Encoding")
emb = model.encode(["hello world", "test test"], batch_size=32, show_progress_bar=False)
print("Encoded", emb.shape)
