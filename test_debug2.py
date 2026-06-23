import faiss
import numpy as np
dim = 384
index = faiss.IndexFlatL2(dim)
emb = np.random.rand(2, dim).astype("float32")
print("Adding to index")
index.add(emb)
print("Added")
