import tempfile
print("Importing RAGEngine")
from lokum_engine.rag.engine import RAGEngine
print("Creating engine")
temp_dir = tempfile.mkdtemp()
engine = RAGEngine(storage_dir=temp_dir, quality="mid")
print("Engine created")
docs = ["The quick brown fox jumps over the lazy dog.", "A fast dark colored fox leaps above a sleepy hound."]
files = []
for i, d in enumerate(docs):
    p = temp_dir + f"/doc_{i}.txt"
    with open(p, "w") as f: f.write(d)
    files.append(p)
print("Ingesting")
engine.ingest_documents(files)
print("Ingested")
print("Querying")
engine.query_with_sources("fast fox", k=2)
print("Done")
