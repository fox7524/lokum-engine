from lokum_engine.rag.engine import normalize_rag_quality, get_rag_quality_profile

q1 = normalize_rag_quality("custom")
print(f"normalize_rag_quality('custom') -> {q1}")

prof = get_rag_quality_profile("custom")
print(f"get_rag_quality_profile('custom') -> {prof.name}")
