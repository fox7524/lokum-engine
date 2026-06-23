from sentence_transformers import CrossEncoder
import os
print("Loading model")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
print("Predicting")
scores = model.predict([["hello world", "test test"], ["fast fox", "dark colored fox"]])
print("Predicted", scores)
