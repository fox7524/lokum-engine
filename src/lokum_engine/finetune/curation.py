import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)

def deduplicate_dataset(input_file: str, output_file: str, text_key: str = "text") -> dict:
    """
    Deduplicates a JSONL dataset based on exact text matches.
    Future enhancements can include MinHash/Jaccard similarity.
    
    Returns:
        dict: Stats on original count, deduplicated count, and removed count.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    seen_hashes = set()
    original_count = 0
    removed_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            original_count += 1
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                if not text:
                    # Keep lines that don't have the text key, or it's empty
                    f_out.write(line)
                    continue
                    
                import hashlib
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                
                if text_hash in seen_hashes:
                    removed_count += 1
                    continue
                    
                seen_hashes.add(text_hash)
                f_out.write(line)
                
            except json.JSONDecodeError:
                # If it's malformed, keep it and let the training engine fail or handle it
                f_out.write(line)
                
    return {
        "original_count": original_count,
        "deduplicated_count": original_count - removed_count,
        "removed_count": removed_count
    }

def auto_score_dataset(input_file: str, output_file: str, scoring_fn: Callable[[str], float], threshold: float = 0.5) -> dict:
    """
    Uses an LLM-as-a-judge scoring function to filter out low-quality rows.
    
    Args:
        scoring_fn: A function that takes text and returns a float score (0.0 to 1.0)
        threshold: Minimum score required to keep the row.
    """
    original_count = 0
    removed_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            original_count += 1
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                
                score = scoring_fn(text)
                if score >= threshold:
                    obj["_curation_score"] = score
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    removed_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to score row: {e}")
                f_out.write(line)
                
    return {
        "original_count": original_count,
        "kept_count": original_count - removed_count,
        "removed_count": removed_count
    }
