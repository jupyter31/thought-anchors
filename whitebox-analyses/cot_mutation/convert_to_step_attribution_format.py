"""
Convert rollouts_A.json and rollouts_C.json into the directory structure expected by step_attribution.py.

Expected output structure:
analysis_dir/
  correct_base_solution/  (or incorrect_base_solution/)
    problem_0/
      problem.json
      base_solution.json
      chunks_labeled.json
      chunk_0/
        solutions.json  (list of rollout variations)
      chunk_1/
        solutions.json
      ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def split_solution_into_chunks(solution_text: str) -> List[str]:
    """
    Split a solution into chunks for rollout generation.
    Simplified version copied from utils.py to avoid import issues.
    
    Args:
        solution_text: The full solution text
        
    Returns:
        List of chunks
    """
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    
    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()
    
    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]
    
    # Split the text into chunks
    chunks = []
    current_chunk = ""
    
    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]
        
        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break
        
        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True
        
        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = ""
        
        i += 1
    
    # Add any remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def load_rollouts(path: Path) -> List[Dict[str, Any]]:
    """Load rollouts from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_problem_structure(
    rollout: Dict[str, Any],
    condition: str,
    output_dir: Path,
    problem_idx: int = 0
) -> None:
    """
    Create the problem directory structure for a single rollout.
    
    Args:
        rollout: The rollout dictionary with uid, prompt, solution, cot, answer, meta
        condition: "A" or "C"
        output_dir: Base output directory
        problem_idx: Problem index for the directory name
    """
    # Create problem directory
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract components
    uid = rollout.get("uid", f"unknown_{problem_idx}")
    prompt = rollout.get("prompt", "")
    solution = rollout.get("solution", "")
    cot = rollout.get("cot", "")
    answer = rollout.get("answer", "")
    meta = rollout.get("meta", {})
    
    # Create problem.json
    problem_data = {
        "uid": uid,
        "condition": condition,
        "prompt": prompt,
        "meta": meta
    }
    with (problem_dir / "problem.json").open("w", encoding="utf-8") as f:
        json.dump(problem_data, f, indent=2, ensure_ascii=False)
    
    # Create base_solution.json (the original solution)
    base_solution_data = {
        "solution": solution,
        "cot": cot,
        "answer": answer
    }
    with (problem_dir / "base_solution.json").open("w", encoding="utf-8") as f:
        json.dump(base_solution_data, f, indent=2, ensure_ascii=False)
    
    # Split solution into chunks
    chunks = split_solution_into_chunks(cot if cot else solution)
    
    # Create chunks_labeled.json
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "chunk_id": i,
            "text": chunk,
            "label": None  # No label for now
        })
    
    with (problem_dir / "chunks_labeled.json").open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    # Create chunk directories with the base solution as the only "rollout"
    for i, chunk in enumerate(chunks):
        chunk_dir = problem_dir / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # For now, we only have one solution (the base solution)
        # In a full rollout experiment, you would have multiple variations here
        solutions_data = [{
            "solution": solution,
            "cot": cot,
            "answer": answer,
            "chunk_text": chunk
        }]
        
        with (chunk_dir / "solutions.json").open("w", encoding="utf-8") as f:
            json.dump(solutions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {problem_dir} with {len(chunks)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert rollouts JSON files to step_attribution.py format"
    )
    parser.add_argument(
        "--rollouts_a",
        required=True,
        help="Path to rollouts_A.json"
    )
    parser.add_argument(
        "--rollouts_c",
        required=True,
        help="Path to rollouts_C.json"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for converted structure"
    )
    parser.add_argument(
        "--correct_base",
        action="store_true",
        help="Use correct_base_solution subdirectory (default: incorrect_base_solution)"
    )
    
    args = parser.parse_args()
    
    # Load rollouts
    rollouts_a = load_rollouts(Path(args.rollouts_a))
    rollouts_c = load_rollouts(Path(args.rollouts_c))
    
    print(f"Loaded {len(rollouts_a)} rollouts from condition A")
    print(f"Loaded {len(rollouts_c)} rollouts from condition C")
    
    # Determine subdirectory
    subdir = "correct_base_solution" if args.correct_base else "incorrect_base_solution"
    
    # Create output directories
    output_a = Path(args.output_dir) / "condition_A" / subdir
    output_c = Path(args.output_dir) / "condition_C" / subdir
    
    # Process condition A
    print("\nProcessing condition A...")
    for idx, rollout in enumerate(rollouts_a):
        create_problem_structure(rollout, "A", output_a, idx)
    
    # Process condition C
    print("\nProcessing condition C...")
    for idx, rollout in enumerate(rollouts_c):
        create_problem_structure(rollout, "C", output_c, idx)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Condition A: {output_a}")
    print(f"  Condition C: {output_c}")
    print(f"\nTo run step_attribution.py on these:")
    print(f"  python step_attribution.py --analysis_dir {output_a.parent} [other flags]")


if __name__ == "__main__":
    main()
