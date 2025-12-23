from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils_bridge import pair_A_C, build_anchor_rollout


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_pairs(args: argparse.Namespace) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if args.sample_a or args.sample_c:
        if not (args.sample_a and args.sample_c):
            raise ValueError("Provide both --sample_a and --sample_c or neither.")
        a = read_json(Path(args.sample_a))
        c = read_json(Path(args.sample_c))
        return [(a, c)]

    if args.input:
        rows = read_jsonl(Path(args.input))
        return pair_A_C(rows)

    raise ValueError("Must supply either --input JSONL or --sample_a/--sample_c JSON files.")


def maybe_load_tokenizer(tokenizer_id: Optional[str]):
    if not tokenizer_id:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("Install transformers to use --tokenizer option") from e

    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export thought-anchors-compatible rollouts for CoT mutation.")
    ap.add_argument("--input", help="Path to JSONL with combined A/C rows")
    ap.add_argument("--sample_a", help="Single JSON file for condition A")
    ap.add_argument("--sample_c", help="Single JSON file for condition C")
    ap.add_argument("--outdir", required=True, help="Output directory for rollouts_*.jsonl")
    ap.add_argument("--tokenizer", help="HF tokenizer name/path to render prompts via chat template")
    args = ap.parse_args()

    pairs = load_pairs(args)
    tokenizer = maybe_load_tokenizer(args.tokenizer)

    out_A, out_C = [], []
    for a, c in pairs:
        rA = build_anchor_rollout(a, c, condition="A", tokenizer=tokenizer)
        rC = build_anchor_rollout(a, c, condition="C", tokenizer=tokenizer)
        out_A.append(rA.__dict__)
        out_C.append(rC.__dict__)

    outdir = Path(args.outdir)

    # If single pair via --sample_a/--sample_c, emit JSON arrays; otherwise keep JSONL for large batches.
    if args.sample_a and args.sample_c:
        write_json(outdir / "rollouts_A.json", out_A)
        write_json(outdir / "rollouts_C.json", out_C)
    else:
        write_jsonl(outdir / "rollouts_A.jsonl", out_A)
        write_jsonl(outdir / "rollouts_C.jsonl", out_C)


if __name__ == "__main__":
    main()