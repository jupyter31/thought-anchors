from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class AnchorRollout:
    """A single item consumable by thought-anchors-style scripts."""

    uid: str
    condition: str  # "A" or "C"
    prompt: str  # prompt text ending with "<think>\n"
    solution: str  # reasoning + answer (like math-rollouts)
    cot: str  # extracted reasoning content (no tags)
    answer: str  # extracted short answer (or full final answer)
    meta: Dict[str, Any]


def _contains_think_block(text: str) -> bool:
    return "<think>" in text or "</think>" in text


def _infer_tool_name(content: Any, supplied_name: Optional[str]) -> str:
    if supplied_name:
        return supplied_name
    if isinstance(content, dict):
        for key in ("tool", "name", "label"):
            if content.get(key):
                return str(content[key])
    if isinstance(content, str):
        match = _TOOL_NAME_RE.search(content)
        if match:
            return match.group(1)
    return "tool"


def _render_message(role: str, content: str, name: Optional[str] = None) -> str:
    """Deterministic role rendering (fallback when tokenizer template is unavailable)."""

    if role == "system":
        return f"<<SYSTEM>>\n{content.strip()}\n<</SYSTEM>>\n"
    if role == "user":
        return f"<<USER>>\n{content.strip()}\n<</USER>>\n"
    if role == "assistant":
        if content and content.strip():
            return f"<<ASSISTANT>>\n{content.strip()}\n<</ASSISTANT>>\n"
        return ""
    if role == "tool":
        tool_name = _infer_tool_name(content, name)
        return (
            f"<<TOOL:{tool_name}>>\n"
            f"{content.strip()}\n"
            f"<</TOOL:{tool_name}>>\n"
        )
    return f"<<{role.upper()}>>\n{content.strip()}\n<</{role.upper()}>>\n"


def _strip_think_from_assistant(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop embedded <think> content from assistant messages so CoT is only in solution."""

    cleaned: List[Dict[str, Any]] = []
    for m in msgs:
        m_copy = dict(m)
        content = m_copy.get("content") or ""
        if m_copy.get("role") == "assistant" and _contains_think_block(str(content)):
            m_copy["content"] = ""
        cleaned.append(m_copy)
    return cleaned


def render_chat_with_tools(
    messages: List[Dict[str, Any]],
    tokenizer: Any = None,
    add_generation_prompt: bool = False,
    template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Render messages into a model-specific prompt.

    - If a tokenizer is provided, prefer its chat template (model-specific formatting).
    - If the tokenizer lacks tool support or raises, fall back to deterministic tags.
    - Assistant messages containing <think> are cleared so the CoT lives in `solution`.
    """

    cleaned_messages = _strip_think_from_assistant(messages)

    if tokenizer is not None:
        try:
            extra = template_kwargs or {}
            text = tokenizer.apply_chat_template(
                cleaned_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **extra,
            )
            return text if text.endswith("\n") else text + "\n"
        except Exception:
            # Fall back to deterministic rendering if the template fails (e.g., no tool role support).
            pass

    # Fallback deterministic rendering.
    chunks: List[str] = []
    for m in cleaned_messages:
        role = m.get("role", "")
        name = m.get("name")
        content = m.get("content") or ""

        if role == "tool" and not name:
            name = _infer_tool_name(content, None)

        chunks.append(_render_message(role, str(content), name=name))

    return "".join(chunks).strip() + "\n"


def extract_cot_from_sample(sample: Dict[str, Any], condition: str) -> str:
    """Choose which field is the 'CoT' for anchor analysis."""

    if condition == "A":
        return (sample.get("trace_A") or sample.get("reasoning_text") or "").strip()
    if condition == "C":
        return (sample.get("mutated_cot") or "").strip()
    return ""


def extract_short_answer(final_answer: str) -> str:
    """Try to get the minimal answer string (fallback to full text)."""

    if not final_answer:
        return ""

    m = re.search(r"\*\*Answer:\*\*\s*(.+)", final_answer, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip().splitlines()[0].strip()
        return ans

    m = re.search(r"Final Answer:\s*(.+)", final_answer, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip().splitlines()[0].strip()
        return ans

    return final_answer.strip()


def build_anchor_rollout(
    sample_A: Dict[str, Any],
    sample_C: Dict[str, Any],
    condition: str,
    tokenizer: Any = None,
    template_kwargs: Optional[Dict[str, Any]] = None,
) -> AnchorRollout:
    """Build a thought-anchors-style rollout for either A or C with frozen tools."""

    sample = sample_A if condition == "A" else sample_C
    uid = sample.get("sample_id") or sample_A.get("sample_id") or sample_C.get("sample_id")

    prompt_text = render_chat_with_tools(
        sample["messages"], tokenizer=tokenizer, add_generation_prompt=False, template_kwargs=template_kwargs
    )

    cot = extract_cot_from_sample(sample, condition=condition)
    final_answer = sample.get("final_answer_text") or sample.get("final_answer") or ""
    answer = extract_short_answer(final_answer)

    prompt = prompt_text + "\n<think>\n"
    solution = cot + "\n</think>\n" + answer + "\n"

    meta = {
        "sample_id": uid,
        "mutation_family": sample.get("mutation_family"),
        "mutation_type": sample.get("mutation_type"),
        "directive": sample.get("directive"),
        "model": sample.get("mutation_spec", {}).get("model") or sample.get("model"),
    }

    return AnchorRollout(
        uid=uid,
        condition=condition,
        prompt=prompt,
        solution=solution,
        cot=cot,
        answer=answer,
        meta=meta,
    )


def pair_A_C(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return list of (A_row, C_row) pairs with same sample_id."""

    A = {r["sample_id"]: r for r in rows if r.get("condition") == "A"}
    C = {r["sample_id"]: r for r in rows if r.get("condition") == "C"}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for sid, a in A.items():
        c = C.get(sid)
        if c is not None:
            pairs.append((a, c))

    return pairs


__all__ = [
    "AnchorRollout",
    "render_chat_with_tools",
    "extract_cot_from_sample",
    "extract_short_answer",
    "build_anchor_rollout",
    "pair_A_C",
]