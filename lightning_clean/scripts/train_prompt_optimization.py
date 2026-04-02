#!/usr/bin/env python3
"""
Prompt optimization with Agent Lightning + vERL.

Policy:
    Qwen/Qwen3-8B served by veRL/vLLM. This model emits a complete candidate
    prompt string for the optimized ReAct instructions slot instead of a
    symbolic JSON edit program.

Frozen downstream agent:
    A separate, fixed LLM endpoint plus BM25 Wikipedia tools. The generated
    prompt is injected into the downstream HoVer-style claim verification loop.

    Uses DIRECT OpenAI SDK calls to the RITS endpoint (no Mellea).

Reward:
    Top-5 recall of predicted supporting Wikipedia titles, with light shaping
    that discourages malformed or bloated prompts while giving a small bonus to
    successful prompts that explore beyond the seed in a controlled way.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import requests
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# veRL/agentlightning uses the vLLM async V1 engine path in this stack.
# Set this before importing torch/agentlightning so child workers inherit a
# self-consistent engine mode.
os.environ.setdefault("VLLM_USE_V1", "1")

import ray
import torch
from openai import OpenAI
from openai import APIStatusError, APITimeoutError, APIConnectionError
from pydantic import BaseModel

import agentlightning as agl
from agentlightning import LitAgent, NamedResources, PromptTemplate, Rollout, emit_reward


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]


def _find_ancestor_with_relative_path(start: Path, relative_path: str) -> Optional[Path]:
    for candidate in start.parents:
        if (candidate / relative_path).exists():
            return candidate
    return None


WORKSPACE_ROOT = (
    _find_ancestor_with_relative_path(SCRIPT_PATH, "dspy_tutorial/agents_tutorial_bm25.py")
    or _find_ancestor_with_relative_path(SCRIPT_PATH, "../dspy_tutorial/agents_tutorial_bm25.py")
    or PROJECT_DIR.parent
)
LIGHTNING_DIR = _find_ancestor_with_relative_path(SCRIPT_PATH, "examples") or PROJECT_DIR
DSPY_TUTORIAL_DIR = WORKSPACE_ROOT / "dspy_tutorial"
EXAMPLES_DIR = LIGHTNING_DIR / "examples"

for path in (DSPY_TUTORIAL_DIR, EXAMPLES_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

RITS_ENDPOINTS = {
    "qwen38b": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "gptoss120b": ("openai/gpt-oss-120b", "gpt-oss-120b"),
}

RITS_HTTP_TIMEOUT: float = 60.0
RITS_RETRIES: int = 3

# Known context window sizes for common policy models in this codebase.
# Used to clamp max response length so vLLM does not reject requests as too long.
MODEL_CONTEXT_SIZE: Dict[str, int] = {
    "qwen/qwen3-8b": 2048,
    "qwen3-8b": 2048,
    "qwen/qwen3-14b": 3072,
    "qwen3-14b": 3072,
}

# vLLM chat serving adds template/system overhead beyond the user prompt body.
# Keep a conservative slack so veRL rollout budgets do not exceed the real
# context limit once the prompt is wrapped for chat completion.
CHAT_TEMPLATE_TOKEN_SLACK = 96


def _get_model_context_size(model: str) -> int:
    model_key = model.lower().strip()
    if model_key in MODEL_CONTEXT_SIZE:
        return MODEL_CONTEXT_SIZE[model_key]

    for known_key, context_size in MODEL_CONTEXT_SIZE.items():
        if model_key.endswith(known_key) or known_key in model_key:
            return context_size

    return 4096

def _clamp_max_tokens_for_model(model: str, requested_max_tokens: int) -> int:
    max_context = _get_model_context_size(model)

    # Conservative defaults: keep < 50% of context for generation when possible.
    if max_context <= 1536:
        generation_cap = 384
    elif max_context <= 3072:
        generation_cap = 1024
    else:
        generation_cap = requested_max_tokens

    safe_max_tokens = min(requested_max_tokens, generation_cap)
    if safe_max_tokens < requested_max_tokens:
        print(
            f"[warn] Capping max_tokens for model {model} from {requested_max_tokens} "
            f"to {safe_max_tokens} to avoid max-context overflow",
            flush=True,
        )
    return max(1, safe_max_tokens)


def _compute_safe_verl_token_budgets(
    *,
    model: str,
    requested_prompt_length: int,
    requested_response_length: int,
) -> tuple[int, int]:
    max_context = _get_model_context_size(model)

    safe_response_length = _clamp_max_tokens_for_model(model, requested_response_length)

    # Reserve room for the chat template, role markers, and small tokenizer drift.
    max_prompt_budget = max(256, max_context - safe_response_length - CHAT_TEMPLATE_TOKEN_SLACK)
    safe_prompt_length = min(requested_prompt_length, max_prompt_budget)

    # Recompute the response cap after clamping the prompt budget so the two
    # values are guaranteed to fit together within the actual model window.
    safe_response_length = min(
        safe_response_length,
        max(1, max_context - safe_prompt_length - CHAT_TEMPLATE_TOKEN_SLACK),
    )
    return safe_prompt_length, safe_response_length

EXPLORATION_PROFILES: List[tuple[str, str]] = [
    (
        "structured_rewrite",
        "Rewrite the prompt with clear numbered sections, decision trees, "
        "and explicit phase transitions (e.g., Discovery phase -> Verification "
        "phase -> Completion phase). Make the structure guide the agent through "
        "a logical workflow.",
    ),
    (
        "behavioral",
        "Focus on adding behavioral guidance: when to search vs lookup, "
        "when to stop searching, how to handle ambiguous entities, "
        "what to do when search results are poor, and how to avoid common "
        "mistakes like redundant lookups or premature finishing.",
    ),
    (
        "exemplar",
        "Add worked decision templates or mini-examples that show the agent "
        "how to reason. For instance, show the pattern for 'if the claim "
        "mentions X and Y, search for X first, then use results to find Y.' "
        "Use placeholder examples, not real claims.",
    ),
    (
        "concise",
        "Compress the prompt to its essentials. Remove all ambiguity through "
        "brevity and precision. Every sentence must earn its place. "
        "Aim for maximum information density with minimum word count.",
    ),
    (
        "elaborate",
        "Expand the prompt with detailed instructions covering edge cases, "
        "prompting best practices, and explicit quality criteria. "
        "Add guidance on multi-hop reasoning, entity disambiguation, "
        "and evidence sufficiency checks.",
    ),
]

class ReActStep(BaseModel):
    next_thought: str
    next_tool_name: str
    next_tool_args: Dict[str, Any]


class ExtractTitlesOut(BaseModel):
    titles: List[str]


@dataclass
class DownstreamRuntime:
    search_wikipedia: Any
    lookup_wikipedia: Any
    endpoint: str
    model_name: str
    rits_base_url: str
    rits_api_key: str
    demos: List[Dict[str, Any]]
    extract_instructions: str
    agent_temperature: float
    max_agent_steps: int


@dataclass
class EpisodeResult:
    reward: float
    claim: str
    gold_titles: List[str]
    pred_titles: List[str]
    trajectory: str
    finished: bool
    n_steps: int
    extract_prompt: str
    stop_reason: str


@dataclass
class PromptGenerationResult:
    raw_text: str
    sanitized_text: str
    used_fallback: bool
    rejection_reason: str


_RUNTIME: Optional[DownstreamRuntime] = None
_RUNTIME_LOCK = threading.Lock()


def get_rits_base_url(endpoint: str) -> str:
    return f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{endpoint}/v1"


def _tool_schema_text() -> str:
    return (
        "(1) search_wikipedia: <desc>Returns top search results.</desc> "
        'Args: {"query": "string"}\n'
        "(2) lookup_wikipedia: <desc>Returns the text of a Wikipedia page.</desc> "
        'Args: {"title": "string"}\n'
        '(3) finish: <desc>Marks the task as complete.</desc> Args: {}\n'
    )


def _format_demos(demos: List[Dict[str, Any]], max_demos: int = 5) -> str:
    if not demos:
        return ""
    blocks = []
    for demo in demos[:max_demos]:
        claim = demo.get("claim")
        trajectory = demo.get("trajectory", "")
        if claim:
            blocks.append(f"### Example\nClaim: {claim}\nTrajectory:\n{trajectory}\n")
    return "\n".join(blocks).strip()


def build_react_prompt(
    *,
    instructions: str,
    claim: str,
    trajectory: str,
    demos: List[Dict[str, Any]],
) -> str:
    demos_txt = _format_demos(demos)
    demos_block = f"{demos_txt}\n\n" if demos_txt else ""
    return (
        f"{instructions}\n\n"
        "You are an Agent. In each episode, you will be given the field `claim` as input, "
        "and you can see your past `trajectory`.\n"
        "Your goal is to use the supplied tools to gather what you need.\n\n"
        "Tools:\n"
        f"{_tool_schema_text()}\n\n"
        f"{demos_block}"
        f"Claim: {claim}\n\n"
        f"Trajectory:\n{trajectory}\n\n"
        "**OUTPUT RULES:**\n"
        "- Do NOT use <think>...</think> tags - output JSON directly!\n"
        "- Return EXACTLY a JSON object with these keys:\n"
        '  - "next_thought": string (your reasoning for this step)\n'
        '  - "next_tool_name": one of ["search_wikipedia","lookup_wikipedia","finish"]\n'
        '  - "next_tool_args": JSON object matching the chosen tool schema\n\n'
        "Output (JSON only):"
    ).strip()


def build_extract_prompt(
    *,
    instructions: Optional[str],
    claim: str,
    trajectory: str,
) -> str:
    instr = instructions or "Find all Wikipedia titles relevant to verifying (or refuting) the claim."
    return (
        f"{instr}\n\n"
        "Given the claim and the full trajectory (tool calls + observations), "
        "output the final list of Wikipedia article titles that should be returned.\n\n"
        "**CRITICAL OUTPUT RULES:**\n"
        '- Output EXACTLY one JSON object: {"titles": ["Title1", "Title2", "..."]}\n'
        "- Do NOT use <think>...</think> tags - output JSON directly!\n"
        "- Do NOT include explanations, reasoning, or any other text.\n"
        "- Your response must start with '{' and end with '}'.\n"
        "- Titles must be strings (Wikipedia article titles), no duplicates.\n\n"
        f"Claim: {claim}\n\n"
        f"Trajectory:\n{trajectory}\n\n"
        "Output: JSON only"
    ).strip()


def top5_recall(pred_titles: List[str], gold_titles: List[str]) -> float:
    if not gold_titles:
        return 0.0
    return sum(1 for gold in gold_titles if gold in pred_titles[:5]) / len(gold_titles)


def _safe_slug(value: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (slug or "example")[:limit]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_program_prompts(program_json_path: str) -> Dict[str, Any]:
    with open(program_json_path, "r") as handle:
        program = json.load(handle)

    react = program.get("react", {})
    react_sig = react.get("signature", {})
    react_demos = react.get("demos", []) or []

    extract = program.get("extract.predict", {})
    extract_sig = extract.get("signature", {})

    return {
        "react_instructions": react_sig.get("instructions", "").strip(),
        "extract_instructions": extract_sig.get("instructions"),
        "demos": react_demos,
    }


def load_hover_data(
    n_train: int = 256,
    n_dev: int = 100,
    only_3hop: bool = True,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset("vincentkoc/hover-parquet")

    def process_split(split_data: Any, max_n: int) -> List[Dict[str, Any]]:
        hpqa_ids = set()
        examples: List[Dict[str, Any]] = []

        for ex in split_data:
            if only_3hop and ex.get("num_hops") != 3:
                continue
            if ex.get("hpqa_id") in hpqa_ids:
                continue
            hpqa_ids.add(ex["hpqa_id"])

            supporting_facts = ex.get("supporting_facts", [])
            if not supporting_facts:
                continue
            titles = list({fact["key"] for fact in supporting_facts if "key" in fact})
            if not titles:
                continue

            examples.append({"claim": ex["claim"], "titles": titles, "hpqa_id": ex.get("hpqa_id")})

        random.Random(seed).shuffle(examples)
        return examples[:max_n] if max_n > 0 else examples

    trainset = process_split(dataset["train"], n_train)
    devset = process_split(dataset["validation"], n_dev)
    testset = devset
    print(f"[data] Loaded {len(trainset)} train and {len(devset)} dev examples")
    return trainset, devset, testset


def _normalize_remote_model(model_name: str) -> str:
    if model_name.startswith("openai/"):
        return model_name.split("/", 1)[1]
    return model_name


def _strip_thinking_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    fenced_match = re.search(r"```(?:json|text|markdown)?\s*([\s\S]*?)```", text)
    if fenced_match:
        text = fenced_match.group(1).strip()
    if not text.strip().startswith("{"):
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            text = brace_match.group(0)
    return text.strip()


def _extract_tagged_prompt(text: str) -> str:
    match = re.search(r"<prompt>\s*([\s\S]*?)\s*</prompt>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"^([\s\S]*?)\s*</prompt>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _sanitize_generated_prompt(
    raw_text: str,
    fallback_prompt: str,
    claim: str,
) -> PromptGenerationResult:
    text = raw_text.strip()
    if not text:
        return PromptGenerationResult(
            raw_text=raw_text,
            sanitized_text=fallback_prompt,
            used_fallback=True,
            rejection_reason="empty_raw_text",
        )

    # Only strip <think> blocks here — do NOT apply _strip_thinking_tags, which
    # also extracts JSON fragments. The generated prompt legitimately contains
    # JSON-like tool-schema examples ({"query": "..."}), and the JSON extractor
    # would corrupt the entire prompt text to just that fragment.
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = _extract_tagged_prompt(text)
    text = re.sub(r"^(Here is|Here's|Revised prompt:|Improved prompt:|Candidate prompt:)\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*['\"]|['\"]\s*$", "", text).strip()

    filtered_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() in {
            "<seed_prompt>", "</seed_prompt>", "<prompt>", "</prompt>",
            "<reference>", "</reference>",
        }:
            continue
        if stripped.startswith((
            "Optimization targets:", "Claim context", "Exploration mode",
            "Hard constraints:", "## What makes a great prompt",
            "## Your generation style", "## Reference prompt",
            "## Claim context", "## Output instructions",
        )):
            continue
        filtered_lines.append(line.rstrip())

    text = "\n".join(filtered_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return PromptGenerationResult(
            raw_text=raw_text,
            sanitized_text=fallback_prompt,
            used_fallback=True,
            rejection_reason="empty_after_cleanup",
        )

    if text.startswith("{") or text.startswith("["):
        return PromptGenerationResult(
            raw_text=raw_text,
            sanitized_text=fallback_prompt,
            used_fallback=True,
            rejection_reason="json_like_output",
        )

    if claim and claim in text:
        return PromptGenerationResult(
            raw_text=raw_text,
            sanitized_text=fallback_prompt,
            used_fallback=True,
            rejection_reason="claim_leakage",
        )

    if len(text.split()) < 16:
        return PromptGenerationResult(
            raw_text=raw_text,
            sanitized_text=fallback_prompt,
            used_fallback=True,
            rejection_reason="too_short_after_cleanup",
        )

    if text.endswith(("<seed_prompt>", "<prompt>", "Claim context", "Hard constraints:")):
        return PromptGenerationResult(
            raw_text=raw_text,
            sanitized_text=fallback_prompt,
            used_fallback=True,
            rejection_reason="likely_truncated",
        )

    return PromptGenerationResult(
        raw_text=raw_text,
        sanitized_text=text,
        used_fallback=False,
        rejection_reason="",
    )


def _compute_prompt_metrics(seed_prompt: str, generated_prompt: str) -> Dict[str, Any]:
    seed_word_count = len(seed_prompt.split())
    generated_word_count = len(generated_prompt.split())
    similarity_ratio = difflib.SequenceMatcher(a=seed_prompt, b=generated_prompt).ratio()
    novelty_ratio = 1.0 - similarity_ratio
    return {
        "seed_word_count": seed_word_count,
        "generated_word_count": generated_word_count,
        "word_delta": generated_word_count - seed_word_count,
        "seed_char_count": len(seed_prompt),
        "generated_char_count": len(generated_prompt),
        "char_delta": len(generated_prompt) - len(seed_prompt),
        "similarity_ratio": round(float(similarity_ratio), 6),
        "novelty_ratio": round(float(novelty_ratio), 6),
    }


def _stable_rollout_key(*, claim: str, rollout_id: str, attempt_id: str) -> str:
    return f"{rollout_id}:{attempt_id}:{claim}"


def _parse_temperature_schedule(raw_value: str) -> List[float]:
    values: List[float] = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid temperature value {stripped!r} in schedule {raw_value!r}") from exc
    if not values:
        raise ValueError("Policy temperature schedule must contain at least one numeric value.")
    return values
def _select_rollout_temperature(
    *,
    claim: str,
    rollout_id: str,
    attempt_id: str,
    base_temperature: float,
    vary_temperature: bool,
    temperature_schedule: List[float],
) -> float:
    if not vary_temperature:
        return base_temperature
    key = _stable_rollout_key(claim=claim, rollout_id=rollout_id, attempt_id=attempt_id)
    idx = zlib.crc32(key.encode("utf-8")) % len(temperature_schedule)
    return float(temperature_schedule[idx])


def build_candidate_prompt_request(
    *,
    seed_prompt: str,
    claim: str,
    exploration_profile: Dict[str, str],
) -> str:
    # Previous version kept for reference. It provided a detailed optimization
    # rubric, which made the generator rely on hand-authored prompt-engineering
    # guidance rather than mostly learning from the seed prompt and exploration
    # profile.
    #
    # return (
    #     "You are a prompt engineer. Your task is to write the best possible "
    #     "instruction prompt for a frozen fact-checking agent.\n\n"
    #     "The agent receives a `claim` and must find the exact Wikipedia article "
    #     "titles needed to verify or refute it. The agent has three tools:\n"
    #     "- search_wikipedia(query): returns top search results with article titles and snippets\n"
    #     "- lookup_wikipedia(title): returns the full text of a Wikipedia page\n"
    #     "- finish(): signals the agent has gathered enough evidence\n\n"
    #     "The agent works in a ReAct loop: it outputs a thought, picks a tool, "
    #     "sees the result, and repeats until it calls finish.\n\n"
    #     "## What makes a great prompt for this agent\n\n"
    #     "A great prompt:\n"
    #     "1. **Gives the agent a clear mental model** of the task: it must find "
    #     "2-3 specific Wikipedia articles, not just answer the claim.\n"
    #     "2. **Provides decision-making guidance**: when to search (need to "
    #     "discover a title) vs. lookup (already know the title), and when to stop.\n"
    #     "3. **Describes common failure modes** and how to avoid them: searching "
    #     "too broadly, looking up the same page twice, finishing before gathering "
    #     "enough evidence, or failing to decompose multi-hop claims.\n"
    #     "4. **Uses clear structure** (numbered steps, phases, or decision rules) "
    #     "so the agent can follow a systematic workflow.\n"
    #     "5. **Is specific and actionable** rather than vague.\n"
    #     "6. **Keeps only what helps.** Every sentence should improve the agent's "
    #     "behavior. Remove filler.\n\n"
    #     f"## Your generation style: {exploration_profile['name']}\n\n"
    #     f"{exploration_profile['description']}\n\n"
    #     "## Reference prompt (for inspiration, not copying)\n\n"
    #     "Below is a reference prompt that achieves reasonable performance. Use it "
    #     "to understand the task and tool interface, but feel free to depart from "
    #     "its structure, wording, and approach entirely. Your goal is to write a "
    #     "BETTER prompt, not a similar one.\n\n"
    #     f"<reference>\n{seed_prompt}\n</reference>\n\n"
    #     "## Claim context (for understanding what the agent faces -- do NOT "
    #     "embed this specific claim in your prompt)\n\n"
    #     f"{claim}\n\n"
    #     "## Output instructions\n\n"
    #     "- Write ONLY the new prompt inside <prompt>...</prompt> tags.\n"
    #     "- The prompt must be general-purpose (work for ANY claim, not just the one above).\n"
    #     "- Do NOT explain your changes or reasoning.\n"
    #     "- Do NOT include <think> tags, markdown fences, or extra XML.\n\n"
    #     "<prompt>"
    # )
    return (
        "You are a prompt engineer. Your task is to write an improved "
        "instruction prompt for a frozen fact-checking agent.\n\n"
        "The agent receives a `claim` and must find the exact Wikipedia article "
        "titles needed to verify or refute it. The agent has three tools:\n"
        "- search_wikipedia(query): returns top search results with article titles and snippets\n"
        "- lookup_wikipedia(title): returns the full text of a Wikipedia page\n"
        "- finish(): signals the agent has gathered enough evidence\n\n"
        "The agent works in a ReAct loop: it outputs a thought, picks a tool, "
        "sees the result, and repeats until it calls finish.\n\n"
        f"## Exploration profile: {exploration_profile['name']}\n\n"
        f"{exploration_profile['description']}\n\n"
        "## Seed prompt\n\n"
        "Use the following baseline prompt as the starting point. Produce a new "
        "prompt that is still general-purpose, but improved according to the "
        "exploration profile.\n\n"
        f"<reference>\n{seed_prompt}\n</reference>\n\n"
        "## Claim context\n\n"
        "This claim is provided only so you can understand the benchmark setting "
        "the downstream agent will face. Do not specialize the new prompt to this "
        "particular claim.\n\n"
        f"{claim}\n\n"
        "## Output instructions\n\n"
        "- Write ONLY the new prompt inside <prompt>...</prompt> tags.\n"
        "- The prompt must remain general-purpose and preserve the same task and tool interface.\n"
        "- Do NOT explain your changes or reasoning.\n"
        "- Do NOT include <think> tags, markdown fences, or extra XML.\n\n"
        "<prompt>"
    )


def _default_react_instructions() -> str:
    return (
        "** Given a `claim` and the current `trajectory`, decide what information is still missing, "
        "choose an appropriate tool, and specify its arguments, repeating this process until you have "
        "gathered the Wikipedia pages that contain the exact three distinct article titles needed to "
        "verify (or refute) the claim. Then signal completion with `finish`.\n\n"
        "Read the claim carefully, identify the required entities, use `search_wikipedia` to discover "
        "article titles, use `lookup_wikipedia` when you already know the title and need evidence, avoid "
        "repeating prior calls, and call `finish` only when the needed pages have been retrieved."
    )


def _coerce_model(model_cls: type[BaseModel], raw: Any) -> BaseModel:
    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, dict):
        return model_cls.model_validate(raw)
    return model_cls.model_validate_json(_strip_thinking_tags(str(raw)))


def _rits_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{base_url}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "RITS_API_KEY": api_key,
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=RITS_HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text").strip()
    return (content or "").strip()



def _rits_structured_output(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    model_cls: type[BaseModel],
    temperature: float,
    max_tokens: int,
    schema_hint: str,
    retries: int = RITS_RETRIES,
) -> BaseModel:
    attempt_messages = list(messages)
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        raw_text = _rits_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=attempt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return _coerce_model(model_cls, raw_text)
        except Exception as exc:
            last_error = exc
            cleaned = _strip_thinking_tags(raw_text)
            attempt_messages.extend(
                [
                    {"role": "assistant", "content": cleaned or raw_text or ""},
                    {
                        "role": "user",
                        "content": f"Return only valid JSON matching this schema: {schema_hint}",
                    },
                ]
            )
            print(f"  [rits] parse attempt {attempt + 1}/{retries} failed: {exc!r}", flush=True)

    raise RuntimeError(f"Failed to parse {model_cls.__name__} after {retries} attempts: {last_error}")


def _chat_completion_text(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    min_tokens: Optional[int] = None,
    disable_thinking: bool = False,
) -> str:
    extra_body: Dict[str, Any] = {}
    if min_tokens is not None:
        extra_body["min_tokens"] = min_tokens
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    safe_max_tokens = _clamp_max_tokens_for_model(model, max_tokens)
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": safe_max_tokens,
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "".join(part.text for part in content if getattr(part, "type", None) == "text").strip()
    return (content or "").strip()


def _build_runtime(
    *,
    wiki_dir: str,
    agent_model: str,
    agent_endpoint: str,
    extract_instructions: str,
    demos: List[Dict[str, Any]],
    agent_temperature: float,
    max_agent_steps: int,
) -> DownstreamRuntime:
    api_key = os.getenv("RITS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("RITS_API_KEY or OPENAI_API_KEY must be set for the frozen downstream agent.")

    base_url = get_rits_base_url(agent_endpoint)
    model_name = _normalize_remote_model(agent_model)


    print(f"[runtime] Using frozen RITS endpoint via raw HTTP: {base_url} model={model_name}", flush=True)

    try:
        from agents_tutorial_bm25 import build_tools_bm25

        search_wikipedia, lookup_wikipedia = build_tools_bm25(
            wiki_dir=wiki_dir,
            bm25_n_threads=1,
            cache_docs=True,
        )
        print(f"[runtime] Initialized BM25 tools from {wiki_dir}")
    except Exception as exc:
        hover_tools_dir = WORKSPACE_ROOT / "mellea_hover_baseline" / "src"
        hover_tools_dir_str = str(hover_tools_dir)
        if hover_tools_dir_str not in sys.path:
            sys.path.insert(0, hover_tools_dir_str)
        from hover_mellea.tools import lookup_wikipedia, search_wikipedia

        print(f"[runtime] Falling back to Wikipedia API tools because BM25 init failed: {exc}")

    return DownstreamRuntime(
        search_wikipedia=search_wikipedia,
        lookup_wikipedia=lookup_wikipedia,
        endpoint=agent_endpoint,
        model_name=model_name,
        rits_base_url=base_url,
        rits_api_key=api_key,
        demos=demos,
        extract_instructions=extract_instructions,
        agent_temperature=agent_temperature,
        max_agent_steps=max_agent_steps,
    )


def ensure_runtime(
    *,
    wiki_dir: str,
    agent_model: str,
    agent_endpoint: str,
    extract_instructions: str,
    demos: List[Dict[str, Any]],
    agent_temperature: float,
    max_agent_steps: int,
) -> DownstreamRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = _build_runtime(
                wiki_dir=wiki_dir,
                agent_model=agent_model,
                agent_endpoint=agent_endpoint,
                extract_instructions=extract_instructions,
                demos=demos,
                agent_temperature=agent_temperature,
                max_agent_steps=max_agent_steps,
            )
        return _RUNTIME


def run_frozen_hover_episode(
    *,
    runtime: DownstreamRuntime,
    task: Dict[str, Any],
    react_instructions: str,
) -> EpisodeResult:
    claim = task["claim"]
    gold_titles = task["titles"]
    finished = False
    n_steps = 0
    stop_reason = "max_steps"

    def run_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "search_wikipedia":
            result = runtime.search_wikipedia(**tool_args)
            if isinstance(result, list):
                return "\n".join(str(item) for item in result)
            return str(result)
        if tool_name == "lookup_wikipedia":
            return str(runtime.lookup_wikipedia(**tool_args))
        if tool_name == "finish":
            return "Completed."
        raise ValueError(f"Unknown tool: {tool_name}")

    def append_step(trajectory: str, step_idx: int, step: ReActStep, observation: str) -> str:
        return (
            trajectory
            + ("" if not trajectory or trajectory.endswith("\n") else "\n")
            + f"[[ ## thought_{step_idx} ## ]]\n{step.next_thought}\n\n"
            + f"[[ ## tool_name_{step_idx} ## ]]\n{step.next_tool_name}\n\n"
            + f"[[ ## tool_args_{step_idx} ## ]]\n{json.dumps(step.next_tool_args)}\n\n"
            + f"[[ ## observation_{step_idx} ## ]]\n{observation}\n"
        )

    trajectory = ""
    previous_calls: set[str] = set()

    print(f"[episode] start claim={claim[:90]!r}", flush=True)

    react_schema_hint = '{"next_thought": "string", "next_tool_name": "search_wikipedia|lookup_wikipedia|finish", "next_tool_args": {...}}'
    extract_schema_hint = '{"titles": ["Title1", "Title2", ...]}'

    for step_idx in range(runtime.max_agent_steps):
        react_prompt = build_react_prompt(
            instructions=react_instructions,
            claim=claim,
            trajectory=trajectory,
            demos=runtime.demos,
        )
        if previous_calls:
            recent_calls = list(previous_calls)[-5:]
            react_prompt += f"\n\nAvoid repeating prior tool calls: {recent_calls}"

        print(f"[episode] step {step_idx + 1}/{runtime.max_agent_steps}: requesting ReAct step", flush=True)

        try:
            step = cast(
                ReActStep,
                _rits_structured_output(
                    base_url=runtime.rits_base_url,
                    api_key=runtime.rits_api_key,
                    model=runtime.model_name,
                    messages=[{"role": "user", "content": react_prompt}],
                    model_cls=ReActStep,
                    temperature=runtime.agent_temperature,
                    max_tokens=1024,
                    schema_hint=react_schema_hint,
                ),
            )
            print(f"[episode] step {step_idx + 1}: parsed OK", flush=True)
        except Exception as exc:
            print(f"[episode] step {step_idx + 1}: RITS/parse failed: {exc!r}", flush=True)
            step = ReActStep(
                next_thought=f"RITS call failed: {str(exc)[:100]}",
                next_tool_name="finish",
                next_tool_args={},
            )

        if step.next_tool_name not in {"search_wikipedia", "lookup_wikipedia", "finish"}:
            step = ReActStep(next_thought="Invalid tool requested.", next_tool_name="finish", next_tool_args={})

        call_key = f"{step.next_tool_name}:{json.dumps(step.next_tool_args, sort_keys=True)}"
        if call_key in previous_calls and step.next_tool_name != "finish":
            trajectory = append_step(trajectory, step_idx, step, "Loop detected; forcing termination.")
            n_steps = step_idx + 1
            stop_reason = "loop_detected"
            break
        previous_calls.add(call_key)

        print(f"[episode] step {step_idx + 1}: tool={step.next_tool_name} args={step.next_tool_args}", flush=True)
        try:
            observation = run_tool(step.next_tool_name, step.next_tool_args)
        except Exception as exc:
            observation = f"Tool error: {exc!r}"

        trajectory = append_step(trajectory, step_idx, step, observation)
        n_steps = step_idx + 1
        if step.next_tool_name == "finish":
            finished = True
            stop_reason = "finish"
            break

    extract_prompt = build_extract_prompt(
        instructions=runtime.extract_instructions,
        claim=claim,
        trajectory=trajectory,
    )
    print("[episode] extracting titles", flush=True)
    try:
        extracted = cast(
            ExtractTitlesOut,
            _rits_structured_output(
                base_url=runtime.rits_base_url,
                api_key=runtime.rits_api_key,
                model=runtime.model_name,
                messages=[{"role": "user", "content": extract_prompt}],
                model_cls=ExtractTitlesOut,
                temperature=runtime.agent_temperature,
                max_tokens=384,
                schema_hint=extract_schema_hint,
            ),
        )
        raw_titles = extracted.titles
    except Exception as exc:
        print(f"[episode] extraction failed: {exc!r}", flush=True)
        raw_titles = []

    seen: set[str] = set()
    pred_titles: List[str] = []
    for title in raw_titles:
        clean = title.strip()
        if clean and clean not in seen:
            seen.add(clean)
            pred_titles.append(clean)

    reward = float(top5_recall(pred_titles, gold_titles))
    return EpisodeResult(
        reward=reward,
        claim=claim,
        gold_titles=list(gold_titles),
        pred_titles=pred_titles,
        trajectory=trajectory,
        finished=finished,
        n_steps=n_steps,
        extract_prompt=extract_prompt,
        stop_reason=stop_reason,
    )


class PromptOptimizationAgent(LitAgent[Dict[str, Any]]):
    def __init__(
        self,
        *,
        wiki_dir: str,
        agent_model: str,
        agent_endpoint: str,
        extract_instructions: str,
        demos: List[Dict[str, Any]],
        policy_temperature: float,
        vary_policy_temperature: bool,
        policy_temperature_schedule: List[float],
        policy_max_tokens: int,
        max_policy_edits: int,
        exploration_bonus_scale: float,
        exploration_bonus_max_novelty: float,
        drift_penalty_threshold: float,
        drift_penalty_scale: float,
        extra_word_penalty_per_50_words: float,
        agent_temperature: float,
        max_agent_steps: int,
        output_dir: str,
        seed_react_instructions: str,
    ) -> None:
        super().__init__()
        self.wiki_dir = wiki_dir
        self.agent_model = agent_model
        self.agent_endpoint = agent_endpoint
        self.extract_instructions = extract_instructions
        self.demos = demos
        self.policy_temperature = policy_temperature
        self.vary_policy_temperature = vary_policy_temperature
        self.policy_temperature_schedule = list(policy_temperature_schedule)
        self.policy_max_tokens = policy_max_tokens
        self.max_policy_edits = max_policy_edits
        self.exploration_bonus_scale = exploration_bonus_scale
        self.exploration_bonus_max_novelty = exploration_bonus_max_novelty
        self.drift_penalty_threshold = drift_penalty_threshold
        self.drift_penalty_scale = drift_penalty_scale
        self.extra_word_penalty_per_50_words = extra_word_penalty_per_50_words
        self.agent_temperature = agent_temperature
        self.max_agent_steps = max_agent_steps
        self.output_dir = Path(output_dir)
        self.seed_react_instructions = seed_react_instructions.strip() or _default_react_instructions()

    def _shape_reward(
        self,
        *,
        base_reward: float,
        prompt_generation: PromptGenerationResult,
        prompt_metrics: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any]]:
        word_delta = max(0, int(prompt_metrics.get("word_delta", 0)))
        novelty_ratio = float(prompt_metrics.get("novelty_ratio", 0.0))

        fallback_penalty = 0.1 if prompt_generation.used_fallback else 0.0
        length_penalty = self.extra_word_penalty_per_50_words * (word_delta / 50.0)
        exploration_bonus = (
            self.exploration_bonus_scale
            * min(max(novelty_ratio, 0.0), self.exploration_bonus_max_novelty)
            * max(base_reward, 0.0)
        )
        excess_novelty = max(0.0, novelty_ratio - self.drift_penalty_threshold)
        drift_penalty = self.drift_penalty_scale * excess_novelty * max(0.0, 1.0 - base_reward)

        training_reward = float(base_reward + exploration_bonus - drift_penalty - length_penalty - fallback_penalty)
        return training_reward, {
            "base_reward": base_reward,
            "training_reward": training_reward,
            "fallback_penalty": fallback_penalty,
            "length_penalty": length_penalty,
            "exploration_bonus": exploration_bonus,
            "drift_penalty": drift_penalty,
        }

    def _persist_rollout_artifact(
        self,
        *,
        task: Dict[str, Any],
        rollout: Rollout,
        generated_prompt: str,
        prompt_generation: PromptGenerationResult,
        prompt_request: str,
        episode: EpisodeResult,
        prompt_metrics: Dict[str, Any],
        reward_breakdown: Dict[str, Any],
        exploration_profile: Dict[str, str],
    ) -> None:
        split = str(task.get("split", "unknown"))
        example_idx = int(task.get("example_idx", -1))
        claim_slug = _safe_slug(task["claim"], limit=48)
        attempt_id = getattr(rollout.attempt, "attempt_id", "attempt")
        artifact_path = self.output_dir / "rollouts" / split / (
            f"example_{example_idx:04d}_rollout_{rollout.rollout_id}_{attempt_id}_{claim_slug}.json"
        )
        payload = {
            "rollout_id": rollout.rollout_id,
            "attempt_id": attempt_id,
            "split": split,
            "example_idx": example_idx,
            "hpqa_id": task.get("hpqa_id"),
            "claim": task["claim"],
            "gold_titles": task["titles"],
            "generated_prompt": generated_prompt,
            "raw_generated_prompt": prompt_generation.raw_text,
            "sanitized_generated_prompt": prompt_generation.sanitized_text,
            "used_fallback_prompt": prompt_generation.used_fallback,
            "prompt_rejection_reason": prompt_generation.rejection_reason,
            "prompt_request": prompt_request,
            "prompt_metrics": prompt_metrics,
            "reward_breakdown": reward_breakdown,
            "exploration_profile": exploration_profile,
            "policy_temperature": reward_breakdown.get("policy_temperature"),
            "deprecated_max_policy_edits": self.max_policy_edits,
            "episode": asdict(episode),
        }
        write_json(artifact_path, payload)

    def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float:
        t0 = time.time()
        print(
            f"[rollout {rollout.rollout_id}] start example_idx={task.get('example_idx')} claim={task['claim'][:80]!r}",
            flush=True,
        )

        runtime = ensure_runtime(
            wiki_dir=self.wiki_dir,
            agent_model=self.agent_model,
            agent_endpoint=self.agent_endpoint,
            extract_instructions=self.extract_instructions,
            demos=self.demos,
            agent_temperature=self.agent_temperature,
            max_agent_steps=self.max_agent_steps,
        )
        print(f"[rollout {rollout.rollout_id}] runtime ready in {time.time() - t0:.1f}s", flush=True)

        llm = cast(agl.LLM, resources["main_llm"])
        # Keep the trainable policy on the traced OpenAI-compatible path.
        # Frozen downstream model calls must stay off this path and use raw HTTP.
        policy_client = OpenAI(
            base_url=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            api_key=os.getenv("OPENAI_API_KEY", os.getenv("RITS_API_KEY", "dummy")),
        )

        rollout_id = str(rollout.rollout_id)
        attempt_id = str(getattr(rollout.attempt, "attempt_id", "attempt"))
        stable_key = _stable_rollout_key(claim=task["claim"], rollout_id=rollout_id, attempt_id=attempt_id)
        profile_idx = zlib.crc32(stable_key.encode("utf-8")) % len(EXPLORATION_PROFILES)
        profile_name, profile_description = EXPLORATION_PROFILES[profile_idx]
        exploration_profile = {"name": profile_name, "description": profile_description}
        rollout_policy_temperature = _select_rollout_temperature(
            claim=task["claim"],
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            base_temperature=self.policy_temperature,
            vary_temperature=self.vary_policy_temperature,
            temperature_schedule=self.policy_temperature_schedule,
        )
        prompt_request = build_candidate_prompt_request(
            seed_prompt=self.seed_react_instructions,
            claim=task["claim"],
            exploration_profile=exploration_profile,
        )

        raw_generated_prompt = _chat_completion_text(
            policy_client,
            model=llm.model,
            messages=[{"role": "user", "content": prompt_request}],
            temperature=rollout_policy_temperature,
            max_tokens=self.policy_max_tokens,
            min_tokens=32,
            disable_thinking=True,
        ).strip()

        if not raw_generated_prompt:
            print(
                f"[rollout {rollout.rollout_id}] empty policy response; assigning zero reward",
                flush=True,
            )
            emit_reward(0.0)
            return 0.0
        prompt_generation = _sanitize_generated_prompt(
            raw_generated_prompt,
            self.seed_react_instructions,
            task["claim"],
        )
        generated_prompt = prompt_generation.sanitized_text
        prompt_metrics = _compute_prompt_metrics(self.seed_react_instructions, generated_prompt)

        if prompt_generation.used_fallback:
            print(
                "[rollout "
                f"{rollout.rollout_id}] using seed fallback prompt "
                f"(reason={prompt_generation.rejection_reason or 'unknown'})",
                flush=True,
            )

        print(
            f"[rollout {rollout.rollout_id}] generated prompt len={len(generated_prompt)} "
            f"novelty={prompt_metrics['novelty_ratio']:.3f} "
            f"profile={exploration_profile['name']} temp={rollout_policy_temperature:.2f} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

        try:
            episode = run_frozen_hover_episode(
                runtime=runtime,
                task=task,
                react_instructions=generated_prompt,
            )
            training_reward, reward_breakdown = self._shape_reward(
                base_reward=episode.reward,
                prompt_generation=prompt_generation,
                prompt_metrics=prompt_metrics,
            )
            reward_breakdown["policy_temperature"] = rollout_policy_temperature
        except Exception as exc:
            print(f"[rollout {rollout.rollout_id}] episode crashed: {exc!r}", flush=True)
            episode = EpisodeResult(
                reward=0.0,
                claim=task["claim"],
                gold_titles=list(task["titles"]),
                pred_titles=[],
                trajectory="",
                finished=False,
                n_steps=0,
                extract_prompt="",
                stop_reason="episode_crash",
            )
            training_reward = 0.0
            reward_breakdown = {
                "base_reward": 0.0,
                "training_reward": 0.0,
                "fallback_penalty": 0.0,
                "length_penalty": 0.0,
                "exploration_bonus": 0.0,
                "drift_penalty": 0.0,
                "policy_temperature": rollout_policy_temperature,
            }

        emit_reward(training_reward)
        try:
            self._persist_rollout_artifact(
                task=task,
                rollout=rollout,
                generated_prompt=generated_prompt,
                prompt_generation=prompt_generation,
                prompt_request=prompt_request,
                episode=episode,
                prompt_metrics=prompt_metrics,
                reward_breakdown=reward_breakdown,
                exploration_profile=exploration_profile,
            )
        except Exception as exc:
            print(
                "[warn] Failed to persist rollout artifact "
                f"for {rollout.rollout_id}/{getattr(rollout.attempt, 'attempt_id', 'attempt')}: {exc!r}",
                flush=True,
            )
        return training_reward


def _write_rollout_summary(output_dir: Path) -> None:
    rollouts_dir = output_dir / "rollouts"
    if not rollouts_dir.exists():
        return

    artifact_files = sorted(rollouts_dir.rglob("*.json"))
    if not artifact_files:
        return

    results_rows: List[Dict[str, Any]] = []
    prompts_rows: List[Dict[str, Any]] = []
    traces_rows: List[Dict[str, Any]] = []

    for path in artifact_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        episode = data.get("episode", {})
        reward_breakdown = data.get("reward_breakdown", {})
        results_rows.append(
            {
                "rollout_id": data.get("rollout_id"),
                "attempt_id": data.get("attempt_id"),
                "split": data.get("split"),
                "example_idx": data.get("example_idx"),
                "hpqa_id": data.get("hpqa_id"),
                "claim": data.get("claim"),
                "gold_titles": data.get("gold_titles"),
                "pred_titles": episode.get("pred_titles"),
                "episode_reward": episode.get("reward"),
                "training_reward": reward_breakdown.get("training_reward", episode.get("reward")),
                "n_steps": episode.get("n_steps"),
                "finished": episode.get("finished"),
                "stop_reason": episode.get("stop_reason"),
            }
        )
        prompts_rows.append(
            {
                "rollout_id": data.get("rollout_id"),
                "attempt_id": data.get("attempt_id"),
                "split": data.get("split"),
                "example_idx": data.get("example_idx"),
                "claim": data.get("claim"),
                "prompt_request": data.get("prompt_request"),
                "raw_generated_prompt": data.get("raw_generated_prompt"),
                "sanitized_generated_prompt": data.get("sanitized_generated_prompt"),
                "generated_prompt": data.get("generated_prompt"),
                "used_fallback_prompt": data.get("used_fallback_prompt"),
                "prompt_rejection_reason": data.get("prompt_rejection_reason"),
                "prompt_metrics": data.get("prompt_metrics"),
                "reward_breakdown": reward_breakdown,
                "exploration_profile": data.get("exploration_profile"),
            }
        )
        traces_rows.append(
            {
                "rollout_id": data.get("rollout_id"),
                "attempt_id": data.get("attempt_id"),
                "split": data.get("split"),
                "example_idx": data.get("example_idx"),
                "claim": data.get("claim"),
                "trajectory": episode.get("trajectory"),
                "extract_prompt": episode.get("extract_prompt"),
                "stop_reason": episode.get("stop_reason"),
            }
        )

    write_jsonl(rollouts_dir / "results.jsonl", results_rows)
    write_jsonl(rollouts_dir / "prompts.jsonl", prompts_rows)
    write_jsonl(rollouts_dir / "traces.jsonl", traces_rows)

    episode_rewards = [row["episode_reward"] for row in results_rows if row.get("episode_reward") is not None]
    training_rewards = [row["training_reward"] for row in results_rows if row.get("training_reward") is not None]
    summary = {
        "n_rollouts": len(results_rows),
        "n_train": sum(1 for row in results_rows if row.get("split") == "train"),
        "n_val": sum(1 for row in results_rows if row.get("split") == "validation"),
        "mean_episode_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
        "mean_training_reward": sum(training_rewards) / len(training_rewards) if training_rewards else 0.0,
        "max_episode_reward": max(episode_rewards) if episode_rewards else 0.0,
        "max_training_reward": max(training_rewards) if training_rewards else 0.0,
        "min_episode_reward": min(episode_rewards) if episode_rewards else 0.0,
        "min_training_reward": min(training_rewards) if training_rewards else 0.0,
        "n_nonzero_episode_reward": sum(1 for reward in episode_rewards if reward > 0),
        "n_fallback_prompts": sum(1 for row in prompts_rows if row.get("used_fallback_prompt")),
    }
    write_json(rollouts_dir / "summary.json", summary)
    print(f"[summary] Rollout summary: {summary}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt optimization with Agent Lightning + vERL (full prompt generation)")
    parser.add_argument(
        "--wiki-dir",
        default=str(WORKSPACE_ROOT / "dspy_tutorial" / "wiki2017"),
        help="Path to the Wikipedia corpus directory.",
    )
    parser.add_argument(
        "--program-json",
        default=str(WORKSPACE_ROOT / "dspy_tutorial" / "mellea_tutorial" / "programs" / "dspy_optimized" / "best_program.json"),
        help="Optional DSPy program JSON used for fixed demos and extraction instructions.",
    )
    parser.add_argument("--n-train", type=int, default=256, help="Number of HoVer training examples.")
    parser.add_argument("--n-dev", type=int, default=100, help="Number of HoVer validation examples.")
    parser.add_argument("--n-iterations", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--group-size", type=int, default=2, help="GRPO rollout group size.")
    parser.add_argument(
        "--tasks-per-step",
        type=int,
        default=2,
        help="Distinct tasks per optimizer step; train_batch_size=group_size*tasks_per_step.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Actor learning rate.")
    parser.add_argument("--kl-coef", type=float, default=0.0, help="KL coefficient.")
    parser.add_argument(
        "--policy-model",
        default="Qwen/Qwen3-8B",
        help="HF model path for the trainable policy.",
    )
    parser.add_argument(
        "--agent-model",
        default="qwen38b",
        help="Frozen downstream model alias or explicit model name.",
    )
    parser.add_argument(
        "--agent-endpoint",
        default=None,
        help="Optional override for the frozen agent RITS endpoint.",
    )
    parser.add_argument("--policy-temperature", type=float, default=0.3, help="Sampling temperature for the policy.")
    parser.add_argument(
        "--vary-policy-temperature",
        action="store_true",
        help="Sweep policy temperatures across rollouts using --policy-temperature-schedule.",
    )
    parser.add_argument(
        "--policy-temperature-schedule",
        type=str,
        default="0.3,0.5,0.7,0.9,1.0",
        help="Comma-separated temperatures used when --vary-policy-temperature is enabled.",
    )
    parser.add_argument("--policy-max-tokens", type=int, default=1024, help="Max tokens for full-prompt generation.")
    parser.add_argument(
        "--max-policy-edits",
        type=int,
        default=2,
        help="Deprecated compatibility flag. Ignored because the policy now emits a full prompt string.",
    )
    parser.add_argument(
        "--exploration-bonus-scale",
        type=float,
        default=0.05,
        help="Small reward bonus scale for successful prompts that differ meaningfully from the seed.",
    )
    parser.add_argument(
        "--exploration-bonus-max-novelty",
        type=float,
        default=0.35,
        help="Maximum novelty ratio counted toward the exploration bonus.",
    )
    parser.add_argument(
        "--drift-penalty-threshold",
        type=float,
        default=0.55,
        help="Start penalizing prompt drift once novelty exceeds this ratio.",
    )
    parser.add_argument(
        "--drift-penalty-scale",
        type=float,
        default=0.1,
        help="Penalty scale for large prompt drift when the prompt does not pay off.",
    )
    parser.add_argument(
        "--extra-word-penalty-per-50-words",
        type=float,
        default=0.01,
        help="Penalty for prompt bloat, applied per 50 extra words over the seed.",
    )
    parser.add_argument("--agent-temperature", type=float, default=0.0, help="Sampling temperature for the frozen agent.")
    parser.add_argument("--max-agent-steps", type=int, default=6, help="Max ReAct steps for the frozen agent.")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory fraction reserved for KV cache and serving.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional explicit vLLM max_model_len override. Leave unset to let vLLM choose from the available KV cache.",
    )
    parser.add_argument("--ray-cpus", type=int, default=8, help="CPUs reserved for Ray.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint/test frequency.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for verl weight checkpoints (default_local_dir). "
                             "Set to a GPFS path to avoid NFS quota/corruption issues.")
    parser.add_argument("--save-freq", type=int, default=0,
                        help="Save a checkpoint every N global steps (0 = never, -1 = every step).")
    parser.add_argument("--resume-mode", default="auto",
                        choices=["disable", "auto", "resume_path"],
                        help="Checkpoint resume mode passed to verl trainer.")
    parser.add_argument("--resume-from-path", default=None,
                        help="Explicit checkpoint path to resume from (requires --resume-mode resume_path).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_agent_endpoint(agent_model_arg: str, endpoint_override: Optional[str]) -> tuple[str, str]:
    if agent_model_arg in RITS_ENDPOINTS:
        model_name, default_endpoint = RITS_ENDPOINTS[agent_model_arg]
        return model_name, endpoint_override or default_endpoint
    if not endpoint_override:
        raise ValueError("When --agent-model is not a known alias, --agent-endpoint must be set.")
    return agent_model_arg, endpoint_override


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = PROJECT_DIR / "runs" / f"prompt_opt_full_prompt_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    os.environ.setdefault("OPENAI_API_KEY", os.getenv("RITS_API_KEY", "dummy"))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print("=" * 60)
    print("Prompt Optimization with Agent Lightning + vERL")
    print("(full prompt-string policy over a seed prompt)")
    print("=" * 60)
    print(f"[env] Python: {sys.executable}")
    print(f"[env] Torch: {torch.__version__}")
    print(f"[env] Agent Lightning: {agl.__version__}")

    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available.")
        return 1
    print(f"[env] CUDA devices: {torch.cuda.device_count()}")
    print(f"[env] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")

    train_batch_size = args.group_size * args.tasks_per_step
    if args.n_train < train_batch_size:
        print(
            f"[FAIL] n_train={args.n_train} is smaller than train_batch_size={train_batch_size}. "
            "Increase --n-train or reduce --group-size/--tasks-per-step."
        )
        return 1

    output_dir = build_output_dir(args)
    print(f"[config] Output directory: {output_dir}")
    policy_temperature_schedule = _parse_temperature_schedule(args.policy_temperature_schedule)

    if args.program_json and Path(args.program_json).exists():
        prompt_config = load_program_prompts(args.program_json)
        demos = prompt_config["demos"]
        seed_react_instructions = prompt_config["react_instructions"] or _default_react_instructions()
        extract_instructions = prompt_config["extract_instructions"] or (
            "Extract all Wikipedia titles relevant to verifying or refuting the claim."
        )
    else:
        prompt_config = {"react_instructions": "", "extract_instructions": None, "demos": []}
        demos = []
        seed_react_instructions = _default_react_instructions()
        extract_instructions = "Extract all Wikipedia titles relevant to verifying or refuting the claim."

    agent_model_name, agent_endpoint = resolve_agent_endpoint(args.agent_model, args.agent_endpoint)
    print(f"[config] Policy model: {args.policy_model}")
    print(f"[config] Frozen agent: {agent_model_name} via {agent_endpoint}")
    print(f"[config] Train batch size: {train_batch_size} ({args.tasks_per_step} tasks x group_size {args.group_size})")

    trainset, devset, _ = load_hover_data(
        n_train=args.n_train,
        n_dev=args.n_dev,
        seed=args.seed,
    )
    trainset = [{**example, "split": "train", "example_idx": idx} for idx, example in enumerate(trainset)]
    devset = [{**example, "split": "validation", "example_idx": idx} for idx, example in enumerate(devset)]

    ray_temp_dir = os.environ.get("RAY_tmpdir", f"/tmp/ray_{os.getpid()}")
    print(f"[init] Ray temp dir: {ray_temp_dir}")
    ray.init(
        _temp_dir=ray_temp_dir,
        num_cpus=args.ray_cpus,
        object_store_memory=10 * 1024**3,
        ignore_reinit_error=True,
    )
    print(f"[init] Ray resources: {ray.cluster_resources()}")

    run_config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy_model": args.policy_model,
        "agent_model": agent_model_name,
        "agent_endpoint": agent_endpoint,
        "wiki_dir": args.wiki_dir,
        "program_json": args.program_json,
        "n_train": len(trainset),
        "n_dev": len(devset),
        "n_iterations": args.n_iterations,
        "group_size": args.group_size,
        "tasks_per_step": args.tasks_per_step,
        "train_batch_size": train_batch_size,
        "learning_rate": args.learning_rate,
        "policy_temperature": args.policy_temperature,
        "vary_policy_temperature": args.vary_policy_temperature,
        "policy_temperature_schedule": policy_temperature_schedule,
        "policy_max_tokens": args.policy_max_tokens,
        "max_policy_edits": args.max_policy_edits,
        "exploration_bonus_scale": args.exploration_bonus_scale,
        "exploration_bonus_max_novelty": args.exploration_bonus_max_novelty,
        "drift_penalty_threshold": args.drift_penalty_threshold,
        "drift_penalty_scale": args.drift_penalty_scale,
        "extra_word_penalty_per_50_words": args.extra_word_penalty_per_50_words,
        "kl_coef": args.kl_coef,
        "max_agent_steps": args.max_agent_steps,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_len": args.vllm_max_model_len,
        "seed_prompt_source": "program_json" if prompt_config.get("react_instructions") else "default_react_instructions",
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "program_prompts.json",
        {
            "react_instructions": seed_react_instructions,
            "extract_instructions": extract_instructions,
            "react_demos": demos,
        },
    )
    write_jsonl(output_dir / "train_examples.jsonl", trainset)
    write_jsonl(output_dir / "dev_examples.jsonl", devset)

    agent = PromptOptimizationAgent(
        wiki_dir=args.wiki_dir,
        agent_model=agent_model_name,
        agent_endpoint=agent_endpoint,
        extract_instructions=extract_instructions,
        demos=demos,
        policy_temperature=args.policy_temperature,
        vary_policy_temperature=args.vary_policy_temperature,
        policy_temperature_schedule=policy_temperature_schedule,
        policy_max_tokens=args.policy_max_tokens,
        max_policy_edits=args.max_policy_edits,
        exploration_bonus_scale=args.exploration_bonus_scale,
        exploration_bonus_max_novelty=args.exploration_bonus_max_novelty,
        drift_penalty_threshold=args.drift_penalty_threshold,
        drift_penalty_scale=args.drift_penalty_scale,
        extra_word_penalty_per_50_words=args.extra_word_penalty_per_50_words,
        agent_temperature=args.agent_temperature,
        max_agent_steps=args.max_agent_steps,
        output_dir=str(output_dir),
        seed_react_instructions=seed_react_instructions,
    )

    model_max_context = _get_model_context_size(args.policy_model)
    # requested_prompt_length = 1024
    requested_prompt_length = 2048
    max_prompt_length, max_response_length = _compute_safe_verl_token_budgets(
        model=args.policy_model,
        requested_prompt_length=requested_prompt_length,
        requested_response_length=args.policy_max_tokens,
    )
    if max_prompt_length < requested_prompt_length or max_response_length < args.policy_max_tokens:
        print(
            "[warn] Adjusted veRL token budgets to fit the policy model context: "
            f"prompt_length={max_prompt_length} (requested {requested_prompt_length}), "
            f"response_length={max_response_length} (requested {args.policy_max_tokens}), "
            f"context={model_max_context}, slack={CHAT_TEMPLATE_TOKEN_SLACK}",
            flush=True,
        )

    verl_config = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": args.kl_coef > 0,
        },
        "data": {
            "train_batch_size": train_batch_size,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "truncation": "right",
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 2,
                "n": args.group_size,
                "log_prob_micro_batch_size_per_gpu": 1,
                "name": "vllm",
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_model_len": args.vllm_max_model_len,
            },
            "actor": {
                "ppo_mini_batch_size": train_batch_size,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {"lr": args.learning_rate},
                "use_kl_loss": args.kl_coef > 0,
                "kl_loss_coef": args.kl_coef,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": args.policy_model,
                "use_remove_padding": False,
                "enable_gradient_checkpointing": True,
                "torch_dtype": "bfloat16",
            },
        },
        "trainer": {
            "n_gpus_per_node": 2,
            "nnodes": 1,
            "total_epochs": args.n_iterations,
            "save_freq": args.save_freq,
            "default_local_dir": args.checkpoint_dir or str(output_dir / "checkpoints"),
            "resume_mode": args.resume_mode,
            "resume_from_path": args.resume_from_path,
            "test_freq": args.checkpoint_every,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "PromptOptimization",
            "experiment_name": output_dir.name,
        },
    }

    initial_resources = {
        "prompt_template": PromptTemplate(template="", engine="f-string"),
    }

    trainer = agl.Trainer(
        algorithm=agl.VERL(config=verl_config),
        n_runners=1,
        initial_resources=initial_resources,
    )

    try:
        trainer.fit(agent, train_dataset=trainset, val_dataset=devset)
    except Exception as exc:
        print("[FAIL] Training crashed.")
        crash_info = {
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "crash_info.json").write_text(json.dumps(crash_info, indent=2))
        raise
    finally:
        ray.shutdown()

    _write_rollout_summary(output_dir)
    final_results = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "policy_model": args.policy_model,
        "agent_model": agent_model_name,
        "rollout_results_jsonl": str(output_dir / "rollouts" / "results.jsonl"),
        "rollout_prompts_jsonl": str(output_dir / "rollouts" / "prompts.jsonl"),
        "rollout_traces_jsonl": str(output_dir / "rollouts" / "traces.jsonl"),
        "rollout_summary_json": str(output_dir / "rollouts" / "summary.json"),
        "train_examples_path": str(output_dir / "train_examples.jsonl"),
        "dev_examples_path": str(output_dir / "dev_examples.jsonl"),
        "program_prompts_path": str(output_dir / "program_prompts.json"),
    }
    write_json(output_dir / "final_results.json", final_results)

    print("=" * 60)
    print("Training finished successfully")
    print("=" * 60)
    print(json.dumps(final_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
