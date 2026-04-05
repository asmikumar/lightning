#!/usr/bin/env python3
"""Run IFEval using a Qwen model served through IBM RITS.

This script:
1. Loads the English IFEval benchmark examples.
2. Sends each prompt to a Qwen model through an OpenAI-compatible RITS endpoint.
3. Evaluates the responses with the built-in IFEval evaluator.
4. Writes aggregate metrics plus raw generations/per-example outputs to disk.

Example:
    export RITS_API_KEY=...
    python run_ifeval_rits.py \
        --model Qwen/Qwen3-8B \
        --endpoint qwen3-8b \
        --limit 50 \
        --output_dir ./ifeval_runs/qwen3_8b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen on IFEval through RITS.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model id to send in the RITS request.")
    parser.add_argument("--endpoint", default="qwen3-8b", help="RITS endpoint slug.")
    parser.add_argument("--api_key_env", default="RITS_API_KEY", help="Environment variable containing the RITS API key.")
    parser.add_argument("--language", default="en", choices=["en", "ru"], help="IFEval language.")
    parser.add_argument("--dataset_mode", default="default", choices=["default", "hf"], help="Use ifeval's built-in dataset loader or load google/IFEval through datasets.")
    parser.add_argument("--split", default="train", help="Dataset split when --dataset_mode=hf.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N examples after loading. 0 means all.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max_tokens", type=int, default=768, help="Maximum completion tokens.")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for sampling.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=4, help="Number of retries per prompt on transient API failures.")
    parser.add_argument("--sleep_on_error", type=float, default=3.0, help="Seconds to sleep between retries.")
    parser.add_argument("--system_prompt", default="", help="Optional system prompt prepended to every request.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics, responses, and detailed outputs.")
    parser.add_argument("--save_every", type=int, default=25, help="Flush intermediate responses to disk every N examples.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing responses.jsonl in output_dir if present.")
    return parser.parse_args()


def get_rits_base_url(endpoint: str) -> str:
    return f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{endpoint}/v1"


def make_client(*, endpoint: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(
        base_url=get_rits_base_url(endpoint),
        api_key=api_key,
        default_headers={"RITS_API_KEY": api_key},
        timeout=timeout,
    )


def maybe_slice(items: List[Any], limit: int) -> List[Any]:
    if limit and limit > 0:
        return items[:limit]
    return items


def load_examples(*, language: str, dataset_mode: str, split: str) -> List[Any]:
    from ifeval import get_default_dataset

    if dataset_mode == "default":
        return list(get_default_dataset(language))

    from datasets import load_dataset
    from ifeval import InputExample

    if language != "en":
        raise ValueError("The HF loading path in this script currently expects English IFEval examples.")

    ds = load_dataset("google/IFEval", split=split)
    examples = []
    for row in ds:
        examples.append(
            InputExample(
                key=row["key"],
                prompt=row["prompt"],
                instruction_id_list=list(row["instruction_id_list"]),
                kwargs=list(row["kwargs"]),
            )
        )
    return examples


def read_existing_responses(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["prompt"]] = row["response"]
    return out


class RITSGenerator:
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        max_retries: int,
        sleep_on_error: float,
        system_prompt: str,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.sleep_on_error = sleep_on_error
        self.system_prompt = system_prompt.strip()

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(prompt),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                if isinstance(content, list):
                    text = "".join(
                        part.text for part in content if getattr(part, "type", None) == "text"
                    ).strip()
                else:
                    text = (content or "").strip()
                return text
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
                last_err = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.sleep_on_error)
        raise RuntimeError(f"Generation failed after {self.max_retries} attempts: {last_err!r}")



def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)



def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(dict(row)), ensure_ascii=False) + "\n")



def evaluate_examples(
    *,
    examples: List[Any],
    generator: RITSGenerator,
    output_dir: Path,
    save_every: int,
    resume: bool,
) -> Dict[str, Any]:
    from ifeval import Evaluator, instruction_registry, ru_instruction_registry

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    existing = read_existing_responses(responses_path) if resume else {}
    responses: Dict[str, str] = dict(existing)

    generation_rows: List[Dict[str, Any]] = []
    if existing:
        for ex in examples:
            if ex.prompt in existing:
                generation_rows.append({"prompt": ex.prompt, "response": existing[ex.prompt]})

    start_index = len(generation_rows)
    pending_count = len(examples) - start_index
    print(f"Loaded {len(examples)} examples; {len(existing)} existing responses; {pending_count} to generate.")

    for idx, ex in enumerate(tqdm(examples, desc="Generating"), start=1):
        if ex.prompt in responses:
            continue
        text = generator.generate(ex.prompt)
        responses[ex.prompt] = text
        generation_rows.append({"prompt": ex.prompt, "response": text})
        if save_every > 0 and len(generation_rows) % save_every == 0:
            write_jsonl(responses_path, generation_rows)

    write_jsonl(responses_path, generation_rows)

    registry = instruction_registry if getattr(examples[0], "instruction_id_list", None) is not None else instruction_registry
    if hasattr(examples[0], "instruction_id_list") and any("ru:" in inst for inst in getattr(examples[0], "instruction_id_list", [])):
        registry = ru_instruction_registry

    evaluator = Evaluator(registry)
    report, all_outputs = evaluator.evaluate(examples, responses)

    write_json(output_dir / "report.json", report)
    write_json(output_dir / "all_outputs.json", all_outputs)

    detailed_rows = []
    for ex, out in zip(examples, all_outputs):
        detailed_rows.append(
            {
                "key": getattr(ex, "key", None),
                "prompt": ex.prompt,
                "instruction_id_list": getattr(ex, "instruction_id_list", None),
                "kwargs": getattr(ex, "kwargs", None),
                "response": responses.get(ex.prompt, ""),
                "evaluation": to_jsonable(out),
            }
        )
    write_jsonl(output_dir / "detailed_outputs.jsonl", detailed_rows)

    return {
        "report": report,
        "responses_path": str(responses_path),
        "all_outputs_path": str(output_dir / "all_outputs.json"),
        "detailed_outputs_path": str(output_dir / "detailed_outputs.jsonl"),
    }



def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} (or OPENAI_API_KEY) before running this script."
        )

    output_dir = Path(args.output_dir)
    client = make_client(endpoint=args.endpoint, api_key=api_key, timeout=args.timeout)
    generator = RITSGenerator(
        client=client,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        max_retries=args.max_retries,
        sleep_on_error=args.sleep_on_error,
        system_prompt=args.system_prompt,
    )

    examples = load_examples(language=args.language, dataset_mode=args.dataset_mode, split=args.split)
    examples = maybe_slice(examples, args.limit)
    if not examples:
        raise RuntimeError("No IFEval examples were loaded.")

    metadata = {
        "model": args.model,
        "endpoint": args.endpoint,
        "base_url": get_rits_base_url(args.endpoint),
        "language": args.language,
        "dataset_mode": args.dataset_mode,
        "split": args.split,
        "num_examples": len(examples),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "resume": args.resume,
        "system_prompt": args.system_prompt,
    }
    write_json(output_dir / "run_config.json", metadata)

    result = evaluate_examples(
        examples=examples,
        generator=generator,
        output_dir=output_dir,
        save_every=args.save_every,
        resume=args.resume,
    )

    report = result["report"]
    strict = report["eval_results_strict"]
    loose = report["eval_results_loose"]

    print("\n=== IFEval results ===")
    print(f"Strict prompt accuracy:      {strict['prompt_accuracy']:.4f}")
    print(f"Strict instruction accuracy: {strict['instruction_accuracy']:.4f}")
    print(f"Loose prompt accuracy:       {loose['prompt_accuracy']:.4f}")
    print(f"Loose instruction accuracy:  {loose['instruction_accuracy']:.4f}")
    print(f"Saved responses to:          {result['responses_path']}")
    print(f"Saved report to:             {output_dir / 'report.json'}")
    print(f"Saved per-example outputs to:{result['detailed_outputs_path']}")


if __name__ == "__main__":
    main()
