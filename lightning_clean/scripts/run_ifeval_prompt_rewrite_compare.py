
#!/usr/bin/env python3
"""Compare three IFEval conditions:

1. Original IFEval prompts as-is.
2. Prompts rewritten by base Qwen 3 8B, then answered by an evaluation model.
3. Prompts rewritten by a fine-tuned Qwen 3 8B checkpoint (or served endpoint), then answered by an evaluation model.

The evaluator is the built-in `ifeval.Evaluator`.

Typical usage with a local fine-tuned checkpoint:
    export RITS_API_KEY=...
    python run_ifeval_prompt_rewrite_compare.py \
      --eval_model Qwen/Qwen3-8B \
      --eval_endpoint qwen3-8b \
      --base_rewriter_model Qwen/Qwen3-8B \
      --base_rewriter_endpoint qwen3-8b \
      --ft_rewriter_checkpoint /path/to/checkpoint \
      --output_dir ./ifeval_rewrite_compare

If the fine-tuned model is also served on a RITS/OpenAI-compatible endpoint:
    python run_ifeval_prompt_rewrite_compare.py \
      --ft_rewriter_backend openai \
      --ft_rewriter_model Qwen/Qwen3-8B \
      --ft_rewriter_endpoint qwen3-8b-ft \
      ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3-way IFEval comparison with prompt rewriting.")

    # Data / eval
    parser.add_argument("--language", default="en", choices=["en", "ru"], help="IFEval language.")
    parser.add_argument("--dataset_mode", default="default", choices=["default", "hf"], help="Use ifeval's built-in dataset loader or load google/IFEval through datasets.")
    parser.add_argument("--split", default="train", help="Dataset split when --dataset_mode=hf.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N examples after loading. 0 means all.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics, prompts, and raw generations.")

    # Shared API
    parser.add_argument("--api_key_env", default="RITS_API_KEY", help="Environment variable containing the RITS API key.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=4, help="Number of retries per prompt on transient API failures.")
    parser.add_argument("--sleep_on_error", type=float, default=3.0, help="Seconds to sleep between retries.")
    parser.add_argument("--save_every", type=int, default=25, help="Flush intermediate outputs every N examples.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL files if present.")

    # Answer-generation model used for evaluation responses
    parser.add_argument("--eval_model", default="Qwen/Qwen3-8B", help="Model id used to answer IFEval prompts.")
    parser.add_argument("--eval_endpoint", default="qwen3-8b", help="RITS endpoint slug for the answer model.")
    parser.add_argument("--eval_temperature", type=float, default=0.0, help="Sampling temperature for the answer model.")
    parser.add_argument("--eval_max_tokens", type=int, default=768, help="Maximum completion tokens for the answer model.")
    parser.add_argument("--eval_top_p", type=float, default=1.0, help="top_p for answer generation.")
    parser.add_argument("--eval_system_prompt", default="", help="Optional system prompt prepended to answer-generation requests.")

    # Base prompt-rewriter
    parser.add_argument("--base_rewriter_model", default="Qwen/Qwen3-8B", help="Base model used to rewrite prompts.")
    parser.add_argument("--base_rewriter_endpoint", default="qwen3-8b", help="RITS endpoint slug for the base rewriter.")
    parser.add_argument("--base_rewriter_temperature", type=float, default=0.2, help="Sampling temperature for the base rewriter.")
    parser.add_argument("--base_rewriter_max_tokens", type=int, default=512, help="Maximum completion tokens for the base rewriter.")
    parser.add_argument("--base_rewriter_top_p", type=float, default=1.0, help="top_p for the base rewriter.")
    parser.add_argument("--base_rewriter_system_prompt", default="", help="Optional system prompt for the base rewriter.")

    # Fine-tuned prompt-rewriter
    parser.add_argument("--ft_rewriter_backend", default="local", choices=["local", "openai"], help="Whether the fine-tuned rewriter is loaded locally from a checkpoint or called via an OpenAI-compatible endpoint.")
    parser.add_argument("--ft_rewriter_checkpoint", default="", help="Local checkpoint path for the fine-tuned rewriter when --ft_rewriter_backend=local.")
    parser.add_argument("--ft_rewriter_model", default="Qwen/Qwen3-8B", help="Model id for the fine-tuned rewriter when using an OpenAI-compatible endpoint, or tokenizer/model fallback base when loading local checkpoint.")
    parser.add_argument("--ft_rewriter_endpoint", default="", help="Endpoint slug for the fine-tuned rewriter when --ft_rewriter_backend=openai.")
    parser.add_argument("--ft_rewriter_temperature", type=float, default=0.2, help="Sampling temperature for the fine-tuned rewriter.")
    parser.add_argument("--ft_rewriter_max_tokens", type=int, default=512, help="Maximum completion tokens for the fine-tuned rewriter.")
    parser.add_argument("--ft_rewriter_top_p", type=float, default=1.0, help="top_p for the fine-tuned rewriter.")
    parser.add_argument("--ft_rewriter_system_prompt", default="", help="Optional system prompt for the fine-tuned rewriter.")
    parser.add_argument("--ft_rewriter_device", default="cuda", help="Device for a local fine-tuned rewriter checkpoint.")
    parser.add_argument("--ft_rewriter_dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"], help="Torch dtype for a local fine-tuned rewriter checkpoint.")

    # Prompt rewriting policy
    parser.add_argument("--rewriter_style", default="instruction_following", choices=["instruction_following", "minimal", "structured"], help="How aggressively to rewrite prompts.")
    parser.add_argument("--keep_original_if_empty", action="store_true", help="If a rewriter returns an empty/invalid prompt, fall back to the original prompt instead of failing.")
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


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_prompt_response_map(path: Path, key_field: str = "prompt", value_field: str = "response") -> Dict[str, Any]:
    rows = read_jsonl_rows(path)
    return {row[key_field]: row[value_field] for row in rows if key_field in row and value_field in row}


class OpenAIChatGenerator:
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
        system_prompt: str = "",
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
                    text = "".join(part.text for part in content if getattr(part, "type", None) == "text").strip()
                else:
                    text = (content or "").strip()
                return text
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
                last_err = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.sleep_on_error)
        raise RuntimeError(f"Generation failed after {self.max_retries} attempts: {last_err!r}")


class LocalHFGenerator:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        fallback_model_name: str,
        device: str,
        dtype: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        system_prompt: str = "",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt.strip()

        if not checkpoint_path:
            raise ValueError("--ft_rewriter_checkpoint must be set when --ft_rewriter_backend=local.")

        dtype_map = {
            "auto": None,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self.uses_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    def generate(self, prompt: str) -> str:
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        if self.uses_chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
        else:
            text = ""
            if self.system_prompt:
                text += f"System: {self.system_prompt}\n\n"
            text += f"User: {prompt}\n\nAssistant:"
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]

        input_ids = input_ids.to(self.model.device)
        attention_mask = self.torch.ones_like(input_ids)

        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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


def sanitize_rewritten_prompt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    tagged = re.search(r"<rewritten_prompt>\s*([\s\S]*?)\s*</rewritten_prompt>", text, flags=re.IGNORECASE)
    if tagged:
        text = tagged.group(1).strip()

    fenced = re.search(r"```(?:text|markdown)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    text = re.sub(r"^(Here is|Here's|Rewritten prompt:|Improved prompt:)\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip()


def build_rewriter_request(original_prompt: str, style: str) -> str:
    style_instruction = {
        "instruction_following": (
            "Rewrite the prompt so a model is more likely to follow every instruction correctly. "
            "Keep the task, all constraints, and all required content intact. "
            "You may clarify wording, reorganize the request, make constraints more explicit, "
            "and reduce ambiguity, but do not change the underlying assignment."
        ),
        "minimal": (
            "Make the smallest possible edits that improve instruction clarity and compliance. "
            "Do not alter the task, requested content, or constraints."
        ),
        "structured": (
            "Rewrite the prompt into a clearer, more structured version that preserves all original requirements exactly. "
            "Use explicit wording and organization to improve instruction following."
        ),
    }[style]

    return (
        "You are improving a user prompt for an instruction-following benchmark.\n\n"
        f"{style_instruction}\n\n"
        "Hard constraints:\n"
        "- Preserve the meaning of the task.\n"
        "- Preserve every original requirement and restriction.\n"
        "- Do not add new factual content that answers the task.\n"
        "- Do not add extra requirements unrelated to the original prompt.\n"
        "- Output only the rewritten prompt, wrapped in <rewritten_prompt>...</rewritten_prompt> tags.\n\n"
        "Original prompt:\n"
        "<original_prompt>\n"
        f"{original_prompt}\n"
        "</original_prompt>\n\n"
        "<rewritten_prompt>"
    )


def evaluate_condition(
    *,
    condition_name: str,
    examples: List[Any],
    answer_generator: OpenAIChatGenerator,
    prompt_map: Dict[str, str],
    output_dir: Path,
    save_every: int,
    resume: bool,
) -> Dict[str, Any]:
    from ifeval import Evaluator, instruction_registry, ru_instruction_registry

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    existing = read_prompt_response_map(responses_path) if resume else {}
    responses: Dict[str, str] = dict(existing)
    generation_rows = read_jsonl_rows(responses_path) if resume and responses_path.exists() else []

    for idx, ex in enumerate(tqdm(examples, desc=f"Answering: {condition_name}"), start=1):
        effective_prompt = prompt_map[ex.prompt]
        if ex.prompt in responses:
            continue
        response_text = answer_generator.generate(effective_prompt)
        responses[ex.prompt] = response_text
        generation_rows.append(
            {
                "key": getattr(ex, "key", None),
                "original_prompt": ex.prompt,
                "effective_prompt": effective_prompt,
                "response": response_text,
            }
        )
        if save_every > 0 and len(generation_rows) % save_every == 0:
            write_jsonl(responses_path, generation_rows)

    write_jsonl(responses_path, generation_rows)

    registry = instruction_registry
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
                "original_prompt": ex.prompt,
                "effective_prompt": prompt_map[ex.prompt],
                "instruction_id_list": getattr(ex, "instruction_id_list", None),
                "kwargs": getattr(ex, "kwargs", None),
                "response": responses.get(ex.prompt, ""),
                "evaluation": to_jsonable(out),
            }
        )
    write_jsonl(output_dir / "detailed_outputs.jsonl", detailed_rows)

    return {
        "condition_name": condition_name,
        "report": report,
        "responses_path": str(responses_path),
        "all_outputs_path": str(output_dir / "all_outputs.json"),
        "detailed_outputs_path": str(output_dir / "detailed_outputs.jsonl"),
    }


def generate_rewritten_prompts(
    *,
    condition_name: str,
    examples: List[Any],
    rewriter: Any,
    style: str,
    output_dir: Path,
    save_every: int,
    resume: bool,
    keep_original_if_empty: bool,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rewrites_path = output_dir / "rewritten_prompts.jsonl"
    existing_rows = read_jsonl_rows(rewrites_path) if resume else []
    rewrites: Dict[str, str] = {row["original_prompt"]: row["rewritten_prompt"] for row in existing_rows if "original_prompt" in row and "rewritten_prompt" in row}
    rows = list(existing_rows)

    for ex in tqdm(examples, desc=f"Rewriting: {condition_name}"):
        if ex.prompt in rewrites:
            continue

        request = build_rewriter_request(ex.prompt, style=style)
        raw_text = rewriter.generate(request)
        rewritten = sanitize_rewritten_prompt(raw_text)

        if not rewritten:
            if keep_original_if_empty:
                rewritten = ex.prompt
            else:
                raise RuntimeError(f"{condition_name} returned an empty rewritten prompt for key={getattr(ex, 'key', None)}")

        rewrites[ex.prompt] = rewritten
        rows.append(
            {
                "key": getattr(ex, "key", None),
                "original_prompt": ex.prompt,
                "rewriter_request": request,
                "raw_rewriter_output": raw_text,
                "rewritten_prompt": rewritten,
                "changed": rewritten != ex.prompt,
            }
        )
        if save_every > 0 and len(rows) % save_every == 0:
            write_jsonl(rewrites_path, rows)

    write_jsonl(rewrites_path, rows)
    return rewrites


def build_generators(args: argparse.Namespace, api_key: str) -> Tuple[OpenAIChatGenerator, OpenAIChatGenerator, Any]:
    eval_client = make_client(endpoint=args.eval_endpoint, api_key=api_key, timeout=args.timeout)
    eval_generator = OpenAIChatGenerator(
        client=eval_client,
        model=args.eval_model,
        temperature=args.eval_temperature,
        max_tokens=args.eval_max_tokens,
        top_p=args.eval_top_p,
        max_retries=args.max_retries,
        sleep_on_error=args.sleep_on_error,
        system_prompt=args.eval_system_prompt,
    )

    base_rewriter_client = make_client(endpoint=args.base_rewriter_endpoint, api_key=api_key, timeout=args.timeout)
    base_rewriter = OpenAIChatGenerator(
        client=base_rewriter_client,
        model=args.base_rewriter_model,
        temperature=args.base_rewriter_temperature,
        max_tokens=args.base_rewriter_max_tokens,
        top_p=args.base_rewriter_top_p,
        max_retries=args.max_retries,
        sleep_on_error=args.sleep_on_error,
        system_prompt=args.base_rewriter_system_prompt,
    )

    if args.ft_rewriter_backend == "openai":
        if not args.ft_rewriter_endpoint:
            raise ValueError("--ft_rewriter_endpoint is required when --ft_rewriter_backend=openai.")
        ft_client = make_client(endpoint=args.ft_rewriter_endpoint, api_key=api_key, timeout=args.timeout)
        ft_rewriter = OpenAIChatGenerator(
            client=ft_client,
            model=args.ft_rewriter_model,
            temperature=args.ft_rewriter_temperature,
            max_tokens=args.ft_rewriter_max_tokens,
            top_p=args.ft_rewriter_top_p,
            max_retries=args.max_retries,
            sleep_on_error=args.sleep_on_error,
            system_prompt=args.ft_rewriter_system_prompt,
        )
    else:
        ft_rewriter = LocalHFGenerator(
            checkpoint_path=args.ft_rewriter_checkpoint,
            fallback_model_name=args.ft_rewriter_model,
            device=args.ft_rewriter_device,
            dtype=args.ft_rewriter_dtype,
            temperature=args.ft_rewriter_temperature,
            max_tokens=args.ft_rewriter_max_tokens,
            top_p=args.ft_rewriter_top_p,
            system_prompt=args.ft_rewriter_system_prompt,
        )

    return eval_generator, base_rewriter, ft_rewriter


def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key and args.ft_rewriter_backend != "local":
        raise RuntimeError(f"Set {args.api_key_env} (or OPENAI_API_KEY) before running this script.")
    if not api_key:
        api_key = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(language=args.language, dataset_mode=args.dataset_mode, split=args.split)
    examples = maybe_slice(examples, args.limit)
    if not examples:
        raise RuntimeError("No IFEval examples were loaded.")

    eval_generator, base_rewriter, ft_rewriter = build_generators(args, api_key)

    metadata = {
        "language": args.language,
        "dataset_mode": args.dataset_mode,
        "split": args.split,
        "num_examples": len(examples),
        "rewriter_style": args.rewriter_style,
        "keep_original_if_empty": args.keep_original_if_empty,
        "eval_model": args.eval_model,
        "eval_endpoint": args.eval_endpoint,
        "base_rewriter_model": args.base_rewriter_model,
        "base_rewriter_endpoint": args.base_rewriter_endpoint,
        "ft_rewriter_backend": args.ft_rewriter_backend,
        "ft_rewriter_model": args.ft_rewriter_model,
        "ft_rewriter_endpoint": args.ft_rewriter_endpoint,
        "ft_rewriter_checkpoint": args.ft_rewriter_checkpoint,
        "resume": args.resume,
    }
    write_json(output_dir / "run_config.json", metadata)
    write_jsonl(
        output_dir / "dataset_examples.jsonl",
        [
            {
                "key": getattr(ex, "key", None),
                "prompt": ex.prompt,
                "instruction_id_list": getattr(ex, "instruction_id_list", None),
                "kwargs": getattr(ex, "kwargs", None),
            }
            for ex in examples
        ],
    )

    baseline_prompt_map = {ex.prompt: ex.prompt for ex in examples}

    base_rewrites = generate_rewritten_prompts(
        condition_name="base_qwen_rewrite",
        examples=examples,
        rewriter=base_rewriter,
        style=args.rewriter_style,
        output_dir=output_dir / "base_qwen_rewrite",
        save_every=args.save_every,
        resume=args.resume,
        keep_original_if_empty=args.keep_original_if_empty,
    )
    ft_rewrites = generate_rewritten_prompts(
        condition_name="ft_qwen_rewrite",
        examples=examples,
        rewriter=ft_rewriter,
        style=args.rewriter_style,
        output_dir=output_dir / "ft_qwen_rewrite",
        save_every=args.save_every,
        resume=args.resume,
        keep_original_if_empty=args.keep_original_if_empty,
    )

    baseline_result = evaluate_condition(
        condition_name="original_prompt",
        examples=examples,
        answer_generator=eval_generator,
        prompt_map=baseline_prompt_map,
        output_dir=output_dir / "original_prompt",
        save_every=args.save_every,
        resume=args.resume,
    )
    base_result = evaluate_condition(
        condition_name="base_qwen_rewrite",
        examples=examples,
        answer_generator=eval_generator,
        prompt_map=base_rewrites,
        output_dir=output_dir / "base_qwen_rewrite",
        save_every=args.save_every,
        resume=args.resume,
    )
    ft_result = evaluate_condition(
        condition_name="ft_qwen_rewrite",
        examples=examples,
        answer_generator=eval_generator,
        prompt_map=ft_rewrites,
        output_dir=output_dir / "ft_qwen_rewrite",
        save_every=args.save_every,
        resume=args.resume,
    )

    summary = {
        "original_prompt": baseline_result["report"],
        "base_qwen_rewrite": base_result["report"],
        "ft_qwen_rewrite": ft_result["report"],
    }
    write_json(output_dir / "comparison_summary.json", summary)

    compact = {}
    for name, result in [
        ("original_prompt", baseline_result),
        ("base_qwen_rewrite", base_result),
        ("ft_qwen_rewrite", ft_result),
    ]:
        report = result["report"]
        compact[name] = {
            "strict_prompt_accuracy": report["eval_results_strict"]["prompt_accuracy"],
            "strict_instruction_accuracy": report["eval_results_strict"]["instruction_accuracy"],
            "loose_prompt_accuracy": report["eval_results_loose"]["prompt_accuracy"],
            "loose_instruction_accuracy": report["eval_results_loose"]["instruction_accuracy"],
        }
    write_json(output_dir / "comparison_summary_compact.json", compact)

    print("\n=== 3-way IFEval comparison ===")
    for name, vals in compact.items():
        print(f"\n[{name}]")
        print(f"  Strict prompt accuracy:      {vals['strict_prompt_accuracy']:.4f}")
        print(f"  Strict instruction accuracy: {vals['strict_instruction_accuracy']:.4f}")
        print(f"  Loose prompt accuracy:       {vals['loose_prompt_accuracy']:.4f}")
        print(f"  Loose instruction accuracy:  {vals['loose_instruction_accuracy']:.4f}")

    print(f"\nSaved comparison summary to: {output_dir / 'comparison_summary_compact.json'}")


if __name__ == "__main__":
    main()
