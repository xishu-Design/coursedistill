import argparse
import json
import os
import signal
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from verifier import extract_boxed_answer, verify_response


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_with_timeout(seconds: float, func, *args, **kwargs):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return func(*args, **kwargs)

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"operation timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def resolve_dtype(value: str):
    import torch

    value = (value or "bfloat16").lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def apply_chat_prompt(tokenizer, problem: str, student_thinking: bool) -> str:
    user_message = (
        f"Problem: {problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    messages = [{"role": "user", "content": user_message}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=student_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{user_message}\n\nAssistant:"


def wrap_latex_if_needed(answer: str) -> str:
    answer = str(answer or "").strip()
    if not answer:
        return answer
    if answer.startswith("$") or answer.startswith(r"\(") or answer.startswith(r"\["):
        return answer
    if "\\" in answer or "{" in answer or "^" in answer or "_" in answer:
        return f"${answer}$"
    return answer


class AnswerGrader:
    def __init__(self, name: str, prm800k_grading_dir: str | None = None):
        self.name = name
        self._math_verify = None
        self._prm800k_grade_answer = None

        if name == "math_verify":
            from math_verify import parse, verify
            try:
                from math_verify import ExprExtractionConfig, LatexExtractionConfig
            except ImportError:
                from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

            self._math_verify = {
                "parse": parse,
                "verify": verify,
                "configs": [LatexExtractionConfig(), ExprExtractionConfig()],
            }
        elif name == "prm800k":
            self._add_prm800k_grading_path(prm800k_grading_dir)
            try:
                from grading.grader import grade_answer
            except ImportError as exc:
                raise ImportError(
                    "Could not import OpenAI PRM800K official grader. "
                    "Clone https://github.com/openai/prm800k and pass either "
                    "--prm800k_grading_dir /path/to/prm800k/prm800k or "
                    "--prm800k_grading_dir /path/to/prm800k/prm800k/grading. "
                    "For a non-official fallback, use --grader math_verify or --grader exact."
                ) from exc
            self._prm800k_grade_answer = grade_answer
        elif name != "exact":
            raise ValueError(f"Unsupported grader: {name}")

    @staticmethod
    def _add_prm800k_grading_path(prm800k_grading_dir: str | None) -> None:
        if not prm800k_grading_dir:
            return

        root = Path(prm800k_grading_dir).resolve()
        candidates = []
        if (root / "grading" / "grader.py").exists():
            candidates.append(root)
        if (root / "grader.py").exists() and root.name == "grading":
            candidates.append(root.parent)
        if (root / "prm800k" / "grading" / "grader.py").exists():
            candidates.append(root / "prm800k")

        if not candidates:
            candidates.append(root)

        for candidate in reversed(candidates):
            sys.path.insert(0, str(candidate))

    def grade(self, response: str, answer: str) -> bool:
        if self.name == "exact":
            return verify_response(response, answer, "exact_match")

        if self.name == "prm800k":
            prediction = extract_boxed_answer(response)
            return bool(self._prm800k_grade_answer(prediction, str(answer)))

        parser = self._math_verify["parse"]
        verifier = self._math_verify["verify"]
        configs = self._math_verify["configs"]
        try:
            gold = parser(wrap_latex_if_needed(answer), extraction_config=configs)
            prediction = parser(response, extraction_config=configs)
            return bool(verifier(gold, prediction))
        except Exception:
            return verify_response(response, answer, "exact_match")


def load_student_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": resolve_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"{adapter_path} does not look like a PEFT adapter directory; missing adapter_config.json"
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        if args.merge_adapter:
            model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


class VLLMModel:
    def __init__(self, llm, lora_request=None):
        self.llm = llm
        self.lora_request = lora_request


def _normalize_vllm_dtype(dtype: str) -> str:
    dtype = (dtype or "bfloat16").lower()
    if dtype == "bf16":
        return "bfloat16"
    if dtype == "fp16":
        return "float16"
    if dtype == "fp32":
        return "float32"
    return dtype


def _truncate_prompts_for_vllm(tokenizer, prompts: list[str], max_prompt_length: int) -> list[str]:
    truncated = []
    for prompt in prompts:
        token_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        truncated.append(tokenizer.decode(token_ids, skip_special_tokens=False))
    return truncated


def load_vllm_model(args):
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "model": args.model_name_or_path,
        "tokenizer": args.model_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": _normalize_vllm_dtype(args.dtype),
        "max_model_len": args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "seed": args.seed,
        "enable_lora": bool(args.adapter_path),
    }
    if args.adapter_path:
        llm_kwargs["max_lora_rank"] = args.vllm_max_lora_rank
    if args.vllm_max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
    if args.vllm_enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    lora_request = None
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"{adapter_path} does not look like a PEFT adapter directory; missing adapter_config.json"
            )
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("student_adapter", 1, str(adapter_path))
    return tokenizer, VLLMModel(llm, lora_request)


def load_eval_model(args):
    if getattr(args, "backend", "hf") == "vllm":
        return load_vllm_model(args)
    return load_student_model(args)


def generate_batch(tokenizer, model, prompts: list[str], args) -> list[str]:
    if isinstance(model, VLLMModel):
        from vllm import SamplingParams

        do_sample = args.do_sample or args.num_samples > 1
        top_k = args.top_k if args.top_k and args.top_k > 0 else -1
        sampling_params = SamplingParams(
            n=args.num_samples,
            temperature=args.temperature if do_sample else 0.0,
            top_p=args.top_p if do_sample else 1.0,
            top_k=top_k,
            max_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        vllm_prompts = _truncate_prompts_for_vllm(tokenizer, prompts, args.max_prompt_length)
        request_outputs = model.llm.generate(
            vllm_prompts,
            sampling_params,
            lora_request=model.lora_request,
        )
        return [
            completion.text
            for request_output in request_outputs
            for completion in request_output.outputs
        ]

    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    do_sample = args.do_sample or args.num_samples > 1
    num_return_sequences = 1 if getattr(args, "sequential_samples", False) else args.num_samples
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    if args.repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty

    prompt_width = encoded["input_ids"].shape[1]

    if getattr(args, "sequential_samples", False) and args.num_samples > 1:
        per_prompt_outputs = [[] for _ in prompts]
        with torch.inference_mode():
            for sample_round in range(args.num_samples):
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"generate sample round {sample_round + 1}/{args.num_samples}",
                    flush=True,
                )
                output_ids = model.generate(**encoded, **generation_kwargs)
                completion_ids = output_ids[:, prompt_width:]
                decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                for prompt_index, completion in enumerate(decoded):
                    per_prompt_outputs[prompt_index].append(completion)
        return [completion for prompt_outputs in per_prompt_outputs for completion in prompt_outputs]

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    completion_ids = output_ids[:, prompt_width:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)


def summarize(rows: list[dict[str, Any]], num_samples: int) -> dict[str, Any]:
    total = len({row["unique_id"] for row in rows})
    avg_correct = []
    first_correct = []
    pass_correct = []
    by_subject = defaultdict(lambda: {"count": 0, "avg_correct": 0.0, "first_correct": 0, "pass_correct": 0})
    by_level = defaultdict(lambda: {"count": 0, "avg_correct": 0.0, "first_correct": 0, "pass_correct": 0})

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["unique_id"]].append(row)

    for _, samples in grouped.items():
        samples = sorted(samples, key=lambda item: item["sample_index"])
        sample_correct = [bool(sample["correct"]) for sample in samples]
        avg = sum(sample_correct) / max(1, len(sample_correct))
        first = bool(samples[0]["correct"])
        passed = any(sample_correct)
        avg_correct.append(avg)
        first_correct.append(first)
        pass_correct.append(passed)

        subject = samples[0].get("subject", "")
        level = str(samples[0].get("level", ""))
        by_subject[subject]["count"] += 1
        by_subject[subject]["avg_correct"] += avg
        by_subject[subject]["first_correct"] += int(first)
        by_subject[subject]["pass_correct"] += int(passed)
        by_level[level]["count"] += 1
        by_level[level]["avg_correct"] += avg
        by_level[level]["first_correct"] += int(first)
        by_level[level]["pass_correct"] += int(passed)

    def finalize(bucket):
        output = {}
        for key, value in sorted(bucket.items()):
            count = max(1, value["count"])
            output[key] = {
                "count": value["count"],
                f"avg_at_{num_samples}": value["avg_correct"] / count,
                f"pass_at_{num_samples}": value["pass_correct"] / count,
                f"first_at_{num_samples}": value["first_correct"] / count,
            }
        return output

    return {
        "count": total,
        "num_samples": num_samples,
        f"avg_at_{num_samples}": sum(avg_correct) / max(1, total),
        f"pass_at_{num_samples}": sum(pass_correct) / max(1, total),
        f"first_at_{num_samples}": sum(first_correct) / max(1, total),
        "by_subject": finalize(by_subject),
        "by_level": finalize(by_level),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base or LoRA student model on MATH-500.")
    parser.add_argument("--model_name_or_path", required=True, help="Base model path, e.g. /data0/shared/Qwen3-1.7B.")
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT LoRA adapter checkpoint/output directory.")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--test_file", default="data/lite_cie/test.jsonl")
    parser.add_argument("--output_dir", default="outputs/math500_eval")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--grader", choices=["prm800k", "math_verify", "exact"], default="prm800k")
    parser.add_argument("--prm800k_grading_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--student_thinking", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument(
        "--sequential_samples",
        action="store_true",
        help="Generate num_samples one round at a time to reduce peak KV-cache memory.",
    )
    parser.add_argument("--log_grading", action="store_true", help="Print per-sample grading progress.")
    parser.add_argument("--grade_timeout_sec", type=float, default=0.0, help="Mark a sample wrong if grading exceeds this many seconds.")
    parser.add_argument("--save_every", type=int, default=0, help="Write partial generations/metrics every N finished batches.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    test_rows = read_jsonl(Path(args.test_file))
    if args.limit > 0:
        test_rows = test_rows[: args.limit]

    missing = [field for field in ("problem", "answer") if any(field not in row for row in test_rows)]
    if missing:
        raise ValueError(f"{args.test_file} is missing required MATH-500 fields: {sorted(set(missing))}")
    for row_index, row in enumerate(test_rows):
        if "unique_id" not in row:
            row["unique_id"] = str(row.get("id", row_index))

    run_name = args.run_name
    if not run_name:
        model_label = Path(args.adapter_path or args.model_name_or_path).name
        run_name = f"{model_label}_math500_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_eval_model(args)
    grader = AnswerGrader(args.grader, args.prm800k_grading_dir)

    outputs = []
    for start in range(0, len(test_rows), args.batch_size):
        batch = test_rows[start : start + args.batch_size]
        batch_end = min(start + len(batch), len(test_rows))
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"start batch {start + 1}-{batch_end}/{len(test_rows)}",
            flush=True,
        )
        prompts = [apply_chat_prompt(tokenizer, row["problem"], args.student_thinking) for row in batch]
        completions = generate_batch(tokenizer, model, prompts, args)
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"generated batch {start + 1}-{batch_end}/{len(test_rows)}",
            flush=True,
        )

        for row_index, row in enumerate(batch):
            for sample_index in range(args.num_samples):
                completion = completions[row_index * args.num_samples + sample_index]
                if args.log_grading:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"grade row {start + row_index + 1}/{len(test_rows)} sample {sample_index + 1}/{args.num_samples}",
                        flush=True,
                    )
                try:
                    correct = call_with_timeout(args.grade_timeout_sec, grader.grade, completion, row["answer"])
                except TimeoutError as exc:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"grade timeout row {start + row_index + 1}/{len(test_rows)} "
                        f"sample {sample_index + 1}/{args.num_samples}: {exc}",
                        flush=True,
                    )
                    correct = False
                outputs.append(
                    {
                        "unique_id": row["unique_id"],
                        "subject": row.get("subject", ""),
                        "level": row.get("level", ""),
                        "problem": row["problem"],
                        "answer": row["answer"],
                        "sample_index": sample_index,
                        "prompt": prompts[row_index],
                        "completion": completion,
                        "extracted_answer": extract_boxed_answer(completion),
                        "correct": correct,
                    }
                )
        done = min(start + len(batch), len(test_rows))
        running = summarize(outputs, args.num_samples)
        print(
            f"[{done}/{len(test_rows)}] avg@{args.num_samples}={running[f'avg_at_{args.num_samples}']:.4f} "
            f"pass@{args.num_samples}={running[f'pass_at_{args.num_samples}']:.4f}",
            flush=True,
        )
        batch_number = start // args.batch_size + 1
        if args.save_every > 0 and batch_number % args.save_every == 0:
            write_jsonl(run_dir / "generations.partial.jsonl", outputs)
            (run_dir / "metrics.partial.json").write_text(
                json.dumps(running, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] wrote partial outputs to {run_dir}",
                flush=True,
            )

    metrics = summarize(outputs, args.num_samples)
    metrics.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "adapter_path": args.adapter_path,
            "test_file": args.test_file,
            "grader": args.grader,
            "max_new_tokens": args.max_new_tokens,
            "student_thinking": args.student_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        }
    )

    write_jsonl(run_dir / "generations.jsonl", outputs)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nMATH-500 evaluation complete", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print(f"Avg@{args.num_samples}: {metrics[f'avg_at_{args.num_samples}']:.4f}", flush=True)
    print(f"Pass@{args.num_samples}: {metrics[f'pass_at_{args.num_samples}']:.4f}", flush=True)
    print(f"First@{args.num_samples}: {metrics[f'first_at_{args.num_samples}']:.4f}", flush=True)


if __name__ == "__main__":
    main()
