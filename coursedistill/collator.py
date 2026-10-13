from __future__ import annotations

import torch


class CourseDistillDataCollator:
    """Collator for online course distillation.

    It only builds prompts and carries raw privileged information. It does not
    precompute rollouts, outcomes, spans, or teacher scores.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.tokenizer.padding_side = "right"

    def _apply_chat_template(self, messages, enable_thinking: bool) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            content = messages[-1]["content"] if messages else ""
            return f"{content}\n\nAssistant:"

    def _student_prompt(self, problem: str) -> str:
        user_message = (
            f"Problem: {problem}\n\n"
            "Give the necessary reasoning concisely, and put your final answer within \\boxed{}."
        )
        return self._apply_chat_template(
            [{"role": "user", "content": user_message}],
            enable_thinking=self.student_thinking,
        )

    def _teacher_prompt(self, problem: str, privileged_info: str = "") -> str:
        if privileged_info.strip():
            user_message = (
                f"Problem: {problem}\n\n"
                "Privileged information available to the teacher:\n"
                f"{privileged_info.strip()}\n\n"
                "Use the privileged information only to evaluate the student's reasoning. "
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        else:
            user_message = (
                f"Problem: {problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        return self._apply_chat_template(
            [{"role": "user", "content": user_message}],
            enable_thinking=self.teacher_thinking,
        )

    def __call__(self, features: list[dict]):
        student_prompts = []
        base_teacher_prompts = []
        privileged_teacher_prompts = []
        sample_ids = []
        student_inputs = []
        ground_truths = []
        verifier_types = []
        reference_solutions = []

        for feature in features:
            problem = feature.get("student_input") or feature.get("problem") or feature.get("question") or ""
            reference_solution = feature.get("reference_solution") or feature.get("solution") or ""
            privileged_info = reference_solution

            student_prompts.append(self._student_prompt(problem))
            base_teacher_prompts.append(self._teacher_prompt(problem, ""))
            privileged_teacher_prompts.append(self._teacher_prompt(problem, privileged_info))
            sample_ids.append(str(feature.get("id", "")))
            student_inputs.append(str(problem))
            ground_truths.append(str(feature.get("ground_truth", "")))
            verifier_types.append(str(feature.get("verifier_type", "exact_match")))
            reference_solutions.append(str(reference_solution))

        encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths)
        encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )

        return {
            "sample_ids": sample_ids,
            "student_inputs": student_inputs,
            "student_prompts": encoded["input_ids"],
            "student_prompt_attention_mask": encoded["attention_mask"],
            "student_prompt_length": max_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(prompt_lengths, dtype=torch.long),
            "teacher_base_context_texts": base_teacher_prompts,
            "teacher_privileged_context_texts": privileged_teacher_prompts,
            "reference_solution": reference_solutions,
            "ground_truth": ground_truths,
            "verifier_type": verifier_types,
        }
