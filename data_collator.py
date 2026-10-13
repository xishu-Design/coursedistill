import torch


class LiteCIEDataCollator:
    """
    Data collator for sample-wise Lite-CIE-OPD.

    It builds the student c0 prompt plus raw teacher context texts c0..cK. The trainer
    chooses or searches adjacent transitions c^{j-1}->c^j during on-policy scoring.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        ladder_steps=4,
        student_thinking=False,
        teacher_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ladder_steps = max(1, int(ladder_steps))
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.transition_prompt = (
            "\n\nUse the available context only as evidence. Now solve the problem in your own words. "
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        print(f"[LiteCIEDataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[LiteCIEDataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[LiteCIEDataCollator] ladder_steps: {self.ladder_steps}")

    def _apply_chat_template(self, messages, enable_thinking):
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

    def _student_prompt(self, student_input):
        user_message = (
            f"Problem: {student_input}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        return self._apply_chat_template(
            [{"role": "user", "content": user_message}],
            enable_thinking=self.student_thinking,
        )

    def _teacher_context_prompt(self, student_input, evidence):
        if evidence:
            user_message = (
                f"Problem: {student_input}\n\n"
                "Privileged evidence available to the teacher:\n"
                f"{evidence}"
                f"{self.transition_prompt}"
            )
        else:
            user_message = (
                f"Problem: {student_input}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        return self._apply_chat_template(
            [{"role": "user", "content": user_message}],
            enable_thinking=self.teacher_thinking,
        )

    def _normalize_ladder(self, ladder):
        if ladder is None:
            ladder = []
        if isinstance(ladder, str):
            ladder = [ladder]
        else:
            normalized = []
            for item in list(ladder):
                if isinstance(item, dict):
                    level = item.get("level", "")
                    name = item.get("name", "")
                    content = item.get("content", "")
                    prefix = f"z{level} {name}:".strip()
                    normalized.append(f"{prefix} {content}".strip())
                else:
                    normalized.append(str(item))
            ladder = normalized
        if not ladder:
            ladder = [""]
        ladder = ladder[: self.ladder_steps]
        while len(ladder) < self.ladder_steps:
            ladder.append("")
        return ladder

    def __call__(self, features):
        student_prompts = []
        teacher_context_texts = []
        teacher_ladder_pieces = []
        sample_ids = []
        student_inputs = []
        ground_truths = []
        verifier_types = []

        for feature in features:
            student_input = feature.get("student_input") or feature.get("problem") or feature.get("question") or ""
            ladder = self._normalize_ladder(feature.get("privileged_ladder"))

            student_prompts.append(self._student_prompt(student_input))
            student_inputs.append(str(student_input))
            teacher_ladder_pieces.append(ladder)
            sample_ids.append(str(feature.get("id", "")))
            ground_truths.append(str(feature.get("ground_truth", "")))
            verifier_types.append(str(feature.get("verifier_type", "exact_match")))

            contexts = [self._teacher_context_prompt(student_input, "")]
            cumulative = []
            for piece in ladder:
                if piece:
                    cumulative.append(piece)
                contexts.append(self._teacher_context_prompt(student_input, "\n".join(cumulative)))
            teacher_context_texts.append(contexts)

        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)

        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        return {
            "sample_ids": sample_ids,
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
            "teacher_context_texts": teacher_context_texts,
            "teacher_ladder_pieces": teacher_ladder_pieces,
            "student_inputs": student_inputs,
            "ground_truth": ground_truths,
            "verifier_type": verifier_types,
            "lite_cie_ladder_steps": self.ladder_steps,
        }
