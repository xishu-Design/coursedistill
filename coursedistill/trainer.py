from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from accelerate.utils import DistributedType, gather_object, is_peft_model
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from lite_cie_trainer import LiteCIEOPDTrainer
from verifier import verify_response

from .span_utils import SpanConfig, build_candidate_span_masks


class OnlineCLSSampler(Sampler[int]):
    """Online student-side curriculum sampler backed by trainer EMA CLS memory."""

    def __init__(self, trainer: "CourseDistillTrainer", data_source):
        self.trainer = trainer
        self.data_source = data_source
        self.num_samples = len(data_source)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_id(self, index: int) -> str:
        try:
            row = self.data_source[index]
            key = row.get("id") if isinstance(row, dict) else None
        except Exception:
            key = None
        return str(key or index)

    def __iter__(self):
        trainer = self.trainer
        seed = int(getattr(trainer.args, "seed", 42) or 42)
        step = int(getattr(trainer.state, "global_step", 0) or 0)
        rank = int(getattr(trainer.accelerator, "process_index", 0) or 0)
        generator = torch.Generator()
        generator.manual_seed(seed + 1000003 * rank + 9973 * max(0, step))

        progress = trainer._training_progress()
        memory = getattr(trainer, "_curriculum_memory", {})
        if (
            not bool(getattr(trainer, "curriculum_sampler_enabled", False))
            or progress < getattr(trainer, "curriculum_sampler_warmup_ratio", 0.0)
            or not memory
        ):
            yield from torch.randperm(self.num_samples, generator=generator).tolist()
            return

        floor = float(getattr(trainer, "curriculum_sampler_floor", 0.10))
        power = float(getattr(trainer, "curriculum_sampler_power", 1.0))
        weights = torch.full((self.num_samples,), max(1e-6, floor), dtype=torch.float32)
        for idx in range(self.num_samples):
            entry = memory.get(self._sample_id(idx))
            if not entry:
                continue
            cls_value = max(0.0, float(entry.get("cls", 0.0) or 0.0))
            weights[idx] = max(1e-6, floor + (1.0 - floor) * (cls_value ** power))
        if not bool(torch.isfinite(weights).all().item()) or float(weights.sum().item()) <= 0:
            yield from torch.randperm(self.num_samples, generator=generator).tolist()
            return

        replacement = bool(getattr(trainer, "curriculum_sampler_replacement", True))
        unique_fraction = float(getattr(trainer, "curriculum_sampler_unique_fraction", 0.0) or 0.0)
        unique_fraction = min(1.0, max(0.0, unique_fraction))
        unique_count = int(round(self.num_samples * unique_fraction))
        if unique_count > 0:
            unique_indices = torch.randperm(self.num_samples, generator=generator)[:unique_count]
            weighted_count = max(0, self.num_samples - unique_count)
            if weighted_count > 0:
                weighted_indices = torch.multinomial(weights, weighted_count, replacement=True, generator=generator)
                indices = torch.cat([unique_indices, weighted_indices], dim=0)
                indices = indices[torch.randperm(indices.numel(), generator=generator)]
            else:
                indices = unique_indices
        else:
            indices = torch.multinomial(weights, self.num_samples, replacement=replacement, generator=generator)
        trainer._metric("course/sampler_weight_mean", weights.mean())
        trainer._metric("course/sampler_weight_max", weights.max())
        trainer._metric("course/sampler_memory_coverage", (weights > floor + 1e-6).float().mean())
        trainer._metric("course/sampler_unique_fraction", torch.tensor(unique_fraction, dtype=torch.float32))
        yield from indices.tolist()


class CourseDistillTrainer(LiteCIEOPDTrainer):
    """Online Reward-Ordered Privileged Span Curriculum Distillation.

    This trainer reuses the rollout/vLLM utilities from LiteCIEOPDTrainer but
    replaces the training step and loss. No offline rollout, span, or teacher
    score preprocessing is used.
    """

    _tag_names = ["trl", "course-distill-opd"]
    _name = "Course-Distill-OPD"

    def __init__(self, *args, **kwargs):
        teacher_model = kwargs.pop("teacher_model", None)
        teacher_model_name_or_path = kwargs.pop("teacher_model_name_or_path", None)
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model_name_or_path = teacher_model_name_or_path
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.to(self.accelerator.device)
        self.span_distill_coef = max(0.0, float(getattr(self.args, "span_distill_coef", 0.0) or 0.0))
        self.span_coef_floor = max(0.0, float(getattr(self.args, "span_coef_floor", 0.0) or 0.0))
        self.span_decay_start_ratio = min(1.0, max(0.0, float(getattr(self.args, "span_decay_start_ratio", 0.25) or 0.0)))
        self.span_decay_end_ratio = min(1.0, max(self.span_decay_start_ratio, float(getattr(self.args, "span_decay_end_ratio", 0.85) or 0.0)))
        self.grpo_coef = max(0.0, float(getattr(self.args, "grpo_coef", 1.0) or 0.0))
        self.grpo_loss_clip = max(0.0, float(getattr(self.args, "grpo_loss_clip", 0.5) or 0.0))
        self.grpo_logp_clip = max(0.0, float(getattr(self.args, "grpo_logp_clip", 10.0) or 0.0))
        self.error_suppress_coef = max(0.0, float(getattr(self.args, "error_suppress_coef", 0.0) or 0.0))
        self.error_suppress_max_ratio = max(0.0, float(getattr(self.args, "error_suppress_max_ratio", 0.25) or 0.0))
        self.error_suppress_max_span_ratio = max(
            0.0,
            float(getattr(self.args, "error_suppress_max_span_ratio", 0.0) or 0.0),
        )
        self.error_suppress_start_ratio = min(
            1.0,
            max(0.0, float(getattr(self.args, "error_suppress_start_ratio", 0.20) or 0.0)),
        )
        self.curriculum_alpha_start = max(0.0, float(getattr(self.args, "curriculum_alpha_start", 2.0) or 0.0))
        self.curriculum_alpha_end = max(0.0, float(getattr(self.args, "curriculum_alpha_end", 0.5) or 0.0))
        self.curriculum_routing_rule = str(
            getattr(self.args, "curriculum_routing_rule", "cls") or "cls"
        ).strip().lower()
        if self.curriculum_routing_rule not in {"cls", "uncertainty"}:
            raise ValueError(
                "curriculum_routing_rule must be either 'cls' or 'uncertainty', "
                f"got {self.curriculum_routing_rule!r}."
            )
        self.curriculum_stage1_ratio = min(
            1.0,
            max(0.0, float(getattr(self.args, "curriculum_stage1_ratio", 0.25) or 0.0)),
        )
        self.curriculum_stage2_ratio = min(
            1.0,
            max(
                self.curriculum_stage1_ratio,
                float(getattr(self.args, "curriculum_stage2_ratio", 0.70) or 0.0),
            ),
        )
        self.curriculum_stage3_ratio = min(
            1.0,
            max(
                self.curriculum_stage2_ratio,
                float(getattr(self.args, "curriculum_stage3_ratio", 0.88) or 0.0),
            ),
        )
        self.curriculum_memory_momentum = min(
            0.99,
            max(0.0, float(getattr(self.args, "curriculum_memory_momentum", 0.80) or 0.0)),
        )
        self.curriculum_memory_blend = min(
            1.0,
            max(0.0, float(getattr(self.args, "curriculum_memory_blend", 0.50) or 0.0)),
        )
        self.curriculum_weight_floor = min(
            1.0,
            max(0.0, float(getattr(self.args, "curriculum_weight_floor", 0.0) or 0.0)),
        )
        self.curriculum_review_mix_max = min(
            0.35,
            max(0.0, float(getattr(self.args, "curriculum_review_mix_max", 0.35) or 0.0)),
        )
        self.curriculum_sampler_enabled = bool(getattr(self.args, "curriculum_sampler_enabled", False))
        self.curriculum_sampler_warmup_ratio = min(
            1.0,
            max(0.0, float(getattr(self.args, "curriculum_sampler_warmup_ratio", 0.15) or 0.0)),
        )
        self.curriculum_sampler_floor = min(
            1.0,
            max(0.0, float(getattr(self.args, "curriculum_sampler_floor", 0.10) or 0.0)),
        )
        self.curriculum_sampler_power = max(
            0.0,
            float(getattr(self.args, "curriculum_sampler_power", 1.0) or 0.0),
        )
        self.curriculum_sampler_replacement = bool(getattr(self.args, "curriculum_sampler_replacement", True))
        self.curriculum_sampler_unique_fraction = min(
            1.0,
            max(0.0, float(getattr(self.args, "curriculum_sampler_unique_fraction", 0.0) or 0.0)),
        )
        self.adaptive_curriculum_enabled = bool(getattr(self.args, "adaptive_curriculum_enabled", False))
        self.adaptive_curriculum_min_coverage = min(
            1.0,
            max(0.0, float(getattr(self.args, "adaptive_curriculum_min_coverage", 0.20) or 0.0)),
        )
        self.adaptive_support_min_correct_rate = min(
            1.0,
            max(0.0, float(getattr(self.args, "adaptive_support_min_correct_rate", 0.42) or 0.0)),
        )
        self.adaptive_reason_min_correct_rate = min(
            1.0,
            max(0.0, float(getattr(self.args, "adaptive_reason_min_correct_rate", 0.50) or 0.0)),
        )
        self.all_wrong_penalty = max(0.0, float(getattr(self.args, "all_wrong_penalty", 0.0) or 0.0))
        self.all_wrong_data_weight = max(0.0, float(getattr(self.args, "all_wrong_data_weight", 0.0) or 0.0))
        self.all_correct_bonus = max(0.0, float(getattr(self.args, "all_correct_bonus", 0.0) or 0.0))
        self.all_correct_data_weight = max(0.0, float(getattr(self.args, "all_correct_data_weight", 0.0) or 0.0))
        self.advantage_clip = max(0.0, float(getattr(self.args, "advantage_clip", 0.0) or 0.0))
        self.wrong_length_boost = max(0.0, float(getattr(self.args, "wrong_length_boost", 0.0) or 0.0))
        self.truncated_negative_weight = min(
            1.0,
            max(0.0, float(getattr(self.args, "truncated_negative_weight", 0.0) or 0.0)),
        )
        self.correct_near_full_positive_weight = min(
            1.0,
            max(0.0, float(getattr(self.args, "correct_near_full_positive_weight", 1.0) or 0.0)),
        )
        self.post_answer_grpo_weight = min(
            1.0,
            max(0.0, float(getattr(self.args, "post_answer_grpo_weight", 1.0) or 0.0)),
        )
        self.answer_stop_loss_coef = max(
            0.0,
            float(getattr(self.args, "answer_stop_loss_coef", 0.0) or 0.0),
        )
        self.answer_stop_grace_tokens = max(
            0,
            int(getattr(self.args, "answer_stop_grace_tokens", 2) or 0),
        )
        self.support_start_ratio = min(1.0, max(0.0, float(getattr(self.args, "support_start_ratio", 0.35) or 0.0)))
        self.reason_start_ratio = min(1.0, max(0.0, float(getattr(self.args, "reason_start_ratio", 0.45) or 0.0)))
        span_evidence_floor_arg = getattr(self.args, "span_evidence_floor", -1.0)
        self.span_evidence_floor = min(
            1.0,
            max(-1.0, float(-1.0 if span_evidence_floor_arg is None else span_evidence_floor_arg)),
        )
        self.span_rank_floor = min(1.0, max(0.0, float(getattr(self.args, "span_rank_floor", 0.10) or 0.0)))
        self.span_gain_weight_floor = min(
            1.0,
            max(0.0, float(getattr(self.args, "span_gain_weight_floor", 1.0) or 0.0)),
        )
        self.span_gain_weight_scale = max(
            1e-6,
            float(getattr(self.args, "span_gain_weight_scale", 1.0) or 1.0),
        )
        self.span_traj_normalize = bool(getattr(self.args, "span_traj_normalize", False))
        self.span_max_traj_weight = max(
            1e-6,
            float(getattr(self.args, "span_max_traj_weight", 1.0) or 1.0),
        )
        final_span_weight_arg = getattr(self.args, "final_span_weight", -1.0)
        contrast_span_weight_arg = getattr(self.args, "contrast_span_weight", -1.0)
        context_span_weight_arg = getattr(self.args, "context_span_weight", -1.0)
        self.final_span_weight = max(
            -1.0,
            float(-1.0 if final_span_weight_arg is None else final_span_weight_arg),
        )
        self.contrast_span_weight = max(
            -1.0,
            float(-1.0 if contrast_span_weight_arg is None else contrast_span_weight_arg),
        )
        self.context_span_weight = max(
            -1.0,
            float(-1.0 if context_span_weight_arg is None else context_span_weight_arg),
        )
        self.answer_span_weight = max(0.0, float(getattr(self.args, "answer_span_weight", 1.0) or 0.0))
        self.diff_span_weight = max(0.0, float(getattr(self.args, "diff_span_weight", 1.0) or 0.0))
        self.calculation_span_weight = max(0.0, float(getattr(self.args, "calculation_span_weight", 0.0) or 0.0))
        self.support_span_weight = max(0.0, float(getattr(self.args, "support_span_weight", 0.5) or 0.0))
        self.reason_span_weight = max(0.0, float(getattr(self.args, "reason_span_weight", 0.0) or 0.0))
        if self.final_span_weight >= 0:
            self.answer_span_weight = self.final_span_weight
            self.calculation_span_weight = self.final_span_weight
        if self.contrast_span_weight >= 0:
            self.diff_span_weight = self.contrast_span_weight
        if self.context_span_weight >= 0:
            self.support_span_weight = self.context_span_weight
            self.reason_span_weight = self.context_span_weight
        self.use_privileged_gain = bool(getattr(self.args, "use_privileged_gain", True))
        self.teacher_step_window = max(1, int(getattr(self.args, "teacher_step_window", 33) or 33))
        if self.teacher_step_window % 2 == 0:
            self.teacher_step_window += 1
        self.teacher_student_gap_coef = max(
            0.0,
            float(getattr(self.args, "teacher_student_gap_coef", 0.0) or 0.0),
        )
        self.teacher_privileged_confidence_coef = max(
            0.0,
            float(getattr(self.args, "teacher_privileged_confidence_coef", 0.0) or 0.0),
        )
        self.mixed_balance_bonus = max(
            0.0,
            float(getattr(self.args, "mixed_balance_bonus", 0.0) or 0.0),
        )
        self.near_full_negative_ratio = min(
            1.0,
            max(0.0, float(getattr(self.args, "near_full_negative_ratio", 1.0) or 1.0)),
        )
        self.teacher_step_rank_floor = min(
            1.0,
            max(0.0, float(getattr(self.args, "teacher_step_rank_floor", 0.05) or 0.0)),
        )
        if self.span_evidence_floor >= 0:
            self.span_rank_floor = self.span_evidence_floor
            self.teacher_step_rank_floor = self.span_evidence_floor
        self.teacher_token_weight_clip = max(
            0.0,
            float(getattr(self.args, "teacher_token_weight_clip", 1.0) or 0.0),
        )
        self.trace_context_topk = max(0, int(getattr(self.args, "trace_context_topk", 0) or 0))
        self.trace_context_max_tokens = max(1, int(getattr(self.args, "trace_context_max_tokens", 64) or 64))
        self.teacher_score_micro_batch = max(
            1,
            int(getattr(self.args, "teacher_score_micro_batch", 2) or 2),
        )
        self.teacher_group_support_floor = min(
            1.0,
            max(0.0, float(getattr(self.args, "teacher_group_support_floor", 0.10) or 0.0)),
        )
        self.teacher_support_max = max(
            1.0,
            float(getattr(self.args, "teacher_support_max", 1.45) or 1.45),
        )
        self.teacher_grpo_magnitude_coef = min(
            1.0,
            max(0.0, float(getattr(self.args, "teacher_grpo_magnitude_coef", 1.0) or 0.0)),
        )
        self.teacher_correct_length_dampen = min(
            1.0,
            max(0.0, float(getattr(self.args, "teacher_correct_length_dampen", 0.0) or 0.0)),
        )
        self.positive_teacher_span_only = bool(getattr(self.args, "positive_teacher_span_only", True))
        self.positive_reason_span_only = bool(getattr(self.args, "positive_reason_span_only", True))
        self.no_answer_full_length_weight = max(0.0, float(getattr(self.args, "no_answer_full_length_weight", 0.0) or 0.0))
        self.verifier_confidence_enabled = bool(getattr(self.args, "verifier_confidence_enabled", False))
        self.verifier_answer_marker_negative_confidence = min(
            1.0,
            max(0.0, float(getattr(self.args, "verifier_answer_marker_negative_confidence", 0.85) or 0.0)),
        )
        self.verifier_truncated_answer_negative_confidence = min(
            1.0,
            max(0.0, float(getattr(self.args, "verifier_truncated_answer_negative_confidence", 0.65) or 0.0)),
        )
        self.verifier_correct_no_answer_confidence = min(
            1.0,
            max(0.0, float(getattr(self.args, "verifier_correct_no_answer_confidence", 0.75) or 0.0)),
        )
        self.span_debug_rate = max(0.0, float(getattr(self.args, "span_debug_rate", 0.01) or 0.0))
        self.span_cfg = SpanConfig(
            answer_span_tokens=max(1, int(getattr(self.args, "answer_span_tokens", 48) or 48)),
            diff_span_tokens=max(1, int(getattr(self.args, "diff_span_tokens", 64) or 64)),
            diff_span_top_k=max(1, int(getattr(self.args, "diff_span_top_k", 3) or 3)),
            calculation_span_tokens=max(1, int(getattr(self.args, "calculation_span_tokens", 64) or 64)),
            support_span_tokens=max(1, int(getattr(self.args, "support_span_tokens", 96) or 96)),
            reason_span_tokens=max(1, int(getattr(self.args, "reason_span_tokens", 128) or 128)),
        )
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5
        self._textual_logs = {
            "prompt": deque(maxlen=self.accelerator.num_processes * self.args.per_device_train_batch_size),
            "completion": deque(maxlen=self.accelerator.num_processes * self.args.per_device_train_batch_size),
        }
        self._on_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._curriculum_memory: dict[str, dict[str, float]] = {}

        if self.accelerator.is_main_process:
            print(f"\n{'=' * 80}")
            print("COURSE-DISTILL MODE ENABLED")
            print("Online rollout/verifier/curriculum/span/teacher scoring: enabled")
            if self.teacher_model is not None:
                print(f"Separate privileged teacher: {self.teacher_model_name_or_path}")
            else:
                print("Separate privileged teacher: disabled")
            print(f"GRPO coefficient: {self.grpo_coef}")
            print(f"GRPO guards: loss_clip={self.grpo_loss_clip}, logp_clip={self.grpo_logp_clip}")
            print(f"Span distill coefficient: {self.span_distill_coef} -> floor {self.span_coef_floor}")
            print(
                "Error suppress: "
                f"coef={self.error_suppress_coef}, cap ratio={self.error_suppress_max_ratio}, "
                f"span cap={self.error_suppress_max_span_ratio}, start={self.error_suppress_start_ratio}"
            )
            print(f"Curriculum alpha: {self.curriculum_alpha_start} -> {self.curriculum_alpha_end}")
            print(
                "Curriculum stages: "
                f"0<{self.curriculum_stage1_ratio}, "
                f"1<{self.curriculum_stage2_ratio}, "
                f"2<{self.curriculum_stage3_ratio}, 3"
            )
            print(
                "Curriculum memory: "
                f"momentum={self.curriculum_memory_momentum}, blend={self.curriculum_memory_blend}, "
                f"active_floor={self.curriculum_weight_floor}"
            )
            print(
                "Curriculum sampler: "
                f"enabled={self.curriculum_sampler_enabled}, warmup={self.curriculum_sampler_warmup_ratio}, "
                f"floor={self.curriculum_sampler_floor}, power={self.curriculum_sampler_power}, "
                f"replacement={self.curriculum_sampler_replacement}, unique_fraction={self.curriculum_sampler_unique_fraction}, "
                f"review_mix_max={self.curriculum_review_mix_max}"
            )
            print(
                "Adaptive curriculum gates: "
                f"enabled={self.adaptive_curriculum_enabled}, coverage={self.adaptive_curriculum_min_coverage}, "
                f"support_p={self.adaptive_support_min_correct_rate}, reason_p={self.adaptive_reason_min_correct_rate}"
            )
            print(
                "Homogeneous reward fallback: "
                f"all_wrong_penalty={self.all_wrong_penalty}, all_wrong_weight={self.all_wrong_data_weight}, "
                f"all_correct_bonus={self.all_correct_bonus}, all_correct_weight={self.all_correct_data_weight}, "
                f"adv_clip={self.advantage_clip}"
            )
            print(f"Wrong length boost: {self.wrong_length_boost}")
            print(f"Truncated negative weight: {self.truncated_negative_weight}")
            print(
                "Correct stop/length guard: "
                f"near_full_pos_weight={self.correct_near_full_positive_weight}, "
                f"post_answer_grpo_weight={self.post_answer_grpo_weight}, "
                f"stop_loss_coef={self.answer_stop_loss_coef}, "
                f"stop_grace={self.answer_stop_grace_tokens}"
            )
            print(f"Use privileged gain: {self.use_privileged_gain}")
            print(
                "Teacher step calibration: "
                f"window={self.teacher_step_window}, floor={self.teacher_step_rank_floor}, "
                f"clip={self.teacher_token_weight_clip}, student_gap_coef={self.teacher_student_gap_coef}, "
                f"priv_conf_coef={self.teacher_privileged_confidence_coef}, "
                f"score_micro_batch={self.teacher_score_micro_batch}"
            )
            print(
                "TRACE-style context routing: "
                f"topk={self.trace_context_topk}, max_tokens={self.trace_context_max_tokens}"
            )
            print(f"Mixed balance bonus: {self.mixed_balance_bonus}")
            print(f"Near-full negative ratio: {self.near_full_negative_ratio}")
            print(f"Teacher group support floor: {self.teacher_group_support_floor}")
            print(f"Teacher support max: {self.teacher_support_max}")
            print(f"Teacher GRPO magnitude coefficient: {self.teacher_grpo_magnitude_coef}")
            print(f"Teacher correct length dampen: {self.teacher_correct_length_dampen}")
            if self.span_evidence_floor >= 0:
                print(f"Merged teacher rank evidence floor: {self.span_evidence_floor}")
            print(f"Teacher gain-shaped span weight: floor={self.span_gain_weight_floor}, scale={self.span_gain_weight_scale}")
            print(f"Span trajectory normalization: enabled={self.span_traj_normalize}, max_sum={self.span_max_traj_weight}")
            print(f"Positive-teacher span only: {self.positive_teacher_span_only}")
            print(f"Positive-gain ReasonSpan only: {self.positive_reason_span_only}")
            print(f"No-answer full-length negative weight: {self.no_answer_full_length_weight}")
            print(
                "Verifier confidence gates: "
                f"enabled={self.verifier_confidence_enabled}, "
                f"answer_neg={self.verifier_answer_marker_negative_confidence}, "
                f"trunc_answer_neg={self.verifier_truncated_answer_negative_confidence}, "
                f"correct_no_answer={self.verifier_correct_no_answer_confidence}"
            )
            print(
                "Span tokens: "
                f"answer={self.span_cfg.answer_span_tokens}, "
                f"diff={self.span_cfg.diff_span_tokens}, diff_top_k={self.span_cfg.diff_span_top_k}, "
                f"calculation={self.span_cfg.calculation_span_tokens}, "
                f"support={self.span_cfg.support_span_tokens}, "
                f"reason={self.span_cfg.reason_span_tokens}"
            )
            if self.final_span_weight >= 0 or self.contrast_span_weight >= 0 or self.context_span_weight >= 0:
                print(
                    "Merged span priors: "
                    f"final={self.final_span_weight}, "
                    f"contrast={self.contrast_span_weight}, context={self.context_span_weight}"
                )
            print(
                "Span priors: "
                f"answer={self.answer_span_weight}, diff={self.diff_span_weight}, "
                f"calculation={self.calculation_span_weight}, "
                f"support={self.support_span_weight}, reason={self.reason_span_weight}"
            )
            print(f"Span starts: support={self.support_start_ratio}, reason={self.reason_start_ratio}")
            print(f"{'=' * 80}\n")

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if dataset is not None and hasattr(dataset, "column_names"):
            columns = set(dataset.column_names)
            if {"student_input", "ground_truth"}.issubset(columns):
                return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "id",
            "student_input",
            "ground_truth",
            "verifier_type",
            "reference_solution",
            "privileged_ladder",
            "problem",
            "solution",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.curriculum_sampler_enabled and dataset is not None:
            return OnlineCLSSampler(self, dataset)
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

    def _teacher_adapter_context(self, model):
        if self.teacher_model is not None:
            return nullcontext()
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _metric(self, name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = value.detach().float().mean().item()
        self._metrics["train"][name].append(float(value))

    def _compute_outcomes(self, inputs, completion_texts: list[str], parent_indices: torch.Tensor):
        parent_list = parent_indices.detach().cpu().tolist()
        outcomes = []
        for completion, parent in zip(completion_texts, parent_list):
            outcomes.append(
                verify_response(
                    completion,
                    inputs["ground_truth"][parent],
                    inputs["verifier_type"][parent],
                )
            )
        return torch.tensor(outcomes, device=parent_indices.device, dtype=torch.bool)

    def _completion_has_answer_marker(self, text: str) -> bool:
        lowered = str(text).lower()
        markers = (
            "\\boxed",
            "boxed",
            "final answer",
            "final:",
            "answer:",
            "the answer is",
            "so the answer",
            "答案",
            "最终答案",
            "故答案",
        )
        return any(marker in lowered for marker in markers)

    def _completion_answer_end_token_count(self, text: str) -> int:
        """Best-effort token offset immediately after the final answer expression."""
        raw_text = str(text)
        if not raw_text:
            return -1
        special_positions = [
            pos
            for token in ("<|im_end|>", "<|endoftext|>")
            for pos in [raw_text.find(token)]
            if pos >= 0
        ]
        visible_text = raw_text[: min(special_positions)] if special_positions else raw_text
        lowered = visible_text.lower()
        marker_idx = max(lowered.rfind("\\boxed"), lowered.rfind("boxed"))
        end_char = -1
        if marker_idx >= 0:
            brace_idx = visible_text.find("{", marker_idx)
            if brace_idx >= 0:
                depth = 0
                for idx in range(brace_idx, len(visible_text)):
                    char = visible_text[idx]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth <= 0:
                            end_char = idx + 1
                            break
            if end_char < 0:
                line_end = visible_text.find("\n", marker_idx)
                end_char = line_end if line_end >= 0 else len(visible_text)
        else:
            phrase_positions = [
                lowered.rfind(marker)
                for marker in ("final answer", "answer:", "the answer is", "so the answer")
            ]
            marker_idx = max(phrase_positions)
            if marker_idx >= 0:
                line_end = visible_text.find("\n", marker_idx)
                end_char = line_end if line_end >= 0 else len(visible_text)
        if end_char <= 0:
            return -1
        try:
            token_ids = self.processing_class(
                visible_text[:end_char],
                add_special_tokens=False,
            ).input_ids
        except Exception:
            return -1
        return int(len(token_ids)) if token_ids else -1

    def _training_progress(self) -> float:
        max_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return 0.0
        return min(1.0, max(0.0, float(self.state.global_step) / float(max_steps)))

    def _curriculum_alpha(self) -> float:
        stage = self._curriculum_stage()
        if stage <= 0:
            return self.curriculum_alpha_start
        if stage == 1:
            return 0.70 * self.curriculum_alpha_start + 0.30 * self.curriculum_alpha_end
        if stage == 2:
            return 0.35 * self.curriculum_alpha_start + 0.65 * self.curriculum_alpha_end
        return self.curriculum_alpha_end

    def _curriculum_stage(self) -> int:
        progress = self._training_progress()
        if progress < self.curriculum_stage1_ratio:
            return 0
        if progress < self.curriculum_stage2_ratio:
            return 1
        if progress < self.curriculum_stage3_ratio:
            return 2
        return 3

    def _curriculum_memory_state(self) -> tuple[float, float, float]:
        if not self._curriculum_memory:
            return 0.0, 0.0, 0.0
        train_size = max(1, len(self.train_dataset) if self.train_dataset is not None else len(self._curriculum_memory))
        p_values = [float(entry.get("p", 0.0) or 0.0) for entry in self._curriculum_memory.values()]
        coverage = min(1.0, len(self._curriculum_memory) / train_size)
        mean_p = sum(p_values) / max(1, len(p_values))
        mixed_rate = sum(1.0 for p in p_values if 0.0 < p < 1.0) / max(1, len(p_values))
        return coverage, mean_p, mixed_rate

    def _adaptive_span_stage_ready(self, span_stage: str) -> bool:
        coverage, mean_p, mixed_rate = self._curriculum_memory_state()
        self._metric("course/adaptive_memory_coverage", torch.tensor(coverage, device=self.accelerator.device))
        self._metric("course/adaptive_memory_correct_rate", torch.tensor(mean_p, device=self.accelerator.device))
        self._metric("course/adaptive_memory_mixed_rate", torch.tensor(mixed_rate, device=self.accelerator.device))
        if not self.adaptive_curriculum_enabled:
            return True
        if coverage < self.adaptive_curriculum_min_coverage:
            return False
        if span_stage == "support":
            return mean_p >= self.adaptive_support_min_correct_rate
        if span_stage == "reason":
            return mean_p >= self.adaptive_reason_min_correct_rate
        return True

    def _effective_span_coef(self) -> float:
        progress = self._training_progress()
        if progress <= self.span_decay_start_ratio:
            return self.span_distill_coef
        if progress >= self.span_decay_end_ratio:
            return self.span_coef_floor
        span = max(1e-6, self.span_decay_end_ratio - self.span_decay_start_ratio)
        decay = (progress - self.span_decay_start_ratio) / span
        coef = (1.0 - decay) * self.span_distill_coef + decay * self.span_coef_floor
        return max(self.span_coef_floor, coef)

    def _verifier_confidence_weights(
        self,
        advantages: torch.Tensor,
        completion_has_answer_marker: torch.Tensor,
        near_full_length_mask: torch.Tensor,
    ) -> torch.Tensor:
        confidence = torch.ones_like(advantages)
        if not self.verifier_confidence_enabled:
            return confidence
        positive_mask = advantages > 0
        negative_mask = advantages < 0
        answer_negative = negative_mask & completion_has_answer_marker
        truncated_answer_negative = answer_negative & near_full_length_mask
        correct_no_answer = positive_mask & (~completion_has_answer_marker)
        confidence = torch.where(
            answer_negative,
            torch.full_like(confidence, self.verifier_answer_marker_negative_confidence),
            confidence,
        )
        confidence = torch.where(
            truncated_answer_negative,
            torch.full_like(confidence, self.verifier_truncated_answer_negative_confidence),
            confidence,
        )
        confidence = torch.where(
            correct_no_answer,
            torch.full_like(confidence, self.verifier_correct_no_answer_confidence),
            confidence,
        )
        self._metric("verifier/confidence_mean", confidence.mean())
        self._metric("verifier/answer_marker_negative_rate", answer_negative.float().mean())
        self._metric("verifier/truncated_answer_negative_rate", truncated_answer_negative.float().mean())
        self._metric("verifier/correct_no_answer_rate", correct_no_answer.float().mean())
        self._metric(
            "train/verifier_confidence_multiplier",
            confidence[advantages != 0].mean()
            if bool((advantages != 0).any().item())
            else torch.ones((), device=advantages.device, dtype=advantages.dtype),
        )
        return confidence

    def _encode_teacher_contexts(self, texts: list[str], device: torch.device, generation_ids: torch.Tensor):
        encoded = self._tokenize_contexts(texts, device, generation_ids.shape[1])
        input_ids = torch.cat([encoded.input_ids, generation_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        if self.processing_class.pad_token_id is not None:
            attention_mask[input_ids == self.processing_class.pad_token_id] = 0
        return input_ids, attention_mask, encoded.input_ids.shape[1]

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs,
                self.generation_config,
                self.processing_class.pad_token_id,
            )
            (
                generated_ids,
                generated_attention_mask,
                _,
                prompt_texts,
                completion_texts,
                parent_indices,
                rollout_prompt_len,
                prompt_lengths_per_sequence,
                completion_lengths_per_sequence,
            ) = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                (
                    generated_ids,
                    generated_attention_mask,
                    _,
                    parent_indices,
                    rollout_prompt_len,
                    prompt_lengths_per_sequence,
                    completion_lengths_per_sequence,
                ) = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
            prompt_texts = self._repeat_prompt_texts(inputs)
            completion_ids = generated_ids[:, rollout_prompt_len:]
            completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        parent_indices = parent_indices.to(generated_ids.device)
        inputs["student_prompt_length"] = int(rollout_prompt_len)
        inputs["student_prompt_lengths_per_sequence"] = prompt_lengths_per_sequence.to(generated_ids.device)
        inputs["completion_lengths_per_sequence"] = completion_lengths_per_sequence.to(generated_ids.device)
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask
        inputs["labels"] = self._make_labels(inputs, generated_ids, parent_indices)
        inputs["sample_parent_indices"] = parent_indices
        inputs["outcomes"] = self._compute_outcomes(inputs, completion_texts, parent_indices)
        inputs["completion_has_answer_marker"] = torch.tensor(
            [self._completion_has_answer_marker(completion) for completion in completion_texts],
            device=parent_indices.device,
            dtype=torch.bool,
        )
        inputs["answer_end_token_positions"] = torch.tensor(
            [self._completion_answer_end_token_count(completion) for completion in completion_texts],
            device=parent_indices.device,
            dtype=torch.long,
        )
        inputs["num_original_samples"] = len(inputs["ground_truth"])

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))
        for prompt, completion, outcome, parent in zip(
            prompt_texts,
            completion_texts,
            inputs["outcomes"].detach().cpu().tolist(),
            parent_indices.detach().cpu().tolist(),
        ):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "sample_id": inputs["sample_ids"][parent],
                    "outcome": bool(outcome),
                    "prompt": prompt,
                    "completion": completion,
                }
            )

        if self.accelerator.is_main_process and random.random() < self.span_debug_rate and completion_texts:
            sample_idx = random.randint(0, len(completion_texts) - 1)
            print(f"\n{'=' * 80}")
            print(f"COURSE-DISTILL ONLINE SAMPLE (step={self.state.global_step})")
            print(prompt_texts[sample_idx])
            print(completion_texts[sample_idx])
            print(f"outcome={bool(inputs['outcomes'][sample_idx].item())}")
            print(f"{'=' * 80}\n")

        loss = super(LiteCIEOPDTrainer, self).training_step(model, inputs, num_items_in_batch)

        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        self._on_policy_loss_total += loss_scalar
        self._on_policy_step_equiv += 1.0 / ga
        return loss

    def _sample_key(self, sample_ids, group_idx: int) -> str:
        if isinstance(sample_ids, (list, tuple)) and group_idx < len(sample_ids):
            key = str(sample_ids[group_idx])
            if key:
                return key
        return f"rank{self.accelerator.process_index}:group{group_idx}"

    def _group_rates_and_advantages(self, outcomes, parent_indices, num_groups, dtype, sample_ids=None):
        device = outcomes.device
        rates = torch.zeros(num_groups, device=device, dtype=dtype)
        cls = torch.zeros(num_groups, device=device, dtype=dtype)
        memory_rates = torch.zeros(num_groups, device=device, dtype=dtype)
        memory_cls = torch.zeros(num_groups, device=device, dtype=dtype)
        effective_cls = torch.zeros(num_groups, device=device, dtype=dtype)
        has_mixed = torch.zeros(num_groups, device=device, dtype=torch.bool)
        all_wrong_groups = torch.zeros(num_groups, device=device, dtype=torch.bool)
        all_correct_groups = torch.zeros(num_groups, device=device, dtype=torch.bool)
        update_groups = torch.zeros(num_groups, device=device, dtype=torch.bool)
        advantages = torch.zeros_like(outcomes, dtype=dtype)

        outcomes_f = outcomes.to(dtype=dtype)
        for group_idx in range(num_groups):
            group_mask = parent_indices == group_idx
            if not bool(group_mask.any().item()):
                continue
            group_rewards = outcomes_f[group_mask]
            p = group_rewards.mean()
            rates[group_idx] = p
            pair_capacity = torch.minimum(p, 1.0 - p)
            base_cls = 4.0 * pair_capacity * (1.0 - p)
            review_cls = 4.0 * p * (1.0 - p)
            if self.curriculum_routing_rule == "uncertainty":
                cls_value = review_cls
            else:
                stage = self._curriculum_stage()
                if stage <= 0:
                    review_mix = 0.0
                elif stage == 1:
                    review_mix = 0.15
                elif stage == 2:
                    review_mix = 0.35
                else:
                    review_mix = 0.25
                review_mix = min(review_mix, self.curriculum_review_mix_max)
                cls_value = (1.0 - review_mix) * base_cls + review_mix * review_cls
            cls[group_idx] = cls_value
            variance = p * (1.0 - p)
            if bool((variance > 0).item()):
                has_mixed[group_idx] = True
                advantages[group_mask] = (group_rewards - p) / torch.sqrt(variance + 1e-6)
            elif bool((p <= 0.0).item()):
                all_wrong_groups[group_idx] = True
                if self.all_wrong_penalty > 0:
                    advantages[group_mask] = -self.all_wrong_penalty
            elif bool((p >= 1.0).item()):
                all_correct_groups[group_idx] = True
                if self.all_correct_bonus > 0:
                    advantages[group_mask] = self.all_correct_bonus

            key = self._sample_key(sample_ids, group_idx)
            current_p = float(p.detach().float().item())
            current_cls = float(cls_value.detach().float().item())
            previous = self._curriculum_memory.get(key)
            if previous is None:
                ema_p = current_p
                ema_cls = current_cls
                seen = 1.0
            else:
                momentum = self.curriculum_memory_momentum
                ema_p = momentum * previous.get("p", current_p) + (1.0 - momentum) * current_p
                ema_cls = momentum * previous.get("cls", current_cls) + (1.0 - momentum) * current_cls
                seen = previous.get("seen", 0.0) + 1.0
            self._curriculum_memory[key] = {
                "p": ema_p,
                "cls": ema_cls,
                "seen": seen,
            }
            memory_rates[group_idx] = torch.as_tensor(ema_p, device=device, dtype=dtype)
            memory_cls[group_idx] = torch.as_tensor(ema_cls, device=device, dtype=dtype)

        alpha = self._curriculum_alpha()
        blend = self.curriculum_memory_blend
        blended_cls = (1.0 - blend) * cls + blend * memory_cls
        effective_cls = torch.where(has_mixed, blended_cls, torch.zeros_like(blended_cls))
        data_weights = torch.pow(effective_cls.clamp(min=0.0), alpha)
        if self.curriculum_weight_floor > 0:
            data_weights = torch.where(
                has_mixed,
                self.curriculum_weight_floor + (1.0 - self.curriculum_weight_floor) * data_weights,
                data_weights,
            )
        if self.mixed_balance_bonus > 0:
            balance = (4.0 * rates * (1.0 - rates)).clamp(min=0.0, max=1.0)
            data_weights = torch.where(
                has_mixed,
                data_weights * (1.0 + self.mixed_balance_bonus * balance),
                data_weights,
            )
        if self.all_wrong_penalty > 0 and self.all_wrong_data_weight > 0:
            data_weights = torch.where(
                all_wrong_groups,
                torch.full_like(data_weights, self.all_wrong_data_weight),
                data_weights,
            )
        if self.all_correct_bonus > 0 and self.all_correct_data_weight > 0:
            data_weights = torch.where(
                all_correct_groups,
                torch.full_like(data_weights, self.all_correct_data_weight),
                data_weights,
            )
        if self.advantage_clip > 0:
            advantages = advantages.clamp(min=-self.advantage_clip, max=self.advantage_clip)
        update_groups = has_mixed | ((all_wrong_groups | all_correct_groups) & (data_weights > 0))
        return (
            rates,
            cls,
            memory_rates,
            memory_cls,
            effective_cls,
            data_weights,
            advantages,
            has_mixed,
            all_wrong_groups,
            all_correct_groups,
            update_groups,
        )

    def _teacher_token_scores(
        self,
        model,
        inputs,
        sampled_token_ids,
        response_mask,
        generation_ids,
        parent_indices,
        student_log_probs_sampled=None,
    ):
        parent_list = parent_indices.detach().cpu().tolist()
        device = sampled_token_ids.device
        priv_texts = [inputs["teacher_privileged_context_texts"][parent] for parent in parent_list]
        priv_ids, priv_attention, priv_prompt_len = self._encode_teacher_contexts(priv_texts, device, generation_ids)
        teacher_forward_model = self.teacher_model if self.teacher_model is not None else model

        with torch.no_grad(), self._teacher_adapter_context(model):
            logp_priv = self._teacher_sampled_logps(
                teacher_forward_model,
                priv_ids,
                priv_attention,
                priv_prompt_len,
                sampled_token_ids,
            )

            if self.use_privileged_gain:
                base_texts = [inputs["teacher_base_context_texts"][parent] for parent in parent_list]
                base_ids, base_attention, base_prompt_len = self._encode_teacher_contexts(base_texts, device, generation_ids)
                logp_base = self._teacher_sampled_logps(
                    teacher_forward_model,
                    base_ids,
                    base_attention,
                    base_prompt_len,
                    sampled_token_ids,
                )
                token_scores = logp_priv - logp_base
                del logp_base
                empty_cache()
            else:
                token_scores = logp_priv

            if self.teacher_privileged_confidence_coef > 0:
                den = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                centered_priv = logp_priv - (logp_priv * response_mask).sum(dim=-1, keepdim=True) / den
                self._metric(
                    "teacher/privileged_confidence_centered_mean",
                    (centered_priv * response_mask).sum() / response_mask.sum().clamp(min=1.0),
                )
                token_scores = token_scores + self.teacher_privileged_confidence_coef * centered_priv

            if self.teacher_student_gap_coef > 0 and student_log_probs_sampled is not None:
                teacher_student_gap = logp_priv - student_log_probs_sampled.detach()
                self._metric(
                    "span/teacher_student_gap_mean",
                    (teacher_student_gap * response_mask).sum() / response_mask.sum().clamp(min=1.0),
                )
                token_scores = token_scores + self.teacher_student_gap_coef * teacher_student_gap

        return token_scores * response_mask

    def _teacher_sampled_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        sampled_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute teacher log-probs for sampled response tokens in micro-batches.

        The full logits tensor is the dominant memory spike for a large teacher:
        batch * sequence * vocab. We only need logits that predict generated
        tokens, so we ask supported HF models for `logits_to_keep=response_len+1`
        and split the trajectory batch dimension.
        """
        response_len = sampled_token_ids.shape[1]
        outputs = []
        micro_batch = max(1, self.teacher_score_micro_batch)
        for start in range(0, input_ids.shape[0], micro_batch):
            end = min(input_ids.shape[0], start + micro_batch)
            ids = input_ids[start:end]
            attention = attention_mask[start:end]
            token_ids = sampled_token_ids[start:end]
            try:
                model_outputs = model(
                    input_ids=ids,
                    attention_mask=attention,
                    use_cache=False,
                    logits_to_keep=response_len + 1,
                )
            except TypeError:
                model_outputs = model(
                    input_ids=ids,
                    attention_mask=attention,
                    use_cache=False,
                )

            logits = model_outputs.logits
            if logits.shape[1] == response_len + 1:
                logits = logits[:, :-1, :]
            elif logits.shape[1] == ids.shape[1]:
                logits = logits[:, prompt_len - 1 : -1, :]
            elif logits.shape[1] > response_len:
                logits = logits[:, -response_len - 1 : -1, :]
            else:
                logits = logits[:, -response_len:, :]

            log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            outputs.append(torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1))
            del model_outputs, logits, log_probs
            empty_cache()
        return torch.cat(outputs, dim=0)

    def _mask_average_scores(self, token_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        den = mask.sum(dim=-1).clamp(min=1.0)
        return (token_scores * mask).sum(dim=-1) / den

    def _rank_span_weights(self, scores, active_mask):
        weights = torch.zeros_like(scores)
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return weights
        if active_indices.numel() == 1:
            weights[active_indices] = 1.0
            return weights
        active_scores = scores[active_indices]
        order = torch.argsort(active_scores, stable=True)
        ranks = torch.empty_like(active_scores)
        denom = max(1, active_scores.numel() - 1)
        ranks[order] = torch.arange(active_scores.numel(), device=scores.device, dtype=scores.dtype) / denom
        weights[active_indices] = self.span_rank_floor + (1.0 - self.span_rank_floor) * ranks
        return weights

    def _privileged_gain_weights(self, scores: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Softly downweight teacher-unsupported spans without dropping signal."""
        weights = torch.ones_like(scores)
        if not self.use_privileged_gain or self.span_gain_weight_floor >= 1.0:
            return weights
        active_scores = scores[active_mask]
        if active_scores.numel() == 0:
            return weights
        scale = active_scores.detach().abs().mean().clamp(min=1e-6) * self.span_gain_weight_scale
        shaped = torch.sigmoid(scores / scale)
        shaped = self.span_gain_weight_floor + (1.0 - self.span_gain_weight_floor) * shaped
        return torch.where(active_mask, shaped, weights)

    def _local_step_scores(self, token_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        window = max(1, self.teacher_step_window)
        if window <= 1:
            return token_scores * mask
        if window % 2 == 0:
            window += 1
        pad = window // 2
        score_num = F.avg_pool1d(
            (token_scores * mask).unsqueeze(1),
            kernel_size=window,
            stride=1,
            padding=pad,
            count_include_pad=False,
        ).squeeze(1)
        score_den = F.avg_pool1d(
            mask.unsqueeze(1),
            kernel_size=window,
            stride=1,
            padding=pad,
            count_include_pad=False,
        ).squeeze(1).clamp(min=1e-6)
        return score_num / score_den

    def _rank_token_weights(self, scores: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(scores)
        for row in range(scores.shape[0]):
            active_indices = torch.nonzero(active_mask[row], as_tuple=False).flatten()
            if active_indices.numel() == 0:
                continue
            if active_indices.numel() == 1:
                weights[row, active_indices] = 1.0
                continue
            active_scores = scores[row, active_indices]
            order = torch.argsort(active_scores, stable=True)
            ranks = torch.empty_like(active_scores)
            denom = max(1, active_scores.numel() - 1)
            ranks[order] = torch.arange(active_scores.numel(), device=scores.device, dtype=scores.dtype) / denom
            weights[row, active_indices] = self.teacher_step_rank_floor + (1.0 - self.teacher_step_rank_floor) * ranks
        if self.teacher_token_weight_clip > 0:
            weights = weights.clamp(max=self.teacher_token_weight_clip)
        return weights

    def _trace_context_rerank_mask(
        self,
        token_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        active_mask: torch.Tensor,
        metric_prefix: str,
    ) -> torch.Tensor:
        """Keep only top teacher-supported contiguous windows inside explainable candidates."""
        if self.trace_context_topk <= 0:
            return candidate_mask
        filtered = torch.zeros_like(candidate_mask)
        max_tokens = max(1, self.trace_context_max_tokens)
        active_rows = 0
        selected_rows = 0
        selected_segments = 0

        for row in range(candidate_mask.shape[0]):
            if not bool(active_mask[row].detach().item()):
                continue
            indices = torch.nonzero(candidate_mask[row] > 0, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            active_rows += 1

            segments: list[tuple[int, int]] = []
            start = int(indices[0].item())
            prev = start
            for token_idx in indices[1:].detach().cpu().tolist():
                token_idx = int(token_idx)
                if token_idx == prev + 1:
                    prev = token_idx
                    continue
                segments.append((start, prev + 1))
                start = token_idx
                prev = token_idx
            segments.append((start, prev + 1))

            scored_segments: list[tuple[float, int, int]] = []
            for start_idx, end_idx in segments:
                segment_mask = candidate_mask[row, start_idx:end_idx]
                denom = segment_mask.sum().clamp(min=1.0)
                score = (token_scores[row, start_idx:end_idx] * segment_mask).sum() / denom
                scored_segments.append((float(score.detach().float().item()), start_idx, end_idx))
            scored_segments.sort(key=lambda item: item[0], reverse=True)

            remaining = max_tokens
            kept_for_row = 0
            for _, start_idx, end_idx in scored_segments:
                if remaining <= 0 or kept_for_row >= self.trace_context_topk:
                    break
                segment_len = end_idx - start_idx
                if segment_len > remaining:
                    local_mask = candidate_mask[row, start_idx:end_idx] > 0
                    local_scores = token_scores[row, start_idx:end_idx].detach().float()
                    local_scores = local_scores.masked_fill(~local_mask, -float("inf"))
                    peak = int(torch.argmax(local_scores).item()) if bool(local_mask.any().item()) else segment_len // 2
                    window = remaining
                    local_start = max(0, min(segment_len - window, peak - window // 2))
                    local_end = min(segment_len, local_start + window)
                    local_start = max(0, local_end - window)
                    keep_start = start_idx + local_start
                    keep_end = start_idx + local_end
                else:
                    keep_start = start_idx
                    keep_end = end_idx
                if keep_end <= keep_start:
                    continue
                filtered[row, keep_start:keep_end] = candidate_mask[row, keep_start:keep_end]
                remaining -= keep_end - keep_start
                kept_for_row += 1
                selected_segments += 1
            if kept_for_row > 0:
                selected_rows += 1

        candidate_tokens = candidate_mask.sum().clamp(min=1.0)
        self._metric(f"span/{metric_prefix}_trace_keep_rate", filtered.sum() / candidate_tokens)
        self._metric(
            f"span/{metric_prefix}_trace_selected_row_rate",
            torch.as_tensor(
                selected_rows / max(1, active_rows),
                device=candidate_mask.device,
                dtype=candidate_mask.dtype,
            ),
        )
        self._metric(
            f"span/{metric_prefix}_trace_segments_per_row",
            torch.as_tensor(
                selected_segments / max(1, selected_rows),
                device=candidate_mask.device,
                dtype=candidate_mask.dtype,
            ),
        )
        return filtered

    def _teacher_group_support_weights(
        self,
        teacher_token_scores: torch.Tensor,
        span_masks: dict[str, torch.Tensor],
        outcomes: torch.Tensor,
        parent_indices: torch.Tensor,
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use teacher evidence only as a same-prompt magnitude gate.

        The verifier still determines the update sign through GRPO advantages.
        Here the teacher compares verified-correct trajectories against wrong
        rollouts from the same problem, then scales only the correct trajectory
        span update. This keeps the method close to RLSD's scorer role without
        letting teacher preferences flip reward direction.
        """
        correct_mask = ((span_masks["final"] + span_masks["diff"] + span_masks.get("calculation", 0.0)) > 0).to(
            dtype=teacher_token_scores.dtype
        )
        wrong_mask = span_masks["error"]
        score_mask = torch.where(outcomes.unsqueeze(-1), correct_mask, wrong_mask)
        valid = score_mask.sum(dim=-1) > 0
        traj_scores = self._mask_average_scores(teacher_token_scores, score_mask)
        support = torch.ones_like(traj_scores)
        margins = torch.zeros_like(traj_scores)
        supported_correct = torch.zeros_like(outcomes, dtype=torch.bool)

        for group_idx in range(num_groups):
            group_mask = parent_indices == group_idx
            pos_mask = group_mask & outcomes & valid
            neg_mask = group_mask & (~outcomes) & valid
            if not (bool(pos_mask.any().item()) and bool(neg_mask.any().item())):
                continue
            group_valid = group_mask & valid
            wrong_baseline = traj_scores[neg_mask].mean()
            margin = traj_scores[pos_mask] - wrong_baseline
            centered = traj_scores[group_valid] - traj_scores[group_valid].mean()
            scale = centered.detach().abs().mean().clamp(min=1e-6)
            # Teacher evidence is allowed to boost verified-correct rollouts, not
            # suppress them. The verifier/GRPO reward is the only source of sign.
            positive_support = torch.relu(torch.tanh(margin / scale))
            if self.teacher_group_support_floor > 0:
                positive_support = torch.where(
                    margin > 0,
                    self.teacher_group_support_floor
                    + (1.0 - self.teacher_group_support_floor) * positive_support,
                    torch.zeros_like(positive_support),
                )
            raw_support = 1.0 + positive_support
            support[pos_mask] = raw_support.clamp(max=self.teacher_support_max)
            margins[pos_mask] = margin
            supported_correct[pos_mask] = True

        return support, traj_scores, margins, supported_correct

    def _grpo_loss(self, student_log_probs_sampled, response_mask, advantages, traj_weights):
        traj_logp = (student_log_probs_sampled * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0)
        if self.grpo_logp_clip > 0:
            clipped_traj_logp = traj_logp.clamp(min=-self.grpo_logp_clip, max=0.0)
            self._metric("objective/grpo_traj_logp_clip_rate", (clipped_traj_logp != traj_logp).float().mean())
            traj_logp = clipped_traj_logp
        pg_weights = traj_weights.detach() * advantages.detach()
        active = pg_weights.abs() > 0
        if not bool(active.any().item()):
            return student_log_probs_sampled.sum() * 0.0
        raw_loss = -(pg_weights[active] * traj_logp[active]).sum() / pg_weights[active].abs().sum().clamp(min=1e-6)
        self._metric("objective/grpo_loss_unclipped", raw_loss)
        if self.grpo_loss_clip > 0:
            loss = raw_loss.clamp(min=-self.grpo_loss_clip, max=self.grpo_loss_clip)
            self._metric("objective/grpo_loss_clip_active", (loss != raw_loss).float())
        else:
            loss = raw_loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        response_mask = (shifted_labels != -100).to(dtype=torch.float32)
        generation_ids = sampled_token_ids
        lengths = response_mask.sum(dim=-1).clamp(min=1.0)
        max_completion_length = max(1, int(self.args.max_completion_length))
        full_length_mask = lengths >= max_completion_length
        near_full_length_mask = lengths >= (float(max_completion_length) * self.near_full_negative_ratio)

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        student_log_probs_sampled = torch.gather(
            student_log_probs,
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        student_eos_log_probs = None
        if isinstance(eos_token_id, int) and 0 <= eos_token_id < student_log_probs.shape[-1]:
            student_eos_log_probs = student_log_probs[..., int(eos_token_id)]
        del outputs_student, student_log_probs
        empty_cache()

        outcomes = inputs["outcomes"].to(student_log_probs_sampled.device)
        parent_indices = inputs["sample_parent_indices"].to(student_log_probs_sampled.device)
        num_groups = int(inputs["num_original_samples"])
        dtype = student_log_probs_sampled.dtype

        (
            group_rates,
            group_cls,
            group_memory_rates,
            group_memory_cls,
            group_effective_cls,
            group_data_weights,
            advantages,
            mixed_groups,
            all_wrong_groups,
            all_correct_groups,
            update_groups,
        ) = self._group_rates_and_advantages(
            outcomes,
            parent_indices,
            num_groups,
            dtype=dtype,
            sample_ids=inputs.get("sample_ids"),
        )
        traj_data_weights = group_data_weights[parent_indices]

        completion_has_answer_marker = inputs.get("completion_has_answer_marker")
        if completion_has_answer_marker is None:
            completion_has_answer_marker = torch.zeros_like(outcomes, dtype=torch.bool)
        else:
            completion_has_answer_marker = completion_has_answer_marker.to(
                device=student_log_probs_sampled.device,
                dtype=torch.bool,
            )
        answer_end_positions = inputs.get("answer_end_token_positions")
        if answer_end_positions is None:
            answer_end_positions = torch.full_like(outcomes, -1, dtype=torch.long)
        else:
            answer_end_positions = answer_end_positions.to(
                device=student_log_probs_sampled.device,
                dtype=torch.long,
            )
        completion_width = sampled_token_ids.shape[1]
        completion_positions = torch.arange(completion_width, device=student_log_probs_sampled.device).unsqueeze(0)
        valid_answer_end = (answer_end_positions >= 0) & (answer_end_positions < lengths.to(dtype=torch.long))
        post_answer_start = (answer_end_positions + self.answer_stop_grace_tokens).clamp(min=0)
        post_answer_mask = (
            valid_answer_end.unsqueeze(-1)
            & outcomes.unsqueeze(-1)
            & (completion_positions >= post_answer_start.unsqueeze(-1))
        ).to(dtype=student_log_probs_sampled.dtype) * response_mask
        stop_target_positions = answer_end_positions.clamp(min=0, max=max(0, completion_width - 1))
        stop_target_token_active = response_mask.gather(dim=1, index=stop_target_positions.unsqueeze(-1)).squeeze(-1) > 0
        answer_stop_target_mask = outcomes & valid_answer_end & stop_target_token_active
        response_tokens = response_mask.sum().clamp(min=1.0)
        self._metric("rollout/post_answer_token_rate", post_answer_mask.sum() / response_tokens)
        self._metric("rollout/answer_stop_target_rate", answer_stop_target_mask.float().mean())
        self._metric("train/rollout_post_answer_token_fraction", post_answer_mask.sum() / response_tokens)
        self._metric("train/rollout_answer_stop_target_fraction", answer_stop_target_mask.float().mean())
        verifier_confidence = self._verifier_confidence_weights(
            advantages,
            completion_has_answer_marker,
            near_full_length_mask,
        )
        advantages = advantages * verifier_confidence.detach()

        curriculum_stage = self._curriculum_stage()
        support_step_ready = self._training_progress() >= self.support_start_ratio
        reason_step_ready = self._training_progress() >= self.reason_start_ratio
        support_state_ready = self._adaptive_span_stage_ready("support")
        reason_state_ready = self._adaptive_span_stage_ready("reason")
        include_support = support_step_ready and support_state_ready
        include_reason = reason_step_ready and reason_state_ready
        self._metric("course/support_stage_step_ready", torch.tensor(float(support_step_ready), device=student_log_probs_sampled.device))
        self._metric("course/support_stage_state_ready", torch.tensor(float(support_state_ready), device=student_log_probs_sampled.device))
        self._metric("course/reason_stage_step_ready", torch.tensor(float(reason_step_ready), device=student_log_probs_sampled.device))
        self._metric("course/reason_stage_state_ready", torch.tensor(float(reason_state_ready), device=student_log_probs_sampled.device))
        span_masks = build_candidate_span_masks(
            sampled_token_ids=sampled_token_ids,
            response_mask=response_mask,
            outcomes=outcomes,
            parent_indices=parent_indices,
            tokenizer=self.processing_class,
            cfg=self.span_cfg,
            include_support=include_support,
            include_reason=include_reason,
        )
        span_mask = span_masks["combined"]
        grpo_mask = response_mask
        if self.post_answer_grpo_weight < 1.0:
            grpo_mask = grpo_mask * torch.where(
                post_answer_mask > 0,
                torch.full_like(grpo_mask, self.post_answer_grpo_weight),
                torch.ones_like(grpo_mask),
            )

        correct_active = outcomes & (span_mask.sum(dim=-1) > 0) & (advantages > 0)
        span_coef = self._effective_span_coef()

        span_loss = student_log_probs_sampled.sum() * 0.0
        teacher_support = torch.ones_like(advantages, dtype=student_log_probs_sampled.dtype)
        teacher_support_active = torch.zeros_like(outcomes, dtype=torch.bool)
        span_logged = False
        if bool(correct_active.any().item()):
            teacher_token_scores = self._teacher_token_scores(
                model,
                inputs,
                sampled_token_ids,
                response_mask,
                generation_ids,
                parent_indices,
                student_log_probs_sampled=student_log_probs_sampled,
            )
            (
                teacher_support,
                teacher_traj_scores,
                teacher_support_margins,
                teacher_support_active,
            ) = self._teacher_group_support_weights(
                teacher_token_scores,
                span_masks,
                outcomes,
                parent_indices,
                num_groups,
            )
            if self.teacher_correct_length_dampen > 0:
                length_ratio = (lengths / max_completion_length).clamp(min=0.0, max=1.0)
                length_scale = (1.0 - self.teacher_correct_length_dampen * length_ratio).clamp(
                    min=1.0 - self.teacher_correct_length_dampen,
                    max=1.0,
                )
                effective_scale = torch.where(outcomes, length_scale, torch.ones_like(length_scale))
                teacher_support = 1.0 + (teacher_support - 1.0) * effective_scale.to(dtype=teacher_support.dtype)
                self._metric(
                    "teacher/correct_length_dampen_mean",
                    effective_scale[teacher_support_active].mean()
                    if bool(teacher_support_active.any().item())
                    else torch.ones((), device=student_log_probs_sampled.device),
                )
            if bool(teacher_support_active.any().item()):
                self._metric("teacher/group_support_weight_mean", teacher_support[teacher_support_active].mean())
                self._metric("teacher/group_support_margin_mean", teacher_support_margins[teacher_support_active].mean())
                self._metric("teacher/group_support_positive_margin_rate", (teacher_support_margins[teacher_support_active] > 0).float().mean())
                self._metric("teacher/group_support_traj_score_mean", teacher_traj_scores[teacher_support_active].mean())
                self._metric("train/teacher_group_support_weight_for_correct", teacher_support[teacher_support_active].mean())
                self._metric("train/teacher_correct_vs_wrong_support_margin", teacher_support_margins[teacher_support_active].mean())

            if self.trace_context_topk > 0:
                trace_span_names = ["context"] if self.context_span_weight >= 0 else ["support", "reason"]
                span_masks = dict(span_masks)
                for trace_name in trace_span_names:
                    if trace_name == "support" and not include_support:
                        continue
                    if trace_name == "reason" and not include_reason:
                        continue
                    candidate_mask = span_masks.get(trace_name)
                    if candidate_mask is None or not bool((candidate_mask.sum() > 0).item()):
                        continue
                    self._metric(
                        f"span/{trace_name}_candidate_token_rate",
                        candidate_mask.sum() / response_mask.sum().clamp(min=1.0),
                    )
                    trace_active = outcomes & (advantages > 0) & (candidate_mask.sum(dim=-1) > 0)
                    if self.positive_teacher_span_only:
                        trace_active = trace_active & (teacher_support > 1.0)
                    trace_scores = self._mask_average_scores(teacher_token_scores, candidate_mask)
                    if self.use_privileged_gain and self.positive_reason_span_only:
                        trace_active = trace_active & (trace_scores > 0)
                    span_masks[trace_name] = self._trace_context_rerank_mask(
                        teacher_token_scores,
                        candidate_mask,
                        trace_active,
                        trace_name,
                    )

            if span_coef > 0:
                token_weights = torch.zeros_like(student_log_probs_sampled)
                teacher_weight_logs = []
                teacher_score_logs = []
                span_specs = []
                if self.final_span_weight >= 0:
                    span_specs.append(("final", self.final_span_weight))
                else:
                    span_specs.append(("answer", self.answer_span_weight))
                    span_specs.append(("calculation", self.calculation_span_weight))
                span_specs.append(("diff", self.diff_span_weight))
                if self.context_span_weight >= 0:
                    span_specs.append(("context", self.context_span_weight if (include_support or include_reason) else 0.0))
                else:
                    span_specs.append(("support", self.support_span_weight if include_support else 0.0))
                    span_specs.append(("reason", self.reason_span_weight if include_reason else 0.0))
                for span_name, span_prior in span_specs:
                    if span_prior <= 0:
                        continue
                    type_mask = span_masks[span_name]
                    type_active = outcomes & (type_mask.sum(dim=-1) > 0) & (advantages > 0)
                    if self.positive_teacher_span_only:
                        type_active = type_active & (teacher_support > 1.0)
                    type_scores = self._mask_average_scores(teacher_token_scores, type_mask)
                    if span_name in ("reason", "context") and self.use_privileged_gain and self.positive_reason_span_only:
                        type_active = type_active & (type_scores > 0)
                    if self.use_privileged_gain:
                        candidate_active = type_active
                        if bool(candidate_active.any().item()):
                            self._metric(
                                f"span/{span_name}_positive_gain_rate",
                                (type_scores[candidate_active] > 0).float().mean(),
                            )
                    if not bool(type_active.any().item()):
                        continue
                    type_weights = self._rank_span_weights(type_scores, type_active)
                    type_gain_weights = self._privileged_gain_weights(type_scores, type_active)
                    type_token_active = type_mask.bool() & type_active.unsqueeze(-1)
                    type_step_scores = self._local_step_scores(teacher_token_scores, type_mask)
                    type_step_weights = self._rank_token_weights(type_step_scores, type_token_active)
                    token_weights = token_weights + (
                        type_mask
                        * type_weights.unsqueeze(-1).detach()
                        * type_gain_weights.unsqueeze(-1).detach()
                        * type_step_weights.detach()
                        * float(span_prior)
                        * teacher_support.unsqueeze(-1).detach()
                        * advantages.clamp(min=0.0).unsqueeze(-1).detach()
                        * traj_data_weights.unsqueeze(-1).detach()
                    )
                    teacher_weight_logs.append(type_weights[type_active].mean())
                    teacher_score_logs.append(type_scores[type_active].mean())
                    self._metric(f"span/{span_name}_teacher_weight_mean", type_weights[type_active].mean())
                    self._metric(f"span/{span_name}_gain_weight_mean", type_gain_weights[type_active].mean())
                    self._metric(f"span/{span_name}_teacher_score_mean", type_scores[type_active].mean())
                    if bool(type_token_active.any().item()):
                        self._metric(
                            f"span/{span_name}_step_weight_mean",
                            type_step_weights[type_token_active].mean(),
                        )

                if self.span_traj_normalize:
                    traj_span_sum = token_weights.sum(dim=-1, keepdim=True)
                    active_traj = traj_span_sum.squeeze(-1) > 0
                    scale = torch.minimum(
                        torch.ones_like(traj_span_sum),
                        torch.full_like(traj_span_sum, self.span_max_traj_weight)
                        / traj_span_sum.detach().clamp(min=1e-6),
                    )
                    token_weights = token_weights * scale.detach()
                    self._metric(
                        "span/traj_normalization_scale_mean",
                        scale.squeeze(-1)[active_traj].mean()
                        if bool(active_traj.any().item())
                        else torch.ones((), device=student_log_probs_sampled.device),
                    )
                    self._metric(
                        "train/span_traj_normalized_active_rate",
                        active_traj.float().mean(),
                    )

                active_tokens = token_weights > 0
                if bool(active_tokens.any().item()):
                    span_logged = True
                    span_loss = -(student_log_probs_sampled * token_weights).sum() / token_weights.sum().clamp(min=1e-6)
                    if teacher_weight_logs:
                        self._metric("span/teacher_weight_mean", torch.stack(teacher_weight_logs).mean())
                    if teacher_score_logs:
                        self._metric("span/teacher_score_mean", torch.stack(teacher_score_logs).mean())
                    self._metric("span/token_rate", active_tokens.float().sum() / response_mask.sum().clamp(min=1.0))
                    self._metric("span/loss", span_loss)
                    self._metric(
                        "train/span_teacher_selected_token_fraction",
                        active_tokens.float().sum() / response_mask.sum().clamp(min=1.0),
                    )
                    if teacher_weight_logs:
                        self._metric("train/span_teacher_selection_weight", torch.stack(teacher_weight_logs).mean())
                    if teacher_score_logs:
                        self._metric("train/span_teacher_score_on_correct_spans", torch.stack(teacher_score_logs).mean())
                    self._metric("train/span_positive_distillation_loss", span_loss)

        if not span_logged:
            self._metric("span/token_rate", torch.zeros((), device=student_log_probs_sampled.device))
            self._metric(
                "train/span_teacher_selected_token_fraction",
                torch.zeros((), device=student_log_probs_sampled.device),
            )

        grpo_teacher_multiplier = 1.0 + self.teacher_grpo_magnitude_coef * (teacher_support.detach() - 1.0)
        positive_teacher_multiplier = torch.where(
            advantages > 0,
            grpo_teacher_multiplier,
            torch.ones_like(teacher_support),
        )
        correct_near_full_mask = (advantages > 0) & near_full_length_mask
        correct_length_multiplier = torch.where(
            correct_near_full_mask,
            torch.full_like(positive_teacher_multiplier, self.correct_near_full_positive_weight),
            torch.ones_like(positive_teacher_multiplier),
        )
        negative_mask = advantages < 0
        truncated_negative_mask = negative_mask & near_full_length_mask
        no_answer_full_length_mask = truncated_negative_mask & (~completion_has_answer_marker)
        truncated_negative_multiplier = torch.where(
            truncated_negative_mask,
            torch.full_like(positive_teacher_multiplier, self.truncated_negative_weight),
            torch.ones_like(positive_teacher_multiplier),
        )
        if self.no_answer_full_length_weight > 0:
            truncated_negative_multiplier = torch.where(
                no_answer_full_length_mask,
                torch.full_like(positive_teacher_multiplier, self.no_answer_full_length_weight),
                truncated_negative_multiplier,
            )
        wrong_length_multiplier = torch.ones_like(positive_teacher_multiplier)
        if self.wrong_length_boost > 0:
            length_ratio = (lengths / max_completion_length).clamp(min=0.0, max=1.0)
            non_truncated_negative_mask = negative_mask & (~full_length_mask)
            wrong_length_multiplier = torch.where(
                non_truncated_negative_mask,
                1.0 + self.wrong_length_boost * length_ratio,
                wrong_length_multiplier,
            )
            self._metric(
                "rollout/wrong_length_multiplier_mean",
                wrong_length_multiplier[negative_mask].mean() if bool(negative_mask.any().item()) else torch.ones((), device=lengths.device),
            )
            self._metric(
                "train/rollout_wrong_length_penalty_multiplier",
                wrong_length_multiplier[negative_mask].mean() if bool(negative_mask.any().item()) else torch.ones((), device=lengths.device),
            )
        self._metric("rollout/truncated_negative_rate", truncated_negative_mask.float().mean())
        self._metric("rollout/near_full_length_rate", near_full_length_mask.float().mean())
        self._metric("rollout/no_answer_full_length_negative_rate", no_answer_full_length_mask.float().mean())
        self._metric("rollout/correct_near_full_rate", correct_near_full_mask.float().mean())
        self._metric(
            "rollout/correct_near_full_multiplier_mean",
            correct_length_multiplier[advantages > 0].mean()
            if bool((advantages > 0).any().item())
            else torch.ones((), device=lengths.device),
        )
        self._metric(
            "rollout/truncated_negative_multiplier_mean",
            truncated_negative_multiplier[negative_mask].mean() if bool(negative_mask.any().item()) else torch.ones((), device=lengths.device),
        )
        self._metric("train/rollout_truncated_negative_fraction", truncated_negative_mask.float().mean())
        self._metric("train/rollout_near_full_fraction", near_full_length_mask.float().mean())
        self._metric("train/rollout_no_answer_full_length_negative_fraction", no_answer_full_length_mask.float().mean())
        self._metric("train/rollout_correct_near_full_fraction", correct_near_full_mask.float().mean())
        self._metric(
            "train/rollout_correct_near_full_multiplier",
            correct_length_multiplier[advantages > 0].mean()
            if bool((advantages > 0).any().item())
            else torch.ones((), device=lengths.device),
        )
        self._metric(
            "train/rollout_truncated_negative_multiplier",
            truncated_negative_multiplier[negative_mask].mean() if bool(negative_mask.any().item()) else torch.ones((), device=lengths.device),
        )
        grpo_traj_weights = (
            traj_data_weights
            * positive_teacher_multiplier
            * correct_length_multiplier.detach()
            * wrong_length_multiplier.detach()
            * truncated_negative_multiplier.detach()
        )
        grpo_loss = self._grpo_loss(student_log_probs_sampled, grpo_mask, advantages, grpo_traj_weights)
        loss = self.grpo_coef * grpo_loss + span_coef * span_loss
        if bool(teacher_support_active.any().item()):
            self._metric("teacher/grpo_positive_multiplier_mean", positive_teacher_multiplier[teacher_support_active].mean())
            self._metric("train/teacher_grpo_positive_multiplier", positive_teacher_multiplier[teacher_support_active].mean())
        self._metric(
            "teacher/grpo_magnitude_coef",
            torch.tensor(self.teacher_grpo_magnitude_coef, device=student_log_probs_sampled.device),
        )

        answer_stop_loss = student_log_probs_sampled.sum() * 0.0
        answer_stop_weighted_loss = student_log_probs_sampled.sum() * 0.0
        if (
            self.answer_stop_loss_coef > 0
            and student_eos_log_probs is not None
            and bool(answer_stop_target_mask.any().item())
        ):
            stop_log_probs = student_eos_log_probs.gather(
                dim=1,
                index=stop_target_positions.unsqueeze(-1),
            ).squeeze(-1)
            stop_weights = (
                advantages.clamp(min=0.0).detach()
                * traj_data_weights.detach()
                * teacher_support.detach()
                * correct_length_multiplier.detach()
            )
            stop_active = answer_stop_target_mask & (stop_weights > 0)
            if bool(stop_active.any().item()):
                answer_stop_loss = -(
                    stop_log_probs[stop_active] * stop_weights[stop_active]
                ).sum() / stop_weights[stop_active].sum().clamp(min=1e-6)
                answer_stop_weighted_loss = self.answer_stop_loss_coef * answer_stop_loss
                loss = loss + answer_stop_weighted_loss
        self._metric("objective/answer_stop_loss", answer_stop_loss)
        self._metric("objective/answer_stop_weighted_loss", answer_stop_weighted_loss)
        self._metric("train/loss_answer_stop_after_final", answer_stop_weighted_loss)

        error_suppress_loss = student_log_probs_sampled.sum() * 0.0
        error_suppress_weighted_loss = student_log_probs_sampled.sum() * 0.0
        error_suppress_scale = torch.ones((), device=student_log_probs_sampled.device, dtype=student_log_probs_sampled.dtype)
        error_weights = (
            span_masks["error"]
            * (~outcomes).to(dtype=student_log_probs_sampled.dtype).unsqueeze(-1)
            * (-advantages).clamp(min=0.0).unsqueeze(-1).detach()
            * traj_data_weights.unsqueeze(-1).detach()
            * truncated_negative_multiplier.unsqueeze(-1).detach()
        )
        error_active = error_weights > 0
        error_suppress_enabled = self._training_progress() >= self.error_suppress_start_ratio
        self._metric(
            "span/error_suppress_enabled",
            torch.tensor(float(error_suppress_enabled), device=student_log_probs_sampled.device),
        )
        self._metric(
            "train/span_error_suppression_enabled",
            torch.tensor(float(error_suppress_enabled), device=student_log_probs_sampled.device),
        )
        if self.error_suppress_coef > 0 and error_suppress_enabled and bool(error_active.any().item()):
            error_suppress_loss = (student_log_probs_sampled.exp() * error_weights).sum() / error_weights.sum().clamp(min=1e-6)
            raw_error_weighted_loss = self.error_suppress_coef * error_suppress_loss
            cap = None
            if self.error_suppress_max_ratio > 0:
                anchor = (self.grpo_coef * grpo_loss.detach()).abs() + (span_coef * span_loss.detach()).abs()
                cap = self.error_suppress_max_ratio * anchor
            if self.error_suppress_max_span_ratio > 0:
                span_anchor = (span_coef * span_loss.detach()).abs()
                if bool((span_anchor > 0).item()):
                    span_cap = self.error_suppress_max_span_ratio * span_anchor
                    cap = span_cap if cap is None else torch.minimum(cap, span_cap)
            if cap is not None:
                error_suppress_scale = torch.minimum(
                    torch.ones_like(raw_error_weighted_loss),
                    cap / raw_error_weighted_loss.detach().abs().clamp(min=1e-8),
                )
                error_suppress_weighted_loss = raw_error_weighted_loss * error_suppress_scale.detach()
            else:
                error_suppress_weighted_loss = raw_error_weighted_loss
            loss = loss + error_suppress_weighted_loss
            self._metric("span/error_token_rate", error_active.float().sum() / response_mask.sum().clamp(min=1.0))
            self._metric("span/error_suppress_loss", error_suppress_loss)
            self._metric("span/error_suppress_scale", error_suppress_scale)
            self._metric("train/span_error_token_fraction", error_active.float().sum() / response_mask.sum().clamp(min=1.0))
        else:
            self._metric("span/error_token_rate", torch.zeros((), device=student_log_probs_sampled.device))
            self._metric("train/span_error_token_fraction", torch.zeros((), device=student_log_probs_sampled.device))

        pass_at_m = []
        avg_at_8 = []
        pass_at_8 = []
        avg_at_8_second = []
        pass_at_8_second = []
        for group_idx in range(num_groups):
            group_mask = parent_indices == group_idx
            if torch.any(group_mask):
                group_outcomes = outcomes[group_mask].float()
                pass_at_m.append(group_outcomes.any().float())
                if group_outcomes.numel() >= 8:
                    first8 = group_outcomes[:8]
                    avg_at_8.append(first8.mean())
                    pass_at_8.append(first8.any().float())
                if group_outcomes.numel() >= 16:
                    second8 = group_outcomes[8:16]
                    avg_at_8_second.append(second8.mean())
                    pass_at_8_second.append(second8.any().float())
        if pass_at_m:
            self._metric("rollout/pass_at_m", torch.stack(pass_at_m).mean())
        if avg_at_8:
            self._metric("rollout/avg_at_8_first", torch.stack(avg_at_8).mean())
            self._metric("rollout/pass_at_8_first", torch.stack(pass_at_8).mean())
            self._metric("train/rollout_avg_at_8_first", torch.stack(avg_at_8).mean())
            self._metric("train/rollout_pass_at_8_first", torch.stack(pass_at_8).mean())
        if avg_at_8_second:
            self._metric("rollout/avg_at_8_second", torch.stack(avg_at_8_second).mean())
            self._metric("rollout/pass_at_8_second", torch.stack(pass_at_8_second).mean())
            self._metric("train/rollout_avg_at_8_second", torch.stack(avg_at_8_second).mean())
            self._metric("train/rollout_pass_at_8_second", torch.stack(pass_at_8_second).mean())

        self._metric("objective/grpo_loss", grpo_loss)
        self._metric("objective/span_weighted_loss", span_coef * span_loss)
        self._metric("objective/error_suppress_weighted_loss", error_suppress_weighted_loss)
        self._metric("objective/error_suppress_raw_weighted_loss", self.error_suppress_coef * error_suppress_loss)
        self._metric("rollout/answer_accuracy", outcomes.float().mean())
        self._metric("course/group_cls_mean", group_cls.mean())
        self._metric("course/memory_cls_mean", group_memory_cls.mean())
        self._metric("course/effective_cls_mean", group_effective_cls.mean())
        self._metric("course/data_weight_mean", group_data_weights.mean())
        self._metric(
            "course/data_weight_active_mean",
            group_data_weights[mixed_groups].mean() if bool(mixed_groups.any().item()) else torch.zeros((), device=group_data_weights.device),
        )
        self._metric(
            "course/data_weight_update_mean",
            group_data_weights[update_groups].mean() if bool(update_groups.any().item()) else torch.zeros((), device=group_data_weights.device),
        )
        balance_scores = (4.0 * group_rates * (1.0 - group_rates)).clamp(min=0.0, max=1.0)
        self._metric(
            "course/mixed_balance_score_mean",
            balance_scores[mixed_groups].mean() if bool(mixed_groups.any().item()) else torch.zeros((), device=group_rates.device),
        )
        self._metric(
            "course/mixed_balance_bonus",
            torch.tensor(self.mixed_balance_bonus, device=student_log_probs_sampled.device),
        )
        self._metric("course/mixed_group_rate", mixed_groups.float().mean())
        self._metric("course/all_wrong_group_rate", all_wrong_groups.float().mean())
        self._metric("course/all_correct_group_rate", all_correct_groups.float().mean())
        self._metric("course/update_group_rate", update_groups.float().mean())
        self._metric("course/group_correct_rate_mean", group_rates.mean())
        self._metric("course/memory_correct_rate_mean", group_memory_rates.mean())
        self._metric("course/active_weighted_group_rate", (group_data_weights > 0).float().mean())
        self._metric("course/memory_size", torch.tensor(float(len(self._curriculum_memory)), device=student_log_probs_sampled.device))
        self._metric("course/alpha", torch.tensor(self._curriculum_alpha(), device=student_log_probs_sampled.device))
        self._metric("course/stage", torch.tensor(float(curriculum_stage), device=student_log_probs_sampled.device))
        logged_review_mix = min(
            0.0 if curriculum_stage <= 0 else 0.15 if curriculum_stage == 1 else 0.35 if curriculum_stage == 2 else 0.25,
            self.curriculum_review_mix_max,
        )
        self._metric(
            "course/review_mix",
            torch.tensor(
                logged_review_mix,
                device=student_log_probs_sampled.device,
            ),
        )
        self._metric("span/effective_coef", torch.tensor(span_coef, device=student_log_probs_sampled.device))
        self._metric("span/answer_token_rate", span_masks["answer"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/diff_token_rate", span_masks["diff"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/calculation_token_rate", span_masks["calculation"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/final_token_rate", span_masks["final"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/support_token_rate", span_masks["support"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/reason_token_rate", span_masks["reason"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/context_token_rate", span_masks["context"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("span/error_candidate_token_rate", span_masks["error"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("rollout/full_length_rate", (lengths >= self.args.max_completion_length).float().mean())
        self._metric("train/rollout_answer_accuracy", outcomes.float().mean())
        if pass_at_m:
            self._metric("train/rollout_pass_at_m", torch.stack(pass_at_m).mean())
        self._metric("train/rollout_full_length_fraction", (lengths >= self.args.max_completion_length).float().mean())
        self._metric("train/course_raw_current_learnability_score", group_cls.mean())
        self._metric("train/course_memory_learnability_score", group_memory_cls.mean())
        self._metric("train/course_effective_learnability_score", group_effective_cls.mean())
        self._metric("train/course_effective_training_weight", group_data_weights.mean())
        self._metric(
            "train/course_effective_training_weight_on_mixed_groups",
            group_data_weights[mixed_groups].mean() if bool(mixed_groups.any().item()) else torch.zeros((), device=group_data_weights.device),
        )
        self._metric(
            "train/course_effective_training_weight_on_update_groups",
            group_data_weights[update_groups].mean() if bool(update_groups.any().item()) else torch.zeros((), device=group_data_weights.device),
        )
        self._metric("train/course_active_group_fraction", (group_data_weights > 0).float().mean())
        self._metric("train/course_mixed_correct_wrong_group_fraction", mixed_groups.float().mean())
        self._metric("train/course_all_wrong_group_fraction", all_wrong_groups.float().mean())
        self._metric("train/course_all_correct_group_fraction", all_correct_groups.float().mean())
        self._metric("train/course_update_group_fraction", update_groups.float().mean())
        self._metric("train/course_group_correct_rate", group_rates.mean())
        self._metric("train/course_memory_correct_rate", group_memory_rates.mean())
        self._metric("train/course_alpha_schedule", torch.tensor(self._curriculum_alpha(), device=student_log_probs_sampled.device))
        self._metric("train/course_explicit_stage", torch.tensor(float(curriculum_stage), device=student_log_probs_sampled.device))
        self._metric(
            "train/course_review_mix_for_near_solved_groups",
            torch.tensor(
                logged_review_mix,
                device=student_log_probs_sampled.device,
            ),
        )
        self._metric("train/loss_grpo_reward_ordering", grpo_loss)
        self._metric("train/loss_span_weighted_distillation", span_coef * span_loss)
        self._metric("train/loss_error_span_confidence_suppression", error_suppress_weighted_loss)
        self._metric("train/loss_error_span_raw_before_cap", self.error_suppress_coef * error_suppress_loss)
        self._metric("train/span_effective_coef", torch.tensor(span_coef, device=student_log_probs_sampled.device))
        self._metric("train/span_answer_token_fraction", span_masks["answer"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("train/span_diff_token_fraction", span_masks["diff"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("train/span_calculation_token_fraction", span_masks["calculation"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("train/span_support_token_fraction", span_masks["support"].sum() / response_mask.sum().clamp(min=1.0))
        self._metric("train/span_reason_token_fraction", span_masks["reason"].sum() / response_mask.sum().clamp(min=1.0))

        if return_outputs:
            class MinimalOutput:
                def __init__(self, loss_value):
                    self.loss = loss_value

            return loss, MinimalOutput(loss)
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            vec = torch.tensor([self._on_policy_loss_total, self._on_policy_step_equiv], dtype=torch.float64, device=device)
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            on_sum, on_eq = vec.tolist()
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            self._on_policy_loss_total = 0.0
            self._on_policy_step_equiv = 0.0
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        super(LiteCIEOPDTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()
