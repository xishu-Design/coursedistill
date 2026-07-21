from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from coursedistill.collator import CourseDistillDataCollator
from coursedistill.config import CourseDistillConfig
from coursedistill.data import load_course_distill_dataset
from coursedistill.trainer import CourseDistillTrainer


@dataclass
class CourseDistillScriptArguments(ScriptArguments):
    data_dir: str = field(
        default="./coursedistill/data",
        metadata={"help": "Dedicated raw course-distill data directory containing train.jsonl."},
    )
    data_split: str = field(
        default="train",
        metadata={"help": "JSONL split name under data_dir, usually train."},
    )
    max_train_samples: int = field(
        default=0,
        metadata={"help": "Optional cap for smoke runs. 0 means all samples."},
    )
    num_student_samples: int = field(
        default=8,
        metadata={"help": "Number of on-policy student trajectories M per problem."},
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={"help": "Use the frozen base model as privileged teacher by disabling LoRA adapters."},
    )
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name suffix for output_dir and logging."},
    )
    student_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable Qwen thinking mode for student rollout."},
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={"help": "Whether to enable Qwen thinking mode for privileged teacher scoring."},
    )
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional separate privileged teacher model path. When set, student rollouts/training "
                "use model_name_or_path while teacher span scoring uses this frozen model."
            )
        },
    )
    teacher_model_revision: str | None = field(
        default=None,
        metadata={"help": "Optional revision for teacher_model_name_or_path."},
    )


def _resolve_dtype(model_args):
    dtype_value = getattr(model_args, "torch_dtype", None) or getattr(model_args, "dtype", None)
    if dtype_value is None:
        return torch.bfloat16
    if not isinstance(dtype_value, str):
        return dtype_value
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_value.lower(), torch.bfloat16)


def _report_to_list(report_to) -> list[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [name.strip() for name in report_to.split(",") if name.strip()]
    return list(report_to)


def _configure_swanlab(training_args) -> bool:
    report_to = _report_to_list(getattr(training_args, "report_to", None))
    if getattr(training_args, "disable_swanlab", False):
        filtered = [name for name in report_to if name not in {"all", "none", "swanlab"}]
        training_args.report_to = filtered or ["none"]
        return False

    os.environ.setdefault("SWANLAB_PROJECT", getattr(training_args, "swanlab_project", "opdrevi"))
    workspace = getattr(training_args, "swanlab_workspace", None)
    if workspace:
        os.environ.setdefault("SWANLAB_WORKSPACE", workspace)

    if not report_to or "none" in report_to or "all" in report_to:
        training_args.report_to = ["swanlab"]
    elif "swanlab" not in report_to:
        training_args.report_to = report_to + ["swanlab"]
    else:
        training_args.report_to = report_to
    return True


if __name__ == "__main__":
    parser = TrlParser((CourseDistillScriptArguments, CourseDistillConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires --use_peft.")

    model_name = model_args.model_name_or_path.rstrip("/").split("/")[-1]
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )

    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        run_name = (
            f"coursedistill_{model_name}_M{script_args.num_student_samples}_"
            f"lr{lr_str}_bs{effective_batch_size}_c{training_args.max_completion_length}"
        )

    if not getattr(training_args, "run_name", None) or training_args.run_name == training_args.output_dir:
        training_args.run_name = run_name
    swanlab_enabled = _configure_swanlab(training_args)

    model_dtype = _resolve_dtype(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_course_distill_dataset(
        data_dir=script_args.data_dir,
        split=script_args.data_split,
        max_samples=script_args.max_train_samples if script_args.max_train_samples > 0 else None,
        seed=training_args.seed,
    )

    data_collator = CourseDistillDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    print(f"\n{'=' * 80}")
    print("COURSE-DISTILL RUN CONFIGURATION")
    print(f"SwanLab Tracking: {'enabled' if swanlab_enabled else 'disabled'}")
    print(f"Run Name: {training_args.run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Data: {Path(script_args.data_dir) / (script_args.data_split + '.jsonl')}")
    print(f"Loaded samples: {len(train_dataset)}")
    print(f"Student samples M: {script_args.num_student_samples}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Model dtype: {model_dtype}")
    print(f"Student model: {model_args.model_name_or_path}")
    print(f"Teacher model: {script_args.teacher_model_name_or_path or 'student base via fixed_teacher'}")
    print(f"Span distill coef: {training_args.span_distill_coef}")
    print(f"GRPO coef: {training_args.grpo_coef}")
    print(f"GRPO guards: loss_clip={training_args.grpo_loss_clip}, logp_clip={training_args.grpo_logp_clip}")
    print(f"Curriculum alpha: {training_args.curriculum_alpha_start} -> {training_args.curriculum_alpha_end}")
    print(
        "Curriculum stages: "
        f"{training_args.curriculum_stage1_ratio}, "
        f"{training_args.curriculum_stage2_ratio}, "
        f"{training_args.curriculum_stage3_ratio}"
    )
    print(
        "Curriculum memory: "
        f"momentum={training_args.curriculum_memory_momentum}, "
        f"blend={training_args.curriculum_memory_blend}, "
        f"active_floor={training_args.curriculum_weight_floor}"
    )
    print(
        "Homogeneous reward fallback: "
        f"all_wrong_penalty={training_args.all_wrong_penalty}, "
        f"all_wrong_weight={training_args.all_wrong_data_weight}, "
        f"all_correct_bonus={training_args.all_correct_bonus}, "
        f"all_correct_weight={training_args.all_correct_data_weight}, "
        f"adv_clip={training_args.advantage_clip}"
    )
    print(f"Wrong length boost: {training_args.wrong_length_boost}")
    print(f"Truncated negative weight: {training_args.truncated_negative_weight}")
    print(
        "Teacher span scoring: "
        f"privileged_gain={training_args.use_privileged_gain}, "
        f"error_suppress_cap_ratio={training_args.error_suppress_max_ratio}, "
        f"error_suppress_start={training_args.error_suppress_start_ratio}, "
        f"step_window={training_args.teacher_step_window}, "
        f"student_gap_coef={training_args.teacher_student_gap_coef}, "
        f"group_support_floor={training_args.teacher_group_support_floor}, "
        f"score_micro_batch={training_args.teacher_score_micro_batch}, "
        f"positive_teacher_span_only={training_args.positive_teacher_span_only}"
    )
    print(f"{'=' * 80}\n")

    teacher_model = None
    if script_args.teacher_model_name_or_path:
        teacher_kwargs = dict(
            revision=script_args.teacher_model_revision or model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation or "flash_attention_2",
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
        print(f"Loading separate privileged teacher: {script_args.teacher_model_name_or_path}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            script_args.teacher_model_name_or_path,
            **teacher_kwargs,
        )

    trainer = CourseDistillTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        teacher_model=teacher_model,
        teacher_model_name_or_path=script_args.teacher_model_name_or_path,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        num_student_samples=script_args.num_student_samples,
        model_revision=model_args.model_revision,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
