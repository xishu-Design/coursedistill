from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class LiteCIEConfig(SFTConfig):
    """Training arguments for text-only Lite-CIE-OPD."""

    max_completion_length: int = field(
        default=1024,
        metadata={"help": "Maximum number of new text tokens sampled from the student under c0."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling and log-prob temperature."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling parameter for student text rollouts."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling parameter. 0 disables top-k for transformers generation."},
    )
    min_p: float = field(
        default=0.0,
        metadata={"help": "Optional vLLM min-p sampling parameter."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Optional repetition penalty used by vLLM generation."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Optional presence penalty used by vLLM generation."},
    )
    student_entropy_coef: float = field(
        default=0.004,
        metadata={"help": "Coefficient for the target-entropy hinge penalty that prevents student entropy collapse."},
    )
    sdpo_preference_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for trajectory-level SDPO preference optimization. Correct rollouts are "
                "ranked above wrong rollouts within utility-positive evidence groups."
            )
        },
    )
    sdpo_distill_coef: float = field(
        default=0.006,
        metadata={
            "help": (
                "Coefficient for evidence-gated SDPO distillation. Correct rollouts absorb token-level "
                "teacher guidance only where selected privileged evidence improves teacher confidence."
            )
        },
    )
    sdpo_require_logp_causal_gate: bool = field(
        default=True,
        metadata={
            "help": (
                "Require selected evidence to separate correct from wrong rollout log-prob gains before "
                "it can drive token-level SDPO distillation."
            )
        },
    )
    sdpo_low_entropy_distill: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply reverse-KL-like dense SDPO distillation mainly on low teacher-entropy tokens; "
                "high-entropy regions are left to the entropy-maintenance objective."
            )
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout in the policy model before training."},
    )

    use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vLLM for text-only student rollout generation."},
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={"help": "vLLM mode: colocate or server."},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.6,
        metadata={"help": "GPU memory utilization for colocated vLLM."},
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for colocated vLLM."},
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Enable vLLM sleep mode between rollout and weight sync."},
    )
    vllm_guided_decoding_regex: str | None = field(
        default=None,
        metadata={"help": "Optional regex for guided decoding."},
    )
    vllm_sync_frequency: int = field(
        default=1,
        metadata={"help": "Synchronize training weights to vLLM every N optimizer steps."},
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "vLLM server host when vllm_mode=server."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "vLLM server port when vllm_mode=server."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "vLLM server connection timeout in seconds."},
    )

    swanlab_project: str = field(
        default="cpd-anonymous",
        metadata={"help": "SwanLab project name."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "Optional SwanLab workspace."},
    )
    disable_swanlab: bool = field(
        default=False,
        metadata={"help": "Disable SwanLab tracking for smoke tests or local debug runs."},
    )
