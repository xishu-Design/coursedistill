from dataclasses import dataclass, field

from lite_cie_config import LiteCIEConfig


@dataclass
class CourseDistillConfig(LiteCIEConfig):
    """Training arguments for online Reward-Ordered Privileged Span Curriculum Distillation."""

    span_distill_coef: float = field(
        default=0.01,
        metadata={"help": "Coefficient for teacher-weighted positive span distillation."},
    )
    span_coef_floor: float = field(
        default=0.0,
        metadata={"help": "Minimum coefficient after span distillation annealing."},
    )
    span_decay_start_ratio: float = field(
        default=0.25,
        metadata={"help": "Training progress ratio where span distillation coefficient starts decaying."},
    )
    span_decay_end_ratio: float = field(
        default=0.85,
        metadata={"help": "Training progress ratio where span distillation coefficient reaches the floor."},
    )
    grpo_coef: float = field(
        default=1.0,
        metadata={"help": "Coefficient for GRPO-style rollout objective."},
    )
    grpo_loss_clip: float = field(
        default=0.5,
        metadata={
            "help": (
                "Symmetric clamp for the scalar GRPO loss. This prevents rare all-negative/noisy "
                "rollout batches from producing destructive policy updates."
            )
        },
    )
    grpo_logp_clip: float = field(
        default=10.0,
        metadata={
            "help": (
                "Clamp the per-trajectory average sampled-token log probability to [-value, 0] "
                "inside the GRPO objective. Set <=0 to disable."
            )
        },
    )
    error_suppress_coef: float = field(
        default=0.0,
        metadata={"help": "Coefficient for local confidence suppression on wrong error spans."},
    )
    error_suppress_max_ratio: float = field(
        default=0.10,
        metadata={"help": "Cap error-span suppression to this ratio of the active reward/span objective magnitude."},
    )
    error_suppress_max_span_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional extra cap for error-span suppression relative to the positive span objective. "
                "Set >0 to keep suppression as a guardrail instead of letting it dominate teacher-guided learning."
            )
        },
    )
    error_suppress_start_ratio: float = field(
        default=0.20,
        metadata={
            "help": (
                "Training progress ratio before local wrong-span confidence suppression is enabled. "
                "This avoids suppressing early truncated rollouts too aggressively."
            )
        },
    )
    curriculum_alpha_start: float = field(
        default=1.8,
        metadata={"help": "Initial exponent for CLS(x)^alpha online data curriculum."},
    )
    curriculum_alpha_end: float = field(
        default=0.8,
        metadata={"help": "Final exponent for CLS(x)^alpha online data curriculum."},
    )
    curriculum_routing_rule: str = field(
        default="cls",
        metadata={
            "help": (
                "Group routing score used by the online curriculum. Supported values are "
                "'cls' for 4*min(p,1-p)*(1-p) and 'uncertainty' for 4*p*(1-p)."
            )
        },
    )
    curriculum_stage1_ratio: float = field(
        default=0.25,
        metadata={"help": "Progress ratio ending stage 0: answer/diff bootstrap."},
    )
    curriculum_stage2_ratio: float = field(
        default=0.70,
        metadata={"help": "Progress ratio starting stage 2: support-span expansion."},
    )
    curriculum_stage3_ratio: float = field(
        default=0.88,
        metadata={"help": "Progress ratio starting stage 3: consolidation."},
    )
    curriculum_memory_momentum: float = field(
        default=0.65,
        metadata={"help": "EMA momentum for online per-problem curriculum value memory."},
    )
    curriculum_memory_blend: float = field(
        default=0.40,
        metadata={"help": "Blend ratio between current rollout CLS and online EMA CLS for mixed groups."},
    )
    curriculum_weight_floor: float = field(
        default=0.0,
        metadata={
            "help": (
                "Minimum relative curriculum weight for active mixed groups. "
                "Keeps online learning broad enough while CLS(x)^alpha still controls priority."
            )
        },
    )
    curriculum_review_mix_max: float = field(
        default=0.35,
        metadata={
            "help": (
                "Maximum amount of symmetric 4p(1-p) review mixed into the asymmetric CLS score. "
                "Set 0 for the main data-selection method so sampling remains governed by "
                "4*min(p,1-p)*(1-p)."
            )
        },
    )
    curriculum_sampler_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable online student-side data selection at the sampler level. The sampler uses "
                "EMA CLS memory from previous rollouts to sample high-utility problems more often."
            )
        },
    )
    curriculum_sampler_warmup_ratio: float = field(
        default=0.15,
        metadata={"help": "Training progress before the online curriculum sampler starts using EMA CLS weights."},
    )
    curriculum_sampler_floor: float = field(
        default=0.10,
        metadata={"help": "Minimum sampler probability floor for problems without strong EMA CLS evidence."},
    )
    curriculum_sampler_power: float = field(
        default=1.0,
        metadata={"help": "Exponent applied to EMA CLS inside the online curriculum sampler."},
    )
    curriculum_sampler_replacement: bool = field(
        default=True,
        metadata={"help": "Sample with replacement when online curriculum sampling is enabled."},
    )
    curriculum_sampler_unique_fraction: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of each sampler epoch reserved for a uniform without-replacement pass. "
                "The rest is sampled by EMA CLS. This preserves data coverage while still prioritizing "
                "high-utility curriculum examples."
            )
        },
    )
    adaptive_curriculum_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Gate later support/reason span stages by online student-state memory in addition to step progress. "
                "This avoids expanding broad spans before the model has enough stable correct anchors."
            )
        },
    )
    adaptive_curriculum_min_coverage: float = field(
        default=0.20,
        metadata={"help": "Minimum fraction of train problems seen in curriculum memory before adaptive gates open."},
    )
    adaptive_support_min_correct_rate: float = field(
        default=0.42,
        metadata={"help": "Minimum EMA group correct rate required before SupportSpan expansion is allowed."},
    )
    adaptive_reason_min_correct_rate: float = field(
        default=0.50,
        metadata={"help": "Minimum EMA group correct rate required before Reason/ContextSpan expansion is allowed."},
    )
    all_wrong_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Small absolute negative verifier advantage for groups where every rollout is wrong. "
                "Set 0 for guarded mixed-only training; positive values can be unstable without a correct anchor."
            )
        },
    )
    all_wrong_data_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Relative data weight for all-wrong groups when all_wrong_penalty is enabled. "
                "This is not a curriculum score; keep 0 unless explicitly testing hard-negative fallback."
            )
        },
    )
    all_correct_bonus: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional small absolute positive verifier advantage for all-correct groups. "
                "Keep 0 for the main method so solved groups are not overfit."
            )
        },
    )
    all_correct_data_weight: float = field(
        default=0.0,
        metadata={"help": "Relative data weight for all-correct groups when all_correct_bonus is enabled."},
    )
    advantage_clip: float = field(
        default=3.0,
        metadata={
            "help": (
                "Clamp absolute trajectory advantages to reduce rare verifier/noisy-rollout spikes. "
                "Set <=0 to disable clipping."
            )
        },
    )
    wrong_length_boost: float = field(
        default=0.0,
        metadata={
            "help": (
                "Extra GRPO magnitude for wrong trajectories proportional to completion length. "
                "Verifier still controls sign; this only suppresses long wrong rollouts more strongly."
            )
        },
    )
    truncated_negative_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Multiplier for negative-advantage trajectories that reach max_completion_length. "
                "This treats likely truncation as noisy reward instead of a fully trusted wrong answer."
            )
        },
    )
    correct_near_full_positive_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplier for positive-advantage trajectories that are near max_completion_length. "
                "Values below 1 keep verifier direction but avoid over-reinforcing verbose correct rollouts."
            )
        },
    )
    post_answer_grpo_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Token-level GRPO mask multiplier for verified-correct tokens after the detected final answer. "
                "Values below 1 stop correct long tails from receiving the same positive reinforcement."
            )
        },
    )
    answer_stop_loss_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Small auxiliary loss on verified-correct rollouts that encourages <|im_end|> immediately after "
                "the detected final answer. The verifier still determines which trajectories are positive."
            )
        },
    )
    answer_stop_grace_tokens: int = field(
        default=2,
        metadata={
            "help": (
                "Number of tokens after the detected final answer that are exempted from post-answer GRPO "
                "downweighting. The answer-stop target itself is placed at the detected answer end."
            )
        },
    )
    answer_span_tokens: int = field(
        default=48,
        metadata={"help": "Maximum token window used for AnswerSpan."},
    )
    diff_span_tokens: int = field(
        default=64,
        metadata={"help": "Maximum token window used after correct/wrong divergence for DiffSpan."},
    )
    diff_span_top_k: int = field(
        default=3,
        metadata={
            "help": (
                "Number of most-similar opposite-outcome rollouts used to propose DiffSpan candidates. "
                "Lower values reduce non-causal formatting divergence noise."
            )
        },
    )
    calculation_span_tokens: int = field(
        default=64,
        metadata={"help": "Maximum final arithmetic/symbolic merge candidate tokens around AnswerSpan."},
    )
    support_span_tokens: int = field(
        default=96,
        metadata={"help": "Maximum token window immediately before AnswerSpan used for SupportSpan."},
    )
    reason_span_tokens: int = field(
        default=128,
        metadata={
            "help": (
                "Maximum structured reasoning-step candidate tokens before the answer. "
                "These are not distilled unless the privileged teacher ranks them inside active correct spans."
            )
        },
    )
    support_start_ratio: float = field(
        default=0.35,
        metadata={"help": "Training progress ratio after which SupportSpan is enabled."},
    )
    reason_start_ratio: float = field(
        default=0.45,
        metadata={"help": "Training progress ratio after which broader ReasonStepSpan candidates are enabled."},
    )
    span_evidence_floor: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Optional shared floor for teacher rank evidence. When >=0, it overrides both "
                "span_rank_floor and teacher_step_rank_floor, reducing duplicated ranking hyperparameters."
            )
        },
    )
    span_rank_floor: float = field(
        default=0.10,
        metadata={"help": "Minimum rank-normalized teacher span weight for active correct spans."},
    )
    span_gain_weight_floor: float = field(
        default=1.0,
        metadata={
            "help": (
                "Minimum multiplicative weight for privileged-gain span evidence. The default 1.0 disables "
                "gain-shaped reweighting; lower values downweight teacher-unsupported candidate spans without "
                "hard-filtering them."
            )
        },
    )
    span_gain_weight_scale: float = field(
        default=1.0,
        metadata={"help": "Temperature scale for privileged-gain span evidence reweighting."},
    )
    span_traj_normalize: bool = field(
        default=False,
        metadata={
            "help": (
                "Normalize positive span weights at the trajectory level before computing span loss. "
                "This keeps span distillation as local credit assignment instead of letting long spans "
                "behave like sequence-level SFT and hurt pass diversity."
            )
        },
    )
    span_max_traj_weight: float = field(
        default=1.0,
        metadata={"help": "Maximum summed positive span weight per trajectory when span_traj_normalize is enabled."},
    )
    final_span_weight: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Optional merged prior for AnswerSpan and CalculationSpan. When >=0, the trainer uses "
                "a deduplicated FinalSpan instead of separately weighting answer/calculation spans."
            )
        },
    )
    contrast_span_weight: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Optional merged prior for correct-vs-wrong contrast spans. When >=0, it overrides "
                "diff_span_weight."
            )
        },
    )
    context_span_weight: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Optional merged prior for SupportSpan and ReasonSpan. When >=0, the trainer uses "
                "a deduplicated ContextSpan instead of separately weighting support/reason spans."
            )
        },
    )
    answer_span_weight: float = field(
        default=1.0,
        metadata={"help": "Prior weight for AnswerSpan tokens."},
    )
    diff_span_weight: float = field(
        default=1.0,
        metadata={"help": "Prior weight for DiffSpan tokens."},
    )
    calculation_span_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Prior weight for final arithmetic/symbolic merge spans near the answer. "
                "Teacher scoring still reranks these candidates and verifier advantage controls sign."
            )
        },
    )
    support_span_weight: float = field(
        default=0.5,
        metadata={"help": "Prior weight for SupportSpan tokens."},
    )
    reason_span_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Prior weight for broader structured reasoning-step spans. "
                "Use a small value so teacher scoring expands positive distillation without becoming SFT."
            )
        },
    )
    use_privileged_gain: bool = field(
        default=True,
        metadata={
            "help": (
                "Score spans by teacher privileged gain logP_T(.|x,r)-logP_T(.|x). "
                "False uses privileged teacher log probability only."
            )
        },
    )
    teacher_step_window: int = field(
        default=33,
        metadata={"help": "Local token window for OPSD-style teacher step calibration inside selected spans."},
    )
    teacher_student_gap_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for OPSD-style logP_teacher(token)-logP_student(token) evidence "
                "inside span scoring. Direction still comes only from verifier advantage."
            )
        },
    )
    teacher_privileged_confidence_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Small coefficient for centered privileged-teacher log-prob confidence inside span scoring. "
                "This helps a stronger teacher provide magnitude evidence even when privileged gain is compressed; "
                "verifier rewards still control update sign."
            )
        },
    )
    mixed_balance_bonus: float = field(
        default=0.0,
        metadata={
            "help": (
                "Extra multiplier for balanced mixed correct/wrong groups, proportional to 4p(1-p). "
                "This increases effective training density without assigning gradients to all-correct/all-wrong groups."
            )
        },
    )
    near_full_negative_ratio: float = field(
        default=1.0,
        metadata={
            "help": (
                "Treat wrong rollouts with length >= ratio * max_completion_length as near-truncated negatives. "
                "Set below 1.0 to penalize nearly full no-answer outputs before they exactly hit the cap."
            )
        },
    )
    teacher_group_support_floor: float = field(
        default=0.10,
        metadata={
            "help": (
                "Minimum teacher support multiplier for verified-correct trajectories in mixed groups. "
                "Teacher support is computed within each prompt against wrong rollouts and only changes magnitude."
            )
        },
    )
    teacher_support_max: float = field(
        default=1.45,
        metadata={
            "help": (
                "Maximum teacher support multiplier applied to verified-correct trajectories. "
                "This prevents teacher magnitude calibration from concentrating probability mass too aggressively."
            )
        },
    )
    teacher_grpo_magnitude_coef: float = field(
        default=1.0,
        metadata={
            "help": (
                "How much teacher same-prompt support should scale the trajectory-level GRPO magnitude. "
                "The verifier still controls sign; teacher support only changes the magnitude of correct rollouts."
            )
        },
    )
    teacher_correct_length_dampen: float = field(
        default=0.0,
        metadata={
            "help": (
                "Reduce teacher magnitude amplification on long verified-correct trajectories. "
                "Verifier still controls sign; this only avoids over-amplifying verbose accidental positives."
            )
        },
    )
    teacher_step_rank_floor: float = field(
        default=0.05,
        metadata={"help": "Minimum local step rank weight inside active teacher-scored spans."},
    )
    teacher_token_weight_clip: float = field(
        default=1.0,
        metadata={"help": "Maximum local teacher step weight multiplier. Set <=0 to disable clipping."},
    )
    trace_context_topk: int = field(
        default=0,
        metadata={
            "help": (
                "For ContextSpan/ReasonSpan, keep only the top-k teacher-ranked contiguous trace windows per "
                "correct trajectory. Set <=0 to keep the full context mask."
            )
        },
    )
    trace_context_max_tokens: int = field(
        default=64,
        metadata={"help": "Maximum total context/reason tokens kept per trajectory after top-k trace routing."},
    )
    teacher_score_micro_batch: int = field(
        default=2,
        metadata={
            "help": (
                "Micro-batch size for teacher scoring forward passes. This is important when using "
                "a larger separate teacher model because full-vocab logits dominate peak memory."
            )
        },
    )
    positive_teacher_span_only: bool = field(
        default=True,
        metadata={
            "help": (
                "Only distill correct spans when the teacher gives positive same-prompt support. "
                "Verifier still controls sign; teacher support only opens or scales span magnitude."
            )
        },
    )
    positive_reason_span_only: bool = field(
        default=True,
        metadata={
            "help": (
                "For broader ReasonStepSpan candidates, require positive privileged gain before distillation. "
                "This keeps the extra reasoning signal from becoming noisy full-trajectory imitation."
            )
        },
    )
    no_answer_full_length_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Negative-advantage multiplier for full-length wrong rollouts that do not contain an answer marker. "
                "Values below 1 guard noisy truncations; values above 1 actively penalize rambling no-answer truncations "
                "while leaving ambiguous truncations controlled by truncated_negative_weight."
            )
        },
    )
    verifier_confidence_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Apply conservative confidence multipliers to verifier advantages for high-risk cases. "
                "This does not change reward sign; it only reduces magnitude where extraction/verifier noise is likely."
            )
        },
    )
    verifier_answer_marker_negative_confidence: float = field(
        default=0.85,
        metadata={
            "help": (
                "Confidence multiplier for negative-advantage trajectories that contain an explicit answer marker. "
                "These can include verifier/extraction false negatives, so the negative sign is softened."
            )
        },
    )
    verifier_truncated_answer_negative_confidence: float = field(
        default=0.65,
        metadata={
            "help": (
                "Confidence multiplier for near/full-length negative trajectories that still contain an answer marker. "
                "This is the most ambiguous false-negative-prone subset."
            )
        },
    )
    verifier_correct_no_answer_confidence: float = field(
        default=0.75,
        metadata={
            "help": (
                "Confidence multiplier for positive-advantage trajectories without an explicit final answer marker. "
                "This prevents accidental verifier positives from dominating span distillation."
            )
        },
    )
    span_debug_rate: float = field(
        default=0.01,
        metadata={"help": "Probability of printing one online span/course debug sample."},
    )
