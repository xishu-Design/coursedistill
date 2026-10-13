import copy
import json
import os
import random
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import disable_dropout_in_model, empty_cache, ensure_master_addr_port

from data_collator import LiteCIEDataCollator
from verifier import verify_response


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


class EMAUpdateCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class VLLMSyncCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
            and getattr(self.trainer.accelerator, "sync_gradients", True)
        ):
            self.trainer._move_model_to_vllm()
            self.trainer._last_vllm_sync_step = state.global_step


def token_entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = logits / temperature
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def topk_entropy_from_logits(logits: torch.Tensor, k: int = 128, temperature: float = 1.0) -> torch.Tensor:
    k = min(k, logits.shape[-1])
    topk_logits, _ = torch.topk(logits / temperature, k=k, dim=-1)
    topk_log_probs = F.log_softmax(topk_logits, dim=-1)
    topk_probs = torch.exp(topk_log_probs)
    return -(topk_probs * topk_log_probs).sum(dim=-1)


def flatten_completion_ids(completion_ids):
    if not completion_ids:
        return completion_ids
    first = completion_ids[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        return [ids for group in completion_ids for ids in group]
    return completion_ids


class LiteCIEOPDTrainer(SFTTrainer):
    _tag_names = ["trl", "lite-cie-opd"]
    _name = "Lite-CIE-OPD"

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        fixed_teacher: bool = False,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        num_student_samples: int = 4,
        ladder_steps: int = 4,
        teacher_entropy_top_k: int = 128,
        model_revision: str | None = None,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = model_revision
        if args is not None:
            dataset_kwargs = dict(getattr(args, "dataset_kwargs", None) or {})
            dataset_kwargs["skip_prepare_dataset"] = True
            args.dataset_kwargs = dataset_kwargs
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        if data_collator is None:
            data_collator = LiteCIEDataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                ladder_steps=ladder_steps,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(self.model)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.fixed_teacher = fixed_teacher
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None
        self.teacher_thinking = teacher_thinking
        self.num_student_samples = max(1, int(num_student_samples))
        self.lite_cie_ladder_steps = max(1, int(ladder_steps))
        self.teacher_entropy_top_k = max(0, int(teacher_entropy_top_k or 0))
        self.student_entropy_coef = max(0.0, float(getattr(args, "student_entropy_coef", 0.0) or 0.0))
        self.sdpo_preference_coef = max(0.0, float(getattr(args, "sdpo_preference_coef", 0.0) or 0.0))
        self.sdpo_distill_coef = max(0.0, float(getattr(args, "sdpo_distill_coef", 0.0) or 0.0))
        self.sdpo_require_logp_causal_gate = bool(getattr(args, "sdpo_require_logp_causal_gate", True))
        self.sdpo_low_entropy_distill = bool(getattr(args, "sdpo_low_entropy_distill", True))
        self.selective_transition_search = True
        self.marginal_ablation_buffer_size = 128

        if self.fixed_teacher and peft_config is None:
            raise ValueError("fixed_teacher=True requires a PEFT config; pass --use_peft.")
        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError("use_ema_teacher and fixed_teacher are mutually exclusive.")
        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))

        self._on_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._ablation_piece_buffer = deque(maxlen=self.marginal_ablation_buffer_size)
        self.transition_prompt = (
            "\n\nUse the available context only as evidence. Now solve the problem in your own words. "
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
        }

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if getattr(self.model.generation_config, "eos_token_id", None) is not None:
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        self._setup_vllm(args)

        if self.accelerator.is_main_process:
            entropy_mode = "full-vocab" if self.teacher_entropy_top_k == 0 else f"top-{self.teacher_entropy_top_k}"
            print(f"\n{'=' * 80}")
            print("LITE-CIE-OPD MODE ENABLED")
            print(f"Student samples per problem M: {self.num_student_samples}")
            print(f"Privileged ladder steps K: {self.lite_cie_ladder_steps}")
            print(f"Teacher entropy logging: {entropy_mode}")
            print(f"Student entropy coefficient: {self.student_entropy_coef}")
            print(f"SDPO preference coefficient: {self.sdpo_preference_coef}")
            print(f"SDPO distillation coefficient: {self.sdpo_distill_coef}")
            print(f"SDPO requires logp-causal group gate: {self.sdpo_require_logp_causal_gate}")
            print(f"SDPO low-entropy dense distillation: {self.sdpo_low_entropy_distill}")
            print(
                "Advanced utility defaults: "
                f"search={self.selective_transition_search}, "
                "dynamic_scales=True, dynamic_ablation=True, "
                "dynamic_counterfactual=True, dynamic_length=True"
            )
            print(f"{'=' * 80}\n")

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if dataset is not None and hasattr(dataset, "column_names"):
            columns = set(dataset.column_names)
            if {"student_input", "privileged_ladder", "ground_truth"}.issubset(columns):
                return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)

    def _setup_vllm(self, args):
        self.use_vllm = args.use_vllm
        if not self.use_vllm:
            return
        if not is_vllm_available():
            raise ImportError("vLLM is not available. Install vLLM or remove --use_vllm.")

        self.vllm_mode = args.vllm_mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
        self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
        self.vllm_sync_frequency = args.vllm_sync_frequency
        self._last_vllm_sync_step = -1

        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    host=args.vllm_server_host,
                    server_port=args.vllm_server_port,
                    connection_timeout=args.vllm_server_timeout,
                )
                self.vllm_client.init_communicator()
        elif self.vllm_mode == "colocate":
            if self.accelerator.num_processes % self.vllm_tensor_parallel_size != 0:
                raise ValueError(
                    "vllm_tensor_parallel_size must divide the accelerate world size evenly."
                )

            if self.vllm_tensor_parallel_size > 1:
                self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(
                            range(
                                i * self.vllm_tensor_parallel_size,
                                (i + 1) * self.vllm_tensor_parallel_size,
                            )
                        )
                        for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                    ]
                )

            os.environ["RANK"] = str(self.accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
            ensure_master_addr_port()

            self.vllm_engine = LLM(
                model=self.model_name_or_path,
                revision=self.model_revision,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_num_seqs=self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * self.num_student_samples,
                max_model_len=args.max_length,
                distributed_executor_backend="external_launcher",
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                enable_sleep_mode=self.vllm_enable_sleep_mode,
            )
            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        self.add_callback(VLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "id",
            "student_input",
            "privileged_ladder",
            "ground_truth",
            "verifier_type",
            "problem",
            "solution",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _update_ema(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        named_params = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            params_list = [param for _, param in named_params]
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=None):
                if self._ema_params is None:
                    self._ema_params = {
                        name: param.detach().float().cpu().clone() for name, param in named_params
                    }
                    return
                for name, param in named_params:
                    ema = self._ema_params[name]
                    ema.mul_(self.ema_decay).add_(
                        param.detach().float().cpu(),
                        alpha=1.0 - self.ema_decay,
                    )
        else:
            if self._ema_params is None:
                self._ema_params = {
                    name: param.detach().clone() for name, param in named_params
                }
                return
            for name, param in named_params:
                ema = self._ema_params[name]
                if ema.device != param.device:
                    ema = ema.to(param.device)
                    self._ema_params[name] = ema
                ema.mul_(self.ema_decay).add_(param.detach(), alpha=1.0 - self.ema_decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        if self._ema_params is None:
            yield
            return

        unwrapped = self.accelerator.unwrap_model(model)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            with deepspeed.zero.GatheredParameters(list(name_to_param.values()), modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name].to(param.device)
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.device:
                    ema = ema.to(param.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def _teacher_adapter_context(self, model):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                if self.is_fsdp_enabled:
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    for name, param in self.model.named_parameters():
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name or "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                self.model.unmerge_adapter()
        else:
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _vllm_stop_token_ids(self) -> list[int]:
        token_ids = set()
        for token_id in (
            getattr(self.processing_class, "eos_token_id", None),
            getattr(self.model.generation_config, "eos_token_id", None),
            getattr(self.generation_config, "eos_token_id", None),
        ):
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(token_id)
            elif isinstance(token_id, (list, tuple)):
                token_ids.update(int(item) for item in token_id if item is not None and int(item) >= 0)

        for token in ("<|im_end|>", "<|endoftext|>"):
            try:
                token_id = self.processing_class.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.processing_class.unk_token_id:
                token_ids.add(token_id)
        return sorted(token_ids)

    def _save_generation_outputs(self, step: int):
        if not self.accelerator.is_main_process or not self._generation_outputs_buffer:
            return
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        output_file = generations_dir / f"generations_step_{step}.json"
        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }
        output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._generation_outputs_buffer.clear()

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        if self.teacher_entropy_top_k > 0:
            return topk_entropy_from_logits(logits, k=self.teacher_entropy_top_k, temperature=self.temperature)
        return token_entropy_from_logits(logits, temperature=self.temperature)

    def _metric(self, name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = value.detach().float().mean().item()
        self._metrics["train"][name].append(float(value))

    def _tokenize_contexts(self, texts: list[str], device: torch.device, completion_len: int):
        prompt_max_length = None
        if self.args.max_length:
            prompt_max_length = max(1, self.args.max_length - completion_len)
        return self.processing_class(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)

    def _apply_chat_template(self, messages, enable_thinking: bool) -> str:
        try:
            return self.processing_class.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.processing_class.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _teacher_context_prompt(self, student_input: str, evidence: str) -> str:
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

    def _marginal_ablation_enabled(self) -> bool:
        return (
            self.marginal_ablation_buffer_size > 0
            and len(self._ablation_piece_buffer) > 0
        )

    def _build_ablation_next_texts(self, inputs, transition_j: int, parent_list: list[int]) -> list[str] | None:
        if not self._marginal_ablation_enabled():
            return None

        student_inputs = inputs.get("student_inputs")
        ladder_pieces = inputs.get("teacher_ladder_pieces")
        if not student_inputs or not ladder_pieces:
            return None

        ablated_texts = []
        buffer_items = list(self._ablation_piece_buffer)
        for row_index, parent in enumerate(parent_list):
            own_pieces = list(ladder_pieces[parent])
            donor = buffer_items[(int(self.state.global_step) + transition_j + row_index) % len(buffer_items)]
            donor_pieces = donor.get("ladder", [])
            donor_piece = ""
            if transition_j - 1 < len(donor_pieces):
                donor_piece = str(donor_pieces[transition_j - 1]).strip()
            if not donor_piece:
                donor_piece = next((str(piece).strip() for piece in donor_pieces if str(piece).strip()), "")
            if not donor_piece:
                return None

            evidence_pieces = [piece for piece in own_pieces[: transition_j - 1] if piece]
            evidence_pieces.append(f"ablated z{transition_j} shuffled_evidence: {donor_piece}")
            evidence = "\n".join(evidence_pieces)
            ablated_texts.append(self._teacher_context_prompt(student_inputs[parent], evidence))
        return ablated_texts

    def _update_ablation_piece_buffer(self, inputs) -> None:
        if self.marginal_ablation_buffer_size <= 0:
            return
        student_inputs = inputs.get("student_inputs") or []
        ladder_pieces = inputs.get("teacher_ladder_pieces") or []
        sample_ids = inputs.get("sample_ids") or [""] * len(ladder_pieces)
        for sample_id, student_input, ladder in zip(sample_ids, student_inputs, ladder_pieces):
            self._ablation_piece_buffer.append(
                {
                    "sample_id": str(sample_id),
                    "student_input": str(student_input),
                    "ladder": list(ladder),
                }
            )

    def _build_transition_candidates(
        self,
        inputs,
        parent_indices: torch.Tensor,
        completion_len: int,
        generation_ids: torch.Tensor,
    ):
        device = parent_indices.device
        contexts = inputs["teacher_context_texts"]
        local_k = max(1, min(self.lite_cie_ladder_steps, len(contexts[0]) - 1))
        if self.selective_transition_search:
            transition_js = list(range(1, local_k + 1))
        else:
            transition_js = [1 + (int(self.state.global_step) % local_k)]

        parent_list = parent_indices.detach().cpu().tolist()
        candidates = []
        for transition_j in transition_js:
            prev_texts = [contexts[parent][transition_j - 1] for parent in parent_list]
            next_texts = [contexts[parent][transition_j] for parent in parent_list]
            prev_encoded = self._tokenize_contexts(prev_texts, device, completion_len)
            next_encoded = self._tokenize_contexts(next_texts, device, completion_len)
            ablated_next_texts = self._build_ablation_next_texts(inputs, transition_j, parent_list)
            ablated_next_encoded = (
                self._tokenize_contexts(ablated_next_texts, device, completion_len)
                if ablated_next_texts is not None
                else None
            )

            prev_ids = torch.cat([prev_encoded.input_ids, generation_ids], dim=1)
            next_ids = torch.cat([next_encoded.input_ids, generation_ids], dim=1)
            prev_attention = torch.ones_like(prev_ids)
            next_attention = torch.ones_like(next_ids)
            if self.processing_class.pad_token_id is not None:
                prev_attention[prev_ids == self.processing_class.pad_token_id] = 0
                next_attention[next_ids == self.processing_class.pad_token_id] = 0

            candidate = {
                "j": transition_j,
                "prev_input_ids": prev_ids,
                "prev_attention_mask": prev_attention,
                "prev_prompt_length": prev_encoded.input_ids.shape[1],
                "next_input_ids": next_ids,
                "next_attention_mask": next_attention,
                "next_prompt_length": next_encoded.input_ids.shape[1],
            }
            if ablated_next_encoded is not None:
                ablated_next_ids = torch.cat([ablated_next_encoded.input_ids, generation_ids], dim=1)
                ablated_next_attention = torch.ones_like(ablated_next_ids)
                if self.processing_class.pad_token_id is not None:
                    ablated_next_attention[ablated_next_ids == self.processing_class.pad_token_id] = 0
                candidate.update(
                    {
                        "ablated_next_input_ids": ablated_next_ids,
                        "ablated_next_attention_mask": ablated_next_attention,
                        "ablated_next_prompt_length": ablated_next_encoded.input_ids.shape[1],
                    }
                )
            candidates.append(candidate)
        return candidates

    def _make_labels(self, inputs, generated_ids: torch.Tensor, parent_indices: torch.Tensor):
        labels = torch.full_like(generated_ids, -100)
        prompt_len = int(inputs["student_prompt_length"])
        labels[:, prompt_len:] = generated_ids[:, prompt_len:]

        if "completion_lengths_per_sequence" in inputs:
            completion_lengths = inputs["completion_lengths_per_sequence"].to(generated_ids.device)
            for row in range(labels.shape[0]):
                completion_len = int(completion_lengths[row].item())
                labels[row, prompt_len + completion_len :] = -100
        elif "student_attention_mask" in inputs:
            labels[inputs["student_attention_mask"].to(generated_ids.device) == 0] = -100
        elif self.processing_class.pad_token_id is not None:
            completion_labels = labels[:, prompt_len:]
            completion_labels[generated_ids[:, prompt_len:] == self.processing_class.pad_token_id] = -100
            labels[:, prompt_len:] = completion_labels
        return labels

    def _compute_outcomes(self, inputs, completion_texts: list[str], parent_indices: torch.Tensor):
        parent_list = parent_indices.detach().cpu().tolist()
        ground_truths = inputs["ground_truth"]
        verifier_types = inputs["verifier_type"]
        outcomes = []
        for completion, parent in zip(completion_texts, parent_list):
            outcomes.append(verify_response(completion, ground_truths[parent], verifier_types[parent]))
        return torch.tensor(outcomes, device=parent_indices.device, dtype=torch.bool)

    def _repeat_prompt_texts(self, inputs):
        prompt_texts = self.processing_class.batch_decode(inputs["student_prompts"], skip_special_tokens=False)
        repeated = []
        for prompt in prompt_texts:
            repeated.extend([prompt] * self.num_student_samples)
        return repeated

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        import time

        start_time = time.time()
        generation_config = copy.deepcopy(generation_config)
        generation_config.num_return_sequences = self.num_student_samples

        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache
        model.config.use_cache = True
        generation_config.use_cache = True

        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            generated_tokens = generated_outputs.sequences
        finally:
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        total_completion_tokens = generated_tokens.shape[0] * (
            generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        )
        print(
            f"generation done - elapsed: {elapsed_time:.2f}s, prompts: {inputs['student_prompts'].shape[0]}, "
            f"samples: {generated_tokens.shape[0]}, completion tokens: {total_completion_tokens}"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        batch_size = inputs["student_prompts"].shape[0]
        parent_indices = torch.arange(batch_size, device=generated_tokens.device).repeat_interleave(
            self.num_student_samples
        )
        prompt_lengths = inputs["student_prompt_lengths_per_example"].to(generated_tokens.device)[parent_indices]
        prompt_len = int(inputs["student_prompt_length"])
        completion_ids = generated_tokens[:, prompt_len:]
        stop_token_ids = set(self._vllm_stop_token_ids())
        completion_lengths = []
        for row in range(completion_ids.shape[0]):
            row_ids = completion_ids[row].detach().cpu().tolist()
            completion_len = len(row_ids)
            for token_idx, token_id in enumerate(row_ids):
                if int(token_id) in stop_token_ids:
                    completion_len = token_idx + 1
                    break
                if pad_token_id is not None and int(token_id) == int(pad_token_id):
                    completion_len = token_idx
                    break
            completion_lengths.append(completion_len)
        completion_lengths = torch.tensor(completion_lengths, device=generated_tokens.device, dtype=torch.long)
        return (
            generated_tokens,
            new_attention_mask,
            new_labels,
            parent_indices,
            inputs["student_prompt_length"],
            prompt_lengths,
            completion_lengths,
        )

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        import time

        device = self.accelerator.device
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                prompt.replace(self.processing_class.pad_token, "") for prompt in prompts_text_for_vllm
            ]

        max_completion_length = generation_config.max_new_tokens
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        top_p = getattr(self.args, "top_p", 1.0)
        repetition_penalty = getattr(self.args, "repetition_penalty", 1.0)
        min_p = getattr(self.args, "min_p", 0.0)
        presence_penalty = getattr(self.args, "presence_penalty", 0.0)

        start_time = time.time()
        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=self.num_student_samples,
                    repetition_penalty=repetition_penalty,
                    temperature=generation_config.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * (len(all_prompts_text) * self.num_student_samples)
            completion_ids = flatten_completion_ids(broadcast_object_list(completion_ids, from_process=0))
            local_count = len(prompts_text_for_vllm) * self.num_student_samples
            process_slice = slice(
                self.accelerator.process_index * local_count,
                (self.accelerator.process_index + 1) * local_count,
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            guided_decoding = None
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines",
                    regex=self.vllm_guided_decoding_regex,
                )
            sampling_params = SamplingParams(
                n=self.num_student_samples,
                repetition_penalty=repetition_penalty,
                temperature=generation_config.temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
                stop_token_ids=self._vllm_stop_token_ids(),
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group)
                all_prompts_text = [prompt for sublist in gathered_prompts for prompt in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(
                    local_rank_in_group * orig_size * self.num_student_samples,
                    (local_rank_in_group + 1) * orig_size * self.num_student_samples,
                )
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        if self.accelerator.is_main_process:
            print(
                f"vLLM generation done - elapsed: {elapsed_time:.2f}s, prompts: {len(prompts_text_for_vllm)}, "
                f"samples: {len(completion_ids)}, completion tokens: {total_completion_tokens}"
            )

        prompt_texts_for_sequences = []
        for prompt in prompts_text_for_vllm:
            prompt_texts_for_sequences.extend([prompt] * self.num_student_samples)

        prompt_tokenized = self.processing_class(
            prompt_texts_for_sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True if self.args.max_length else False,
            max_length=max(1, self.args.max_length - max_completion_length) if self.args.max_length else None,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids
        prompt_lengths = prompt_tokenized.attention_mask.sum(dim=-1)

        padded_completion_ids = []
        completion_lengths = []
        for ids in completion_ids:
            completion_len = min(len(ids), max_completion_length)
            completion_lengths.append(completion_len)
            tensor = torch.tensor(ids, device=device, dtype=prompt_ids.dtype)
            if len(tensor) > max_completion_length:
                tensor = tensor[:max_completion_length]
            elif len(tensor) < max_completion_length:
                padding_needed = max_completion_length - len(tensor)
                tensor = torch.cat(
                    [
                        tensor,
                        torch.full((padding_needed,), pad_token_id, device=device, dtype=tensor.dtype),
                    ]
                )
            padded_completion_ids.append(tensor)

        completion_tensor = torch.stack(padded_completion_ids)
        new_input_ids = torch.cat([prompt_ids, completion_tensor], dim=1)
        completion_lengths = torch.tensor(completion_lengths, device=device, dtype=torch.long)
        completion_positions = torch.arange(max_completion_length, device=device).unsqueeze(0)
        completion_attention_mask = (completion_positions < completion_lengths.unsqueeze(-1)).to(prompt_ids.dtype)
        new_attention_mask = torch.cat([prompt_tokenized.attention_mask, completion_attention_mask], dim=1)
        new_labels = new_input_ids.clone()

        batch_size = inputs["student_prompts"].shape[0]
        parent_indices = torch.arange(batch_size, device=device).repeat_interleave(self.num_student_samples)
        completion_texts = [self.processing_class.decode(ids, skip_special_tokens=False) for ids in completion_ids]
        prompt_texts_with_special = self._repeat_prompt_texts(inputs)
        return (
            new_input_ids,
            new_attention_mask,
            new_labels,
            prompt_texts_with_special,
            completion_texts,
            parent_indices,
            prompt_ids.shape[1],
            prompt_lengths,
            completion_lengths,
        )

    @profiling_decorator
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
        prompt_lengths_per_sequence = prompt_lengths_per_sequence.to(generated_ids.device)
        completion_lengths_per_sequence = completion_lengths_per_sequence.to(generated_ids.device)
        inputs["student_prompt_length"] = int(rollout_prompt_len)
        inputs["student_prompt_lengths_per_sequence"] = prompt_lengths_per_sequence
        inputs["completion_lengths_per_sequence"] = completion_lengths_per_sequence
        generation_ids = generated_ids[:, inputs["student_prompt_length"] :]
        completion_len = generation_ids.shape[1]

        transition_candidates = self._build_transition_candidates(
            inputs,
            parent_indices,
            completion_len,
            generation_ids,
        )
        self._update_ablation_piece_buffer(inputs)
        transition_label = "search" if self.selective_transition_search else transition_candidates[0]["j"]

        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask
        inputs["labels"] = self._make_labels(inputs, generated_ids, parent_indices)
        inputs["transition_candidates"] = transition_candidates
        inputs["sample_parent_indices"] = parent_indices
        inputs["outcomes"] = self._compute_outcomes(inputs, completion_texts, parent_indices)
        inputs["transition_j"] = transition_label
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
                    "transition_j": transition_label,
                    "outcome": bool(outcome),
                    "prompt": prompt,
                    "completion": completion,
                }
            )

        if random.random() < 0.01 and completion_texts:
            sample_idx = random.randint(0, len(completion_texts) - 1)
            print(f"\n{'=' * 80}")
            print(f"LITE-CIE STUDENT SAMPLE (Step {self.state.global_step}, j={transition_label}):")
            print(prompt_texts[sample_idx])
            print(completion_texts[sample_idx])
            print(f"outcome={bool(inputs['outcomes'][sample_idx].item())}")
            print(f"{'=' * 80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)

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

    def _compute_group_gates(
        self,
        e_traj: torch.Tensor,
        outcomes: torch.Tensor,
        parent_indices: torch.Tensor,
        num_groups: int,
    ):
        gates = torch.zeros(num_groups, device=e_traj.device, dtype=e_traj.dtype)
        u_values = torch.zeros(num_groups, device=e_traj.device, dtype=e_traj.dtype)
        valid = torch.zeros(num_groups, device=e_traj.device, dtype=torch.bool)
        pos_means = []
        neg_means = []

        for group_idx in range(num_groups):
            group_mask = parent_indices == group_idx
            if not torch.any(group_mask):
                continue
            group_outcomes = outcomes[group_mask]
            group_e = e_traj[group_mask]
            pos_mask = group_outcomes
            neg_mask = ~group_outcomes
            if pos_mask.any():
                pos_means.append(group_e[pos_mask].mean())
            if neg_mask.any():
                neg_means.append(group_e[neg_mask].mean())
            if pos_mask.any() and neg_mask.any():
                valid[group_idx] = True
                u_value = group_e[pos_mask].mean() - group_e[neg_mask].mean()
                u_values[group_idx] = u_value
                gates[group_idx] = (u_value > 0).to(e_traj.dtype)

        pos_mean = torch.stack(pos_means).mean() if pos_means else None
        neg_mean = torch.stack(neg_means).mean() if neg_means else None
        return gates, u_values, valid, pos_mean, neg_mean

    def _transition_selection_scores(
        self,
        e_traj: torch.Tensor,
        drop_traj: torch.Tensor,
        outcomes: torch.Tensor,
        parent_indices: torch.Tensor,
        u_values: torch.Tensor,
        valid_groups: torch.Tensor,
        num_groups: int,
        ablated_e_traj: torch.Tensor | None = None,
        ablated_drop_traj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.full(
            (num_groups,),
            -torch.inf,
            device=drop_traj.device,
            dtype=drop_traj.dtype,
        )
        final_utilities = torch.zeros(num_groups, device=drop_traj.device, dtype=drop_traj.dtype)
        drop_gaps = torch.zeros(num_groups, device=drop_traj.device, dtype=drop_traj.dtype)
        counterfactual_penalties = torch.zeros(num_groups, device=drop_traj.device, dtype=drop_traj.dtype)
        ablation_null_scores = torch.zeros(num_groups, device=drop_traj.device, dtype=drop_traj.dtype)
        marginal_deltas = torch.zeros(num_groups, device=drop_traj.device, dtype=drop_traj.dtype)
        has_ablation = ablated_e_traj is not None and ablated_drop_traj is not None
        valid_u = u_values[valid_groups]
        logp_scale = (
            valid_u.detach().abs().mean().clamp(min=1e-6)
            if valid_u.numel() > 0
            else torch.ones((), device=drop_traj.device, dtype=drop_traj.dtype)
        )
        drop_scale = drop_traj.detach().abs().mean().clamp(min=1e-6)

        for group_idx in range(num_groups):
            if not bool(valid_groups[group_idx].item()):
                continue
            group_mask = parent_indices == group_idx
            group_outcomes = outcomes[group_mask]
            group_e = e_traj[group_mask]
            group_drop = drop_traj[group_mask]
            pos_mask = group_outcomes
            neg_mask = ~group_outcomes
            if not (pos_mask.any() and neg_mask.any()):
                continue
            drop_gap = group_drop[pos_mask].mean() - group_drop[neg_mask].mean()
            drop_gaps[group_idx] = drop_gap
            raw_score = 0.5 * ((u_values[group_idx] / logp_scale) + (drop_gap / drop_scale))
            score = raw_score

            if has_ablation:
                group_ablate_e = ablated_e_traj[group_mask]
                group_ablate_drop = ablated_drop_traj[group_mask]
                ablate_u = group_ablate_e[pos_mask].mean() - group_ablate_e[neg_mask].mean()
                ablate_drop_gap = group_ablate_drop[pos_mask].mean() - group_ablate_drop[neg_mask].mean()
                ablate_score = 0.5 * ((ablate_u / logp_scale) + (ablate_drop_gap / drop_scale))
                ablation_null_scores[group_idx] = ablate_score
                marginal_delta = raw_score - ablate_score
                marginal_deltas[group_idx] = marginal_delta
                ablation_mix = ablate_score.abs() / (raw_score.abs() + ablate_score.abs() + 1e-6)
                score = (1.0 - ablation_mix) * raw_score + ablation_mix * marginal_delta

            wrong_logp_support = F.relu(group_e[neg_mask].mean())
            wrong_entropy_support = F.relu(group_drop[neg_mask].mean())
            counterfactual_penalty = 0.5 * (
                (wrong_logp_support / logp_scale) + (wrong_entropy_support / drop_scale)
            )
            counterfactual_penalties[group_idx] = counterfactual_penalty
            counterfactual_weight = (
                counterfactual_penalty / (score.detach().abs() + counterfactual_penalty + 1e-6)
            ).clamp(0.0, 1.0)
            score = score - counterfactual_weight * counterfactual_penalty

            final_utilities[group_idx] = score
            if bool((score > 0).item()):
                scores[group_idx] = score
        return scores, final_utilities, drop_gaps, counterfactual_penalties, ablation_null_scores, marginal_deltas

    def _gather_candidate_sequence(
        self,
        candidate_results: list[dict[str, Any]],
        key: str,
        selected_candidates_per_traj: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack([candidate[key] for candidate in candidate_results], dim=0)
        gather_idx = selected_candidates_per_traj.view(1, -1, 1).expand(1, stacked.shape[1], stacked.shape[2])
        return stacked.gather(0, gather_idx).squeeze(0)

    def _gather_candidate_trajectory(
        self,
        candidate_results: list[dict[str, Any]],
        key: str,
        selected_candidates_per_traj: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack([candidate[key] for candidate in candidate_results], dim=0)
        gather_idx = selected_candidates_per_traj.view(1, -1)
        return stacked.gather(0, gather_idx).squeeze(0)

    def _gather_candidate_group(
        self,
        candidate_results: list[dict[str, Any]],
        key: str,
        selected_candidates_per_group: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack([candidate[key] for candidate in candidate_results], dim=0)
        gather_idx = selected_candidates_per_group.view(1, -1)
        return stacked.gather(0, gather_idx).squeeze(0)

    def _compute_evidence_sdpo_distill_loss(
        self,
        student_log_probs_sampled: torch.Tensor,
        teacher_logp_j: torch.Tensor,
        response_mask: torch.Tensor,
        token_evidence_weight: torch.Tensor,
        outcomes: torch.Tensor,
        token_weights: torch.Tensor,
        teacher_entropy: torch.Tensor,
    ) -> torch.Tensor:
        zero = student_log_probs_sampled.sum() * 0.0
        if self.sdpo_distill_coef <= 0:
            return zero

        correct_mask = outcomes.to(dtype=student_log_probs_sampled.dtype).unsqueeze(-1)
        evidence_weights = token_evidence_weight.detach() * correct_mask * response_mask
        if self.sdpo_low_entropy_distill:
            entropy_values = teacher_entropy.detach()[response_mask.bool()]
            if entropy_values.numel() > 0:
                entropy_center = entropy_values.median()
                entropy_scale = entropy_values.sub(entropy_center).abs().mean().clamp(min=1e-6)
                low_entropy_gate = torch.sigmoid((entropy_center - teacher_entropy.detach()) / entropy_scale)
                evidence_weights = evidence_weights * low_entropy_gate
                self._metric("sdpo/low_entropy_gate_mean", low_entropy_gate[response_mask.bool()].mean())
        evidence_weights = evidence_weights * token_weights.detach().clamp(min=0.0)
        response_tokens = response_mask.sum().clamp(min=1.0)
        candidate_mask = evidence_weights > 0
        self._metric("sdpo/candidate_token_rate", candidate_mask.float().sum() / response_tokens)

        if not bool(candidate_mask.any().item()):
            self._metric("sdpo/distill_token_rate", zero)
            return zero

        teacher_student_gap = teacher_logp_j.detach() - student_log_probs_sampled.detach()
        gap_scale = teacher_student_gap[candidate_mask].abs().mean().clamp(min=1e-6)
        soft_gap_weight = torch.sigmoid(teacher_student_gap / gap_scale)
        evidence_scale = evidence_weights[candidate_mask].mean().clamp(min=1e-6)
        gap_center = soft_gap_weight[candidate_mask].mean().clamp(min=1e-6)
        evidence_strength = (evidence_weights / evidence_scale).clamp(max=4.0)
        gap_strength = (soft_gap_weight / gap_center).clamp(min=0.2, max=2.5)
        distill_weights = evidence_weights * evidence_strength * gap_strength
        active_mask = distill_weights > 0
        if not bool(active_mask.any().item()):
            self._metric("sdpo/distill_token_rate", zero)
            return zero

        distill_loss = -(
            student_log_probs_sampled * distill_weights
        ).sum() / distill_weights.sum().clamp(min=1e-6)

        active_traj = distill_weights.sum(dim=-1) > 0
        correct_traj = outcomes.bool()
        correct_active_rate = (
            (active_traj & correct_traj).float().sum() / correct_traj.float().sum().clamp(min=1.0)
        )
        self._metric("sdpo/distill_loss", distill_loss)
        self._metric("sdpo/distill_token_rate", active_mask.float().sum() / response_tokens)
        self._metric("sdpo/distill_weight_mean", distill_weights[active_mask].mean())
        self._metric("sdpo/distill_weight_max", distill_weights[active_mask].max())
        self._metric("sdpo/teacher_student_gap_mean", teacher_student_gap[candidate_mask].mean())
        self._metric("sdpo/soft_gap_weight_mean", soft_gap_weight[candidate_mask].mean())
        self._metric("sdpo/evidence_strength_mean", evidence_strength[candidate_mask].mean())
        self._metric("sdpo/gap_strength_mean", gap_strength[candidate_mask].mean())
        self._metric("sdpo/correct_rollout_active_rate", correct_active_rate)
        return distill_loss

    def _compute_sdpo_preference_loss(
        self,
        student_log_probs_sampled: torch.Tensor,
        response_mask: torch.Tensor,
        token_weights: torch.Tensor,
        outcomes: torch.Tensor,
        parent_indices: torch.Tensor,
        gates: torch.Tensor,
        e_traj: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        zero = student_log_probs_sampled.sum() * 0.0
        if self.sdpo_preference_coef <= 0:
            return zero

        traj_den = token_weights.sum(dim=-1).clamp(min=1.0)
        student_traj_logp = (student_log_probs_sampled * token_weights).sum(dim=-1) / traj_den
        active_losses = []
        group_weights = []
        teacher_margins = []
        policy_margins = []
        pair_count = zero

        for group_idx in range(num_groups):
            group_weight = gates[group_idx].detach()
            if not bool((group_weight > 0).item()):
                continue
            group_mask = parent_indices == group_idx
            if not torch.any(group_mask):
                continue
            group_outcomes = outcomes[group_mask]
            pos_mask = group_outcomes
            neg_mask = ~group_outcomes
            if not (pos_mask.any() and neg_mask.any()):
                continue

            group_logp = student_traj_logp[group_mask]
            group_e = e_traj[group_mask].detach()
            pos_logp = group_logp[pos_mask]
            neg_logp = group_logp[neg_mask]
            policy_margin = pos_logp[:, None] - neg_logp[None, :]
            margin_scale = policy_margin.detach().abs().mean().clamp(min=1e-3)
            pair_loss = F.softplus(-policy_margin / margin_scale).mean()
            active_losses.append(pair_loss * group_weight)
            group_weights.append(group_weight)
            teacher_margins.append(group_e[pos_mask].mean() - group_e[neg_mask].mean())
            policy_margins.append(policy_margin.detach().mean())
            pair_count = pair_count + policy_margin.numel()

        active_group_count = len(active_losses)
        active_group_rate = torch.tensor(
            active_group_count / max(1, num_groups),
            device=student_log_probs_sampled.device,
            dtype=student_log_probs_sampled.dtype,
        )
        self._metric("sdpo/active_group_rate", active_group_rate)
        self._metric("sdpo/pair_count", pair_count / max(1, num_groups))

        if active_group_count == 0:
            self._metric("sdpo/preference_loss", zero)
            self._metric("sdpo/group_weight_mean", zero)
            self._metric("sdpo/policy_margin_mean", zero)
            self._metric("sdpo/teacher_margin_mean", zero)
            return zero

        group_weight_tensor = torch.stack(group_weights)
        preference_loss = torch.stack(active_losses).sum() / group_weight_tensor.sum().clamp(min=1e-6)
        scaled_loss = preference_loss * active_group_rate
        teacher_margin_tensor = torch.stack(teacher_margins)
        policy_margin_tensor = torch.stack(policy_margins)
        self._metric("sdpo/preference_loss", preference_loss)
        self._metric("sdpo/group_weight_mean", group_weight_tensor.mean())
        self._metric("sdpo/policy_margin_mean", policy_margin_tensor.mean())
        self._metric("sdpo/teacher_margin_mean", teacher_margin_tensor.mean())
        return scaled_loss

    def _log_lite_cie_metrics(
        self,
        inputs,
        e_traj,
        outcomes,
        parent_indices,
        gates,
        u_values,
        valid_groups,
        pos_e_mean,
        neg_e_mean,
        h_prev_traj,
        h_j_traj,
        drop_traj,
        student_entropy_traj,
        response_lengths,
    ):
        num_groups = max(1, int(inputs["num_original_samples"]))
        gate_per_traj = gates[parent_indices]
        valid_count = valid_groups.float().sum()
        valid_u = u_values[valid_groups]
        zero = torch.tensor(0.0, device=e_traj.device)
        valid_group_rate = valid_count / num_groups
        gate_rate = gates.mean()
        u_positive_rate = (valid_u > 0).float().mean() if valid_u.numel() > 0 else zero
        accuracy = outcomes.float().mean()
        h_prev_mean = h_prev_traj.mean()
        h_j_mean = h_j_traj.mean()
        drop_mean = drop_traj.mean()
        student_entropy_mean = student_entropy_traj.mean()
        drop_correct = drop_traj[outcomes].mean() if outcomes.any() else None
        drop_wrong = drop_traj[~outcomes].mean() if (~outcomes).any() else None
        drop_gate_on = drop_traj[gate_per_traj > 0].mean() if (gate_per_traj > 0).any() else None
        drop_gate_off = drop_traj[gate_per_traj <= 0].mean() if (gate_per_traj <= 0).any() else None
        evidence_gap = pos_e_mean - neg_e_mean if pos_e_mean is not None and neg_e_mean is not None else None
        drop_correct_wrong_gap = (
            drop_correct - drop_wrong if drop_correct is not None and drop_wrong is not None else None
        )
        drop_gate_gap = drop_gate_on - drop_gate_off if drop_gate_on is not None and drop_gate_off is not None else None
        full_length_rate = (response_lengths >= self.args.max_completion_length).float().mean()

        self._metric("rollout/answer_accuracy", accuracy)
        self._metric("rollout/full_length_rate", full_length_rate)
        self._metric("opd/usable_group_rate", valid_group_rate)
        self._metric("opd/effective_signal_rate", gate_rate)
        self._metric("opd/u_positive_rate", u_positive_rate)
        self._metric("entropy/student_policy", student_entropy_mean)
        self._metric("entropy/student_minus_teacher_before", student_entropy_mean - h_prev_mean)
        self._metric("entropy/teacher_drop", drop_mean)
        self._metric("objective/anti_false_certainty_score", drop_gate_gap)
        self._metric("objective/self_opd_signal_score", evidence_gap)

        passk = []
        for group_idx in range(num_groups):
            group_mask = parent_indices == group_idx
            if torch.any(group_mask):
                passk.append(outcomes[group_mask].any().float())
        if passk:
            pass_at_m = torch.stack(passk).mean()
            self._metric("rollout/pass_at_m", pass_at_m)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        response_mask = (shifted_labels != -100).to(dtype=torch.float32)
        lengths = response_mask.sum(dim=-1).clamp(min=1.0)

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
        student_entropy = self._entropy(student_logits)
        del outputs_student, student_log_probs
        empty_cache()

        outcomes = inputs["outcomes"].to(student_logits.device)
        parent_indices = inputs["sample_parent_indices"].to(student_logits.device)
        num_groups = int(inputs["num_original_samples"])
        candidate_results = []

        with torch.no_grad(), self._teacher_adapter_context(model):
            for candidate in inputs["transition_candidates"]:
                outputs_prev = model(
                    input_ids=candidate["prev_input_ids"],
                    attention_mask=candidate["prev_attention_mask"],
                )
                teacher_logits_prev = outputs_prev.logits[:, candidate["prev_prompt_length"] - 1 : -1, :]
                teacher_logp_prev_all = F.log_softmax(teacher_logits_prev / self.temperature, dim=-1)
                teacher_logp_prev = torch.gather(
                    teacher_logp_prev_all,
                    dim=-1,
                    index=sampled_token_ids.unsqueeze(-1),
                ).squeeze(-1)
                h_prev = self._entropy(teacher_logits_prev)
                del outputs_prev, teacher_logits_prev, teacher_logp_prev_all
                empty_cache()

                outputs_next = model(
                    input_ids=candidate["next_input_ids"],
                    attention_mask=candidate["next_attention_mask"],
                )
                teacher_logits_j = outputs_next.logits[:, candidate["next_prompt_length"] - 1 : -1, :]
                teacher_logp_j_all = F.log_softmax(teacher_logits_j / self.temperature, dim=-1)
                teacher_logp_j = torch.gather(
                    teacher_logp_j_all,
                    dim=-1,
                    index=sampled_token_ids.unsqueeze(-1),
                ).squeeze(-1)
                h_j = self._entropy(teacher_logits_j)
                del outputs_next, teacher_logits_j, teacher_logp_j_all
                empty_cache()

                e_j = teacher_logp_j - teacher_logp_prev
                e_traj = (e_j * response_mask).sum(dim=-1) / lengths
                h_prev_traj = (h_prev * response_mask).sum(dim=-1) / lengths
                h_j_traj = (h_j * response_mask).sum(dim=-1) / lengths
                drop_token = h_prev - h_j
                drop_traj = (drop_token * response_mask).sum(dim=-1) / lengths
                ablated_e_traj = None
                ablated_drop_traj = None
                ablation_available = "ablated_next_input_ids" in candidate
                if ablation_available:
                    outputs_ablate = model(
                        input_ids=candidate["ablated_next_input_ids"],
                        attention_mask=candidate["ablated_next_attention_mask"],
                    )
                    teacher_logits_ablate = outputs_ablate.logits[
                        :, candidate["ablated_next_prompt_length"] - 1 : -1, :
                    ]
                    teacher_logp_ablate_all = F.log_softmax(
                        teacher_logits_ablate / self.temperature,
                        dim=-1,
                    )
                    teacher_logp_ablate = torch.gather(
                        teacher_logp_ablate_all,
                        dim=-1,
                        index=sampled_token_ids.unsqueeze(-1),
                    ).squeeze(-1)
                    h_ablate = self._entropy(teacher_logits_ablate)
                    del outputs_ablate, teacher_logits_ablate, teacher_logp_ablate_all
                    empty_cache()

                    ablated_e = teacher_logp_ablate - teacher_logp_prev
                    ablated_drop_token = h_prev - h_ablate
                    ablated_e_traj = (ablated_e * response_mask).sum(dim=-1) / lengths
                    ablated_drop_traj = (ablated_drop_token * response_mask).sum(dim=-1) / lengths
                    del teacher_logp_ablate, h_ablate, ablated_e, ablated_drop_token
                gates, u_values, valid_groups, pos_e_mean, neg_e_mean = (
                    self._compute_group_gates(
                        e_traj,
                        outcomes,
                        parent_indices,
                        num_groups,
                    )
                )
                (
                    selection_scores,
                    final_utilities,
                    drop_gaps,
                    counterfactual_penalties,
                    ablation_null_scores,
                    marginal_deltas,
                ) = self._transition_selection_scores(
                    e_traj,
                    drop_traj,
                    outcomes,
                    parent_indices,
                    u_values,
                    valid_groups,
                    num_groups,
                    ablated_e_traj=ablated_e_traj,
                    ablated_drop_traj=ablated_drop_traj,
                )

                candidate_results.append(
                    {
                        "j": int(candidate["j"]),
                        "teacher_logp_j": teacher_logp_j,
                        "h_prev": h_prev,
                        "h_j": h_j,
                        "e_j": e_j,
                        "drop_token": drop_token,
                        "e_traj": e_traj,
                        "h_prev_traj": h_prev_traj,
                        "h_j_traj": h_j_traj,
                        "drop_traj": drop_traj,
                        "gates": gates,
                        "u_values": u_values,
                        "valid_groups": valid_groups,
                        "selection_scores": selection_scores,
                        "final_utilities": final_utilities,
                        "drop_gaps": drop_gaps,
                        "counterfactual_penalties": counterfactual_penalties,
                        "ablation_null_scores": ablation_null_scores,
                        "marginal_deltas": marginal_deltas,
                        "ablation_available": torch.tensor(
                            float(ablation_available),
                            device=e_traj.device,
                            dtype=e_traj.dtype,
                        ),
                        "pos_e_mean": pos_e_mean,
                        "neg_e_mean": neg_e_mean,
                    }
                )

        score_stack = torch.stack([candidate["selection_scores"] for candidate in candidate_results], dim=0)
        selected_scores, selected_candidates_per_group = torch.max(score_stack, dim=0)
        selected_group_valid = torch.isfinite(selected_scores)
        selected_candidates_per_traj = selected_candidates_per_group[parent_indices]

        teacher_logp_j = self._gather_candidate_sequence(
            candidate_results,
            "teacher_logp_j",
            selected_candidates_per_traj,
        )
        e_j = self._gather_candidate_sequence(candidate_results, "e_j", selected_candidates_per_traj)
        h_prev = self._gather_candidate_sequence(candidate_results, "h_prev", selected_candidates_per_traj)
        h_j = self._gather_candidate_sequence(candidate_results, "h_j", selected_candidates_per_traj)
        drop_token = self._gather_candidate_sequence(candidate_results, "drop_token", selected_candidates_per_traj)
        e_traj = self._gather_candidate_trajectory(candidate_results, "e_traj", selected_candidates_per_traj)
        h_prev_traj = self._gather_candidate_trajectory(
            candidate_results,
            "h_prev_traj",
            selected_candidates_per_traj,
        )
        h_j_traj = self._gather_candidate_trajectory(candidate_results, "h_j_traj", selected_candidates_per_traj)
        drop_traj = self._gather_candidate_trajectory(candidate_results, "drop_traj", selected_candidates_per_traj)
        gates = self._gather_candidate_group(candidate_results, "gates", selected_candidates_per_group)
        u_values = self._gather_candidate_group(candidate_results, "u_values", selected_candidates_per_group)
        selected_ablation_null_scores = self._gather_candidate_group(
            candidate_results,
            "ablation_null_scores",
            selected_candidates_per_group,
        )
        selected_marginal_deltas = self._gather_candidate_group(
            candidate_results,
            "marginal_deltas",
            selected_candidates_per_group,
        )
        selected_final_utilities = self._gather_candidate_group(
            candidate_results,
            "final_utilities",
            selected_candidates_per_group,
        )
        valid_groups = self._gather_candidate_group(
            candidate_results,
            "valid_groups",
            selected_candidates_per_group,
        )
        raw_group_gates = gates * selected_group_valid.to(gates.dtype)
        score_gates = torch.zeros_like(gates)
        if bool(selected_group_valid.any().item()):
            score_scale = selected_scores[selected_group_valid].detach().abs().mean().clamp(min=1e-6)
            valid_score_gates = torch.sigmoid(selected_scores[selected_group_valid] / score_scale).to(gates.dtype)
            score_gates[selected_group_valid] = valid_score_gates
            self._metric("transition/group_score_weight_mean", valid_score_gates.mean())
        if self.sdpo_require_logp_causal_gate:
            gates = score_gates * raw_group_gates
        else:
            gates = score_gates * selected_group_valid.to(gates.dtype)
        self._metric("transition/raw_group_gate_mean", raw_group_gates.mean())
        self._metric("transition/group_score_weight_active_rate", (score_gates > 0).float().mean())
        self._metric("transition/effective_group_gate_mean", gates.mean())
        valid_groups = valid_groups & selected_group_valid
        gate_per_traj = gates[parent_indices].unsqueeze(-1)

        token_mask = response_mask.bool()
        if bool(token_mask.any().item()):
            token_logp_scale = e_j[token_mask].detach().abs().mean().clamp(min=1e-6)
            token_entropy_scale = drop_token[token_mask].detach().abs().mean().clamp(min=1e-6)
            entropy_positive_rate = ((drop_token > 0) & token_mask).float().sum()
            entropy_positive_rate = entropy_positive_rate / token_mask.float().sum().clamp(min=1.0)
            entropy_floor = (1.0 / (1.0 + entropy_positive_rate)).to(drop_token.dtype)
        else:
            token_logp_scale = torch.ones((), device=e_j.device, dtype=e_j.dtype)
            token_entropy_scale = torch.ones((), device=e_j.device, dtype=e_j.dtype)
            entropy_floor = torch.tensor(0.5, device=e_j.device, dtype=e_j.dtype)
        logp_evidence_weight = torch.sigmoid(e_j / token_logp_scale) * (e_j > 0).to(e_j.dtype)
        entropy_evidence_weight = torch.sigmoid(drop_token / token_entropy_scale)
        entropy_evidence_weight = entropy_floor + (1.0 - entropy_floor) * entropy_evidence_weight
        token_evidence_weight = (gate_per_traj * logp_evidence_weight * entropy_evidence_weight).clamp(0.0, 1.0)
        token_evidence_weight = token_evidence_weight * response_mask
        self._metric("evidence/dynamic_logp_scale", token_logp_scale)
        self._metric("evidence/dynamic_entropy_scale", token_entropy_scale)
        self._metric("evidence/dynamic_entropy_floor", entropy_floor)

        for idx, candidate in enumerate(candidate_results):
            layer_valid = candidate["valid_groups"]
            valid_scores = candidate["selection_scores"][torch.isfinite(candidate["selection_scores"])]
            layer_final_utilities = candidate["final_utilities"][layer_valid]
            layer_u_positive = (
                (candidate["u_values"][layer_valid] > 0).float().mean()
                if bool(layer_valid.any().item())
                else None
            )
            self._metric(f"transition/layer_{candidate['j']}/u_positive_rate", layer_u_positive)
            self._metric(f"transition/layer_{candidate['j']}/effective_signal_rate", candidate["gates"].mean())
            self._metric(f"transition/layer_{candidate['j']}/entropy_drop", candidate["drop_traj"].mean())
            self._metric(
                f"transition/layer_{candidate['j']}/counterfactual_penalty",
                candidate["counterfactual_penalties"][layer_valid].mean() if bool(layer_valid.any().item()) else None,
            )
            self._metric(
                f"transition/layer_{candidate['j']}/ablation_null_score",
                candidate["ablation_null_scores"][layer_valid].mean() if bool(layer_valid.any().item()) else None,
            )
            self._metric(
                f"transition/layer_{candidate['j']}/marginal_delta",
                candidate["marginal_deltas"][layer_valid].mean() if bool(layer_valid.any().item()) else None,
            )
            self._metric(
                f"transition/layer_{candidate['j']}/marginal_positive_rate",
                (
                    (candidate["marginal_deltas"][layer_valid] > 0.0).float().mean()
                    if bool(layer_valid.any().item())
                    else None
                ),
            )
            self._metric(f"transition/layer_{candidate['j']}/ablation_available", candidate["ablation_available"])
            self._metric(
                f"transition/layer_{candidate['j']}/final_utility",
                layer_final_utilities.mean() if layer_final_utilities.numel() > 0 else None,
            )
            self._metric(f"transition/layer_{candidate['j']}/score", valid_scores.mean() if valid_scores.numel() > 0 else None)
            self._metric(
                f"transition/selected_layer_{candidate['j']}_rate",
                ((selected_candidates_per_group == idx) & selected_group_valid).float().mean(),
            )
        selected_js = torch.tensor(
            [candidate["j"] for candidate in candidate_results],
            device=e_j.device,
            dtype=e_j.dtype,
        )[selected_candidates_per_group]
        self._metric(
            "transition/selected_j_mean",
            selected_js[selected_group_valid].mean() if bool(selected_group_valid.any().item()) else None,
        )
        self._metric("transition/selected_valid_rate", selected_group_valid.float().mean())
        self._metric(
            "transition/selected_ablation_null_score",
            (
                selected_ablation_null_scores[selected_group_valid].mean()
                if bool(selected_group_valid.any().item())
                else None
            ),
        )
        self._metric(
            "transition/selected_marginal_delta",
            selected_marginal_deltas[selected_group_valid].mean() if bool(selected_group_valid.any().item()) else None,
        )
        self._metric(
            "transition/selected_final_utility",
            selected_final_utilities[selected_group_valid].mean() if bool(selected_group_valid.any().item()) else None,
        )
        has_tokens = bool(token_mask.any().item())
        self._metric("evidence/token_weight_mean", token_evidence_weight[token_mask].mean() if has_tokens else None)
        self._metric(
            "evidence/token_weight_active_rate",
            (token_evidence_weight[token_mask] > 0.5).float().mean() if has_tokens else None,
        )
        self._metric(
            "evidence/logp_positive_rate",
            ((e_j > 0) & token_mask).float().sum() / token_mask.float().sum().clamp(min=1.0),
        )
        self._metric(
            "evidence/entropy_positive_rate",
            ((drop_token > 0) & token_mask).float().sum() / token_mask.float().sum().clamp(min=1.0),
        )

        loss_mask = response_mask.bool()
        if loss_mask.any():
            token_weights = response_mask
            length_ratio = (lengths / max(1, int(self.args.max_completion_length))).clamp(max=1.0)
            length_center = length_ratio.detach().median()
            overlong_ratio = F.relu(length_ratio - length_center)
            overlong_ratio = overlong_ratio / (1.0 - length_center).clamp(min=1e-6)
            trajectory_weights = 1.0 / (1.0 + overlong_ratio)
            token_weights = token_weights * trajectory_weights.unsqueeze(-1)
            entropy_target = (1.0 - token_evidence_weight.detach()) * h_prev.detach()
            entropy_deficit = (F.relu(entropy_target - student_entropy) * response_mask).sum()
            entropy_deficit = entropy_deficit / response_mask.sum().clamp(min=1.0)
            loss = self.student_entropy_coef * entropy_deficit
            self._metric("objective/student_entropy_deficit", entropy_deficit)

            if self.sdpo_preference_coef > 0:
                preference_loss = self._compute_sdpo_preference_loss(
                    student_log_probs_sampled,
                    response_mask,
                    token_weights,
                    outcomes,
                    parent_indices,
                    gates,
                    e_traj,
                    num_groups,
                )
                loss = loss + self.sdpo_preference_coef * preference_loss
                self._metric("objective/sdpo_preference_weighted_loss", self.sdpo_preference_coef * preference_loss)

            if self.sdpo_distill_coef > 0:
                sdpo_loss = self._compute_evidence_sdpo_distill_loss(
                    student_log_probs_sampled,
                    teacher_logp_j,
                    response_mask,
                    token_evidence_weight,
                    outcomes,
                    token_weights,
                    h_j,
                )
                loss = loss + self.sdpo_distill_coef * sdpo_loss
                self._metric("objective/sdpo_distill_weighted_loss", self.sdpo_distill_coef * sdpo_loss)

            self._metric("rollout/length_weight_mean", token_weights[loss_mask].mean())
            self._metric("rollout/length_penalty_active_rate", (overlong_ratio > 0).float().mean())
        else:
            loss = student_log_probs_sampled.sum() * 0.0

        pos_e_values = []
        neg_e_values = []
        for group_idx in range(num_groups):
            group_mask = parent_indices == group_idx
            if not torch.any(group_mask):
                continue
            group_outcomes = outcomes[group_mask]
            group_e = e_traj[group_mask]
            if group_outcomes.any():
                pos_e_values.append(group_e[group_outcomes].mean())
            if (~group_outcomes).any():
                neg_e_values.append(group_e[~group_outcomes].mean())
        pos_e_mean = torch.stack(pos_e_values).mean() if pos_e_values else None
        neg_e_mean = torch.stack(neg_e_values).mean() if neg_e_values else None

        student_entropy_traj = (student_entropy * response_mask).sum(dim=-1) / lengths
        self._log_lite_cie_metrics(
            inputs,
            e_traj,
            outcomes,
            parent_indices,
            gates,
            u_values,
            valid_groups,
            pos_e_mean,
            neg_e_mean,
            h_prev_traj,
            h_j_traj,
            drop_traj,
            student_entropy_traj,
            lengths,
        )

        del (
            student_logits,
            student_log_probs_sampled,
            teacher_logp_prev,
            teacher_logp_j,
            e_j,
            h_prev,
            h_j,
            student_entropy,
            token_evidence_weight,
        )
        empty_cache()

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
            vec = torch.tensor(
                [self._on_policy_loss_total, self._on_policy_step_equiv],
                dtype=torch.float64,
                device=device,
            )
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
        super().log(logs, start_time)
        self._metrics[mode].clear()
