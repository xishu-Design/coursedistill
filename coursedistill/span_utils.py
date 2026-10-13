from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpanConfig:
    answer_span_tokens: int = 48
    diff_span_tokens: int = 64
    diff_span_top_k: int = 3
    calculation_span_tokens: int = 64
    support_span_tokens: int = 96
    reason_span_tokens: int = 128


def _valid_length(response_mask: torch.Tensor, row: int) -> int:
    return int(response_mask[row].sum().item())


def _longest_common_prefix(a: torch.Tensor, a_len: int, b: torch.Tensor, b_len: int) -> int:
    limit = min(a_len, b_len)
    if limit <= 0:
        return 0
    same = (a[:limit] == b[:limit]).detach().cpu().tolist()
    for idx, ok in enumerate(same):
        if not ok:
            return idx
    return limit


def _answer_bounds(token_ids: torch.Tensor, length: int, tokenizer, answer_window: int) -> tuple[int, int]:
    if length <= 0:
        return 0, 0
    start = max(0, length - max(1, answer_window))
    end = length
    try:
        tokens = tokenizer.convert_ids_to_tokens(token_ids[:length].detach().cpu().tolist())
    except Exception:
        tokens = []
    marker_positions: list[tuple[int, str]] = []
    for idx, token in enumerate(tokens):
        text = str(token).lower()
        if "boxed" in text:
            marker_positions.append((idx, "boxed"))
        elif "final" in text or "answer" in text or "result" in text:
            marker_positions.append((idx, "answer"))
    if marker_positions:
        marker, marker_kind = marker_positions[-1]
        pre = 4 if marker_kind == "boxed" else max(4, answer_window // 8)
        post = max(12, answer_window // 2)
        start = max(0, marker - pre)
        end = min(length, marker + post)
        return start, max(start + 1, end)
    boxed_positions = [
        idx for idx, token in enumerate(tokens)
        if (
            "boxed" in str(token).lower()
            or "final" in str(token).lower()
            or "answer" in str(token).lower()
            or "result" in str(token).lower()
            or "答案" in str(token)
            or "最终" in str(token)
        )
    ]
    if boxed_positions:
        marker = boxed_positions[-1]
        start = max(0, marker - max(1, answer_window // 4))
        end = min(length, marker + max(1, answer_window))
    else:
        critical = _critical_token_mask(token_ids, length, tokenizer)
        if bool(critical.any().item()):
            last_critical = int(torch.nonzero(critical > 0, as_tuple=False).flatten()[-1].item())
            start = max(0, last_critical - max(1, answer_window) + 1)
            end = min(length, last_critical + max(4, answer_window // 8) + 1)
    return start, max(start + 1, end)


def _token_strings(token_ids: torch.Tensor, length: int, tokenizer) -> list[str]:
    try:
        return [str(token).lower() for token in tokenizer.convert_ids_to_tokens(token_ids[:length].detach().cpu().tolist())]
    except Exception:
        return []


def _critical_token_mask(token_ids: torch.Tensor, length: int, tokenizer) -> torch.Tensor:
    mask = torch.zeros(length, device=token_ids.device, dtype=torch.float32)
    if length <= 0:
        return mask
    tokens = _token_strings(token_ids, length, tokenizer)
    math_markers = set("0123456789=+-*/^_\\{}[]()<>|")
    keywords = (
        "boxed",
        "frac",
        "sqrt",
        "sum",
        "prod",
        "case",
        "mod",
        "gcd",
        "lcm",
        "therefore",
        "hence",
        "thus",
        "implies",
        "equation",
        "answer",
        "final",
        "result",
        "solution",
        "thereby",
        "so",
        "答案",
        "最终",
        "所以",
        "因此",
        "可得",
        "故",
        "结论",
    )
    for idx, token in enumerate(tokens[:length]):
        if any(ch in math_markers for ch in token) or any(word in token for word in keywords):
            mask[idx] = 1.0
    return mask


def _reason_anchor_mask(token_ids: torch.Tensor, length: int, tokenizer) -> torch.Tensor:
    """Stricter anchors for reasoning-step candidates.

    Compared with _critical_token_mask, this intentionally ignores plain
    numbers and most punctuation. ReasonSpan should propose a few mathematical
    or logical steps, not every formula-looking token in the trajectory.
    """
    mask = torch.zeros(length, device=token_ids.device, dtype=torch.float32)
    if length <= 0:
        return mask
    tokens = _token_strings(token_ids, length, tokenizer)
    relation_markers = set("=<>")
    operator_markers = set("+-*/^")
    step_keywords = (
        "therefore",
        "hence",
        "thus",
        "implies",
        "equation",
        "substitute",
        "simplify",
        "factor",
        "solve",
        "gives",
        "obtain",
        "conclude",
        "because",
        "since",
        "then",
        "so",
        "frac",
        "sqrt",
        "mod",
        "case",
        "所以",
        "因此",
        "可得",
        "故",
        "结论",
        "代入",
        "化简",
        "解得",
    )
    for idx, token in enumerate(tokens[:length]):
        has_relation = any(ch in relation_markers for ch in token)
        has_operator = any(ch in operator_markers for ch in token)
        has_digit = any(ch.isdigit() for ch in token)
        has_keyword = any(word in token for word in step_keywords)
        if any(word in token for word in ("then", "so", "because", "since", "gives")) and not (
            has_relation or (has_digit and has_operator)
        ):
            has_keyword = False
        if has_keyword or has_relation or (has_digit and has_operator):
            mask[idx] = 1.0
    return mask


def _calculation_anchor_mask(token_ids: torch.Tensor, length: int, tokenizer) -> torch.Tensor:
    """Anchors for the final arithmetic / symbolic merge near the answer."""
    mask = torch.zeros(length, device=token_ids.device, dtype=torch.float32)
    if length <= 0:
        return mask
    tokens = _token_strings(token_ids, length, tokenizer)
    relation_markers = set("=<>")
    operator_markers = set("+-*/^")
    structural_markers = set("\\{}[]()|")
    keywords = (
        "boxed",
        "frac",
        "sqrt",
        "mod",
        "gcd",
        "lcm",
        "answer",
        "final",
        "result",
        "therefore",
        "hence",
        "thus",
    )
    for idx, token in enumerate(tokens[:length]):
        has_digit = any(ch.isdigit() for ch in token)
        has_relation = any(ch in relation_markers for ch in token)
        has_operator = any(ch in operator_markers for ch in token)
        has_structure = any(ch in structural_markers for ch in token)
        has_keyword = any(word in token for word in keywords)
        if has_relation or has_keyword or (has_digit and (has_operator or has_structure)):
            mask[idx] = 1.0
    return mask


def _dilate(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    if mask.numel() == 0 or radius <= 0 or not bool(mask.any().item()):
        return mask
    out = torch.zeros_like(mask)
    indices = torch.nonzero(mask > 0, as_tuple=False).flatten()
    for idx_t in indices:
        idx = int(idx_t.item())
        out[max(0, idx - radius) : min(mask.numel(), idx + radius + 1)] = 1.0
    return out


def _mark_structured_window(
    target: torch.Tensor,
    token_ids: torch.Tensor,
    length: int,
    tokenizer,
    start: int,
    end: int,
    *,
    core_tokens: int,
    critical_radius: int = 3,
) -> None:
    """Mark a small divergence core plus nearby math/logical anchors.

    This keeps DiffSpan interpretable while avoiding a pure formatting LCP from
    selecting a long prose window. The immediate core captures the local branch,
    and the critical-token dilation expands useful supervision around formulas,
    variables, equations, and conclusion markers.
    """
    if length <= 0:
        return
    start = max(0, min(start, max(0, length - 1)))
    end = max(start + 1, min(end, length))
    if end <= start:
        return

    core_end = min(end, start + max(1, core_tokens))
    target[start:core_end] = 1.0

    critical = _critical_token_mask(token_ids, length, tokenizer)
    local = _dilate(critical[start:end], radius=critical_radius)
    if bool(local.any().item()):
        target[start:end] = torch.maximum(target[start:end], local.to(dtype=target.dtype))


def _mark_support_window(
    target: torch.Tensor,
    token_ids: torch.Tensor,
    length: int,
    tokenizer,
    start: int,
    end: int,
) -> None:
    """Mark answer-support anchors immediately preceding the answer span."""
    if end <= start or length <= 0:
        return
    critical = _critical_token_mask(token_ids, length, tokenizer)
    local_support = _dilate(critical[start:end], radius=2)
    if bool(local_support.any().item()):
        target[start:end] = torch.maximum(target[start:end], local_support.to(dtype=target.dtype))
        return
    fallback_start = max(start, end - min(32, end - start))
    target[fallback_start:end] = 1.0


def _mark_calculation_window(
    target: torch.Tensor,
    token_ids: torch.Tensor,
    length: int,
    tokenizer,
    answer_start: int,
    answer_end: int,
    budget: int,
) -> None:
    """Mark final arithmetic/symbolic merge candidates around the answer."""
    if length <= 0 or budget <= 0:
        return
    search_start = max(0, min(answer_start, answer_end) - max(1, int(budget)))
    search_end = min(length, max(answer_end, answer_start + 1))
    if search_end <= search_start:
        return
    anchors = _calculation_anchor_mask(token_ids, length, tokenizer)[search_start:search_end]
    local = _dilate(anchors, radius=2)
    local_indices = torch.nonzero(local > 0, as_tuple=False).flatten()
    if local_indices.numel() == 0:
        fallback_start = max(search_start, search_end - min(24, search_end - search_start))
        target[fallback_start:search_end] = 1.0
        return
    if local_indices.numel() > budget:
        local_indices = local_indices[-budget:]
    target[search_start + local_indices] = 1.0


def _mark_reason_window(
    target: torch.Tensor,
    token_ids: torch.Tensor,
    length: int,
    tokenizer,
    answer_start: int,
    occupied_mask: torch.Tensor,
    budget: int,
) -> None:
    """Mark broader reasoning-step candidates before the final answer.

    This is intentionally a candidate generator, not a teacher-free selector.
    It keeps only structured math/logical neighborhoods and leaves final
    weighting to the privileged teacher score. Tokens outside these candidates
    continue to receive only the trajectory-level GRPO update.
    """
    if length <= 0 or answer_start <= 0 or budget <= 0:
        return

    search_end = min(answer_start, length)
    anchors = torch.nonzero(_reason_anchor_mask(token_ids, length, tokenizer)[:search_end] > 0, as_tuple=False).flatten()
    if anchors.numel() == 0:
        return

    occupied = occupied_mask[:search_end].clamp(min=0.0, max=1.0)
    segments: list[tuple[int, int]] = []
    seg_start = int(anchors[0].item())
    prev = seg_start
    for anchor_t in anchors[1:]:
        anchor = int(anchor_t.item())
        if anchor - prev <= 6:
            prev = anchor
            continue
        segments.append((seg_start, prev + 1))
        seg_start = anchor
        prev = anchor
    segments.append((seg_start, prev + 1))

    # Prefer recent reasoning steps, but keep them as small step-like windows.
    remaining = max(1, int(budget))
    for start, end in sorted(segments, key=lambda item: item[1], reverse=True):
        if remaining <= 0:
            break
        win_start = max(0, start - 3)
        win_end = min(search_end, end + 4)
        local = torch.zeros_like(target[:search_end])
        local[win_start:win_end] = 1.0
        local = local * (1.0 - occupied)
        local_indices = torch.nonzero(local > 0, as_tuple=False).flatten()
        if local_indices.numel() == 0:
            continue
        if local_indices.numel() > remaining:
            local_indices = local_indices[-remaining:]
        target[local_indices] = 1.0
        remaining -= int(local_indices.numel())


def build_candidate_span_masks(
    sampled_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    outcomes: torch.Tensor,
    parent_indices: torch.Tensor,
    tokenizer,
    cfg: SpanConfig,
    include_support: bool,
    include_reason: bool,
) -> dict[str, torch.Tensor]:
    """Build token masks for AnswerSpan, DiffSpan, SupportSpan, and ReasonSpan.

    DiffSpan uses up to three most similar opposite-outcome trajectories from
    the same prompt by longest common prefix. Opposite trajectories are used
    only to locate divergence spans; teacher scoring and distillation remain on
    verified-correct spans. ReasonSpan is a broader structured candidate set
    before the final answer, and is still teacher-reranked before distillation.
    """
    device = sampled_token_ids.device
    dtype = response_mask.dtype
    answer_mask = torch.zeros_like(response_mask, dtype=dtype, device=device)
    diff_mask = torch.zeros_like(response_mask, dtype=dtype, device=device)
    calculation_mask = torch.zeros_like(response_mask, dtype=dtype, device=device)
    support_mask = torch.zeros_like(response_mask, dtype=dtype, device=device)
    reason_mask = torch.zeros_like(response_mask, dtype=dtype, device=device)
    error_mask = torch.zeros_like(response_mask, dtype=dtype, device=device)

    row_count = sampled_token_ids.shape[0]
    lengths = [_valid_length(response_mask, row) for row in range(row_count)]

    for row in range(row_count):
        if not bool(outcomes[row].item()):
            continue
        length = lengths[row]
        if length <= 0:
            continue

        ans_start, ans_end = _answer_bounds(sampled_token_ids[row], length, tokenizer, cfg.answer_span_tokens)
        answer_mask[row, ans_start:ans_end] = 1.0
        _mark_calculation_window(
            calculation_mask[row],
            sampled_token_ids[row],
            length,
            tokenizer,
            ans_start,
            ans_end,
            max(1, cfg.calculation_span_tokens),
        )

        if include_support and ans_start > 0:
            support_start = max(0, ans_start - max(1, cfg.support_span_tokens))
            _mark_support_window(
                support_mask[row],
                sampled_token_ids[row],
                length,
                tokenizer,
                support_start,
                ans_start,
            )

        group = parent_indices[row]
        wrong_rows = torch.nonzero((parent_indices == group) & (~outcomes), as_tuple=False).flatten()
        if wrong_rows.numel() > 0:
            lcps: list[int] = []
            for wrong_row_t in wrong_rows:
                wrong_row = int(wrong_row_t.item())
                lcp = _longest_common_prefix(
                    sampled_token_ids[row],
                    length,
                    sampled_token_ids[wrong_row],
                    lengths[wrong_row],
                )
                lcps.append(lcp)
            if lcps:
                diff_lower_bound = max(0, ans_start - max(1, cfg.support_span_tokens))
                # Use the most similar wrong rollouts. Keeping this configurable
                # lets the trainer trade off signal density against non-causal
                # formatting divergence noise.
                candidate_lcps = sorted(set(lcps), reverse=True)[: max(1, int(cfg.diff_span_top_k))]
                for lcp in candidate_lcps:
                    diff_start = min(max(lcp, diff_lower_bound), max(0, length - 1))
                    diff_end = min(length, diff_start + max(1, cfg.diff_span_tokens))
                    _mark_structured_window(
                        diff_mask[row],
                        sampled_token_ids[row],
                        length,
                        tokenizer,
                        diff_start,
                        diff_end,
                        core_tokens=min(24, max(8, cfg.diff_span_tokens // 4)),
                    )

        if include_reason and ans_start > 0:
            occupied = answer_mask[row] + diff_mask[row] + calculation_mask[row] + support_mask[row]
            _mark_reason_window(
                reason_mask[row],
                sampled_token_ids[row],
                length,
                tokenizer,
                ans_start,
                occupied,
                max(1, cfg.reason_span_tokens),
            )

    for row in range(row_count):
        if bool(outcomes[row].item()):
            continue
        length = lengths[row]
        if length <= 0:
            continue
        group = parent_indices[row]
        correct_rows = torch.nonzero((parent_indices == group) & outcomes, as_tuple=False).flatten()

        ans_start, ans_end = _answer_bounds(sampled_token_ids[row], length, tokenizer, cfg.answer_span_tokens)
        error_mask[row, ans_start:ans_end] = 1.0

        if correct_rows.numel() == 0:
            continue

        lcps: list[int] = []
        for correct_row_t in correct_rows:
            correct_row = int(correct_row_t.item())
            lcp = _longest_common_prefix(
                sampled_token_ids[row],
                length,
                sampled_token_ids[correct_row],
                lengths[correct_row],
            )
            lcps.append(lcp)
        if lcps:
            diff_lower_bound = max(0, ans_start - max(1, cfg.support_span_tokens))
            candidate_lcps = sorted(set(lcps), reverse=True)[: max(1, int(cfg.diff_span_top_k))]
            for lcp in candidate_lcps:
                diff_start = min(max(lcp, diff_lower_bound), max(0, length - 1))
                diff_end = min(length, diff_start + max(1, cfg.diff_span_tokens))
                _mark_structured_window(
                    error_mask[row],
                    sampled_token_ids[row],
                    length,
                    tokenizer,
                    diff_start,
                    diff_end,
                    core_tokens=min(24, max(8, cfg.diff_span_tokens // 4)),
                )

    final_mask = ((answer_mask + calculation_mask) > 0).to(dtype=dtype) * response_mask
    context_mask = ((support_mask + reason_mask) > 0).to(dtype=dtype) * response_mask
    combined = ((final_mask + diff_mask + context_mask) > 0).to(dtype=dtype) * response_mask
    return {
        "answer": answer_mask * response_mask,
        "diff": diff_mask * response_mask,
        "calculation": calculation_mask * response_mask,
        "final": final_mask,
        "support": support_mask * response_mask,
        "reason": reason_mask * response_mask,
        "context": context_mask,
        "error": error_mask * response_mask,
        "combined": combined,
    }
