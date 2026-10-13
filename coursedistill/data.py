from __future__ import annotations

from pathlib import Path


def load_course_distill_dataset(
    data_dir: str | Path,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
):
    """Load dedicated raw course-distill data.

    The data stores only original problem fields and privileged references. All
    rollout, reward, curriculum, span, and teacher-score signals are computed
    online in the trainer.
    """
    from datasets import load_dataset

    data_dir = Path(data_dir)
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {split} data file: {jsonl_path}")

    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset

