from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from mlx_vlm import lora


def _args(**overrides):
    args = NS(
        model_path="dummy",
        dataset="train-data",
        split="train",
        val_dataset=None,
        dataset_config=None,
        image_resize_shape=None,
        custom_prompt_format=None,
        train_mode="sft",
        train_on_completions=False,
        assistant_id=77091,
        batch_size=2,
        iters=10,
        epochs=None,
        learning_rate=2e-5,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=100,
        val_batches=4,
        max_seq_length=2048,
        grad_checkpoint=False,
        grad_clip=None,
        gradient_accumulation_steps=1,
        full_finetune=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        train_vision=False,
        adapter_path=None,
        output_path="adapters.safetensors",
        beta=0.1,
        eps=1e-8,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _run_main(monkeypatch, args, dataset_sizes):
    """Call lora.main with the model, datasets and trainers stubbed out.

    dataset_sizes gives len() for each load_dataset call, training set first.
    Returns the keyword arguments the trainer was called with.
    """
    model = MagicMock()
    model.config.model_type = "qwen2_vl"
    datasets = [MagicMock(__len__=lambda self, n=n: n) for n in dataset_sizes]

    monkeypatch.setattr(lora, "load", lambda *a, **k: (model, MagicMock()))
    monkeypatch.setattr(lora, "load_dataset", lambda *a, **k: datasets.pop(0))
    monkeypatch.setattr(lora, "transform_dataset_to_messages", lambda ds, *a, **k: ds)
    monkeypatch.setattr(lora, "VisionDataset", lambda ds, *a, **k: ds)
    monkeypatch.setattr(lora, "PreferenceVisionDataset", lambda ds, *a, **k: ds)
    monkeypatch.setattr(lora, "setup_model_for_training", lambda *a, **k: model)
    monkeypatch.setattr(lora, "print_trainable_parameters", lambda *a, **k: None)

    trainer = MagicMock()
    monkeypatch.setattr(
        lora, "train_orpo" if args.train_mode == "orpo" else "train", trainer
    )

    lora.main(args)
    trainer.assert_called_once()
    return trainer.call_args.kwargs


@pytest.mark.parametrize("train_mode", ["sft", "orpo"])
def test_val_dataset_is_passed_to_trainer(monkeypatch, train_mode):
    kwargs = _run_main(
        monkeypatch,
        _args(train_mode=train_mode, val_dataset="val-data"),
        dataset_sizes=[8, 4],
    )
    assert kwargs["val_dataset"] is not None


@pytest.mark.parametrize("train_mode", ["sft", "orpo"])
def test_no_val_dataset_disables_validation(monkeypatch, train_mode):
    kwargs = _run_main(monkeypatch, _args(train_mode=train_mode), dataset_sizes=[8])
    assert kwargs["val_dataset"] is None


def test_val_dataset_smaller_than_batch_size_raises(monkeypatch):
    with pytest.raises(ValueError, match="at least batch_size"):
        _run_main(
            monkeypatch,
            _args(val_dataset="val-data", batch_size=4),
            dataset_sizes=[8, 3],
        )


def test_missing_val_attribute_is_tolerated(monkeypatch):
    """LORA.MD documents calling main() with a hand-built namespace."""
    args = _args()
    del args.val_dataset
    kwargs = _run_main(monkeypatch, args, dataset_sizes=[8])
    assert kwargs["val_dataset"] is None
