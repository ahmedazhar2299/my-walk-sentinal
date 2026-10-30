#!/usr/bin/env python3
"""Continue self-supervised pretraining on the prepared public stroke windows."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "Data/model_inputs/public_stroke"
OUTPUT_ROOT = ROOT / "Data/trained_models/public_stroke"


class WindowDataset(Dataset):
    def __init__(self, path: Path):
        self.values = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        return np.array(self.values[index], dtype=np.float32, copy=True)


def limu_collate(batch: list[np.ndarray]) -> tuple[torch.Tensor, ...]:
    """Apply LIMU-BERT-X's acceleration scaling and masked reconstruction input."""
    from utils import span_mask

    sequences = np.stack(batch)
    sequences[..., :3] /= 9.8
    masked = sequences.copy()
    positions = []
    targets = []
    for row in range(len(sequences)):
        pos = np.asarray(span_mask(20, max_gram=10, goal_num_predict=3), dtype=np.int64)
        positions.append(pos)
        targets.append(sequences[row, pos])
        # Follow the original LIMU-BERT masking probability (80% zero mask).
        if np.random.random() < 0.8:
            masked[row, pos] = 0.0
    return (
        torch.from_numpy(masked),
        torch.from_numpy(np.stack(positions)),
        torch.from_numpy(np.stack(targets)),
    )


def hart_collate(batch: list[np.ndarray]) -> dict[str, torch.Tensor]:
    values = torch.from_numpy(np.stack(batch))
    return {
        "past_values": values,
        "past_observed_mask": torch.ones_like(values, dtype=torch.bool),
    }


def make_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name == "limu":
        sys.path.insert(0, str(ROOT / "models/LIMU-BERT-X"))
        from config import PretrainModelConfig
        from models import LIMUBertModel4Pretrain

        config_path = ROOT / "models/LIMU-BERT-X/config/limu_bert.json"
        config = PretrainModelConfig.from_json(json.loads(config_path.read_text())["base_v4"])
        model = LIMUBertModel4Pretrain(config)
        weights = torch.load(
            ROOT / "models/LIMU-BERT-X/weights/limu_pretrain.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(weights, strict=True)
        return model.to(device)

    from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction

    checkpoint = ROOT / "models/HART/finetuning_HART/pretrained_checkpoint"
    return TSPulseForReconstruction.from_pretrained(
        str(checkpoint), local_files_only=True
    ).to(device)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch_num, batch in enumerate(loader):
        if max_batches is not None and batch_num >= max_batches:
            break
        if model_name == "limu":
            masked, positions, targets = (x.to(device) for x in batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(masked, positions)
            loss = torch.nn.functional.mse_loss(predictions, targets)
        else:
            batch = {key: value.to(device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, return_loss=True).loss

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"{model_name} returned a missing or non-finite loss")
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("The data loader yielded no batches")
    if device.type == "mps":
        torch.mps.synchronize()
    return float(np.mean(losses))


def save_model(model: torch.nn.Module, model_name: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if model_name == "limu":
        torch.save(model.state_dict(), destination / "model.pt")
    else:
        model.save_pretrained(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("limu", "hart"))
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one train and one validation batch, then save a disposable checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.max_train_batches = 1
        args.max_validation_batches = 1

    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this PyTorch installation")
    device = torch.device(
        "mps" if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()) else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = INPUT_ROOT / ("LIMU-BERT-X" if args.model == "limu" else "HART")
    train_data = WindowDataset(data_dir / "train.npy")
    val_data = WindowDataset(data_dir / "validation.npy")
    collate = limu_collate if args.model == "limu" else hart_collate
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    model = make_model(args.model, device)
    learning_rate = args.learning_rate or (1e-4 if args.model == "limu" else 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    output_dir = OUTPUT_ROOT / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"model={args.model} device={device} train_windows={len(train_data)} "
        f"validation_windows={len(val_data)} batch_size={args.batch_size} "
        f"learning_rate={learning_rate:g}"
    )
    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = run_epoch(
            model, train_loader, args.model, device, optimizer, args.max_train_batches
        )
        with torch.no_grad():
            validation_loss = run_epoch(
                model, val_loader, args.model, device, None, args.max_validation_batches
            )
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} "
            f"seconds={time.perf_counter() - start:.1f}"
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            save_model(model, args.model, output_dir / "best")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs; best validation loss={best_loss:.6f}")
                break


if __name__ == "__main__":
    main()
