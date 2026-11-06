#!/usr/bin/env python3
"""Extract LIMU-BERT-X and multichannel HART embeddings from lab NW recordings.

This script performs inference only. It keeps every NW recording separate,
uses the model-specific sampling/windowing used during public-stroke training,
and writes one row per window. HART receives all six axes jointly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "Data/lab/XSENS_synchronized"
OUTPUT_ROOT = ROOT / "Data/embeddings/lab_nw"
AXES = ("X", "Y", "Z")
CHANNELS = tuple([f"Accel_{axis}" for axis in AXES] + [f"Gyro_{axis}" for axis in AXES])
HART_CHANNELS = tuple([f"Accel_{axis}" for axis in AXES] + [f"Gyro_{axis}" for axis in AXES])


def resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but this PyTorch installation has no available MPS device")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def recording_metadata(path: Path, input_root: Path) -> dict[str, str]:
    relative = path.relative_to(input_root)
    return {
        "participant_id": relative.parts[0],
        "visit": relative.parts[1],
        "trial": relative.parts[2],
        "source_file": str(relative),
        "recording_file": path.name,
    }


def read_and_resample(path: Path, target_rate: int) -> tuple[np.ndarray, float, int]:
    frame = pd.read_csv(path)
    required = ["Timestamp", *CHANNELS]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    time_s = pd.to_numeric(frame["Timestamp"], errors="raise").to_numpy(dtype=np.float64)
    if len(time_s) < 2 or not np.isfinite(time_s).all():
        raise ValueError(f"{path}: invalid or insufficient timestamps")
    dt = np.diff(time_s)
    # The synchronized Xsens files contain a signed millisecond counter that
    # wraps by 655.35 s in a few recordings. All other intervals are regular
    # 10-ms steps. Resampling uses sample order after validating this pattern.
    wrap = np.isclose(dt, -655.35, rtol=0.0, atol=1e-6)
    ordinary = np.isclose(dt, 0.01, rtol=0.0, atol=1e-6)
    wraps = int(wrap.sum())
    if wraps > 1 or not np.all(wrap | ordinary):
        unexpected = dt[~(wrap | ordinary)][:5]
        raise ValueError(f"{path}: irregular timestamps, unexpected intervals {unexpected.tolist()}")
    source_rate = 1.0 / float(np.median(dt[ordinary]))
    if not math.isclose(source_rate, 100.0, rel_tol=0.02, abs_tol=0.5):
        raise ValueError(f"{path}: expected 100 Hz lab data, found approximately {source_rate:.4f} Hz")

    values = frame.loc[:, CHANNELS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: selected accelerometer/gyroscope channels contain NaN or infinity")

    ratio = Fraction(target_rate / source_rate).limit_denominator(10000)
    resampled = resample_poly(values, up=ratio.numerator, down=ratio.denominator, axis=0)
    return resampled.astype(np.float32), source_rate, wraps


def prepare_limu_windows(records: list[Path], input_root: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    all_windows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for path in records:
        values, source_rate, timestamp_wraps = read_and_resample(path, 20)
        # Training inputs were stored as SI acceleration and divided by 9.8
        # in LIMU's input collate function. Xsens gyroscope output in this
        # synchronized dataset is rad/s; the public VAE gyroscope values are
        # degree/s, so convert to the public-training scale.
        values[:, 3:] *= np.float32(180.0 / np.pi)
        metadata = recording_metadata(path, input_root)
        for start in range(0, len(values) - 20 + 1, 20):
            window = values[start : start + 20].copy()
            window[:, :3] /= np.float32(9.8)
            all_windows.append(window)
            rows.append({
                **metadata,
                "channel": "all_six",
                "window_index": start // 20,
                "window_start_s": start / 20.0,
                "sampling_rate_hz": 20,
                "window_samples": 20,
                "window_seconds": 1.0,
                "timestamp_counter_wraps": timestamp_wraps,
            })
    if not all_windows:
        raise RuntimeError("No LIMU windows were created from NW sacrum files")
    return np.stack(all_windows), rows


def prepare_hart_windows(records: list[Path], input_root: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    all_windows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    window_length, hop = 120, 30
    for path in records:
        values, _, timestamp_wraps = read_and_resample(path, 30)
        metadata = recording_metadata(path, input_root)
        means = values.mean(axis=0)
        stds = np.maximum(values.std(axis=0), np.float32(1e-5))
        standardized = (values - means) / stds
        for start in range(0, len(standardized) - window_length + 1, hop):
            all_windows.append(standardized[start : start + window_length].copy())
            row = {
                **metadata,
                "channel": "all_six",
                "window_index": start // hop,
                "window_start_s": start / 30.0,
                "sampling_rate_hz": 30,
                "window_samples": 120,
                "window_seconds": 4.0,
                "timestamp_counter_wraps": timestamp_wraps,
            }
            for channel_name, mean, std in zip(HART_CHANNELS, means, stds):
                row[f"normalization_mean_{channel_name}"] = float(mean)
                row[f"normalization_std_{channel_name}"] = float(std)
            rows.append(row)
    if not all_windows:
        raise RuntimeError("No HART windows were created from NW sacrum files")
    return np.stack(all_windows), rows


def load_limu(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(ROOT / "models/LIMU-BERT-X"))
    from config import PretrainModelConfig
    from models import LIMUBertModel4Pretrain

    config_data = json.loads((ROOT / "models/LIMU-BERT-X/config/limu_bert.json").read_text())
    config = PretrainModelConfig.from_json(config_data["base_v4"])
    model = LIMUBertModel4Pretrain(config, output_embed=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_hart(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction

    return TSPulseForReconstruction.from_pretrained(
        str(checkpoint), local_files_only=True, num_input_channels=6
    ).to(device).eval()


@torch.inference_mode()
def infer_embeddings(
    model_name: str,
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
        if model_name == "limu":
            hidden = model(batch)  # [batch, 20 time points, 72 hidden units]
            pooled = hidden.mean(dim=1)
        else:
            observed = torch.ones_like(batch, dtype=torch.bool)
            output = model.backbone(
                batch,
                past_observed_mask=observed,
                output_hidden_states=False,
                return_dict=True,
                enable_masking=False,
            )
            # HART returns [batch, channels, patches, hidden units]. Pool patches,
            # then retain all six channel representations in one window vector.
            pooled = output.last_hidden_state.mean(dim=2).flatten(start_dim=1)
        embeddings.append(pooled.float().cpu().numpy())
    result = np.concatenate(embeddings, axis=0)
    if result.shape[0] != len(windows) or not np.isfinite(result).all():
        raise RuntimeError(f"{model_name} produced invalid embeddings: {result.shape}")
    return result


def write_output(destination: Path, rows: list[dict[str, object]], embeddings: np.ndarray) -> None:
    metadata = pd.DataFrame(rows)
    vectors = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{index:03d}" for index in range(embeddings.shape[1])],
    )
    frame = pd.concat([metadata, vectors], axis=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame):,} windows x {embeddings.shape[1]} dimensions: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    records = sorted(args.input_root.glob("SRS*/Visit*/NW*/sacrum/*.csv"))
    if not records:
        raise RuntimeError(f"No NW sacrum CSV files found under {args.input_root}")
    participants = sorted({recording_metadata(path, args.input_root)["participant_id"] for path in records})
    expected = {f"SRS{i:02d}" for i in range(1, 19)}
    missing_participants = sorted(expected - set(participants))
    print(f"NW sacrum recordings: {len(records)} from {len(participants)} participants")
    if missing_participants:
        print("Participants without NW sacrum CSVs: " + ", ".join(missing_participants))

    device = resolve_device(args.device)
    print(f"Inference device: {device}")
    limu_windows, limu_rows = prepare_limu_windows(records, args.input_root)
    hart_windows, hart_rows = prepare_hart_windows(records, args.input_root)
    print(f"Prepared LIMU inputs: {limu_windows.shape}; HART inputs: {hart_windows.shape}")

    limu_checkpoints = {
        "generic": ROOT / "models/LIMU-BERT-X/weights/limu_bert_x.pt",
        "stroke_adapted": ROOT / "Data/trained_models/public_stroke/limu/best/model.pt",
    }
    hart_checkpoints = {
        "generic": ROOT / "models/HART/finetuning_HART/pretrained_checkpoint",
        "stroke_adapted": ROOT / "Data/trained_models/public_stroke/hart/best",
    }
    for checkpoint in [*limu_checkpoints.values(), *hart_checkpoints.values()]:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    for version, checkpoint in limu_checkpoints.items():
        print(f"Extracting LIMU-BERT-X {version} embeddings")
        model = load_limu(checkpoint, device)
        embeddings = infer_embeddings("limu", model, limu_windows, device, args.batch_size)
        write_output(args.output_root / f"limu_{version}.csv", limu_rows, embeddings)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    for version, checkpoint in hart_checkpoints.items():
        print(f"Extracting HART {version} embeddings")
        model = load_hart(checkpoint, device)
        embeddings = infer_embeddings("hart", model, hart_windows, device, args.batch_size)
        write_output(args.output_root / f"hart_{version}.csv", hart_rows, embeddings)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    summary = {
        "task_filter": "NW treadmill walking trials only; sacrum sensor only",
        "participant_ids_found": participants,
        "participant_ids_without_imu_files": missing_participants,
        "source_sampling_rate_hz": 100,
        "limu": {
            "target_rate_hz": 20,
            "window_samples": 20,
            "hop_samples": 20,
            "input_shape": list(limu_windows.shape),
            "embedding_dimension": 72,
            "pooling": "mean over the 20 contextual time-step embeddings",
            "accelerometer_input": "m/s^2 divided by 9.8 to match public-stroke training scale",
            "gyroscope_input": "rad/s converted to degree/s to match public-stroke training scale",
        },
        "hart": {
            "target_rate_hz": 30,
            "window_samples": 120,
            "hop_samples": 30,
            "input_shape": list(hart_windows.shape),
            "embedding_dimension": 288,
            "input_mode": "six channels jointly; channel dimension retained in the embedding",
            "pooling": "mean over unmasked contextual patch embeddings, then flatten six channels",
            "channels": list(HART_CHANNELS),
            "normalization": "per-recording, per-axis z-score",
        },
        "inference_device": str(device),
        "embeddings_are_inference_only": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
