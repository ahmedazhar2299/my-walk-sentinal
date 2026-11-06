# Public stroke self-supervised training

The training script uses participant-level train and validation arrays prepared under `Data/model_inputs/public_stroke`. It starts from the included general pretrained LIMU-BERT-X or HART checkpoint, then continues masked-signal reconstruction on public stroke windows. It does not create a new split or use labels.

## Environment

On Apple silicon, create the isolated environment and install the pinned macOS training dependencies:

```sh
python3 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-macos-training.txt
```

Confirm that PyTorch sees the Apple GPU:

```sh
.venv-training/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

## One batch check

Run a short forward/backward update on one training batch and one validation batch for each model:

```sh
.venv-training/bin/python scripts/train_public_stroke.py limu --smoke
.venv-training/bin/python scripts/train_public_stroke.py hart --smoke
```

Smoke checkpoints are written under the ignored `Data/trained_models/public_stroke` directory. Full training overwrites the corresponding model's history and best checkpoint.

## Full training

Run each model separately. The defaults use batch size 16, up to 30 epochs, and early stopping after five epochs without validation-loss improvement:

```sh
.venv-training/bin/python scripts/train_public_stroke.py limu
.venv-training/bin/python scripts/train_public_stroke.py hart
```

For HART's joint six-channel setup, first build the multivariate windows using the same participant split, then run:

```sh
.venv-training/bin/python scripts/prepare_hart_joint_data.py
.venv-training/bin/python scripts/train_public_stroke.py hart
```

This provides each HART window as `[120 samples, 6 channels]`, with the axes in the order accelerometer X/Y/Z followed by gyroscope X/Y/Z. It starts from the released one-channel HART checkpoint and configures its unchanged TSPulse backbone for six input channels, as supported by HART's multichannel fine-tuning implementation. The stroke-adapted checkpoint is saved under `Data/trained_models/public_stroke/hart/`.

Use `--batch-size` to lower the batch if memory becomes constrained, or `--epochs` and `--patience` to change the training schedule. Model weights and per-epoch loss history are written under `Data/trained_models/public_stroke/<model>/`.

The HART architecture source is not modified. HART uses the released context-120 checkpoint and continues its reconstruction objective on six-channel windows, matching the official multichannel fine-tuning input format. LIMU acceleration is converted from m/s² to g in its input pipeline, matching the model's original normalization; gyroscope values are unchanged.
