## CIFAR-10 ResNet-18 Training Pipeline

A complete PyTorch training pipeline for **CIFAR-10** image classification using a **ResNet-18** backbone. This script handles dataset discovery, automatic extraction, training with augmentation, learning rate scheduling, early stopping, checkpoint resume, and TensorBoard logging – all in a single, self-contained Python file.

## Features

- **Intelligent Dataset Handling** – Automatically locates `cifar-10-python.tar.gz` (current directory or `--data-dir`) and extracts it to `./data/` if not already present.
- **ResNet‑18 Model** – Pre‑trained weights are **not** used; the model is trained from scratch with a custom 10‑class fully‑connected layer.
- **Data Augmentation** – Random cropping (with padding) and horizontal flipping for better generalisation.
- **Learning Rate Scheduling** – `ReduceLROnPlateau` halves the learning rate when validation accuracy plateaus (patience=3).
- **Early Stopping** – Stops training after `--patience` epochs without validation improvement (default 10).
- **Checkpoint Resume** – Resume from any saved `.pth` checkpoint; the learning rate is automatically reduced for fine‑tuning.
- **TensorBoard Logging** – Logs training/validation loss and validation accuracy.
- **GPU Support** – Automatically uses CUDA when available.

## Requirements

- Python ≥ 3.6
- PyTorch ≥ 1.8
- torchvision
- tensorboard (optional, for visualisation)

Install all dependencies:

```bash
pip install torch torchvision tensorboard
```

## Dataset Preparation

Place the CIFAR-10 archive (`cifar-10-python.tar.gz`) in one of the following locations:

- The current working directory, **or**
- Any directory specified via `--data-dir`.

The script will automatically extract it to `./data/cifar-10-batches-py/` if not already present.  
No manual download is required – just obtain the archive from the [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

## Usage

### Basic Training (from scratch)

```bash
python train.py
```

### Resume Training from a Checkpoint

```bash
python train.py --resume CIFAR10_best.pth
```

> **Note:** When resuming, the learning rate is reduced by a factor of 10 (capped at 1e‑5) to allow fine‑tuning.

### Customise Training

```bash
python train.py --data-dir /path/to/dataset \
                --lr 0.01 \
                --batch_size 64 \
                --epochs 150 \
                --patience 15
```

## Command‑Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | `str` | `.` | Directory containing `cifar-10-python.tar.gz`. The script also checks the current folder. |
| `--resume`  | `str` | `None` | Path to a checkpoint file (`.pth`) to resume training from. |
| `--lr`      | `float` | `0.001` | Initial learning rate. |
| `--batch_size` | `int` | `128` | Batch size for training. |
| `--epochs`  | `int` | `100` | Total number of training epochs. |
| `--patience`| `int` | `10` | Number of epochs with no improvement after which early stopping is triggered. |

## Outputs

- **Best Model** – Saved as `CIFAR10_best.pth` whenever validation accuracy improves. Contains:
  - model state dict
  - optimizer state dict
  - scheduler state dict
  - best accuracy
  - random number generator state (for reproducibility)
- **TensorBoard Logs** – Stored in `logs_cifar10_<timestamp>/`. Launch TensorBoard with:
  ```bash
  tensorboard --logdir logs_cifar10_*
  ```
- **Console Logs** – Per‑epoch metrics: training loss, training accuracy, validation loss, validation accuracy, and current learning rate.

## Expected Performance

With the default hyperparameters, the model typically achieves **~94‑95%** validation accuracy after 70-80epochs. Your results may vary depending on random initialisation and hardware.

## License

This project is open‑source and available under the **MIT License**.
```