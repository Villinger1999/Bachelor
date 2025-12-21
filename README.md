# Federated Learning Framework with Attacks and Defenses

A modular framework for conducting federated learning experiments with gradient-based attacks (e.g., iDLG) and defenses (e.g., clipping, pruning).

## Features

- **Unified Training Interface**: Train models normally or using federated learning
- **Flexible Attack System**: Support for gradient inversion attacks (iDLG)
- **Defense Mechanisms**: Implement clipping, small gradient pruning (SGP), and more
- **Comprehensive Evaluation**: Metrics including PSNR, SSIM, accuracy, and loss
- **Visualization Tools**: Flexible visualization of reconstructions, training history, and gradients
- **Configuration System**: Easy setup via command-line arguments or JSON config files

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

#### Normal Training
```bash
python framework_main.py --mode normal --epochs 100 --batch-size 64
```

#### Federated Learning
```bash
python framework_main.py --mode federated --num-clients 6 --num-rounds 10 --local-epochs 1
```

#### Training with Defense
```bash
python framework_main.py --mode federated --defense clipping --defense-percentile 0.1
```

#### Attack with Original Gradients
```bash
python framework_main.py --mode normal --epochs 10 --attack --attack-type iDLG --image-index 0 --gradient-source original
```

#### Attack with Leaked Gradients
```bash
python framework_main.py --mode normal --attack --attack-type iDLG --image-index 0 --gradient-source leaked --gradient-path state_dicts/local_grads_client0_c1_b1_e1.pt
```

#### Full Pipeline: Training + Attack + Evaluation + Visualization
```bash
python framework_main.py \
    --mode federated \
    --num-clients 6 \
    --num-rounds 10 \
    --defense clipping \
    --defense-percentile 0.1 \
    --attack \
    --image-index 0 \
    --gradient-source leaked \
    --gradient-path state_dicts/local_grads_client0_c1_b1_e1.pt \
    --metrics psnr ssim accuracy \
    --save-visualization reconstruction.png \
    --show
```

## Configuration

### Command-Line Arguments

#### Training Options
- `--mode`: Training mode (`normal` or `federated`)
- `--model`: Model name (`LeNet` or `ResNet18`)
- `--dataset`: Dataset (`CIFAR10` or `ImageNet`)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of epochs for normal training (default: 100)
- `--num-rounds`: Number of federated rounds (default: 10)
- `--local-epochs`: Local epochs per round (default: 1)
- `--num-clients`: Number of clients (default: 6)
- `--client-fraction`: Fraction of clients per round (default: 1.0)
- `--lr`: Learning rate (default: 0.01)
- `--device`: Device (`cpu` or `cuda`)

#### Defense Options
- `--defense`: Defense type (`clipping`, `sgp`, `pruning`)
- `--defense-threshold`: Direct threshold value
- `--defense-percentile`: Percentile for threshold calculation

#### Attack Options
- `--attack`: Enable attack
- `--attack-type`: Attack type (`iDLG`)
- `--attack-iterations`: Number of attack iterations (default: 200)
- `--gradient-source`: Gradient source (`original` or `leaked`)
- `--gradient-path`: Path to leaked gradients file
- `--image-index`: Index of image to attack
- `--random-dummy`: Use random dummy initialization
- `--dummy-variance`: Variance for dummy initialization

#### Evaluation Options
- `--metrics`: List of metrics to calculate (`psnr`, `ssim`, `accuracy`, `loss`)

#### Visualization Options
- `--no-visualize`: Disable visualization
- `--save-visualization`: Path to save visualization
- `--show`: Show visualization plots

#### Config File Options
- `--config`: Path to JSON config file (overrides CLI args)
- `--save-config`: Path to save config file

### JSON Configuration

You can save and load configurations from JSON files:

```bash
# Save current config
python framework_main.py --mode federated --save-config config.json

# Load config
python framework_main.py --config config.json
```

Example JSON config:
```json
{
  "training": {
    "mode": "federated",
    "model_name": "LeNet",
    "dataset": "CIFAR10",
    "batch_size": 64,
    "num_rounds": 10,
    "local_epochs": 1,
    "num_clients": 6,
    "device": "cpu"
  },
  "defense": {
    "enabled": true,
    "defense_type": "clipping",
    "percentile": 0.1
  },
  "attack": {
    "enabled": true,
    "attack_type": "iDLG",
    "iterations": 200,
    "gradient_source": "leaked",
    "gradient_path": "state_dicts/local_grads_client0_c1_b1_e1.pt",
    "image_index": 0
  },
  "evaluation": {
    "metrics": ["psnr", "ssim", "accuracy"]
  },
  "visualization": {
    "enabled": true,
    "save_path": "reconstruction.png"
  }
}
```

## Architecture

### Modules

- **`classes/config.py`**: Configuration system with dataclasses and argument parsing
- **`classes/training.py`**: Unified training interface for normal and federated learning
- **`classes/attacks.py`**: Attack implementations (iDLG)
- **`classes/defenses.py`**: Defense mechanisms (Clipping, SGP)
- **`classes/evaluation.py`**: Evaluation metrics (PSNR, SSIM, accuracy, loss)
- **`classes/visualization.py`**: Visualization tools
- **`classes/models.py`**: Model definitions (LeNet, ResNet18)
- **`classes/datasets.py`**: Dataset loading utilities
- **`classes/federated_learning.py`**: Federated learning core (FedAvg, Client, Trainer)
- **`framework_main.py`**: Main entry point

### Workflow

1. **Configuration**: Parse command-line args or load JSON config
2. **Training**: Train model (normal or federated) with optional defense
3. **Attack** (optional): Execute gradient inversion attack
4. **Evaluation** (optional): Calculate metrics (PSNR, SSIM, accuracy, loss)
5. **Visualization** (optional): Generate and save visualizations

## Examples

### Example 1: Federated Learning with Defense
```bash
python framework_main.py \
    --mode federated \
    --num-clients 6 \
    --num-rounds 10 \
    --local-epochs 5 \
    --defense sgp \
    --defense-percentile 0.9 \
    --save-grads \
    --run-id exp1
```

### Example 2: Attack on Leaked Gradients
```bash
python framework_main.py \
    --mode normal \
    --model LeNet \
    --pretrained-path state_dicts/global_model_state_exp1.pt \
    --attack \
    --attack-type iDLG \
    --image-index 5 \
    --gradient-source leaked \
    --gradient-path state_dicts/local_grads_client0_exp1.pt \
    --metrics psnr ssim \
    --save-visualization attack_result.png
```

### Example 3: Full Experiment Pipeline
```bash
# Step 1: Train with federated learning and save gradients
python framework_main.py \
    --mode federated \
    --num-clients 6 \
    --num-rounds 10 \
    --defense clipping \
    --defense-percentile 0.1 \
    --save-grads \
    --run-id experiment1

# Step 2: Attack using leaked gradients
python framework_main.py \
    --mode normal \
    --pretrained-path state_dicts/global_model_state_experiment1.pt \
    --attack \
    --image-index 0 \
    --gradient-source leaked \
    --gradient-path state_dicts/local_grads_client0_experiment1.pt \
    --metrics psnr ssim accuracy \
    --save-visualization results/attack_exp1.png
```

## Advanced Usage

### Programmatic API

You can also use the framework programmatically:

```python
from classes.config import FrameworkConfig, TrainingConfig, AttackConfig
from classes.training import UnifiedTrainer
from classes.attacks import iDLG
from classes.defenses import Clipping
from classes.evaluation import MetricsCalculator

# Create config
config = FrameworkConfig()
config.training.mode = "federated"
config.training.num_clients = 6
config.defense.enabled = True
config.defense.defense_type = "clipping"

# Train
trainer = UnifiedTrainer(
    model_name="LeNet",
    dataset="CIFAR10",
    device="cpu"
)
model, _ = trainer.train_federated(
    num_rounds=10,
    defense=Clipping(threshold=0.1)
)

# Attack
attacker = iDLG(
    model=model,
    label=torch.tensor([5]),
    orig_img=original_image,
    device="cpu"
)
dummy, recon, pred, history, losses = attacker.attack(iterations=200)

# Evaluate
metrics = MetricsCalculator.evaluate_all(
    original=original_image,
    reconstructed=recon,
    metrics=['psnr', 'ssim']
)
```

## Output Files

- **Model checkpoints**: Saved to `state_dicts/` directory
- **Gradients**: Saved to `state_dicts/` if `--save-grads` is used
- **Visualizations**: Saved to specified path or `reconstruction_*.png`
- **Config files**: JSON format if `--save-config` is used

## Contributing

To add new attacks, defenses, or metrics:

1. **New Attack**: Inherit from `Attack` base class in `classes/attacks.py`
2. **New Defense**: Inherit from `Defense` base class in `classes/defenses.py`
3. **New Metric**: Add method to `MetricsCalculator` in `classes/evaluation.py`

## License

This project is part of a Bachelor's thesis.
