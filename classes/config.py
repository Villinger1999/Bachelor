"""
Configuration system for the federated learning framework.
Supports both programmatic configuration and command-line arguments.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training (normal or federated)."""
    # Training mode
    mode: str = "normal"  # "normal" or "federated"
    
    # Model settings
    model_name: str = "LeNet"
    num_classes: int = 10
    pretrained_path: Optional[str] = None
    
    # Dataset settings
    dataset: str = "CIFAR10"  # "CIFAR10" or "ImageNet"
    batch_size: int = 64
    data_root: str = "./data"
    
    # Training hyperparameters
    epochs: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # Federated learning settings
    num_clients: int = 6
    num_rounds: int = 10
    local_epochs: int = 1
    client_fraction: float = 1.0  # C in FedAvg
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Save settings
    save_model: bool = True
    save_grads: bool = False
    output_dir: str = "./state_dicts"
    run_id: Optional[str] = None


@dataclass
class DefenseConfig:
    """Configuration for defenses."""
    enabled: bool = False
    defense_type: Optional[str] = None  # "clipping", "sgp", "pruning"
    threshold: Optional[float] = None
    percentile: Optional[float] = None  # For percentile-based thresholds


@dataclass
class AttackConfig:
    """Configuration for attacks."""
    enabled: bool = False
    attack_type: str = "iDLG"  # "iDLG" or others
    iterations: int = 200
    seed: Optional[int] = None
    clamp: tuple = (0.0, 1.0)
    
    # Gradient source
    gradient_source: str = "original"  # "original" or "leaked"
    gradient_path: Optional[str] = None  # Path to leaked gradients
    
    # Dummy initialization
    random_dummy: bool = True
    dummy_variance: float = 0.0
    
    # Image settings
    image_index: Optional[int] = None
    image_path: Optional[str] = None
    
    # Post-attack FedAvg settings
    use_attack_gradients_in_fedavg: bool = False  # Use attack gradients in FedAvg
    fedavg_learning_rate: float = 0.01  # Learning rate for gradient update
    fedavg_sample_weight: float = 1.0  # Weight for attack gradients in FedAvg


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = None  # ["psnr", "ssim", "accuracy", "loss"]
    data_range: float = 1.0


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    enabled: bool = True
    visualize_reconstruction: bool = True
    visualize_training: bool = False
    visualize_gradients: bool = False
    save_path: Optional[str] = None
    show: bool = False


@dataclass
class FrameworkConfig:
    """Main configuration class combining all sub-configs."""
    training: TrainingConfig = None
    defense: DefenseConfig = None
    attack: AttackConfig = None
    evaluation: EvaluationConfig = None
    visualization: VisualizationConfig = None
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.defense is None:
            self.defense = DefenseConfig()
        if self.attack is None:
            self.attack = AttackConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'training': asdict(self.training),
            'defense': asdict(self.defense),
            'attack': asdict(self.attack),
            'evaluation': asdict(self.evaluation),
            'visualization': asdict(self.visualization)
        }
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FrameworkConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            training=TrainingConfig(**data.get('training', {})),
            defense=DefenseConfig(**data.get('defense', {})),
            attack=AttackConfig(**data.get('attack', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            visualization=VisualizationConfig(**data.get('visualization', {}))
        )


def parse_args() -> FrameworkConfig:
    """Parse command-line arguments and return FrameworkConfig."""
    parser = argparse.ArgumentParser(
        description='Federated Learning Framework with Attacks and Defenses'
    )
    
    # Training arguments
    parser.add_argument('--mode', type=str, default='normal',
                       choices=['normal', 'federated'],
                       help='Training mode: normal or federated')
    parser.add_argument('--model', type=str, default='LeNet',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'ImageNet'],
                       help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (for normal training)')
    parser.add_argument('--num-rounds', type=int, default=10,
                       help='Number of federated rounds')
    parser.add_argument('--local-epochs', type=int, default=1,
                       help='Number of local epochs per round')
    parser.add_argument('--num-clients', type=int, default=6,
                       help='Number of clients (for federated learning)')
    parser.add_argument('--client-fraction', type=float, default=1.0,
                       help='Fraction of clients participating per round')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--save-grads', action='store_true',
                       help='Save gradients during training')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run identifier for saving files')
    parser.add_argument('--pretrained-path', type=str, default=None,
                       help='Path to pretrained model weights')
    
    # Defense arguments
    parser.add_argument('--defense', type=str, default=None,
                       choices=['clipping', 'sgp', 'pruning'],
                       help='Defense type')
    parser.add_argument('--defense-threshold', type=float, default=None,
                       help='Defense threshold')
    parser.add_argument('--defense-percentile', type=float, default=None,
                       help='Defense percentile for threshold calculation')
    
    # Attack arguments
    parser.add_argument('--attack', action='store_true',
                       help='Enable attack')
    parser.add_argument('--attack-type', type=str, default='iDLG',
                       help='Attack type')
    parser.add_argument('--attack-iterations', type=int, default=200,
                       help='Number of attack iterations')
    parser.add_argument('--gradient-source', type=str, default='original',
                       choices=['original', 'leaked'],
                       help='Gradient source for attack')
    parser.add_argument('--gradient-path', type=str, default=None,
                       help='Path to leaked gradients')
    parser.add_argument('--image-index', type=int, default=None,
                       help='Index of image to attack')
    parser.add_argument('--random-dummy', action='store_true',
                       help='Use random dummy initialization (default if not specified)')
    parser.add_argument('--no-random-dummy', action='store_true',
                       help='Disable random dummy initialization (use NoiseGenerator if dummy-variance > 0)')
    parser.add_argument('--dummy-variance', type=float, default=0.0,
                       help='Variance for NoiseGenerator dummy initialization (requires --no-random-dummy)')
    
    # Evaluation arguments
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['psnr', 'ssim', 'accuracy', 'loss'],
                       help='Metrics to calculate')
    
    # Visualization arguments
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save-visualization', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--show', action='store_true',
                       help='Show visualization plots')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file (overrides CLI args)')
    parser.add_argument('--save-config', type=str, default=None,
                       help='Path to save config file')
    
    args = parser.parse_args()
    
    # Create config from args
    config = FrameworkConfig()
    
    # Training config
    config.training.mode = args.mode
    config.training.model_name = args.model
    config.training.dataset = args.dataset
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.num_rounds = args.num_rounds
    config.training.local_epochs = args.local_epochs
    config.training.num_clients = args.num_clients
    config.training.client_fraction = args.client_fraction
    config.training.learning_rate = args.lr
    config.training.device = args.device
    config.training.save_grads = args.save_grads
    config.training.run_id = args.run_id
    config.training.pretrained_path = args.pretrained_path
    
    # Defense config
    if args.defense:
        config.defense.enabled = True
        config.defense.defense_type = args.defense
        config.defense.threshold = args.defense_threshold
        config.defense.percentile = args.defense_percentile
    
    # Attack config
    config.attack.enabled = args.attack
    config.attack.attack_type = args.attack_type
    config.attack.iterations = args.attack_iterations
    config.attack.gradient_source = args.gradient_source
    config.attack.gradient_path = args.gradient_path
    config.attack.image_index = args.image_index
    # Handle dummy initialization: default is random, use --no-random-dummy to enable NoiseGenerator
    # If --no-random-dummy is set, use NoiseGenerator (when dummy_variance > 0), otherwise default to random
    config.attack.random_dummy = not args.no_random_dummy  # Default True, False if --no-random-dummy
    config.attack.dummy_variance = args.dummy_variance
    config.attack.use_attack_gradients_in_fedavg = args.use_attack_gradients_in_fedavg
    config.attack.fedavg_learning_rate = args.fedavg_learning_rate
    config.attack.fedavg_sample_weight = args.fedavg_sample_weight
    
    # Evaluation config
    config.evaluation.metrics = args.metrics
    
    # Visualization config
    config.visualization.enabled = not args.no_visualize
    config.visualization.save_path = args.save_visualization
    config.visualization.show = args.show
    
    # Load from file if provided (overrides CLI args)
    if args.config:
        file_config = FrameworkConfig.load(args.config)
        # Merge: file config takes precedence
        config = file_config
    
    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
    
    return config

