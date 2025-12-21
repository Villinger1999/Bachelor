"""
Main entry point for the Federated Learning Framework.
This script provides a unified interface for training, attacks, defenses, and evaluation.
"""
import torch
import numpy as np
from pathlib import Path
from classes.config import FrameworkConfig, parse_args
from classes.training import UnifiedTrainer
from classes.attacks import iDLG
from classes.defenses import Defense, Clipping, SGP, clipping_threshold, pruning_threshold
from classes.evaluation import MetricsCalculator
from classes.visualization import Visualizer
from classes.helperfunctions import fix_dimension
from classes.federated_learning import fedavg_gradients, apply_gradients_to_model, evaluate_global
from torch.utils.data import DataLoader
import tensorflow as tf


def load_image_for_attack(dataset: str, image_index: int, device: str = "cpu"):
    """
    Load an image from dataset for attack.
    
    Args:
        dataset: Dataset name ("CIFAR10")
        image_index: Index of image to load
        device: Device to load on
    
    Returns:
        Tuple of (image_tensor, label_tensor)
    """
    if dataset == "CIFAR10":
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        
        # Get image and label
        orig_np = x_train[image_index].astype("float32") / 255.0  # (H,W,C) in [0,1]
        orig_tensor = torch.from_numpy(orig_np)  # (H,W,C)
        orig_tensor = orig_tensor.permute(2, 0, 1)  # -> (C,H,W)
        orig_img = orig_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,C,H,W)
        
        label_value = int(y_train[image_index][0])
        label = torch.tensor([label_value], dtype=torch.long, device=device)
        
        return orig_img, label
    else:
        raise ValueError(f"Dataset {dataset} not supported for attack")


def load_gradients(gradient_path: str, device: str = "cpu"):
    """
    Load leaked gradients from file.
    
    Args:
        gradient_path: Path to gradient file
        device: Device to load on
    
    Returns:
        Dictionary with gradients
    """
    grads_dict = torch.load(gradient_path, map_location=device, weights_only=True)
    if "grads_per_sample" in grads_dict:
        grads_dict_values = grads_dict["grads_per_sample"]
        grads_list = [v for v in grads_dict_values.values() if isinstance(v, torch.Tensor)]
        return grads_list, grads_dict
    else:
        # Assume it's already a list
        return grads_dict, None


def create_defense(defense_type: str, threshold: float = None, percentile: float = None, 
                   grads: list = None) -> Defense:
    """
    Create defense instance.
    
    Args:
        defense_type: Type of defense ("clipping", "sgp", "pruning")
        threshold: Direct threshold value
        percentile: Percentile for threshold calculation
        grads: Gradients for percentile calculation
    
    Returns:
        Defense instance
    """
    if defense_type == "clipping" or defense_type == "Clipping":
        if threshold is None:
            if percentile is None or grads is None:
                raise ValueError("Need threshold or (percentile and grads) for clipping")
            threshold = clipping_threshold(grads, percentile)
        return Clipping(threshold=threshold)
    
    elif defense_type == "sgp" or defense_type == "SGP" or defense_type == "pruning":
        if threshold is None:
            if percentile is None or grads is None:
                raise ValueError("Need threshold or (percentile and grads) for SGP")
            threshold = pruning_threshold(grads, percentile)
        return SGP(threshold=threshold)
    
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")


def main():
    """Main function."""
    # Parse configuration
    config = parse_args()
    
    print("=" * 60)
    print("Federated Learning Framework")
    print("=" * 60)
    print(f"Mode: {config.training.mode}")
    print(f"Model: {config.training.model_name}")
    print(f"Dataset: {config.training.dataset}")
    print(f"Device: {config.training.device}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = UnifiedTrainer(
        model_name=config.training.model_name,
        dataset=config.training.dataset,
        num_classes=config.training.num_classes,
        batch_size=config.training.batch_size,
        device=config.training.device,
        pretrained_path=config.training.pretrained_path
    )
    
    # Create defense if enabled
    # Note: Defense threshold will be computed during training/attack if percentile is provided
    defense = None
    defense_config = None
    if config.defense.enabled:
        print(f"\nSetting up defense: {config.defense.defense_type}")
        # If threshold is directly provided, create defense now
        # Otherwise, store config to create defense during training/attack when gradients are available
        if config.defense.threshold is not None:
            defense = create_defense(
                config.defense.defense_type,
                threshold=config.defense.threshold,
                percentile=None
            )
        else:
            # Store config for dynamic creation
            defense_config = config.defense
    
    # Training phase
    if config.training.mode == "normal":
        print("\n" + "=" * 60)
        print("Starting Normal Training")
        print("=" * 60)
        # For normal training, defense is applied during gradient capture
        # If defense_config is set, we'll need to create it dynamically
        model, grads_dict = trainer.train_normal(
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            defense=defense,
            save_grads=config.training.save_grads,
            run_id=config.training.run_id
        )
    elif config.training.mode == "federated":
        print("\n" + "=" * 60)
        print("Starting Federated Learning")
        print("=" * 60)
        # For federated training, defense is applied in Client.train_local
        # The Client class will handle dynamic defense creation if needed
        model, _ = trainer.train_federated(
            num_rounds=config.training.num_rounds,
            local_epochs=config.training.local_epochs,
            learning_rate=config.training.learning_rate,
            num_clients=config.training.num_clients,
            client_fraction=config.training.client_fraction,
            defense=defense,
            save_grads=config.training.save_grads,
            run_id=config.training.run_id
        )
        grads_dict = None  # Gradients saved internally if requested
    else:
        raise ValueError(f"Unknown training mode: {config.training.mode}")
    
    # Attack phase
    if config.attack.enabled:
        print("\n" + "=" * 60)
        print("Starting Attack")
        print("=" * 60)
        
        # Load or prepare image
        if config.attack.image_index is not None:
            orig_img, label = load_image_for_attack(
                config.training.dataset,
                config.attack.image_index,
                config.training.device
            )
        elif config.attack.image_path:
            # Load from file (implement if needed)
            raise NotImplementedError("Loading from image path not yet implemented")
        else:
            raise ValueError("Need image_index or image_path for attack")
        
        # Load gradients
        grads = None
        if config.attack.gradient_source == "leaked":
            if config.attack.gradient_path is None:
                raise ValueError("gradient_path required for leaked gradients")
            grads, grads_dict = load_gradients(
                config.attack.gradient_path,
                config.training.device
            )
        elif config.attack.gradient_source == "original":
            # Gradients will be computed from original image
            grads = None
        
        # Create defense for attack if needed
        attack_defense = None
        if config.defense.enabled and grads is not None:
            # Need to compute threshold from gradients
            attack_defense = create_defense(
                config.defense.defense_type,
                threshold=config.defense.threshold,
                percentile=config.defense.percentile,
                grads=grads
            )
        
        # Initialize attack
        attacker = iDLG(
            model=model,
            label=label,
            seed=config.attack.seed,
            clamp=config.attack.clamp,
            device=config.training.device,
            orig_img=orig_img,
            grads=grads,
            defense=attack_defense,
            random_dummy=config.attack.random_dummy,
            dummy_var=config.attack.dummy_variance,
        )
        
        # Execute attack
        attack_result = attacker.attack(iterations=config.attack.iterations)
        dummy_init, reconstructed, pred_label, history, losses, reconstructed_grads = attack_result
        
        print(f"\nAttack completed!")
        print(f"Predicted label: {pred_label.item()}")
        print(f"True label: {label.item()}")
        print(f"Final loss: {losses[-1]:.6f}")
        
        # Initialize eval_results for later use
        eval_results = {}
        
        # Post-attack FedAvg: Use attack gradients to update model
        if config.attack.use_attack_gradients_in_fedavg and reconstructed_grads is not None:
            print("\n" + "=" * 60)
            print("Post-Attack FedAvg: Updating Model with Attack Gradients")
            print("=" * 60)
            
            # Apply defense to reconstructed gradients if enabled
            defended_grads = reconstructed_grads
            if config.defense.enabled:
                print(f"Applying defense ({config.defense.defense_type}) to attack gradients...")
                if config.defense.threshold is not None:
                    post_attack_defense = create_defense(
                        config.defense.defense_type,
                        threshold=config.defense.threshold,
                        percentile=None
                    )
                elif config.defense.percentile is not None:
                    post_attack_defense = create_defense(
                        config.defense.defense_type,
                        threshold=None,
                        percentile=config.defense.percentile,
                        grads=reconstructed_grads
                    )
                else:
                    post_attack_defense = None
                
                if post_attack_defense is not None:
                    defended_grads = post_attack_defense.apply(reconstructed_grads)
                    print("Defense applied to attack gradients.")
            
            # Apply FedAvg (in this case, just using the single gradient set)
            # If you have other gradients, you can aggregate them here
            aggregated_grads = fedavg_gradients(
                [defended_grads],
                sample_weights=[config.attack.fedavg_sample_weight]
            )
            
            # Update model with aggregated gradients
            print(f"Updating model with learning rate: {config.attack.fedavg_learning_rate}")
            model = apply_gradients_to_model(
                model,
                aggregated_grads,
                learning_rate=config.attack.fedavg_learning_rate
            )
            
            print("Model updated with attack gradients!")
            
            # Evaluate updated model accuracy
            if config.evaluation.metrics and 'accuracy' in config.evaluation.metrics:
                print("\n" + "=" * 60)
                print("Evaluating Updated Model Accuracy")
                print("=" * 60)
                
                # Load test dataset for evaluation
                if config.training.dataset == "CIFAR10":
                    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
                    x_test_torch = torch.tensor(
                        x_test.transpose((0, 3, 1, 2)), dtype=torch.float32
                    ) / 255.0
                    y_test_torch = torch.tensor(y_test.squeeze(), dtype=torch.long)
                    from torch.utils.data import TensorDataset
                    testset = TensorDataset(x_test_torch, y_test_torch)
                    testloader = DataLoader(testset, batch_size=config.training.batch_size, shuffle=False)
                else:
                    # For other datasets, you'd need to load them appropriately
                    print(f"Warning: Accuracy evaluation not implemented for {config.training.dataset}")
                    testloader = None
                
                if testloader is not None:
                    accuracy = evaluate_global(model, testloader, config.training.device)
                    print(f"Model Accuracy after FedAvg update: {accuracy:.4f}")
                    
                    # Add to evaluation results
                    eval_results['accuracy'] = accuracy
        
        # Evaluation (image quality metrics)
        if config.evaluation.metrics:
            print("\n" + "=" * 60)
            print("Evaluation Metrics")
            print("=" * 60)
            
            # Image quality metrics (PSNR, SSIM)
            image_metrics = [m for m in config.evaluation.metrics if m in ['psnr', 'ssim']]
            if image_metrics:
                image_eval = MetricsCalculator.evaluate_all(
                    original=orig_img,
                    reconstructed=reconstructed,
                    metrics=image_metrics,
                    data_range=config.evaluation.data_range
                )
                eval_results.update(image_eval)
            
            # Print all metrics
            for metric_name, metric_value in eval_results.items():
                print(f"{metric_name.upper()}: {metric_value:.6f}")
        
        # Visualization
        if config.visualization.enabled:
            print("\n" + "=" * 60)
            print("Generating Visualizations")
            print("=" * 60)
            
            metadata = {
                "Attack": config.attack.attack_type,
                "Model": config.training.model_name,
                "Gradient Source": config.attack.gradient_source,
                "Defense": config.defense.defense_type if config.defense.enabled else "None"
            }
            
            save_path = config.visualization.save_path
            if save_path is None:
                save_path = f"reconstruction_{config.training.run_id or 'default'}.png"
            
            Visualizer.visualize_reconstruction(
                original=orig_img,
                reconstructed=reconstructed,
                dummy=dummy_init,
                losses=losses,
                pred_label=pred_label,
                true_label=label,
                metadata=metadata,
                save_path=save_path,
                show=config.visualization.show
            )
            
            print(f"Visualization saved to: {save_path}")
    
    print("\n" + "=" * 60)
    print("Framework execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

