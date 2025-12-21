"""
Unified training interface for both normal and federated learning.
"""
import torch
import copy
from typing import Optional, Dict, Tuple, List
from classes.federated_learning import Client, FederatedTrainer, fedavg, evaluate_global
from classes.defenses import Defense
from classes.models import LeNet, get_model
from classes.datasets import get_federated_cifar10, get_federated_imagenet
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf


class UnifiedTrainer:
    """Unified interface for normal and federated training."""
    
    def __init__(
        self,
        model_name: str = "LeNet",
        dataset: str = "CIFAR10",
        num_classes: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize unified trainer.
        
        Args:
            model_name: Name of model ("LeNet" or "ResNet18")
            dataset: Dataset name ("CIFAR10" or "ImageNet")
            num_classes: Number of classes
            batch_size: Batch size
            device: Device to use
            pretrained_path: Path to pretrained model weights
        """
        self.model_name = model_name
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        
        # Initialize model
        if model_name == "LeNet":
            self.model = LeNet(num_classes=num_classes)
        elif model_name == "ResNet18":
            self.model = get_model(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if pretrained_path:
            self.model.load_state_dict(
                torch.load(pretrained_path, map_location=device, weights_only=True)
            )
        
        self.model = self.model.to(device)
    
    def train_normal(
        self,
        epochs: int = 100,
        learning_rate: float = 0.01,
        defense: Optional[Defense] = None,
        save_grads: bool = False,
        run_id: Optional[str] = None
    ) -> Tuple[torch.nn.Module, Optional[Dict]]:
        """
        Train model normally (non-federated).
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            defense: Defense to apply (optional)
            save_grads: Whether to save gradients
            run_id: Run identifier for saving
        
        Returns:
            Trained model and optional gradients dict
        """
        # Load dataset
        if self.dataset == "CIFAR10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            
            x_train_torch = torch.tensor(
                x_train.transpose((0, 3, 1, 2)), dtype=torch.float32
            ) / 255.0
            x_test_torch = torch.tensor(
                x_test.transpose((0, 3, 1, 2)), dtype=torch.float32
            ) / 255.0
            
            y_train_torch = torch.tensor(y_train.squeeze(), dtype=torch.long)
            y_test_torch = torch.tensor(y_test.squeeze(), dtype=torch.long)
            
            trainset = TensorDataset(x_train_torch, y_train_torch)
            testset = TensorDataset(x_test_torch, y_test_torch)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported for normal training")
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        
        # Train model
        local_model = copy.deepcopy(self.model).to(self.device)
        local_model.train()
        
        import torch.nn as nn
        import torch.optim as optim
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        
        captured_grads = None
        captured_labels = None
        
        for epoch in range(epochs):
            running_loss = 0.0
            num_steps = 0
            
            for images, labels in trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Capture gradients from last batch
                if save_grads and epoch == epochs - 1:
                    from classes.federated_learning import grad_state_dict
                    captured_grads_dict = grad_state_dict(local_model)
                    captured_labels = labels.detach().cpu().clone()
                    
                    # Apply defense if provided
                    if defense is not None:
                        # Convert dict to list for defense application
                        captured_grads_list = [v for v in captured_grads_dict.values() if isinstance(v, torch.Tensor)]
                        defended_grads_list = defense.apply(captured_grads_list)
                        # Convert back to dict (assuming order is preserved)
                        defended_grads_dict = {}
                        for i, (key, _) in enumerate(captured_grads_dict.items()):
                            if i < len(defended_grads_list):
                                defended_grads_dict[key] = defended_grads_list[i]
                        captured_grads = defended_grads_dict
                    else:
                        captured_grads = captured_grads_dict
                
                optimizer.step()
                running_loss += loss.item()
                num_steps += 1
            
            avg_loss = running_loss / max(1, num_steps)
            acc = evaluate_global(local_model, testloader, self.device)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
        
        # Prepare gradients dict if saved
        grads_dict = None
        if save_grads and captured_grads is not None:
            grads_dict = {
                "grads_per_sample": captured_grads,
                "labels_per_sample": captured_labels,
                "model_state": copy.deepcopy(local_model.state_dict()),
            }
            if run_id:
                import os
                os.makedirs("state_dicts", exist_ok=True)
                torch.save(
                    grads_dict,
                    f"state_dicts/local_grads_normal_{run_id}.pt"
                )
        
        # Update model
        self.model.load_state_dict(local_model.state_dict())
        
        return self.model, grads_dict
    
    def train_federated(
        self,
        num_rounds: int = 10,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        num_clients: int = 6,
        client_fraction: float = 1.0,
        defense: Optional[Defense] = None,
        save_grads: bool = False,
        run_id: Optional[str] = None
    ) -> Tuple[torch.nn.Module, Optional[List[Dict]]]:
        """
        Train model using federated learning.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Number of local epochs per round
            learning_rate: Learning rate
            num_clients: Number of clients
            client_fraction: Fraction of clients participating per round
            defense: Defense to apply (optional)
            save_grads: Whether to save gradients
            run_id: Run identifier for saving
        
        Returns:
            Trained model and optional list of gradients dicts
        """
        # Load federated dataset
        if self.dataset == "CIFAR10":
            client_datasets, testloader, _ = get_federated_cifar10(
                num_clients, self.batch_size
            )
        elif self.dataset == "ImageNet":
            client_datasets, testloader, _ = get_federated_imagenet(
                num_clients, self.batch_size
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        
        # Create clients
        clients = []
        for i, ds in enumerate(client_datasets):
            clients.append(
                Client(
                    client_id=i,
                    dataset=ds,
                    batch_size=self.batch_size,
                    device=self.device
                )
            )
        
        # Create trainer
        trainer = FederatedTrainer(
            global_model=copy.deepcopy(self.model),
            clients=clients,
            testloader=testloader,
            C=client_fraction,
            device=self.device,
            aggregator=fedavg
        )
        
        # Train
        last_states, trained_model = trainer.train(
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            defense=defense,
            save_grads=save_grads,
            run_id=run_id
        )
        
        # Update model
        self.model = trained_model
        
        return self.model, None  # Gradients are saved internally if requested
    
    def get_model(self) -> torch.nn.Module:
        """Get the current model."""
        return self.model

