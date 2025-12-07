import random
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def get_federated_imagenet(num_clients, batch_size,
                           train_dir='/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/val/', 
                           val_dir='/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train/',
                           train_fraction=0.05, val_fraction=0.005):
    """
    Build federated splits for ImageNet using ImageFolder + Subset.

    Returns:
        client_datasets: list[Dataset] for each client
        testloader: DataLoader for validation subset
        num_classes: int
    """
    
    img_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_full = ImageFolder(train_dir, transform=img_tf)
    val_full = ImageFolder(val_dir, transform=img_tf)

    # train subset
    num_train = len(train_full)
    train_subset_size = int(train_fraction * num_train)
    train_indices = random.sample(range(num_train), train_subset_size)
    trainset = Subset(train_full, train_indices)

    # val/test subset
    num_val = len(val_full)
    val_subset_size = int(val_fraction * num_val)
    val_indices = random.sample(range(num_val), val_subset_size)
    testset = Subset(val_full, val_indices)

    # split train subset into clients
    num_samples = len(trainset)
    client_sizes = [num_samples // num_clients] * num_clients
    client_sizes[-1] += num_samples - sum(client_sizes)
    client_datasets = random_split(trainset, client_sizes)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_classes = len(train_full.classes)

    return client_datasets, testloader, num_classes


def get_federated_cifar10(num_clients, batch_size, root="./data"):
    """
    Simple federated split for CIFAR-10.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_full = datasets.CIFAR10(root=root, train=True,
                                  download=True, transform=tf)
    testset = datasets.CIFAR10(root=root, train=False,
                               download=True, transform=tf)

    num_samples = len(train_full)
    client_sizes = [num_samples // num_clients] * num_clients
    client_sizes[-1] += num_samples - sum(client_sizes)
    client_datasets = random_split(train_full, client_sizes)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_classes = 10

    return client_datasets, testloader, num_classes
