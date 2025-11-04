from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the path to ImageNet
imagenet_path = "/dtu/datasets1/imagenet_object_localization_patched2019/"

resolution=264

# Define transforms (resize, crop, normalize ase needed)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(resolution),
    transforms.ToTensor(),
    # You can add normalization if needed:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = datasets.ImageFolder(root=imagenet_path + "train", transform=transform)
val_dataset = datasets.ImageFolder(root=imagenet_path + "val", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)