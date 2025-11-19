import torch

leaked_grads = torch.load("state_dicts/local_grads_client0_c1_b1_e3.pt", map_location=torch.device('cpu'), weights_only=True)

def print_structure(obj, depth=0):
    indent = "  " * depth
    if isinstance(obj, dict):
        print(f"{indent}Dict with keys: {list(obj.keys())}")
        for key, value in obj.items():
            print(f"{indent}  {key}:")
            print_structure(value, depth + 2)
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}Tensor: dtype={obj.dtype}, shape={obj.shape}")
    else:
        print(f"{indent}{type(obj).__name__}: {obj}")
        
labels = leaked_grads['labels_per_sample']
print(f"\nLabels: {labels}")
print(f"Labels type: {type(labels)}, shape/length: {len(labels) if isinstance(labels, (list, torch.Tensor)) else 'N/A'}")

print(f"Type: {type(leaked_grads)}")
print(leaked_grads)