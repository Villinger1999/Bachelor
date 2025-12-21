"""
Example usage of the Federated Learning Framework.

This script demonstrates various use cases of the framework.
"""
import subprocess
import sys

def run_example(name, command):
    """Run an example command."""
    print(f"\n{'='*60}")
    print(f"Example: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    # Uncomment to actually run:
    # subprocess.run(command)

if __name__ == "__main__":
    print("Federated Learning Framework - Example Usage")
    print("=" * 60)
    
    # Example 1: Normal training
    run_example(
        "Normal Training",
        ["python", "framework_main.py", "--mode", "normal", "--epochs", "10", "--batch-size", "64"]
    )
    
    # Example 2: Federated learning
    run_example(
        "Federated Learning",
        ["python", "framework_main.py", 
         "--mode", "federated",
         "--num-clients", "6",
         "--num-rounds", "10",
         "--local-epochs", "1"]
    )
    
    # Example 3: Training with defense
    run_example(
        "Federated Learning with Clipping Defense",
        ["python", "framework_main.py",
         "--mode", "federated",
         "--num-clients", "6",
         "--num-rounds", "10",
         "--defense", "clipping",
         "--defense-threshold", "0.1",
         "--save-grads",
         "--run-id", "exp1"]
    )
    
    # Example 4: Attack with original gradients
    run_example(
        "Attack using Original Gradients",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "original",
         "--metrics", "psnr", "ssim",
         "--save-visualization", "attack_original.png"]
    )
    
    # Example 5: Attack with leaked gradients
    run_example(
        "Attack using Leaked Gradients",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_exp1.pt",
         "--metrics", "psnr", "ssim", "accuracy",
         "--save-visualization", "attack_leaked.png",
         "--show"]
    )
    
    # Example 5b: Attack with random dummy initialization
    run_example(
        "Attack with Random Dummy Initialization",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_exp1.pt",
         "--random-dummy",
         "--metrics", "psnr", "ssim",
         "--save-visualization", "attack_random_dummy.png"]
    )
    
    # Example 5c: Attack with NoiseGenerator (low variance)
    run_example(
        "Attack with NoiseGenerator - Low Variance (0.01)",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_exp1.pt",
         "--no-random-dummy",
         "--dummy-variance", "0.01",
         "--metrics", "psnr", "ssim",
         "--save-visualization", "attack_noise_0.01.png"]
    )
    
    # Example 5d: Attack with NoiseGenerator (medium variance)
    run_example(
        "Attack with NoiseGenerator - Medium Variance (0.1)",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_exp1.pt",
         "--no-random-dummy",
         "--dummy-variance", "0.1",
         "--metrics", "psnr", "ssim",
         "--save-visualization", "attack_noise_0.1.png"]
    )
    
    # Example 5e: Attack with NoiseGenerator (high variance)
    run_example(
        "Attack with NoiseGenerator - High Variance (0.3)",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_exp1.pt",
         "--no-random-dummy",
         "--dummy-variance", "0.3",
         "--metrics", "psnr", "ssim",
         "--save-visualization", "attack_noise_0.3.png"]
    )
    
    # Example 5f: Comparing dummy initialization strategies
    print("\n" + "=" * 60)
    print("Example: Compare Dummy Initialization Strategies")
    print("=" * 60)
    print("Run multiple attacks with different dummy initializations:")
    print("\n1. Random dummy initialization:")
    print("   python framework_main.py --mode normal --pretrained-path state_dicts/global_model_state_exp1.pt \\")
    print("        --attack --image-index 0 --gradient-source leaked \\")
    print("        --gradient-path state_dicts/local_grads_client0_exp1.pt --random-dummy \\")
    print("        --save-visualization compare_random.png")
    print("\n2. NoiseGenerator with variance 0.01:")
    print("   python framework_main.py --mode normal --pretrained-path state_dicts/global_model_state_exp1.pt \\")
    print("        --attack --image-index 0 --gradient-source leaked \\")
    print("        --gradient-path state_dicts/local_grads_client0_exp1.pt --no-random-dummy --dummy-variance 0.01 \\")
    print("        --save-visualization compare_noise_0.01.png")
    print("\n3. NoiseGenerator with variance 0.1:")
    print("   python framework_main.py --mode normal --pretrained-path state_dicts/global_model_state_exp1.pt \\")
    print("        --attack --image-index 0 --gradient-source leaked \\")
    print("        --gradient-path state_dicts/local_grads_client0_exp1.pt --no-random-dummy --dummy-variance 0.1 \\")
    print("        --save-visualization compare_noise_0.1.png")
    print("=" * 60 + "\n")
    
    # Example 6: Full pipeline
    run_example(
        "Full Pipeline: Train + Attack + Evaluate",
        ["python", "framework_main.py",
         "--mode", "federated",
         "--num-clients", "6",
         "--num-rounds", "10",
         "--defense", "sgp",
         "--defense-percentile", "0.9",
         "--save-grads",
         "--run-id", "full_exp",
         "--attack",
         "--image-index", "5",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_full_exp.pt",
         "--metrics", "psnr", "ssim",
         "--save-visualization", "full_pipeline_result.png"]
    )
    
    # Example 7: Attack with post-attack FedAvg and accuracy evaluation
    run_example(
        "Attack + FedAvg + Accuracy Evaluation",
        ["python", "framework_main.py",
         "--mode", "normal",
         "--pretrained-path", "state_dicts/global_model_state_exp1.pt",
         "--attack",
         "--attack-type", "iDLG",
         "--image-index", "0",
         "--gradient-source", "leaked",
         "--gradient-path", "state_dicts/local_grads_client0_exp1.pt",
         "--use-attack-gradients-in-fedavg",
         "--fedavg-learning-rate", "0.01",
         "--defense", "clipping",
         "--defense-threshold", "0.1",
         "--metrics", "psnr", "ssim", "accuracy",
         "--save-visualization", "attack_fedavg_result.png"]
    )
    
    # Example 8: Using config file
    print("\n" + "=" * 60)
    print("Example: Using JSON Config File")
    print("=" * 60)
    print("Config files are ideal for:")
    print("  - Reproducible experiments")
    print("  - Complex configurations")
    print("  - Running multiple variations")
    print("  - Sharing with collaborators")
    print("\n1. Save current configuration to file:")
    print("   python framework_main.py --mode federated --num-clients 6 \\")
    print("        --defense clipping --defense-threshold 0.1 --save-grads \\")
    print("        --save-config experiments/my_experiment.json")
    print("\n2. Run using the saved config:")
    print("   python framework_main.py --config experiments/my_experiment.json")
    print("\n3. Edit the JSON file to create variations, then run again!")
    print("=" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("Note: These are example commands. Uncomment the subprocess.run()")
    print("in the script to actually execute them.")
    print("=" * 60)

