# Why Use Config Files Instead of Command-Line Arguments?

## Quick Answer

Config files are better for:
1. **Reproducibility** - Save exact experiment settings
2. **Complex configurations** - Easier to manage many parameters
3. **Reusability** - Run the same experiment multiple times
4. **Version control** - Track changes to experiments
5. **Less typing** - Especially for long commands
6. **Experiment organization** - Manage multiple experiments

## Comparison

### Command-Line (Good for quick tests)
```bash
python framework_main.py \
    --mode federated \
    --num-clients 6 \
    --num-rounds 10 \
    --local-epochs 5 \
    --defense sgp \
    --defense-percentile 0.9 \
    --save-grads \
    --run-id exp1 \
    --attack \
    --image-index 0 \
    --gradient-source leaked \
    --gradient-path state_dicts/local_grads_client0_exp1.pt \
    --no-random-dummy \
    --dummy-variance 0.1 \
    --metrics psnr ssim accuracy \
    --save-visualization result.png
```

**Problems:**
- ❌ Very long command (hard to read/edit)
- ❌ Easy to make typos
- ❌ Hard to remember all parameters
- ❌ Can't easily reuse for multiple runs
- ❌ Difficult to share with others

### Config File (Better for experiments)
```json
{
  "training": {
    "mode": "federated",
    "num_clients": 6,
    "num_rounds": 10,
    "local_epochs": 5,
    "save_grads": true,
    "run_id": "exp1"
  },
  "defense": {
    "enabled": true,
    "defense_type": "sgp",
    "percentile": 0.9
  },
  "attack": {
    "enabled": true,
    "image_index": 0,
    "gradient_source": "leaked",
    "gradient_path": "state_dicts/local_grads_client0_exp1.pt",
    "random_dummy": false,
    "dummy_variance": 0.1
  },
  "evaluation": {
    "metrics": ["psnr", "ssim", "accuracy"]
  },
  "visualization": {
    "save_path": "result.png"
  }
}
```

**Then run:**
```bash
python framework_main.py --config config.json
```

**Benefits:**
- ✅ Clean, readable format
- ✅ Easy to edit (just change the JSON)
- ✅ Can reuse: `python framework_main.py --config config.json`
- ✅ Easy to share with collaborators
- ✅ Can version control (track changes in git)

## Real-World Use Cases

### 1. **Reproducibility** (Most Important!)

**Scenario:** You run an experiment and get great results. 3 months later, you need to reproduce it.

**With command-line:**
- ❌ You have to remember/guess all the parameters
- ❌ Might use slightly different values
- ❌ Results won't match

**With config file:**
```bash
# Save your working config
python framework_main.py --mode federated --defense clipping \
    --save-config experiments/exp1_best_results.json

# Later, reproduce exactly
python framework_main.py --config experiments/exp1_best_results.json
```
- ✅ Exact same parameters every time
- ✅ Results are reproducible

### 2. **Running Multiple Variations**

**Scenario:** You want to test different defense thresholds: 0.1, 0.2, 0.3

**With command-line:**
```bash
# You'd have to type this 3 times, changing one value each time
python framework_main.py --mode federated --defense clipping --defense-threshold 0.1 ...
python framework_main.py --mode federated --defense clipping --defense-threshold 0.2 ...
python framework_main.py --mode federated --defense clipping --defense-threshold 0.3 ...
```

**With config files:**
```bash
# Create base config
python framework_main.py --mode federated --defense clipping \
    --save-config base_config.json

# Edit base_config.json to create 3 versions:
# - exp_threshold_0.1.json (change threshold to 0.1)
# - exp_threshold_0.2.json (change threshold to 0.2)
# - exp_threshold_0.3.json (change threshold to 0.3)

# Run all variations
python framework_main.py --config exp_threshold_0.1.json
python framework_main.py --config exp_threshold_0.2.json
python framework_main.py --config exp_threshold_0.3.json
```

### 3. **Sharing Experiments**

**Scenario:** You want to share your experiment setup with a colleague or include it in a paper.

**With command-line:**
- ❌ Have to copy/paste a huge command
- ❌ Easy to miss parameters
- ❌ Hard to document what each parameter does

**With config file:**
```json
{
  "training": {
    "mode": "federated",
    "num_clients": 6,
    "num_rounds": 10,
    "local_epochs": 5,
    "run_id": "paper_experiment_1"
  },
  "defense": {
    "enabled": true,
    "defense_type": "sgp",
    "percentile": 0.9
  },
  "attack": {
    "enabled": true,
    "iterations": 200,
    "gradient_source": "leaked"
  }
}
```
- ✅ Easy to share (just send the JSON file)
- ✅ Self-documenting (clear structure)
- ✅ Can add comments in a separate README

### 4. **Version Control & Experiment Tracking**

**Scenario:** You're running many experiments and want to track what changed.

**With command-line:**
- ❌ Commands in shell history are hard to organize
- ❌ Can't see what changed between experiments
- ❌ Hard to revert to a previous configuration

**With config files:**
```bash
# Organize by experiment
experiments/
  ├── exp1_baseline.json
  ├── exp2_with_defense.json
  ├── exp3_different_variance.json
  └── exp4_final.json

# Track changes with git
git add experiments/exp1_baseline.json
git commit -m "Baseline experiment configuration"
```

- ✅ Clear organization
- ✅ Can see diffs between experiments
- ✅ Easy to revert to previous configs

### 5. **Complex Configurations**

**Scenario:** You have 20+ parameters to set.

**Command-line becomes unreadable:**
```bash
python framework_main.py --mode federated --num-clients 6 --num-rounds 10 \
    --local-epochs 5 --batch-size 64 --lr 0.01 --defense clipping \
    --defense-threshold 0.1 --save-grads --run-id exp1 --attack \
    --attack-type iDLG --attack-iterations 200 --image-index 0 \
    --gradient-source leaked --gradient-path path/to/grads.pt \
    --no-random-dummy --dummy-variance 0.1 --metrics psnr ssim accuracy \
    --save-visualization result.png --device cuda
```

**Config file is organized:**
```json
{
  "training": {
    "mode": "federated",
    "num_clients": 6,
    "num_rounds": 10,
    "local_epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.01,
    "device": "cuda"
  },
  "defense": {
    "enabled": true,
    "defense_type": "clipping",
    "threshold": 0.1
  },
  "attack": {
    "enabled": true,
    "attack_type": "iDLG",
    "iterations": 200,
    "image_index": 0,
    "gradient_source": "leaked",
    "gradient_path": "path/to/grads.pt",
    "random_dummy": false,
    "dummy_variance": 0.1
  },
  "evaluation": {
    "metrics": ["psnr", "ssim", "accuracy"]
  },
  "visualization": {
    "save_path": "result.png"
  }
}
```

## When to Use Each

### Use Command-Line When:
- ✅ Quick testing/exploration
- ✅ Simple commands (few parameters)
- ✅ One-time experiments
- ✅ Interactive development

### Use Config Files When:
- ✅ Running serious experiments
- ✅ Need reproducibility
- ✅ Complex configurations
- ✅ Running multiple variations
- ✅ Sharing with others
- ✅ Version controlling experiments
- ✅ Paper/thesis experiments

## Best Practice Workflow

1. **Start with command-line** for quick testing
2. **Save working config** when you find good parameters:
   ```bash
   python framework_main.py --mode federated --defense clipping \
       --save-config my_experiment.json
   ```
3. **Use config file** for final runs and variations
4. **Organize configs** in an `experiments/` folder
5. **Version control** important configs with git

## Example: Complete Workflow

```bash
# Step 1: Quick test with command-line
python framework_main.py --mode federated --num-clients 6 --attack

# Step 2: Found good parameters? Save config
python framework_main.py --mode federated --num-clients 6 \
    --defense clipping --defense-threshold 0.1 --attack \
    --save-config experiments/baseline.json

# Step 3: Create variations
cp experiments/baseline.json experiments/variation1.json
# Edit variation1.json: change threshold to 0.2

# Step 4: Run all experiments
python framework_main.py --config experiments/baseline.json
python framework_main.py --config experiments/variation1.json

# Step 5: Track in git
git add experiments/
git commit -m "Experiment configurations for defense comparison"
```

## Summary

**Config files are essential for:**
- 🔬 **Scientific reproducibility**
- 📊 **Experiment management**
- 👥 **Collaboration**
- 📝 **Documentation**
- 🔄 **Reusability**

**Command-line is fine for:**
- 🧪 **Quick testing**
- 🎯 **Simple one-off runs**
- 💻 **Interactive development**

For your Bachelor thesis, **definitely use config files** for your final experiments - it will make your work much more professional and reproducible!

