import argparse, os, csv, random
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import copy

from classes.model_change import LeNet
from classes.federated_learning import evaluate_global
from classes.attacks import iDLG
from classes.helperfunctions import compute_ssim_psnr


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = torch.tensor(x_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)
    x_test  = torch.tensor(x_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    y_test  = torch.tensor(y_test.squeeze(), dtype=torch.long)
    return x_train, y_train, x_test, y_test


def load_model(state_dict_path: str, device: str, activation: str):
    model = LeNet(activation=activation)
    sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def load_leaked_grads_list(path: str):
    leaked = torch.load(path, map_location="cpu", weights_only=True)
    grads_dict = leaked["grads_per_sample"]
    return [v for v in grads_dict.values() if isinstance(v, torch.Tensor)]


def run_repeats(img_idx: int, base_seed: int, run_once_fn, repeats: int = 100):
    """
    Runs exactly `repeats` times.
    Returns list of dicts, one per run.
    """
    runs = []
    for b in range(repeats):
        seed = base_seed + 1000 * img_idx + b
        runs.append(run_once_fn(b, seed))
    return runs

def apply_defended_grads(model, defended_grads, lr=0.01, momentum=0.9):
    """
    Apply a single SGD update using a list of defended gradients aligned with model.parameters().
    """
    if defended_grads is None:
        return  # nothing to apply

    # Create an optimizer (or pass one in if you want to reuse it)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    optimizer.zero_grad(set_to_none=True)

    # Write gradients into .grad
    for p, g in zip(model.parameters(), defended_grads):
        if g is None:
            p.grad = None
        else:
            p.grad = g.detach().to(device=p.device, dtype=p.dtype)

    optimizer.step()


def run_scenario(
    scenario_name: str,
    model_path: str,
    leaked_grads_path: Optional[str],
    image_indices: list[int],
    defense: str,
    activation: str,
    def_params: Optional[list[float]],
    iterations: int,
    base_seed: int,
    out_csv: str,
    device: str,
    repeats: int = 100,
):
    x_train, y_train, x_test, y_test = load_cifar10()

    # IMPORTANT: evaluation loader should NOT shuffle
    testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    model = load_model(model_path, device, activation)
    model_acc = evaluate_global(model, testloader, device)

    leaked_grads = None
    if "leaked_grads" in scenario_name:
        if leaked_grads_path is None:
            raise ValueError("leaked_grads_path must be provided for leaked_grads scenarios")
        leaked_grads = load_leaked_grads_list(leaked_grads_path)

    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "label_pred","label_correct",
                "ssim","psnr",
                "final_loss",
                # summary columns (only filled on AVG row)
                "avg_ssim","avg_psnr","std_ssim","std_psnr","label_acc","threshold","model_acc_before","model_acc_after"
            ])

        for img_idx in image_indices:
            label_true = int(y_train[img_idx].item())
            label = torch.tensor([label_true], dtype=torch.long, device=device)
            orig_img = x_train[img_idx].unsqueeze(0).to(device)

            for dp in def_params:

                def run_once(rep_id: int, seed: int):
                    set_all_seeds(seed)

                    attacker = iDLG(
                        model=model,
                        label=label,
                        clamp=(0.0, 1.0),
                        device=device,
                        orig_img=orig_img,
                        grads=None if "orig_grads" in scenario_name else leaked_grads,
                        defense=defense,      # "none"|"clipping"|"sgp"
                        percentile=dp,         # clipping quantile OR sgp keep_ratio OR None
                        random_dummy=True,
                        dummy_var=0.0,
                        seed=seed,
                    )
                    
                    def_save, dummy, recon, label_pred, history, losses = attacker.attack(iterations=iterations)
                    
                    # print("scenario:", scenario_name)
                    # print("passing grads:", "None" if (None if "orig_grads" in scenario_name else leaked_grads) is None else "LEAKED")

                    
                    # if img_idx == 0 and dp == def_params[0]:
                    #     print(def_save)
                    
                    if defense == "none":
                        model_acc_after = model_acc
                    else: 
                        model_copy = copy.deepcopy(model).to(device)
                        apply_defended_grads(model_copy, def_save)
                        model_acc_after = evaluate_global(model_copy, testloader, device)

                    ssim, psnr = compute_ssim_psnr(orig_img, recon)
                    label_correct = int(label_pred == label_true)
                    final_loss = losses[-1] if len(losses) else float("nan")

                    return {
                        "repeat_id": rep_id,
                        "seed": seed,
                        "label_pred": label_pred,
                        "label_correct": label_correct,
                        "ssim": float(ssim),
                        "psnr": float(psnr),
                        "final_loss": float(final_loss),
                        "model_acc_after": float(model_acc_after),
                        "threshold": float(dp) if dp is not None else None
                    }

                runs = run_repeats(img_idx, base_seed, run_once_fn=run_once, repeats=repeats)

                psnrs = [r["psnr"] for r in runs]
                ssims = [r["ssim"] for r in runs]
                label_corrs = [r["label_correct"] for r in runs]
                model_acc_after = [r["model_acc_after"] for r in runs]
                threshold = [r["threshold"] for r in runs]

                avg_psnr = float(np.mean(psnrs))
                avg_ssim = float(np.mean(ssims))
                std_psnr = float(np.std(psnrs))
                std_ssim = float(np.std(ssims))
                label_acc = float(np.mean(label_corrs))
                avg_acc_after = float(np.mean(model_acc_after))

                # write per-run rows
                for r in runs:
                    if r == 0:
                        w.writerow([scenario_name, model_path, model_acc,
                            defense, img_idx, label_true
                        ])
                    w.writerow([
                        r["label_pred"], r["label_correct"],
                        r["ssim"], r["psnr"],
                        r["final_loss"],
                        "", "", "", "", ""
                    ])

                # write one summary row (AVG)
                w.writerow([
                    "AVG", "-",
                    "-", "-",
                    "-", "-",
                    avg_ssim, avg_psnr, std_ssim, std_psnr, label_acc, threshold[0], model_acc, avg_acc_after
                ])

                f.flush()

    print(f"Finished {scenario_name} defense={defense} -> {out_csv}")

def parse_list_of_ints(s: str) -> list[int]:
    # formats: "0,1,2" or "0-9"
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--scenario", default="all",
                    choices=["all",
                             "normal_model_orig_grads",
                             "fl_model_orig_grads",
                             "normal_model_leaked_grads",
                             "fl_model_leaked_grads"])
    ap.add_argument("--normal_model", required=True)
    ap.add_argument("--fl_model", default=None)
    ap.add_argument("--leaked_grads", default=None)
    ap.add_argument("--activation", default="sigmoid")
    ap.add_argument("--images", default="0-9", help="e.g. '0-9' or '0,5,10,25'")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", default="idlg_results.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--defense", default="none", choices=["none","clipping","sgp"])
    ap.add_argument("--def_params", default="none",
                    help="For none: 'none'. For clipping: '0.99996,0.99995'. For sgp: '0.94,0.93' (keep_ratio).")

    args = ap.parse_args()

    images = parse_list_of_ints(args.images)

    if args.def_params == "none":
        def_params = [None]
    else:
        def_params = [float(x) for x in args.def_params.split(",")]

    scenarios = []
    if args.scenario in ("all", "normal_model_orig_grads"):
        scenarios.append(("normal_model_orig_grads", args.normal_model, None))
    if args.scenario in ("all", "fl_model_orig_grads"):
        scenarios.append(("fl_model_orig_grads", args.fl_model, None))
    if args.scenario in ("all", "normal_model_leaked_grads"):
        scenarios.append(("normal_model_leaked_grads", args.normal_model, args.leaked_grads))
    if args.scenario in ("all", "fl_model_leaked_grads"):
        scenarios.append(("fl_model_leaked_grads", args.fl_model, args.leaked_grads))

    for name, model_path, leak_path in scenarios:
        run_scenario(
            scenario_name=name,
            model_path=model_path,
            leaked_grads_path=leak_path,
            activation=args.activation,
            image_indices=images,
            defense=args.defense,
            def_params=def_params,
            iterations=args.iterations,
            base_seed=args.seed,
            out_csv=args.out_csv,
            device=args.device,
            repeats=args.repeats,
        )
