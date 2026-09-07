from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


TEMPLATE_SCRIPT = "run_proposed_tree_guided_moe_only.py"
DEFAULT_PROCESSED_DIR = Path(r"D:\project\data\merge_paper_ready_tree_cost")

CONFIG_GRID = [
    {"name": "A_current_pr_auc_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.30, "lambda_supcon": 0.010, "selection_objective": "PR_AUC"},
    {"name": "B_kd030_sup010_mcc_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.30, "lambda_supcon": 0.010, "selection_objective": "MCC"},
    {"name": "C_kd030_sup010_f1_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.30, "lambda_supcon": 0.010, "selection_objective": "Fraud_F1"},
    {"name": "D_kd030_sup010_recallP80_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.30, "lambda_supcon": 0.010, "selection_objective": "Recall@Precision>=0.80"},
    {"name": "E_kd020_sup005_pr_auc_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.20, "lambda_supcon": 0.005, "selection_objective": "PR_AUC"},
    {"name": "F_kd040_sup010_pr_auc_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.40, "lambda_supcon": 0.010, "selection_objective": "PR_AUC"},
    {"name": "G_kd050_sup010_pr_auc_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.50, "lambda_supcon": 0.010, "selection_objective": "PR_AUC"},
    {"name": "H_kd030_sup020_pr_auc_seed42", "seed": 42, "lr": 3e-4, "dropout": 0.30, "kd_weight": 0.30, "lambda_supcon": 0.020, "selection_objective": "PR_AUC"},
]

EXTRA_SEEDS = [2024, 3407]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def upsert_assignment(code: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=\s*.*$"
    replacement = f"{name} = {value}"
    new_code, n = re.subn(pattern, replacement, code, count=1, flags=re.MULTILINE)
    if n > 0:
        return new_code

    for marker in [r"^DEFAULT_PROCESSED_DIR\s*=.*$", r"^PROCESSED_DIR\s*=.*$", r"^TARGET_COL\s*=.*$"]:
        m = re.search(marker, code, flags=re.MULTILINE)
        if m:
            insert_at = m.end()
            return code[:insert_at] + "\n" + replacement + code[insert_at:]

    return replacement + "\n" + code


def patch_checkpoint_objective(code: str) -> str:
    if "best_val_score = -np.inf" not in code:
        code = code.replace(
            "best_val_pr_auc = -np.inf\n    best_epoch = 0",
            "best_val_pr_auc = -np.inf\n    best_val_score = -np.inf\n    best_epoch = 0",
            1,
        )

    old_if = (
        "        if val_metrics[\"PR_AUC\"] > best_val_pr_auc:\n"
        "            best_val_pr_auc = val_metrics[\"PR_AUC\"]\n"
        "            best_epoch = epoch\n"
        "            patience_counter = 0"
    )
    new_if = (
        "        selection_score = val_metrics.get(SELECTION_OBJECTIVE, val_metrics[\"PR_AUC\"])\n"
        "        if selection_score > best_val_score:\n"
        "            best_val_score = selection_score\n"
        "            best_val_pr_auc = val_metrics[\"PR_AUC\"]\n"
        "            best_epoch = epoch\n"
        "            patience_counter = 0"
    )

    if old_if in code:
        code = code.replace(old_if, new_if, 1)
    else:
        print("[WARN] Checkpoint block not found. Generated script may still select by PR-AUC.")

    return code


def patch_template(template: str, cfg: dict) -> str:
    code = template
    safe_name = cfg["name"]

    code = re.sub(
        r"RUN_EXPERIMENTS\s*=\s*\[[\s\S]*?\]\n\n# Training",
        'RUN_EXPERIMENTS = [\n    "tree_guided_moe_kd_focal_supcon",\n]\n\n# Training',
        code,
        count=1,
    )

    code = upsert_assignment(code, "RANDOM_STATE", str(cfg["seed"]))
    code = upsert_assignment(code, "LEARNING_RATE", str(cfg["lr"]))
    code = upsert_assignment(code, "DROPOUT", str(cfg["dropout"]))
    code = upsert_assignment(code, "KD_WEIGHT", str(cfg["kd_weight"]))
    code = upsert_assignment(code, "LAMBDA_SUPCON", str(cfg["lambda_supcon"]))
    code = upsert_assignment(code, "SELECTION_OBJECTIVE", '"' + cfg["selection_objective"] + '"')

    code = re.sub(
        r'OUTPUT_DIR\s*=\s*PROCESSED_DIR\s*/\s*"[^"]+"',
        f'OUTPUT_DIR = PROCESSED_DIR / "proposed_tree_guided_moe_tuning" / "{safe_name}"',
        code,
        count=1,
    )

    replacements = {
        '"proposed_only_results_running.csv"': f'"{safe_name}_running.csv"',
        '"proposed_only_results_running.json"': f'"{safe_name}_running.json"',
        '"proposed_only_results_final.csv"': f'"{safe_name}_final.csv"',
        '"proposed_only_results_final.json"': f'"{safe_name}_final.json"',
        '"paper_ready_proposed_only_cost_aware_table.csv"': f'"paper_ready_{safe_name}.csv"',
        '"dl_tree_guided_results_running.csv"': f'"{safe_name}_running.csv"',
        '"dl_tree_guided_results_running.json"': f'"{safe_name}_running.json"',
        '"dl_tree_guided_results_final.csv"': f'"{safe_name}_final.csv"',
        '"dl_tree_guided_results_final.json"': f'"{safe_name}_final.json"',
        '"paper_ready_dl_cost_aware_table.csv"': f'"paper_ready_{safe_name}.csv"',
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    code = patch_checkpoint_objective(code)

    if '"FOCAL_GAMMA": FOCAL_GAMMA,' in code and "TUNING_CONFIG_NAME" not in code:
        code = code.replace(
            '"FOCAL_GAMMA": FOCAL_GAMMA,',
            '"FOCAL_GAMMA": FOCAL_GAMMA,\n                    "SELECTION_OBJECTIVE": SELECTION_OBJECTIVE,\n                    "TUNING_CONFIG_NAME": "' + safe_name + '",',
            1,
        )

    if '"Best_Val_PR_AUC": float(best_val_pr_auc),' in code and '"Tuning_Config":' not in code:
        score_line = '"Best_Val_Selection_Score": float(best_val_score),' if "best_val_score" in code else '"Best_Val_Selection_Score": float(best_val_pr_auc),'
        code = code.replace(
            '"Best_Val_PR_AUC": float(best_val_pr_auc),',
            '"Tuning_Config": "' + safe_name + '",\n        "Selection_Objective": SELECTION_OBJECTIVE,\n        ' + score_line + '\n        "Best_Val_PR_AUC": float(best_val_pr_auc),',
            1,
        )

    if 'print("Running ONLY proposed model: tree_guided_moe_kd_focal_supcon")' in code:
        code = code.replace(
            'print("Running ONLY proposed model: tree_guided_moe_kd_focal_supcon")',
            'print("Running ONLY proposed model: tree_guided_moe_kd_focal_supcon")\n    print("Tuning config: ' + safe_name + '")\n    print("Selection objective:", SELECTION_OBJECTIVE)',
            1,
        )

    return code


def run_subprocess(script_path: Path, dry_run: bool = False) -> int:
    cmd = [sys.executable, str(script_path)]
    print("\n" + "=" * 100)
    print(f"[{now()}] RUN {script_path.name}")
    print("=" * 100)
    print(" ".join(cmd))

    if dry_run:
        return 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(script_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    return proc.wait()


def make_grid(extra_seeds: bool):
    grid = list(CONFIG_GRID)
    if extra_seeds:
        base_cfg = CONFIG_GRID[0]
        for seed in EXTRA_SEEDS:
            cfg = dict(base_cfg)
            cfg["seed"] = seed
            cfg["name"] = f"A_current_pr_auc_seed{seed}"
            grid.append(cfg)
    return grid


def combine_results(processed_dir: Path, cfgs: list[dict]):
    tuning_root = processed_dir / "proposed_tree_guided_moe_tuning"
    summary_dir = tuning_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in cfgs:
        final_path = tuning_root / cfg["name"] / f"{cfg['name']}_final.csv"
        if not final_path.exists():
            print("[MISS]", final_path)
            continue
        try:
            df = pd.read_csv(final_path)
            df["Config_Name"] = cfg["name"]
            df["Config_Seed"] = cfg["seed"]
            df["Config_LR"] = cfg["lr"]
            df["Config_Dropout"] = cfg["dropout"]
            df["Config_KD_Weight"] = cfg["kd_weight"]
            df["Config_Lambda_SupCon"] = cfg["lambda_supcon"]
            df["Config_Selection_Objective"] = cfg["selection_objective"]
            rows.append(df)
        except Exception as e:
            print("[ERROR reading]", final_path, e)

    if not rows:
        print("[WARN] No final tuning results found.")
        return

    combined = pd.concat(rows, ignore_index=True, sort=False)

    sort_cols = [c for c in ["PR_AUC", "Fraud_F1", "MCC", "Recall@Precision>=0.80", "ExpectedUtility@5%"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    combined_path = summary_dir / "tuning_results_combined.csv"
    combined.to_csv(combined_path, index=False)
    print("[SAVED]", combined_path)

    paper_cols = [
        "Config_Name", "Config_KD_Weight", "Config_Lambda_SupCon", "Config_Selection_Objective", "Config_Seed",
        "Model", "PR_AUC", "ROC_AUC", "Fraud_Precision", "Fraud_Recall", "Fraud_F1", "MCC",
        "Recall@Precision>=0.80", "Precision@5%", "Recall@5%", "CapturedAmountRate@5%", "ExpectedUtility@5%",
        "Best_Epoch", "Best_Val_Selection_Score", "Best_Val_PR_AUC",
        "Selected_Threshold", "Selected_Threshold_from_Val", "TN", "FP", "FN", "TP",
    ]
    paper_cols = [c for c in paper_cols if c in combined.columns]
    paper = combined[paper_cols].copy()
    paper_path = summary_dir / "paper_ready_tuning_summary.csv"
    paper.to_csv(paper_path, index=False)
    print("[SAVED]", paper_path)

    best = {}
    for metric in ["PR_AUC", "Fraud_F1", "MCC", "Recall@Precision>=0.80", "ExpectedUtility@5%", "CapturedAmountRate@5%"]:
        if metric in combined.columns:
            idx = combined[metric].astype(float).idxmax()
            best[metric] = combined.loc[idx].to_dict()

    best_path = summary_dir / "best_by_metric.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[SAVED]", best_path)

    print("\nTop tuning rows:")
    print(paper.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Tune only proposed Tree-guided MoE model.")
    parser.add_argument("--scripts-dir", type=str, default=".", help="Folder with run_proposed_tree_guided_moe_only.py and dependencies.")
    parser.add_argument("--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Processed data folder.")
    parser.add_argument("--max-configs", type=int, default=None, help="Run only first N configs.")
    parser.add_argument("--start-index", type=int, default=0, help="Start config index.")
    parser.add_argument("--extra-seeds", action="store_true", help="Also run seed 2024 and 3407 for current config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts and print commands only.")
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    processed_dir = Path(args.processed_dir)

    template_path = scripts_dir / TEMPLATE_SCRIPT
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template script: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    generated_dir = scripts_dir / "generated_tuning_runs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    cfgs = make_grid(extra_seeds=args.extra_seeds)
    if args.start_index:
        cfgs = cfgs[args.start_index:]
    if args.max_configs is not None:
        cfgs = cfgs[:args.max_configs]

    print("=" * 100)
    print("TUNE PROPOSED TREE-GUIDED MOE V2")
    print("=" * 100)
    print("Template:", template_path)
    print("Scripts dir:", scripts_dir)
    print("Processed dir:", processed_dir)
    print("Generated dir:", generated_dir)
    print("Number of configs:", len(cfgs))
    print("Start:", now())

    run_status = []
    for cfg in cfgs:
        generated_script = generated_dir / f"{cfg['name']}.py"
        patched = patch_template(template, cfg)
        generated_script.write_text(patched, encoding="utf-8")
        compile(patched, str(generated_script), "exec")

        code = run_subprocess(generated_script, dry_run=args.dry_run)
        run_status.append({"config": cfg, "script": str(generated_script), "return_code": code, "time": now()})

        status_path = processed_dir / "proposed_tree_guided_moe_tuning" / "_tuning_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8")

        if code != 0:
            print(f"[FAILED] {cfg['name']} returned code {code}. Stop tuning.")
            break

    if not args.dry_run:
        successful_cfgs = [s["config"] for s in run_status if s["return_code"] == 0]
        combine_results(processed_dir, successful_cfgs)

    print("End:", now())


if __name__ == "__main__":
    main()
