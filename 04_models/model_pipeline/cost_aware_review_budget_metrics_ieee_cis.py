# # IEEE-CIS Cost-Aware / Review-Budget Evaluation Metrics
# 
# Notebook này dùng cho hướng mới:
# 
# ```text
# Tree-guided MoE Distillation
# + Cost-aware / Review-budget Fraud Detection
# ```
# 
# Vẫn giữ các chỉ số cũ:
# 
# - PR-AUC
# - ROC-AUC
# - Fraud Precision
# - Fraud Recall
# - Fraud F1
# - MCC
# - Recall@Precision≥0.80
# - Confusion matrix
# 
# Bổ sung chỉ số mới theo ngân sách review:
# 
# - Precision@K%
# - Recall@K%
# - F1@K%
# - Fraud captured@K%
# - Captured fraud amount@K%
# - Review count@K%
# - Expected utility@K%
# - Lift@K%
# 
# Ý nghĩa: nếu doanh nghiệp chỉ có thể review top 1%, 3%, 5%, 10% giao dịch rủi ro nhất, mô hình bắt được bao nhiêu gian lận và bao nhiêu giá trị gian lận.

# ## 1. Imports and configuration

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(r"D:\project\data\merge_paper_ready_tree_cost")

TARGET_COL = "isFraud"
META_TEST_NAME = "review_meta_internal_test"

# Prediction files may come from ML baselines, ablation DL models, unified MoE, or tree-guided student later.
PREDICTION_DIRS = [
    PROCESSED_DIR / "ml_baseline_results",
    PROCESSED_DIR / "dl_moe_results",
    PROCESSED_DIR / "unified_moe_results",
    PROCESSED_DIR / "tree_guided_moe_results",
]

OUTPUT_DIR = PROCESSED_DIR / "cost_aware_review_budget_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Review budgets: top K percent transactions to send to manual review.
REVIEW_BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

# Utility assumption.
# utility = captured_fraud_amount - review_cost_per_case * reviewed_count - false_positive_cost * FP
REVIEW_COST_PER_CASE = 1.0
FALSE_POSITIVE_COST = 0.0

MIN_PRECISION_TARGET = 0.80

print("Processed dir:", PROCESSED_DIR)
print("Output dir:", OUTPUT_DIR)

# ## 2. File loading utilities

def resolve_file(base_name: str, folder: Path) -> Path:
    for ext in [".parquet", ".csv.gz", ".csv"]:
        p = folder / f"{base_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {base_name} with .parquet/.csv.gz/.csv in {folder}")

def read_table(base_name_or_path, folder: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = Path(base_name_or_path)
    if not path.exists():
        path = resolve_file(str(base_name_or_path), folder)

    print("[LOAD]", path)

    if path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def find_prediction_files(prediction_dirs):
    files = []
    for d in prediction_dirs:
        if not d.exists():
            print("[SKIP DIR]", d)
            continue

        for p in sorted(d.glob("*.csv")):
            name = p.name.lower()
            if "pred" in name or "prediction" in name:
                files.append(p)

    return files

# ## 3. Load review metadata

meta_test = read_table(META_TEST_NAME)

required_meta_cols = [TARGET_COL, "TransactionAmt_raw", "fraud_value_raw"]
missing = [c for c in required_meta_cols if c not in meta_test.columns]
if missing:
    raise ValueError(f"Missing required columns in review metadata: {missing}")

meta_test[TARGET_COL] = meta_test[TARGET_COL].astype(int)
meta_test["TransactionAmt_raw"] = pd.to_numeric(meta_test["TransactionAmt_raw"], errors="coerce").fillna(0).astype(float)
meta_test["fraud_value_raw"] = pd.to_numeric(meta_test["fraud_value_raw"], errors="coerce").fillna(0).astype(float)

print(meta_test.head())
print("Meta shape:", meta_test.shape)
print("Fraud ratio:", meta_test[TARGET_COL].mean())
print("Total fraud amount:", meta_test["fraud_value_raw"].sum())

# ## 4. Standard fraud metrics

def safe_arrays(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return y_true, y_prob

def threshold_by_best_f1(y_true, y_prob):
    y_true, y_prob = safe_arrays(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)

    best_idx = int(np.nanargmax(f1))

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "f1": float(f1[best_idx]),
    }

def recall_at_precision(y_true, y_prob, min_precision=0.80):
    y_true, y_prob = safe_arrays(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    valid = precision >= min_precision
    if not np.any(valid):
        return {
            "recall": 0.0,
            "precision": 0.0,
            "threshold": np.nan,
        }

    valid_idx = np.where(valid)[0]
    best_i = valid_idx[np.argmax(recall[valid])]

    if best_i >= len(thresholds):
        threshold = 1.0
    else:
        threshold = thresholds[best_i]

    return {
        "recall": float(recall[best_i]),
        "precision": float(precision[best_i]),
        "threshold": float(threshold),
    }

def standard_metrics(y_true, y_prob, threshold=None, min_precision=0.80):
    y_true, y_prob = safe_arrays(y_true, y_prob)

    if threshold is None:
        threshold_info = threshold_by_best_f1(y_true, y_prob)
        threshold = threshold_info["threshold"]
    else:
        threshold_info = {
            "threshold": float(threshold),
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rap = recall_at_precision(y_true, y_prob, min_precision=min_precision)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan

    return {
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "ROC_AUC": float(roc_auc),
        "Fraud_Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Fraud_Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Fraud_F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        f"Recall@Precision>={min_precision:.2f}": float(rap["recall"]),
        f"Threshold@Precision>={min_precision:.2f}": float(rap["threshold"]),
        "Selected_Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

# ## 5. Cost-aware / review-budget metrics

def review_budget_metrics(
    y_true,
    y_prob,
    amount,
    budgets=REVIEW_BUDGETS,
    review_cost_per_case=REVIEW_COST_PER_CASE,
    false_positive_cost=FALSE_POSITIVE_COST,
):
    y_true, y_prob = safe_arrays(y_true, y_prob)
    amount = np.asarray(amount).astype(float)
    amount = np.nan_to_num(amount, nan=0.0, posinf=0.0, neginf=0.0)
    amount = np.clip(amount, 0.0, None)

    n = len(y_true)
    total_fraud = int(y_true.sum())
    total_fraud_amount = float((amount * y_true).sum())
    base_fraud_rate = total_fraud / max(n, 1)

    order = np.argsort(-y_prob)

    rows = []

    for budget in budgets:
        k = max(1, int(np.ceil(n * budget)))
        selected = order[:k]

        reviewed = np.zeros(n, dtype=int)
        reviewed[selected] = 1

        tp = int(((reviewed == 1) & (y_true == 1)).sum())
        fp = int(((reviewed == 1) & (y_true == 0)).sum())
        fn = int(((reviewed == 0) & (y_true == 1)).sum())
        tn = int(((reviewed == 0) & (y_true == 0)).sum())

        precision_k = tp / max(k, 1)
        recall_k = tp / max(total_fraud, 1)
        f1_k = 2 * precision_k * recall_k / max(precision_k + recall_k, 1e-12)

        captured_amount = float((amount[selected] * y_true[selected]).sum())
        captured_amount_rate = captured_amount / max(total_fraud_amount, 1e-12)

        review_cost = review_cost_per_case * k
        fp_cost = false_positive_cost * fp
        expected_utility = captured_amount - review_cost - fp_cost

        lift_k = precision_k / max(base_fraud_rate, 1e-12)

        rows.append({
            "Review_Budget": budget,
            "Review_Budget_Percent": budget * 100,
            "Review_Count": int(k),
            "TP_at_K": tp,
            "FP_at_K": fp,
            "FN_at_K": fn,
            "TN_at_K": tn,
            "Precision@K": float(precision_k),
            "Recall@K": float(recall_k),
            "F1@K": float(f1_k),
            "Lift@K": float(lift_k),
            "Captured_Fraud_Amount@K": float(captured_amount),
            "Captured_Fraud_Amount_Rate@K": float(captured_amount_rate),
            "Review_Cost@K": float(review_cost),
            "False_Positive_Cost@K": float(fp_cost),
            "Expected_Utility@K": float(expected_utility),
        })

    return pd.DataFrame(rows)

def summarize_budget_metrics(budget_df, model_name):
    out = {"Model": model_name}

    for _, row in budget_df.iterrows():
        pct = int(round(row["Review_Budget_Percent"]))
        out[f"Precision@{pct}%"] = row["Precision@K"]
        out[f"Recall@{pct}%"] = row["Recall@K"]
        out[f"F1@{pct}%"] = row["F1@K"]
        out[f"Lift@{pct}%"] = row["Lift@K"]
        out[f"CapturedAmountRate@{pct}%"] = row["Captured_Fraud_Amount_Rate@K"]
        out[f"ExpectedUtility@{pct}%"] = row["Expected_Utility@K"]

    return out

# ## 6. Load prediction files

prediction_files = find_prediction_files(PREDICTION_DIRS)

print("Found prediction files:", len(prediction_files))
for p in prediction_files:
    print("-", p)

# ## 7. Normalize prediction files
# 
# Prediction file cần có ít nhất:
# - `y_true`
# - `y_prob`
# 
# Nếu file có `y_pred` thì vẫn được, nhưng notebook sẽ tự chọn threshold lại theo F1 cho standard metrics.

def read_prediction_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    lower_map = {c.lower(): c for c in df.columns}

    if "y_true" not in lower_map:
        raise ValueError(f"{path} does not contain y_true column.")
    if "y_prob" not in lower_map:
        # Try common alternatives
        candidates = ["prob", "probability", "pred_prob", "fraud_prob", "student_prob", "teacher_prob"]
        found = None
        for c in candidates:
            if c in lower_map:
                found = lower_map[c]
                break
        if found is None:
            raise ValueError(f"{path} does not contain y_prob or probability column.")
        df = df.rename(columns={found: "y_prob"})

    df = df.rename(columns={lower_map["y_true"]: "y_true"})

    if len(df) != len(meta_test):
        raise ValueError(
            f"Prediction length mismatch for {path}: pred={len(df)}, meta={len(meta_test)}"
        )

    return pd.DataFrame({
        "y_true": df["y_true"].astype(int).values,
        "y_prob": pd.to_numeric(df["y_prob"], errors="coerce").fillna(0).astype(float).values,
    })

def model_name_from_path(path: Path) -> str:
    name = path.stem
    name = name.replace("pred_", "")
    name = name.replace("_test_predictions", "")
    name = name.replace("_predictions", "")
    name = name.replace("_", " ")
    return name

# ## 8. Evaluate all models

all_standard_rows = []
all_budget_summary_rows = []
all_budget_detail = []

y_meta = meta_test[TARGET_COL].astype(int).values
amount_meta = meta_test["TransactionAmt_raw"].astype(float).values

for pred_path in prediction_files:
    try:
        model_name = model_name_from_path(pred_path)
        pred = read_prediction_file(pred_path)

        y_true = pred["y_true"].values.astype(int)
        y_prob = pred["y_prob"].values.astype(float)

        if not np.array_equal(y_true, y_meta):
            print(f"[WARN] y_true in prediction file differs from review metadata: {pred_path}")

        std = standard_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=None,
            min_precision=MIN_PRECISION_TARGET,
        )
        std["Model"] = model_name
        std["Prediction_File"] = str(pred_path)
        all_standard_rows.append(std)

        budget_df = review_budget_metrics(
            y_true=y_true,
            y_prob=y_prob,
            amount=amount_meta,
            budgets=REVIEW_BUDGETS,
            review_cost_per_case=REVIEW_COST_PER_CASE,
            false_positive_cost=FALSE_POSITIVE_COST,
        )
        budget_df.insert(0, "Model", model_name)
        budget_df.insert(1, "Prediction_File", str(pred_path))
        all_budget_detail.append(budget_df)

        all_budget_summary_rows.append(summarize_budget_metrics(budget_df, model_name))

        print("[OK]", model_name)

    except Exception as e:
        print("[ERROR]", pred_path)
        print(e)

standard_results = pd.DataFrame(all_standard_rows)
budget_summary = pd.DataFrame(all_budget_summary_rows)
budget_detail = pd.concat(all_budget_detail, ignore_index=True) if all_budget_detail else pd.DataFrame()

if not standard_results.empty:
    standard_results = standard_results.sort_values("PR_AUC", ascending=False)

standard_results

# ## 9. Combined paper-ready table

if not standard_results.empty and not budget_summary.empty:
    combined = standard_results.merge(budget_summary, on="Model", how="left")
else:
    combined = standard_results.copy()

sort_col = "PR_AUC" if "PR_AUC" in combined.columns else combined.columns[0]
combined = combined.sort_values(sort_col, ascending=False)

combined_csv = OUTPUT_DIR / "cost_aware_combined_results.csv"
standard_csv = OUTPUT_DIR / "standard_fraud_metrics.csv"
budget_summary_csv = OUTPUT_DIR / "review_budget_summary.csv"
budget_detail_csv = OUTPUT_DIR / "review_budget_detail.csv"

combined.to_csv(combined_csv, index=False)
standard_results.to_csv(standard_csv, index=False)
budget_summary.to_csv(budget_summary_csv, index=False)
budget_detail.to_csv(budget_detail_csv, index=False)

print("Saved:")
print(combined_csv)
print(standard_csv)
print(budget_summary_csv)
print(budget_detail_csv)

combined.head(20)

# ## 10. Budget-detail view

budget_detail.head(30)

# ## 11. Minimal table for paper

paper_cols = [
    "Model",
    "PR_AUC",
    "ROC_AUC",
    "Fraud_Precision",
    "Fraud_Recall",
    "Fraud_F1",
    "MCC",
    f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",
]

for pct in [1, 3, 5, 10]:
    paper_cols += [
        f"Precision@{pct}%",
        f"Recall@{pct}%",
        f"CapturedAmountRate@{pct}%",
        f"ExpectedUtility@{pct}%",
    ]

paper_cols = [c for c in paper_cols if c in combined.columns]

paper_table = combined[paper_cols].copy()
paper_table_path = OUTPUT_DIR / "paper_ready_cost_aware_table.csv"
paper_table.to_csv(paper_table_path, index=False)

print("Saved paper table:", paper_table_path)
paper_table

# ## 12. Interpretation guide
# 
# Cách đọc kết quả:
# 
# - Nếu mô hình không thắng LightGBM ở PR-AUC nhưng thắng `Precision@5%`, `CapturedAmountRate@5%`, hoặc `ExpectedUtility@5%`, bài vẫn có câu chuyện tốt theo hướng review-budget.
# - `Recall@K%` cho biết trong top K% giao dịch được review, mô hình bắt được bao nhiêu phần trăm fraud thật.
# - `CapturedAmountRate@K%` quan trọng hơn recall nếu doanh nghiệp quan tâm giá trị tiền gian lận.
# - `ExpectedUtility@K%` cho thấy lợi ích thực dụng sau khi trừ chi phí review.