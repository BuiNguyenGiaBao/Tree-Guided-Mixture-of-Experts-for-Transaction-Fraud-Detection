# # IEEE-CIS Fraud Detection – ML Baselines + Tree Teacher + Cost-Aware Metrics
# 
# Notebook này dùng cho hướng bài mới:
# 
# ```text
# Tree-guided MoE Distillation
# + Cost-aware / Review-budget Fraud Detection
# ```
# 
# Nội dung chính:
# 
# 1. Chạy các ML baselines trên `full_train`, `full_val`, `full_internal_test`.
# 2. Dùng `y_train`, `y_val`, `y_internal_test` riêng để tránh lỗi RAM do bảng `full_*` quá lớn.
# 3. Báo chỉ số chuẩn:
#    - PR-AUC
#    - ROC-AUC
#    - Fraud Precision
#    - Fraud Recall
#    - Fraud F1
#    - MCC
#    - Recall@Precision≥0.80
#    - Confusion matrix
# 4. Bổ sung chỉ số cost-aware / review-budget:
#    - Precision@K%
#    - Recall@K%
#    - F1@K%
#    - Lift@K%
#    - Captured Fraud Amount Rate@K%
#    - Expected Utility@K%
# 5. Xuất `teacher_signal_*` để dùng cho Tree-guided MoE Distillation.

# ## 1. Imports and configuration

from pathlib import Path
import json
import time
import warnings
import gc

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ====== CONFIG ======
# Folder output của file data_cleaning_ieee_cis_tree_guided_cost_ready_memory_fixed_v2.ipynb
PROCESSED_DIR = Path(r"D:\project\data\merge_paper_ready_tree_cost")

TARGET_COL = "isFraud"
RANDOM_STATE = 42

# Nếu máy yếu RAM, đặt 200000 hoặc 300000.
# Nếu muốn train full data, để None.
SAMPLE_TRAIN_N = None

# Safe default:
# RF/ExtraTrees khá nặng RAM, nên để False trước.
# Muốn chạy đầy đủ thì đổi thành True.
RUN_LOGISTIC = True
RUN_RANDOM_FOREST = False
RUN_EXTRA_TREES = False
RUN_HIST_GB = True
RUN_XGBOOST = True
RUN_LIGHTGBM = True
RUN_CATBOOST = True

# Threshold metric
MIN_PRECISION_TARGET = 0.80

# Review-budget metrics
REVIEW_BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

# Utility giả định:
# Expected Utility = Captured fraud amount - review_cost_per_case * review_count - false_positive_cost * FP
REVIEW_COST_PER_CASE = 1.0
FALSE_POSITIVE_COST = 0.0

OUTPUT_DIR = PROCESSED_DIR / "ml_baseline_results"
TEACHER_DIR = PROCESSED_DIR / "teacher_signals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEACHER_DIR.mkdir(parents=True, exist_ok=True)

print("Processed dir:", PROCESSED_DIR)
print("ML output dir:", OUTPUT_DIR)
print("Teacher signal dir:", TEACHER_DIR)

# ## 2. Load tables

SUPPORTED_EXTS = [".parquet", ".csv.gz", ".csv"]

def resolve_file(base_name: str, folder: Path = PROCESSED_DIR) -> Path:
    for ext in SUPPORTED_EXTS:
        p = folder / f"{base_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {base_name} in {folder}. Tried {SUPPORTED_EXTS}")

def read_table(base_name: str, folder: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = resolve_file(base_name, folder)
    print(f"[LOAD] {base_name}: {path}")
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def memory_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1024**2, 2)

def downcast_numeric(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    for col in df.columns:
        if col == target_col:
            df[col] = df[col].astype("int8")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def load_y(base_name: str, meta_fallback_name: str):
    try:
        y_df = read_table(base_name)
        y = y_df[TARGET_COL].astype(int)
        return y
    except Exception:
        meta_df = read_table(meta_fallback_name)
        return meta_df[TARGET_COL].astype(int)

# full_* are UNLABELED in the memory-fixed data cleaning file.
X_train = downcast_numeric(read_table("full_train"))
X_val = downcast_numeric(read_table("full_val"))
X_test = downcast_numeric(read_table("full_internal_test"))

y_train = load_y("y_train", "review_meta_train")
y_val = load_y("y_val", "review_meta_val")
y_test = load_y("y_internal_test", "review_meta_internal_test")

review_meta_train = read_table("review_meta_train")
review_meta_val = read_table("review_meta_val")
review_meta_test = read_table("review_meta_internal_test")

# Fallback if old full_* still has target.
for name, X in [
    ("train", X_train),
    ("val", X_val),
    ("test", X_test),
]:
    if TARGET_COL in X.columns:
        print(f"[INFO] {name}: found target in X; dropping it.")
        if name == "train":
            y_train = X_train[TARGET_COL].astype(int)
            X_train = X_train.drop(columns=[TARGET_COL])
        elif name == "val":
            y_val = X_val[TARGET_COL].astype(int)
            X_val = X_val.drop(columns=[TARGET_COL])
        else:
            y_test = X_test[TARGET_COL].astype(int)
            X_test = X_test.drop(columns=[TARGET_COL])

print("X_train:", X_train.shape, "memory MB:", memory_mb(X_train))
print("X_val  :", X_val.shape, "memory MB:", memory_mb(X_val))
print("X_test :", X_test.shape, "memory MB:", memory_mb(X_test))
print("Fraud ratio train:", float(y_train.mean()))
print("Fraud ratio val  :", float(y_val.mean()))
print("Fraud ratio test :", float(y_test.mean()))
print("Review meta test:", review_meta_test.shape)

# ## 3. Align features and optional sampling

# Remove constant columns using train only.
constant_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]

if constant_cols:
    X_train = X_train.drop(columns=constant_cols, errors="ignore")
    X_val = X_val.drop(columns=constant_cols, errors="ignore")
    X_test = X_test.drop(columns=constant_cols, errors="ignore")

# Align columns.
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Dropped constant columns:", len(constant_cols))
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

# Optional sampling để chạy thử nhanh nhưng giữ toàn bộ fraud.
if SAMPLE_TRAIN_N is not None and SAMPLE_TRAIN_N < len(X_train):
    rng = np.random.default_rng(RANDOM_STATE)
    y_arr = y_train.values
    fraud_idx = np.where(y_arr == 1)[0]
    normal_idx = np.where(y_arr == 0)[0]
    n_normal_needed = max(SAMPLE_TRAIN_N - len(fraud_idx), 0)
    normal_sample = rng.choice(
        normal_idx,
        size=min(n_normal_needed, len(normal_idx)),
        replace=False,
    )
    selected_idx = np.concatenate([fraud_idx, normal_sample])
    rng.shuffle(selected_idx)

    X_train_fit = X_train.iloc[selected_idx]
    y_train_fit = y_train.iloc[selected_idx]
else:
    X_train_fit = X_train
    y_train_fit = y_train

print("Training data used:", X_train_fit.shape)
print("Training fraud ratio used:", float(y_train_fit.mean()))

gc.collect()

# ## 4. Standard fraud metrics

def safe_prob(y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(y_prob, 0.0, 1.0)

def prob_to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def predict_proba_positive(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return safe_prob(np.asarray(p)[:, 1])
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return safe_prob(1.0 / (1.0 + np.exp(-s)))
    raise ValueError("Model does not support predict_proba or decision_function.")

def threshold_by_best_f1(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1 = 2 * precision_t * recall_t / np.maximum(precision_t + recall_t, 1e-12)

    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision_t[best_idx]),
        "recall": float(recall_t[best_idx]),
        "f1": float(f1[best_idx]),
    }

def recall_at_precision(y_true, y_prob, min_precision=0.80):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    valid = precision >= min_precision

    if not np.any(valid):
        return {
            "recall_at_precision": 0.0,
            "threshold_at_precision": np.nan,
            "precision_at_threshold": np.nan,
        }

    valid_idx = np.where(valid)[0]
    best_i = valid_idx[np.argmax(recall[valid])]
    threshold = 1.0 if best_i >= len(thresholds) else thresholds[best_i]

    return {
        "recall_at_precision": float(recall[best_i]),
        "threshold_at_precision": float(threshold),
        "precision_at_threshold": float(precision[best_i]),
    }

def evaluate_with_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rap = recall_at_precision(y_true, y_prob, min_precision=MIN_PRECISION_TARGET)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan

    return {
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "ROC_AUC": float(roc_auc),
        "Fraud_Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Fraud_Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Fraud_F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}": float(rap["recall_at_precision"]),
        f"Threshold@Precision>={MIN_PRECISION_TARGET:.2f}": float(rap["threshold_at_precision"]) if np.isfinite(rap["threshold_at_precision"]) else np.nan,
        f"Precision@ThresholdForRecall@P>={MIN_PRECISION_TARGET:.2f}": float(rap["precision_at_threshold"]) if np.isfinite(rap["precision_at_threshold"]) else np.nan,
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
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)

    amount = np.asarray(amount, dtype=float)
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

        selected_flag = np.zeros(n, dtype=np.int8)
        selected_flag[selected] = 1

        tp = int(((selected_flag == 1) & (y_true == 1)).sum())
        fp = int(((selected_flag == 1) & (y_true == 0)).sum())
        fn = int(((selected_flag == 0) & (y_true == 1)).sum())
        tn = int(((selected_flag == 0) & (y_true == 0)).sum())

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
            "Review_Budget": float(budget),
            "Review_Budget_Percent": float(budget * 100),
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

def flatten_budget_metrics(budget_df):
    out = {}
    for _, row in budget_df.iterrows():
        pct = int(round(row["Review_Budget_Percent"]))
        out[f"Precision@{pct}%"] = float(row["Precision@K"])
        out[f"Recall@{pct}%"] = float(row["Recall@K"])
        out[f"F1@{pct}%"] = float(row["F1@K"])
        out[f"Lift@{pct}%"] = float(row["Lift@K"])
        out[f"CapturedAmountRate@{pct}%"] = float(row["Captured_Fraud_Amount_Rate@K"])
        out[f"ExpectedUtility@{pct}%"] = float(row["Expected_Utility@K"])
    return out

if "TransactionAmt_raw" not in review_meta_test.columns:
    raise ValueError("review_meta_internal_test must contain TransactionAmt_raw.")

test_amount = pd.to_numeric(
    review_meta_test["TransactionAmt_raw"],
    errors="coerce",
).fillna(0).astype(float).values

print("Total test transactions:", len(test_amount))
print("Total test fraud amount:", float((test_amount * y_test.values).sum()))

# ## 6. Model definitions

models = {}

pos = int(y_train_fit.sum())
neg = int(len(y_train_fit) - pos)
scale_pos_weight = neg / max(pos, 1)

print("Positive:", pos)
print("Negative:", neg)
print("scale_pos_weight:", scale_pos_weight)

if RUN_LOGISTIC:
    models["Logistic Regression Balanced"] = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

if RUN_RANDOM_FOREST:
    models["Random Forest Balanced"] = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

if RUN_EXTRA_TREES:
    models["Extra Trees Balanced"] = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

if RUN_HIST_GB:
    models["HistGradientBoosting SampleWeight"] = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

if RUN_XGBOOST:
    try:
        from xgboost import XGBClassifier

        models["XGBoost scale_pos_weight"] = XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print("[SKIP] XGBoost not available:", e)

if RUN_LIGHTGBM:
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM scale_pos_weight"] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print("[SKIP] LightGBM not available:", e)

if RUN_CATBOOST:
    try:
        from catboost import CatBoostClassifier

        models["CatBoost class_weights"] = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="PRAUC",
            class_weights=[1.0, scale_pos_weight],
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as e:
        print("[SKIP] CatBoost not available:", e)

print("Models to run:")
for name in models:
    print("-", name)

# ## 7. Fit, evaluate, and save predictions

def add_id_columns(pred_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    for col in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw"]:
        if col in meta_df.columns:
            out[col] = meta_df[col].values
    front_cols = [c for c in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw"] if c in out.columns]
    return out[front_cols + [c for c in out.columns if c not in front_cols]]

def fit_eval_model(model_name, model, sample_weight=None):
    print("\n==============================")
    print("Training model:", model_name)
    print("==============================")

    start = time.time()

    if sample_weight is None:
        model.fit(X_train_fit, y_train_fit)
    else:
        model.fit(X_train_fit, y_train_fit, sample_weight=sample_weight)

    train_time = time.time() - start

    val_prob = predict_proba_positive(model, X_val)
    test_prob = predict_proba_positive(model, X_test)

    best_val = threshold_by_best_f1(y_val, val_prob)
    threshold = best_val["threshold"]

    std = evaluate_with_threshold(y_test, test_prob, threshold)

    budget_detail = review_budget_metrics(
        y_true=y_test.values,
        y_prob=test_prob,
        amount=test_amount,
        budgets=REVIEW_BUDGETS,
        review_cost_per_case=REVIEW_COST_PER_CASE,
        false_positive_cost=FALSE_POSITIVE_COST,
    )
    budget_flat = flatten_budget_metrics(budget_detail)

    metrics = {
        "Model": model_name,
        **std,
        **budget_flat,
        "Selected_Threshold_from_Val": float(threshold),
        "Val_Best_Fraud_Precision": float(best_val["precision"]),
        "Val_Best_Fraud_Recall": float(best_val["recall"]),
        "Val_Best_Fraud_F1": float(best_val["f1"]),
        "Train_Time_Seconds": round(train_time, 3),
    }

    safe_name = (
        model_name.replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace(">=", "ge")
    )

    pred_df = pd.DataFrame({
        "y_true": y_test.values.astype(int),
        "y_prob": test_prob.astype(float),
        "y_pred": (test_prob >= threshold).astype(int),
    })
    pred_df = add_id_columns(pred_df, review_meta_test)
    pred_path = OUTPUT_DIR / f"pred_{safe_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    budget_detail.insert(0, "Model", model_name)
    budget_path = OUTPUT_DIR / f"review_budget_detail_{safe_name}.csv"
    budget_detail.to_csv(budget_path, index=False)

    metrics["Prediction_File"] = str(pred_path)
    metrics["Review_Budget_Detail_File"] = str(budget_path)

    print("Best threshold from validation:", threshold)
    print("Test PR-AUC:", metrics["PR_AUC"])
    print("Test ROC-AUC:", metrics["ROC_AUC"])
    print("Fraud Precision:", metrics["Fraud_Precision"])
    print("Fraud Recall:", metrics["Fraud_Recall"])
    print("Fraud F1:", metrics["Fraud_F1"])
    print("MCC:", metrics["MCC"])
    print(f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}:", metrics[f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}"])
    for pct in [1, 3, 5, 10]:
        key = f"Recall@{pct}%"
        if key in metrics:
            print(f"{key}:", metrics[key], "|", f"CapturedAmountRate@{pct}%:", metrics[f"CapturedAmountRate@{pct}%"])

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])

    return metrics, model

# ## 8. Run ML baselines

results = []
trained_models = {}

for model_name, model in models.items():
    try:
        if model_name == "HistGradientBoosting SampleWeight":
            sw = compute_sample_weight(class_weight="balanced", y=y_train_fit)
            metrics, fitted_model = fit_eval_model(model_name, model, sample_weight=sw)
        else:
            metrics, fitted_model = fit_eval_model(model_name, model)

        results.append(metrics)
        trained_models[model_name] = fitted_model

        running_df = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
        running_df.to_csv(OUTPUT_DIR / "ml_baseline_results_running.csv", index=False)

        gc.collect()

    except Exception as e:
        print("[ERROR] Model failed:", model_name)
        print(e)

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values("PR_AUC", ascending=False)

results_df

# ## 9. Save final ML result tables

results_csv = OUTPUT_DIR / "ml_baseline_results_final.csv"
results_json = OUTPUT_DIR / "ml_baseline_results_final.json"
paper_table_csv = OUTPUT_DIR / "paper_ready_ml_cost_aware_table.csv"

if results_df.empty:
    raise RuntimeError("No model results were produced. Please check model installation/settings.")

results_df.to_csv(results_csv, index=False)

with open(results_json, "w", encoding="utf-8") as f:
    json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

paper_cols = [
    "Model",
    "PR_AUC",
    "ROC_AUC",
    "Fraud_Precision",
    "Fraud_Recall",
    "Fraud_F1",
    "MCC",
    f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",
    "Precision@1%",
    "Recall@1%",
    "CapturedAmountRate@1%",
    "ExpectedUtility@1%",
    "Precision@3%",
    "Recall@3%",
    "CapturedAmountRate@3%",
    "ExpectedUtility@3%",
    "Precision@5%",
    "Recall@5%",
    "CapturedAmountRate@5%",
    "ExpectedUtility@5%",
    "Precision@10%",
    "Recall@10%",
    "CapturedAmountRate@10%",
    "ExpectedUtility@10%",
    "Selected_Threshold_from_Val",
    "TN",
    "FP",
    "FN",
    "TP",
    "Train_Time_Seconds",
]

paper_cols = [c for c in paper_cols if c in results_df.columns]
paper_table = results_df[paper_cols].copy()
paper_table.to_csv(paper_table_csv, index=False)

print("Saved:")
print(results_csv)
print(results_json)
print(paper_table_csv)

paper_table

# ## 10. Export teacher signals for Tree-guided MoE Distillation

def choose_teacher_model(results_df, trained_models):
    # Prefer LightGBM as teacher because it is usually strongest on tabular fraud data.
    priority_keywords = ["LightGBM", "CatBoost", "XGBoost", "HistGradientBoosting"]

    for key in priority_keywords:
        for model_name in trained_models:
            if key.lower() in model_name.lower():
                return model_name, trained_models[model_name]

    # Fallback: best PR-AUC model.
    best_name = results_df.sort_values("PR_AUC", ascending=False).iloc[0]["Model"]
    return best_name, trained_models[best_name]

def build_teacher_signal_df(prob, meta_df, y=None, split_name="train"):
    prob = safe_prob(prob)
    df = pd.DataFrame({
        "split": split_name,
        "teacher_prob": prob.astype(float),
        "teacher_logit": prob_to_logit(prob).astype(float),
    })

    if y is not None:
        df[TARGET_COL] = np.asarray(y).astype(int)

    for col in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw", "fraud_value_raw"]:
        if col in meta_df.columns:
            df[col] = meta_df[col].values

    front_cols = [
        c for c in [
            "row_id", "TransactionID", "split", TARGET_COL,
            "teacher_prob", "teacher_logit",
            "TransactionDT_raw", "TransactionAmt_raw", "fraud_value_raw"
        ]
        if c in df.columns
    ]

    return df[front_cols + [c for c in df.columns if c not in front_cols]]

teacher_name, teacher_model = choose_teacher_model(results_df, trained_models)
print("Selected teacher:", teacher_name)

teacher_train_prob = predict_proba_positive(teacher_model, X_train)
teacher_val_prob = predict_proba_positive(teacher_model, X_val)
teacher_test_prob = predict_proba_positive(teacher_model, X_test)

teacher_train = build_teacher_signal_df(teacher_train_prob, review_meta_train, y_train.values, "train")
teacher_val = build_teacher_signal_df(teacher_val_prob, review_meta_val, y_val.values, "val")
teacher_test = build_teacher_signal_df(teacher_test_prob, review_meta_test, y_test.values, "internal_test")

teacher_train_path = TEACHER_DIR / "teacher_signal_train.csv"
teacher_val_path = TEACHER_DIR / "teacher_signal_val.csv"
teacher_test_path = TEACHER_DIR / "teacher_signal_internal_test.csv"

teacher_train.to_csv(teacher_train_path, index=False)
teacher_val.to_csv(teacher_val_path, index=False)
teacher_test.to_csv(teacher_test_path, index=False)

teacher_meta = {
    "teacher_model": teacher_name,
    "teacher_selection_rule": "Prefer LightGBM, then CatBoost, then XGBoost, then HistGradientBoosting; otherwise best PR-AUC.",
    "teacher_signal_files": {
        "train": str(teacher_train_path),
        "val": str(teacher_val_path),
        "internal_test": str(teacher_test_path),
    },
    "distillation_columns": {
        "probability": "teacher_prob",
        "logit": "teacher_logit",
    },
}

teacher_meta_path = TEACHER_DIR / "teacher_signal_metadata.json"
with open(teacher_meta_path, "w", encoding="utf-8") as f:
    json.dump(teacher_meta, f, ensure_ascii=False, indent=2)

print("Saved teacher signals:")
print(teacher_train_path)
print(teacher_val_path)
print(teacher_test_path)
print(teacher_meta_path)

teacher_train.head()

# ## 11. Interpretation for the paper

print("Paper interpretation template:")
items = [
    "1. PR-AUC and ROC-AUC report ranking quality.",
    "2. Fraud F1 and MCC report decision-level classification balance.",
    "3. Recall@Precision>=0.80 reports fraud detection under a strict precision constraint.",
    "4. Precision@K% and Recall@K% simulate manual review budgets.",
    "5. CapturedAmountRate@K% reports the proportion of fraud value captured under a fixed review budget.",
    "6. ExpectedUtility@K% approximates business utility after review cost.",
    "7. teacher_signal_train/val/internal_test are used by Tree-guided MoE Distillation as soft labels."
]
for item in items:
    print(item)