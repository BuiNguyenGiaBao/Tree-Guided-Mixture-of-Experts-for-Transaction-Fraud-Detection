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
EMBEDDED_TEMPLATE_SCRIPT = 'from __future__ import annotations\n\nimport gc\nimport json\nimport math\nimport random\nimport time\nfrom pathlib import Path\nfrom typing import Dict, List, Optional, Tuple\n\nimport numpy as np\nimport pandas as pd\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader, Dataset, WeightedRandomSampler\n\nfrom sklearn.metrics import (\n    average_precision_score,\n    roc_auc_score,\n    precision_score,\n    recall_score,\n    f1_score,\n    matthews_corrcoef,\n    confusion_matrix,\n    precision_recall_curve,\n)\n\nfrom cnn_branch_updated import TabularCNNBranch\nfrom deepfm_branch_updated import DeepFMBranch\nfrom fraud_losses import BinaryFocalLoss, CombinedFocalSupConLoss\n\n\n# ============================================================\n# CONFIG – SINGLE PROPOSED MODEL ONLY\n# ============================================================\n\n# Folder output của:\n# data_cleaning_ieee_cis_tree_guided_cost_ready_memory_fixed_v2.py\nPROCESSED_DIR = Path(r"D:\\project\\data\\merge_paper_ready_tree_cost")\n\nOUTPUT_DIR = PROCESSED_DIR / "proposed_tree_guided_moe_only_results"\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\nTEACHER_DIR = PROCESSED_DIR / "teacher_signals"\n\nTARGET_COL = "isFraud"\nRANDOM_STATE = 42\n\n# DL experiments.\n# Có thể chạy ít trước bằng cách comment bớt.\nRUN_EXPERIMENTS = [\n    "tree_guided_moe_kd_focal_supcon",\n]\n\n# Training\nBATCH_SIZE = 1024\nEPOCHS = 20\nPATIENCE = 5\nLEARNING_RATE = 3e-4\nWEIGHT_DECAY = 1e-4\nGRAD_CLIP_NORM = 5.0\n\n# Nếu máy yếu hoặc hay NaN, giữ USE_AMP=False.\nUSE_AMP = False\nUSE_WEIGHTED_SAMPLER = True\nNUM_WORKERS = 0\n\n# Nếu muốn chạy thử nhanh:\nSUBSET_TRAIN_N = None\nSUBSET_VAL_N = None\nSUBSET_TEST_N = None\n\n# Threshold and metrics\nTHRESHOLD_OBJECTIVE = "f1"\nMIN_PRECISION_TARGET = 0.80\nTHRESHOLD_MIN_PRECISION = None\n\n# Review-budget metrics\nREVIEW_BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]\nREVIEW_COST_PER_CASE = 1.0\nFALSE_POSITIVE_COST = 0.0\n\n# Model sizes\nCNN_EMBED_DIM = 128\nCNN_CONV_CHANNELS = 128\nCNN_KERNEL_SIZE = 3\nCNN_BILINEAR_RANK = 32\nCNN_OUT_DIM = 128\nCNN_SEQ_LENGTH = 10\n\nDEEPFM_EMBED_DIM = 16\nDEEPFM_DENSE_NUM_FIELDS = 8\nDEEPFM_BRANCH_OUT_DIM = 128\nDEEPFM_HIDDEN = [256, 128]\n\nFUSION_DIM = 128\nGATE_HIDDEN_DIM = 64\nDROPOUT = 0.30\n\n# Loss weights\nFOCAL_ALPHA = 0.85\nFOCAL_GAMMA = 2.0\nLAMBDA_SUPCON = 0.01\nSUPCON_TEMPERATURE = 0.10\n\n# Distillation\nKD_WEIGHT = 0.30\nKD_TEMPERATURE = 2.0\n\n\n# ============================================================\n# REPRODUCIBILITY AND IO\n# ============================================================\n\ndef set_seed(seed: int = RANDOM_STATE) -> None:\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)\n    torch.backends.cudnn.deterministic = False\n    torch.backends.cudnn.benchmark = True\n\n\ndef resolve_file(base_name: str, folder: Path = PROCESSED_DIR) -> Path:\n    for ext in [".parquet", ".csv.gz", ".csv"]:\n        p = folder / f"{base_name}{ext}"\n        if p.exists():\n            return p\n    raise FileNotFoundError(f"Cannot find {base_name} in {folder}")\n\n\ndef read_table(base_name: str, folder: Path = PROCESSED_DIR) -> pd.DataFrame:\n    path = resolve_file(base_name, folder)\n    print(f"[LOAD] {base_name}: {path}")\n    if path.name.endswith(".parquet"):\n        return pd.read_parquet(path)\n    return pd.read_csv(path)\n\n\ndef downcast_df(df: pd.DataFrame) -> pd.DataFrame:\n    for col in df.columns:\n        if col == TARGET_COL:\n            df[col] = df[col].astype("int8")\n        elif pd.api.types.is_float_dtype(df[col]):\n            df[col] = df[col].astype("float32")\n        elif pd.api.types.is_integer_dtype(df[col]):\n            df[col] = pd.to_numeric(df[col], downcast="integer")\n    return df\n\n\ndef maybe_subset_by_y(df: pd.DataFrame, n: Optional[int], seed: int = RANDOM_STATE) -> pd.DataFrame:\n    if n is None or n >= len(df):\n        return df\n    if TARGET_COL not in df.columns:\n        return df.sample(n=n, random_state=seed).reset_index(drop=True)\n\n    fraud = df[df[TARGET_COL] == 1]\n    normal = df[df[TARGET_COL] == 0]\n    n_normal = max(n - len(fraud), 0)\n    normal_sample = normal.sample(n=min(n_normal, len(normal)), random_state=seed)\n    out = pd.concat([fraud, normal_sample], axis=0).sample(frac=1.0, random_state=seed)\n    return out.reset_index(drop=True)\n\n\ndef safe_prob(y_prob):\n    y_prob = np.asarray(y_prob, dtype=float)\n    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)\n    return np.clip(y_prob, 0.0, 1.0)\n\n\n# ============================================================\n# METRICS\n# ============================================================\n\ndef threshold_by_best_f1(y_true, y_prob):\n    y_true = np.asarray(y_true).astype(int)\n    y_prob = safe_prob(y_prob)\n\n    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)\n    p = precision[:-1]\n    r = recall[:-1]\n    f1 = 2 * p * r / np.maximum(p + r, 1e-12)\n\n    best_idx = int(np.nanargmax(f1))\n    return {\n        "threshold": float(thresholds[best_idx]),\n        "precision": float(p[best_idx]),\n        "recall": float(r[best_idx]),\n        "f1": float(f1[best_idx]),\n    }\n\n\ndef recall_at_precision(y_true, y_prob, min_precision=0.80):\n    y_true = np.asarray(y_true).astype(int)\n    y_prob = safe_prob(y_prob)\n\n    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)\n    valid = precision >= min_precision\n\n    if not np.any(valid):\n        return {\n            "recall": 0.0,\n            "precision": np.nan,\n            "threshold": np.nan,\n        }\n\n    valid_idx = np.where(valid)[0]\n    best_i = valid_idx[np.argmax(recall[valid])]\n    threshold = 1.0 if best_i >= len(thresholds) else thresholds[best_i]\n\n    return {\n        "recall": float(recall[best_i]),\n        "precision": float(precision[best_i]),\n        "threshold": float(threshold),\n    }\n\n\ndef evaluate_fraud_metrics(y_true, y_prob, threshold, min_precision=0.80):\n    y_true = np.asarray(y_true).astype(int)\n    y_prob = safe_prob(y_prob)\n    y_pred = (y_prob >= threshold).astype(int)\n\n    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()\n\n    try:\n        pr_auc = average_precision_score(y_true, y_prob)\n    except Exception:\n        pr_auc = np.nan\n\n    try:\n        roc_auc = roc_auc_score(y_true, y_prob)\n    except Exception:\n        roc_auc = np.nan\n\n    rap = recall_at_precision(y_true, y_prob, min_precision=min_precision)\n\n    return {\n        "PR_AUC": float(pr_auc),\n        "ROC_AUC": float(roc_auc),\n        "Fraud_Precision": float(precision_score(y_true, y_pred, zero_division=0)),\n        "Fraud_Recall": float(recall_score(y_true, y_pred, zero_division=0)),\n        "Fraud_F1": float(f1_score(y_true, y_pred, zero_division=0)),\n        "MCC": float(matthews_corrcoef(y_true, y_pred)),\n        f"Recall@Precision>={min_precision:.2f}": float(rap["recall"]),\n        f"Threshold@Precision>={min_precision:.2f}": float(rap["threshold"]) if np.isfinite(rap["threshold"]) else np.nan,\n        "Selected_Threshold": float(threshold),\n        "TN": int(tn),\n        "FP": int(fp),\n        "FN": int(fn),\n        "TP": int(tp),\n    }\n\n\ndef review_budget_metrics(\n    y_true,\n    y_prob,\n    amount,\n    budgets=REVIEW_BUDGETS,\n    review_cost_per_case=REVIEW_COST_PER_CASE,\n    false_positive_cost=FALSE_POSITIVE_COST,\n):\n    y_true = np.asarray(y_true).astype(int)\n    y_prob = safe_prob(y_prob)\n\n    amount = np.asarray(amount, dtype=float)\n    amount = np.nan_to_num(amount, nan=0.0, posinf=0.0, neginf=0.0)\n    amount = np.clip(amount, 0.0, None)\n\n    n = len(y_true)\n    total_fraud = int(y_true.sum())\n    total_fraud_amount = float((amount * y_true).sum())\n    base_fraud_rate = total_fraud / max(n, 1)\n    order = np.argsort(-y_prob)\n\n    rows = []\n\n    for budget in budgets:\n        k = max(1, int(np.ceil(n * budget)))\n        selected = order[:k]\n\n        selected_flag = np.zeros(n, dtype=np.int8)\n        selected_flag[selected] = 1\n\n        tp = int(((selected_flag == 1) & (y_true == 1)).sum())\n        fp = int(((selected_flag == 1) & (y_true == 0)).sum())\n        fn = int(((selected_flag == 0) & (y_true == 1)).sum())\n        tn = int(((selected_flag == 0) & (y_true == 0)).sum())\n\n        precision_k = tp / max(k, 1)\n        recall_k = tp / max(total_fraud, 1)\n        f1_k = 2 * precision_k * recall_k / max(precision_k + recall_k, 1e-12)\n        lift_k = precision_k / max(base_fraud_rate, 1e-12)\n\n        captured_amount = float((amount[selected] * y_true[selected]).sum())\n        captured_amount_rate = captured_amount / max(total_fraud_amount, 1e-12)\n\n        review_cost = review_cost_per_case * k\n        fp_cost = false_positive_cost * fp\n        expected_utility = captured_amount - review_cost - fp_cost\n\n        rows.append({\n            "Review_Budget": float(budget),\n            "Review_Budget_Percent": float(budget * 100),\n            "Review_Count": int(k),\n            "TP_at_K": tp,\n            "FP_at_K": fp,\n            "FN_at_K": fn,\n            "TN_at_K": tn,\n            "Precision@K": float(precision_k),\n            "Recall@K": float(recall_k),\n            "F1@K": float(f1_k),\n            "Lift@K": float(lift_k),\n            "Captured_Fraud_Amount@K": float(captured_amount),\n            "Captured_Fraud_Amount_Rate@K": float(captured_amount_rate),\n            "Review_Cost@K": float(review_cost),\n            "False_Positive_Cost@K": float(fp_cost),\n            "Expected_Utility@K": float(expected_utility),\n        })\n\n    return pd.DataFrame(rows)\n\n\ndef flatten_budget_metrics(budget_df: pd.DataFrame) -> Dict[str, float]:\n    out = {}\n    for _, row in budget_df.iterrows():\n        pct = int(round(row["Review_Budget_Percent"]))\n        out[f"Precision@{pct}%"] = float(row["Precision@K"])\n        out[f"Recall@{pct}%"] = float(row["Recall@K"])\n        out[f"F1@{pct}%"] = float(row["F1@K"])\n        out[f"Lift@{pct}%"] = float(row["Lift@K"])\n        out[f"CapturedAmountRate@{pct}%"] = float(row["Captured_Fraud_Amount_Rate@K"])\n        out[f"ExpectedUtility@{pct}%"] = float(row["Expected_Utility@K"])\n    return out\n\n\n# ============================================================\n# DATASET\n# ============================================================\n\nclass FraudMultiInputDataset(Dataset):\n    def __init__(\n        self,\n        cnn_df: pd.DataFrame,\n        cat_df: pd.DataFrame,\n        num_df: pd.DataFrame,\n        teacher_df: Optional[pd.DataFrame] = None,\n    ):\n        for name, df in [("cnn_df", cnn_df), ("cat_df", cat_df), ("num_df", num_df)]:\n            if TARGET_COL not in df.columns:\n                raise ValueError(f"{TARGET_COL} not found in {name}")\n\n        y_cnn = cnn_df[TARGET_COL].astype(int).to_numpy()\n        y_cat = cat_df[TARGET_COL].astype(int).to_numpy()\n        y_num = num_df[TARGET_COL].astype(int).to_numpy()\n\n        if not (np.array_equal(y_cnn, y_cat) and np.array_equal(y_cnn, y_num)):\n            raise ValueError("Labels in cnn/cat/num files are not aligned.")\n\n        self.y = y_cnn.astype("float32")\n\n        self.x_cnn = cnn_df.drop(columns=[TARGET_COL]).to_numpy(dtype="float32", copy=True)\n        self.x_num = num_df.drop(columns=[TARGET_COL]).to_numpy(dtype="float32", copy=True)\n\n        x_cat = cat_df.drop(columns=[TARGET_COL]).to_numpy(dtype="int64", copy=True)\n        x_cat = x_cat + 1\n        x_cat[x_cat < 0] = 0\n        self.x_cat = x_cat.astype("int64")\n\n        if teacher_df is not None and "teacher_prob" in teacher_df.columns:\n            teacher_prob = pd.to_numeric(teacher_df["teacher_prob"], errors="coerce").fillna(0.0).to_numpy(dtype="float32")\n            teacher_logit = pd.to_numeric(teacher_df.get("teacher_logit", pd.Series(np.zeros(len(teacher_df)))), errors="coerce").fillna(0.0).to_numpy(dtype="float32")\n            if len(teacher_prob) != len(self.y):\n                raise ValueError("teacher_signal length does not match dataset length.")\n            self.teacher_prob = np.clip(teacher_prob, 0.0, 1.0).astype("float32")\n            self.teacher_logit = teacher_logit.astype("float32")\n        else:\n            self.teacher_prob = np.zeros_like(self.y, dtype="float32")\n            self.teacher_logit = np.zeros_like(self.y, dtype="float32")\n\n    def __len__(self):\n        return len(self.y)\n\n    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:\n        return {\n            "x_cnn": torch.from_numpy(self.x_cnn[idx]),\n            "x_cat": torch.from_numpy(self.x_cat[idx]),\n            "x_num": torch.from_numpy(self.x_num[idx]),\n            "y": torch.tensor(self.y[idx], dtype=torch.float32),\n            "teacher_prob": torch.tensor(self.teacher_prob[idx], dtype=torch.float32),\n            "teacher_logit": torch.tensor(self.teacher_logit[idx], dtype=torch.float32),\n        }\n\n\ndef load_split(split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n    cnn = downcast_df(read_table(f"cnn_{split}"))\n    cat = downcast_df(read_table(f"deepfm_cat_{split}"))\n    num = downcast_df(read_table(f"deepfm_num_{split}"))\n    return cnn, cat, num\n\n\ndef load_teacher_split(split: str) -> Optional[pd.DataFrame]:\n    # split: train, val, internal_test\n    base = f"teacher_signal_{split}"\n    try:\n        return downcast_df(read_table(base, TEACHER_DIR))\n    except Exception as e:\n        print(f"[WARN] Teacher signal not found for {split}: {e}")\n        return None\n\n\ndef build_loaders():\n    train_cnn, train_cat, train_num = load_split("train")\n    val_cnn, val_cat, val_num = load_split("val")\n    test_cnn, test_cat, test_num = load_split("internal_test")\n\n    train_teacher = load_teacher_split("train")\n    val_teacher = load_teacher_split("val")\n    test_teacher = load_teacher_split("internal_test")\n\n    if SUBSET_TRAIN_N is not None:\n        idx = maybe_subset_by_y(train_cnn[[TARGET_COL]].reset_index(), SUBSET_TRAIN_N)["index"].to_numpy()\n        train_cnn = train_cnn.iloc[idx].reset_index(drop=True)\n        train_cat = train_cat.iloc[idx].reset_index(drop=True)\n        train_num = train_num.iloc[idx].reset_index(drop=True)\n        if train_teacher is not None:\n            train_teacher = train_teacher.iloc[idx].reset_index(drop=True)\n\n    if SUBSET_VAL_N is not None:\n        idx = maybe_subset_by_y(val_cnn[[TARGET_COL]].reset_index(), SUBSET_VAL_N)["index"].to_numpy()\n        val_cnn = val_cnn.iloc[idx].reset_index(drop=True)\n        val_cat = val_cat.iloc[idx].reset_index(drop=True)\n        val_num = val_num.iloc[idx].reset_index(drop=True)\n        if val_teacher is not None:\n            val_teacher = val_teacher.iloc[idx].reset_index(drop=True)\n\n    if SUBSET_TEST_N is not None:\n        idx = maybe_subset_by_y(test_cnn[[TARGET_COL]].reset_index(), SUBSET_TEST_N)["index"].to_numpy()\n        test_cnn = test_cnn.iloc[idx].reset_index(drop=True)\n        test_cat = test_cat.iloc[idx].reset_index(drop=True)\n        test_num = test_num.iloc[idx].reset_index(drop=True)\n        if test_teacher is not None:\n            test_teacher = test_teacher.iloc[idx].reset_index(drop=True)\n\n    cat_feature_cols = [c for c in train_cat.columns if c != TARGET_COL]\n    categorical_cardinalities = []\n    for col in cat_feature_cols:\n        max_code = int(train_cat[col].max())\n        categorical_cardinalities.append(max_code + 2)\n\n    dims = {\n        "cnn_dim": train_cnn.shape[1] - 1,\n        "deepfm_num_dim": train_num.shape[1] - 1,\n        "num_cat": len(cat_feature_cols),\n        "train_size": len(train_cnn),\n        "val_size": len(val_cnn),\n        "test_size": len(test_cnn),\n    }\n\n    train_ds = FraudMultiInputDataset(train_cnn, train_cat, train_num, train_teacher)\n    val_ds = FraudMultiInputDataset(val_cnn, val_cat, val_num, val_teacher)\n    test_ds = FraudMultiInputDataset(test_cnn, test_cat, test_num, test_teacher)\n\n    if USE_WEIGHTED_SAMPLER:\n        y = train_ds.y.astype(int)\n        class_count = np.bincount(y, minlength=2)\n        class_weight = 1.0 / np.maximum(class_count, 1)\n        sample_weight = class_weight[y]\n        sampler = WeightedRandomSampler(\n            weights=torch.as_tensor(sample_weight, dtype=torch.double),\n            num_samples=len(sample_weight),\n            replacement=True,\n        )\n        shuffle = False\n    else:\n        sampler = None\n        shuffle = True\n\n    train_loader = DataLoader(\n        train_ds,\n        batch_size=BATCH_SIZE,\n        shuffle=shuffle,\n        sampler=sampler,\n        num_workers=NUM_WORKERS,\n        pin_memory=torch.cuda.is_available(),\n        drop_last=False,\n    )\n    val_loader = DataLoader(\n        val_ds,\n        batch_size=BATCH_SIZE * 2,\n        shuffle=False,\n        num_workers=NUM_WORKERS,\n        pin_memory=torch.cuda.is_available(),\n        drop_last=False,\n    )\n    test_loader = DataLoader(\n        test_ds,\n        batch_size=BATCH_SIZE * 2,\n        shuffle=False,\n        num_workers=NUM_WORKERS,\n        pin_memory=torch.cuda.is_available(),\n        drop_last=False,\n    )\n\n    print("DIMS:", dims)\n    print("Train fraud ratio:", float(train_ds.y.mean()))\n    print("Val fraud ratio:", float(val_ds.y.mean()))\n    print("Test fraud ratio:", float(test_ds.y.mean()))\n    print("Teacher signal available train:", bool(np.any(train_ds.teacher_prob > 0)))\n\n    return train_loader, val_loader, test_loader, dims, categorical_cardinalities, test_ds\n\n\n# ============================================================\n# MODEL WRAPPERS\n# ============================================================\n\nclass DeepFMCompatWrapper(nn.Module):\n    def __init__(self, deepfm: nn.Module):\n        super().__init__()\n        self.deepfm = deepfm\n        self.output_dim = getattr(deepfm, "output_dim", None)\n\n    def extract_embedding(\n        self,\n        x_cat: Optional[torch.Tensor] = None,\n        x_num: Optional[torch.Tensor] = None,\n        x_dense: Optional[torch.Tensor] = None,\n        cat_x: Optional[torch.Tensor] = None,\n        num_x: Optional[torch.Tensor] = None,\n        dense_x: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n        if cat_x is None:\n            cat_x = x_cat\n        if num_x is None:\n            num_x = x_num\n        if dense_x is None:\n            dense_x = x_dense\n        return self.deepfm.extract_embedding(cat_x=cat_x, num_x=num_x, dense_x=dense_x)\n\n    def forward(self, *args, **kwargs):\n        return self.extract_embedding(*args, **kwargs)\n\n\nclass CNNOnlyClassifier(nn.Module):\n    def __init__(self, cnn_branch: nn.Module):\n        super().__init__()\n        self.cnn_branch = cnn_branch\n\n    def forward(self, x_cnn, x_cat=None, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):\n        logits, z_cnn, _ = self.cnn_branch(x_cnn, return_embedding=True)\n        logits = logits.view(-1)\n        prob = torch.sigmoid(logits)\n        if not return_dict:\n            return logits\n        return {"logits": logits, "prob": prob, "embedding": z_cnn}\n\n\nclass DeepFMOnlyClassifier(nn.Module):\n    def __init__(self, deepfm: nn.Module):\n        super().__init__()\n        self.deepfm = deepfm\n\n    def forward(self, x_cnn=None, x_cat=None, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):\n        logits, z_fm = self.deepfm(cat_x=x_cat, num_x=None, dense_x=x_dense, return_embedding=True)\n        logits = logits.view(-1)\n        prob = torch.sigmoid(logits)\n        if not return_dict:\n            return logits\n        return {"logits": logits, "prob": prob, "embedding": z_fm}\n\n\nclass ConcatCNNDeepFM(nn.Module):\n    def __init__(self, cnn_branch, deepfm_branch, cnn_dim, deepfm_dim, fusion_dim=128, dropout=0.30):\n        super().__init__()\n        self.cnn_branch = cnn_branch\n        self.deepfm_branch = deepfm_branch\n        self.fusion = nn.Sequential(\n            nn.Linear(cnn_dim + deepfm_dim, fusion_dim),\n            nn.BatchNorm1d(fusion_dim),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(fusion_dim, 1),\n        )\n\n    def forward(self, x_cnn, x_cat, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):\n        _, z_cnn, _ = self.cnn_branch(x_cnn, return_embedding=True)\n        z_fm = self.deepfm_branch.extract_embedding(x_cat=x_cat, x_dense=x_dense)\n        z = torch.cat([z_cnn, z_fm], dim=1)\n        logits = self.fusion(z).view(-1)\n        prob = torch.sigmoid(logits)\n        if not return_dict:\n            return logits\n        return {"logits": logits, "prob": prob, "embedding": z}\n\n\nclass MoEGatedCNNDeepFM(nn.Module):\n    def __init__(\n        self,\n        cnn_branch,\n        deepfm_branch,\n        cnn_dim=128,\n        deepfm_dim=128,\n        fusion_dim=128,\n        gate_hidden_dim=64,\n        dropout=0.30,\n        tree_guided=False,\n    ):\n        super().__init__()\n        self.cnn_branch = cnn_branch\n        self.deepfm_branch = deepfm_branch\n        self.tree_guided = tree_guided\n\n        self.cnn_proj = nn.Sequential(\n            nn.Linear(cnn_dim, fusion_dim),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n        )\n        self.fm_proj = nn.Sequential(\n            nn.Linear(deepfm_dim, fusion_dim),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n        )\n\n        gate_in = fusion_dim * 2 + (1 if tree_guided else 0)\n        self.gate = nn.Sequential(\n            nn.Linear(gate_in, gate_hidden_dim),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(gate_hidden_dim, 2),\n        )\n\n        self.classifier = nn.Sequential(\n            nn.BatchNorm1d(fusion_dim),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(fusion_dim, 1),\n        )\n\n    def forward(self, x_cnn, x_cat, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):\n        _, z_cnn_raw, _ = self.cnn_branch(x_cnn, return_embedding=True)\n        z_fm_raw = self.deepfm_branch.extract_embedding(x_cat=x_cat, x_dense=x_dense)\n\n        z_cnn = self.cnn_proj(z_cnn_raw)\n        z_fm = self.fm_proj(z_fm_raw)\n\n        if self.tree_guided:\n            if teacher_logit is None:\n                teacher_logit = torch.zeros(z_cnn.size(0), device=z_cnn.device)\n            guide = teacher_logit.view(-1, 1)\n            gate_input = torch.cat([z_cnn, z_fm, guide], dim=1)\n        else:\n            gate_input = torch.cat([z_cnn, z_fm], dim=1)\n\n        gate_weight = F.softmax(self.gate(gate_input), dim=1)\n        z = gate_weight[:, 0:1] * z_cnn + gate_weight[:, 1:2] * z_fm\n\n        logits = self.classifier(z).view(-1)\n        prob = torch.sigmoid(logits)\n\n        if not return_dict:\n            return logits\n\n        return {\n            "logits": logits,\n            "prob": prob,\n            "embedding": z,\n            "gate_weight": gate_weight,\n        }\n\n\ndef make_cnn_branch(cnn_dim: int) -> nn.Module:\n    return TabularCNNBranch(\n        tabular_dim=cnn_dim,\n        embed_dim=CNN_EMBED_DIM,\n        conv_channels=CNN_CONV_CHANNELS,\n        kernel_size=CNN_KERNEL_SIZE,\n        bilinear_rank=CNN_BILINEAR_RANK,\n        bilinear_out_dim=CNN_OUT_DIM,\n        num_classes=1,\n        seq_length=CNN_SEQ_LENGTH,\n        dropout=DROPOUT,\n    )\n\n\ndef make_deepfm_branch(cat_cardinalities: List[int], deepfm_num_dim: int) -> nn.Module:\n    return DeepFMBranch(\n        num_classes=2,\n        categorical_cardinalities=cat_cardinalities,\n        num_numerical=0,\n        embed_dim=DEEPFM_EMBED_DIM,\n        deep_hidden=DEEPFM_HIDDEN,\n        dropout=DROPOUT,\n        dense_in_dim=deepfm_num_dim,\n        dense_num_fields=DEEPFM_DENSE_NUM_FIELDS,\n        branch_out_dim=DEEPFM_BRANCH_OUT_DIM,\n    )\n\n\ndef build_model(experiment: str, dims: Dict[str, int], cat_cardinalities: List[int]) -> nn.Module:\n    cnn = make_cnn_branch(dims["cnn_dim"])\n    deepfm = make_deepfm_branch(cat_cardinalities, dims["deepfm_num_dim"])\n    deepfm_compat = DeepFMCompatWrapper(deepfm)\n\n    if experiment.startswith("cnn_only"):\n        return CNNOnlyClassifier(cnn)\n\n    if experiment.startswith("deepfm_only"):\n        return DeepFMOnlyClassifier(deepfm)\n\n    if experiment.startswith("concat"):\n        return ConcatCNNDeepFM(\n            cnn_branch=cnn,\n            deepfm_branch=deepfm_compat,\n            cnn_dim=CNN_OUT_DIM,\n            deepfm_dim=DEEPFM_BRANCH_OUT_DIM,\n            fusion_dim=FUSION_DIM,\n            dropout=DROPOUT,\n        )\n\n    if experiment.startswith("tree_guided_moe"):\n        return MoEGatedCNNDeepFM(\n            cnn_branch=cnn,\n            deepfm_branch=deepfm_compat,\n            cnn_dim=CNN_OUT_DIM,\n            deepfm_dim=DEEPFM_BRANCH_OUT_DIM,\n            fusion_dim=FUSION_DIM,\n            gate_hidden_dim=GATE_HIDDEN_DIM,\n            dropout=DROPOUT,\n            tree_guided=True,\n        )\n\n    if experiment.startswith("moe"):\n        return MoEGatedCNNDeepFM(\n            cnn_branch=cnn,\n            deepfm_branch=deepfm_compat,\n            cnn_dim=CNN_OUT_DIM,\n            deepfm_dim=DEEPFM_BRANCH_OUT_DIM,\n            fusion_dim=FUSION_DIM,\n            gate_hidden_dim=GATE_HIDDEN_DIM,\n            dropout=DROPOUT,\n            tree_guided=False,\n        )\n\n    raise ValueError(f"Unknown experiment: {experiment}")\n\n\n# ============================================================\n# LOSSES\n# ============================================================\n\ndef kd_loss_with_logits(student_logits, teacher_prob, temperature=2.0):\n    teacher_prob = torch.clamp(teacher_prob.float(), 1e-6, 1.0 - 1e-6)\n    teacher_logit = torch.log(teacher_prob / (1.0 - teacher_prob))\n    s = student_logits.float() / temperature\n    t = torch.sigmoid(teacher_logit / temperature)\n    return F.binary_cross_entropy_with_logits(s, t) * (temperature ** 2)\n\n\ndef compute_loss(experiment: str, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:\n    logits = out["logits"].view(-1)\n    y = batch["y"].float()\n    teacher_prob = batch["teacher_prob"].float()\n\n    if "focal" in experiment:\n        cls_loss = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)(logits, y)\n    else:\n        cls_loss = F.binary_cross_entropy_with_logits(logits, y)\n\n    kd = torch.zeros((), device=logits.device)\n    if "kd" in experiment:\n        kd = kd_loss_with_logits(logits, teacher_prob, temperature=KD_TEMPERATURE)\n\n    supcon = torch.zeros((), device=logits.device)\n    if "supcon" in experiment:\n        # Reuse existing supervised contrastive implementation through combined class.\n        # Classification part from this combined loss is ignored here to avoid double counting.\n        comb = CombinedFocalSupConLoss(\n            focal_alpha=FOCAL_ALPHA,\n            focal_gamma=FOCAL_GAMMA,\n            lambda_supcon=1.0,\n            temperature=SUPCON_TEMPERATURE,\n            fraud_anchor_weight=2.0,\n        ).to(logits.device)\n        loss_dict = comb(logits, y, features=out.get("embedding"))\n        supcon = loss_dict["supcon_loss"]\n\n    total = cls_loss + KD_WEIGHT * kd + LAMBDA_SUPCON * supcon\n\n    if not torch.isfinite(total):\n        raise FloatingPointError("NaN/Inf loss detected.")\n\n    return total, {\n        "classification_loss": float(cls_loss.detach().cpu()),\n        "kd_loss": float(kd.detach().cpu()),\n        "supcon_loss": float(supcon.detach().cpu()),\n        "total_loss": float(total.detach().cpu()),\n    }\n\n\n# ============================================================\n# TRAINING\n# ============================================================\n\ndef move_batch(batch, device):\n    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}\n\n\ndef forward_model(model, batch):\n    return model(\n        x_cnn=batch["x_cnn"],\n        x_cat=batch["x_cat"].long(),\n        x_dense=batch["x_num"],\n        teacher_logit=batch["teacher_logit"],\n        return_dict=True,\n    )\n\n\ndef train_one_epoch(model, loader, optimizer, device, scaler=None):\n    model.train()\n    total = {"loss": 0.0, "classification_loss": 0.0, "kd_loss": 0.0, "supcon_loss": 0.0}\n    n = 0\n\n    for batch in loader:\n        batch = move_batch(batch, device)\n        optimizer.zero_grad(set_to_none=True)\n\n        if scaler is not None:\n            with torch.amp.autocast("cuda", enabled=True):\n                out = forward_model(model, batch)\n                loss, loss_log = compute_loss(model.experiment_name, out, batch)\n\n            scaler.scale(loss).backward()\n            scaler.unscale_(optimizer)\n            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)\n            scaler.step(optimizer)\n            scaler.update()\n        else:\n            out = forward_model(model, batch)\n            loss, loss_log = compute_loss(model.experiment_name, out, batch)\n            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)\n            optimizer.step()\n\n        bs = int(batch["y"].size(0))\n        total["loss"] += loss_log["total_loss"] * bs\n        total["classification_loss"] += loss_log["classification_loss"] * bs\n        total["kd_loss"] += loss_log["kd_loss"] * bs\n        total["supcon_loss"] += loss_log["supcon_loss"] * bs\n        n += bs\n\n    return {k: v / max(n, 1) for k, v in total.items()}\n\n\n@torch.no_grad()\ndef collect_predictions(model, loader, device):\n    model.eval()\n    ys, probs = [], []\n    gate_weights = []\n\n    for batch in loader:\n        batch = move_batch(batch, device)\n        out = forward_model(model, batch)\n        ys.append(batch["y"].detach().cpu().numpy())\n        probs.append(out["prob"].view(-1).detach().cpu().numpy())\n        if "gate_weight" in out:\n            gate_weights.append(out["gate_weight"].detach().cpu().numpy())\n\n    y = np.concatenate(ys)\n    p = np.concatenate(probs)\n    gw = np.concatenate(gate_weights) if gate_weights else None\n    return y, p, gw\n\n\ndef attach_meta(pred_df: pd.DataFrame, meta_df: Optional[pd.DataFrame]) -> pd.DataFrame:\n    if meta_df is None:\n        return pred_df\n\n    out = pred_df.copy()\n    for col in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw"]:\n        if col in meta_df.columns and len(meta_df) == len(out):\n            out[col] = meta_df[col].values\n\n    front = [c for c in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw"] if c in out.columns]\n    return out[front + [c for c in out.columns if c not in front]]\n\n\ndef run_experiment(experiment, train_loader, val_loader, test_loader, dims, cat_cardinalities, test_ds, review_meta_test, device):\n    print("\\n" + "=" * 80)\n    print("EXPERIMENT:", experiment)\n    print("=" * 80)\n\n    model = build_model(experiment, dims, cat_cardinalities).to(device)\n    model.experiment_name = experiment\n\n    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n        optimizer,\n        mode="max",\n        factor=0.5,\n        patience=2,\n    )\n    scaler = torch.amp.GradScaler("cuda") if (USE_AMP and device.type == "cuda") else None\n\n    ckpt_path = OUTPUT_DIR / f"best_{experiment}.pt"\n    history_path = OUTPUT_DIR / f"history_{experiment}.csv"\n\n    best_val_pr_auc = -np.inf\n    best_epoch = 0\n    patience_counter = 0\n    history = []\n    start_time = time.time()\n\n    for epoch in range(1, EPOCHS + 1):\n        epoch_start = time.time()\n\n        try:\n            train_log = train_one_epoch(model, train_loader, optimizer, device, scaler)\n        except FloatingPointError as e:\n            print("[STOP]", e)\n            break\n\n        y_val, val_prob, _ = collect_predictions(model, val_loader, device)\n        thr_info = threshold_by_best_f1(y_val, val_prob)\n        val_metrics = evaluate_fraud_metrics(y_val, val_prob, thr_info["threshold"], min_precision=MIN_PRECISION_TARGET)\n\n        scheduler.step(val_metrics["PR_AUC"])\n\n        row = {\n            "experiment": experiment,\n            "epoch": epoch,\n            "train_loss": train_log["loss"],\n            "train_classification_loss": train_log["classification_loss"],\n            "train_kd_loss": train_log["kd_loss"],\n            "train_supcon_loss": train_log["supcon_loss"],\n            "val_PR_AUC": val_metrics["PR_AUC"],\n            "val_ROC_AUC": val_metrics["ROC_AUC"],\n            "val_Fraud_Precision": val_metrics["Fraud_Precision"],\n            "val_Fraud_Recall": val_metrics["Fraud_Recall"],\n            "val_Fraud_F1": val_metrics["Fraud_F1"],\n            "val_MCC": val_metrics["MCC"],\n            f"val_Recall@Precision>={MIN_PRECISION_TARGET:.2f}": val_metrics[f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}"],\n            "val_threshold": thr_info["threshold"],\n            "lr": optimizer.param_groups[0]["lr"],\n            "epoch_seconds": round(time.time() - epoch_start, 2),\n        }\n        history.append(row)\n\n        print(\n            f"Epoch {epoch:03d} | "\n            f"loss={row[\'train_loss\']:.5f} | "\n            f"kd={row[\'train_kd_loss\']:.5f} | "\n            f"supcon={row[\'train_supcon_loss\']:.5f} | "\n            f"val_pr_auc={row[\'val_PR_AUC\']:.5f} | "\n            f"val_f1={row[\'val_Fraud_F1\']:.5f} | "\n            f"val_rec={row[\'val_Fraud_Recall\']:.5f} | "\n            f"val_thr={row[\'val_threshold\']:.5f} | "\n            f"time={row[\'epoch_seconds\']}s"\n        )\n\n        if val_metrics["PR_AUC"] > best_val_pr_auc:\n            best_val_pr_auc = val_metrics["PR_AUC"]\n            best_epoch = epoch\n            patience_counter = 0\n            torch.save({\n                "experiment": experiment,\n                "epoch": epoch,\n                "model_state_dict": model.state_dict(),\n                "dims": dims,\n                "categorical_cardinalities": cat_cardinalities,\n                "config": {\n                    "LEARNING_RATE": LEARNING_RATE,\n                    "WEIGHT_DECAY": WEIGHT_DECAY,\n                    "KD_WEIGHT": KD_WEIGHT,\n                    "LAMBDA_SUPCON": LAMBDA_SUPCON,\n                    "FOCAL_ALPHA": FOCAL_ALPHA,\n                    "FOCAL_GAMMA": FOCAL_GAMMA,\n                },\n            }, ckpt_path)\n        else:\n            patience_counter += 1\n            if patience_counter >= PATIENCE:\n                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")\n                break\n\n    pd.DataFrame(history).to_csv(history_path, index=False)\n\n    checkpoint = torch.load(ckpt_path, map_location=device)\n    model.load_state_dict(checkpoint["model_state_dict"])\n\n    y_val, val_prob, _ = collect_predictions(model, val_loader, device)\n    thr_info = threshold_by_best_f1(y_val, val_prob)\n    selected_threshold = thr_info["threshold"]\n\n    y_test, test_prob, gate_weights = collect_predictions(model, test_loader, device)\n    test_metrics = evaluate_fraud_metrics(\n        y_test,\n        test_prob,\n        selected_threshold,\n        min_precision=MIN_PRECISION_TARGET,\n    )\n\n    amount = None\n    if review_meta_test is not None and "TransactionAmt_raw" in review_meta_test.columns and len(review_meta_test) == len(y_test):\n        amount = pd.to_numeric(review_meta_test["TransactionAmt_raw"], errors="coerce").fillna(0).to_numpy(dtype=float)\n    else:\n        amount = np.zeros(len(y_test), dtype=float)\n\n    budget_df = review_budget_metrics(y_test, test_prob, amount)\n    budget_flat = flatten_budget_metrics(budget_df)\n\n    pred_df = pd.DataFrame({\n        "y_true": y_test.astype(int),\n        "y_prob": test_prob.astype(float),\n        "y_pred": (test_prob >= selected_threshold).astype(int),\n    })\n\n    if gate_weights is not None:\n        pred_df["gate_cnn_weight"] = gate_weights[:, 0]\n        pred_df["gate_deepfm_weight"] = gate_weights[:, 1]\n\n    pred_df = attach_meta(pred_df, review_meta_test)\n\n    pred_path = OUTPUT_DIR / f"pred_{experiment}.csv"\n    budget_path = OUTPUT_DIR / f"review_budget_detail_{experiment}.csv"\n\n    pred_df.to_csv(pred_path, index=False)\n    budget_df.insert(0, "Model", experiment)\n    budget_df.to_csv(budget_path, index=False)\n\n    result = {\n        "Model": experiment,\n        "Best_Epoch": int(best_epoch),\n        "Best_Val_PR_AUC": float(best_val_pr_auc),\n        "Selected_Threshold_from_Val": float(selected_threshold),\n        "Val_Best_Fraud_Precision": float(thr_info["precision"]),\n        "Val_Best_Fraud_Recall": float(thr_info["recall"]),\n        "Val_Best_Fraud_F1": float(thr_info["f1"]),\n        "Train_Time_Seconds": round(time.time() - start_time, 3),\n        "Checkpoint_File": str(ckpt_path),\n        "History_File": str(history_path),\n        "Prediction_File": str(pred_path),\n        "Review_Budget_Detail_File": str(budget_path),\n        **test_metrics,\n        **budget_flat,\n    }\n\n    print("\\nFINAL TEST RESULT")\n    for k in [\n        "PR_AUC", "ROC_AUC", "Fraud_Precision", "Fraud_Recall", "Fraud_F1",\n        "MCC", f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",\n        "Precision@5%", "Recall@5%", "CapturedAmountRate@5%",\n        "Selected_Threshold", "TN", "FP", "FN", "TP",\n    ]:\n        if k in result:\n            print(k, ":", result[k])\n\n    return result\n\n\n# ============================================================\n# MAIN\n# ============================================================\n\ndef save_paper_table(results_df: pd.DataFrame):\n    paper_cols = [\n        "Model",\n        "PR_AUC",\n        "ROC_AUC",\n        "Fraud_Precision",\n        "Fraud_Recall",\n        "Fraud_F1",\n        "MCC",\n        f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",\n        "Precision@1%",\n        "Recall@1%",\n        "CapturedAmountRate@1%",\n        "ExpectedUtility@1%",\n        "Precision@3%",\n        "Recall@3%",\n        "CapturedAmountRate@3%",\n        "ExpectedUtility@3%",\n        "Precision@5%",\n        "Recall@5%",\n        "CapturedAmountRate@5%",\n        "ExpectedUtility@5%",\n        "Precision@10%",\n        "Recall@10%",\n        "CapturedAmountRate@10%",\n        "ExpectedUtility@10%",\n        "Selected_Threshold_from_Val",\n        "TN", "FP", "FN", "TP",\n        "Best_Epoch",\n        "Best_Val_PR_AUC",\n        "Train_Time_Seconds",\n    ]\n    paper_cols = [c for c in paper_cols if c in results_df.columns]\n    results_df[paper_cols].to_csv(OUTPUT_DIR / "paper_ready_proposed_only_cost_aware_table.csv", index=False)\n\n\ndef main():\n    set_seed(RANDOM_STATE)\n\n    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    print("Device:", device)\n    print("Running ONLY proposed model: tree_guided_moe_kd_focal_supcon")\n    print("Output dir:", OUTPUT_DIR)\n\n    train_loader, val_loader, test_loader, dims, cat_cardinalities, test_ds = build_loaders()\n\n    try:\n        review_meta_test = read_table("review_meta_internal_test")\n        if SUBSET_TEST_N is not None:\n            idx = maybe_subset_by_y(review_meta_test[[TARGET_COL]].reset_index(), SUBSET_TEST_N)["index"].to_numpy()\n            review_meta_test = review_meta_test.iloc[idx].reset_index(drop=True)\n    except Exception as e:\n        print("[WARN] Could not load review_meta_internal_test:", e)\n        review_meta_test = None\n\n    all_results = []\n\n    for experiment in RUN_EXPERIMENTS:\n        if "kd" in experiment and not TEACHER_DIR.exists():\n            print("[SKIP] KD experiment needs teacher_signals folder:", experiment)\n            continue\n\n        try:\n            result = run_experiment(\n                experiment=experiment,\n                train_loader=train_loader,\n                val_loader=val_loader,\n                test_loader=test_loader,\n                dims=dims,\n                cat_cardinalities=cat_cardinalities,\n                test_ds=test_ds,\n                review_meta_test=review_meta_test,\n                device=device,\n            )\n            all_results.append(result)\n\n            running_df = pd.DataFrame(all_results).sort_values("PR_AUC", ascending=False)\n            running_df.to_csv(OUTPUT_DIR / "proposed_only_results_running.csv", index=False)\n            with open(OUTPUT_DIR / "proposed_only_results_running.json", "w", encoding="utf-8") as f:\n                json.dump(running_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)\n            save_paper_table(running_df)\n\n            gc.collect()\n            if torch.cuda.is_available():\n                torch.cuda.empty_cache()\n\n        except Exception as e:\n            print(f"[ERROR] Experiment failed: {experiment}")\n            print(type(e).__name__, str(e))\n\n    if not all_results:\n        raise RuntimeError("No DL experiment completed successfully.")\n\n    results_df = pd.DataFrame(all_results).sort_values("PR_AUC", ascending=False)\n\n    final_csv = OUTPUT_DIR / "proposed_only_results_final.csv"\n    final_json = OUTPUT_DIR / "proposed_only_results_final.json"\n\n    results_df.to_csv(final_csv, index=False)\n    with open(final_json, "w", encoding="utf-8") as f:\n        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)\n\n    save_paper_table(results_df)\n\n    print("\\nSaved final results:")\n    print(final_csv)\n    print(final_json)\n    print(OUTPUT_DIR / "paper_ready_proposed_only_cost_aware_table.csv")\n    print(results_df)\n\n\nif __name__ == "__main__":\n    main()\n'
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


DEPENDENCY_FILES = [
    "cnn_branch_updated.py",
    "deepfm_branch_updated.py",
    "fraud_losses.py",
    "moe_gated_fusion.py",
    "proposed_tree_guided_cnnmix_deepfmmix_moe_kd_focal_supcon.py",
]


def copy_dependencies_to_generated_dir(scripts_dir: Path, generated_dir: Path) -> None:
    """
    Generated scripts are executed from generated_tuning_runs.
    Without this, Python may not find cnn_branch_updated / deepfm_branch_updated.
    """
    import shutil

    for filename in DEPENDENCY_FILES:
        src = scripts_dir / filename
        dst = generated_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[COPY] {src} -> {dst}")
        else:
            print(f"[WARN] Dependency not found in scripts_dir: {src}")


def add_import_path_patch(code: str) -> str:
    """
    Insert sys.path fix at top of every generated tuning script.
    This allows generated scripts to import dependencies from:
    - generated_tuning_runs
    - parent folder containing the tuner/dependencies
    """
    patch = (
        "import sys\n"
        "from pathlib import Path as _PathForImport\n"
        "_THIS_DIR = _PathForImport(__file__).resolve().parent\n"
        "_PARENT_DIR = _THIS_DIR.parent\n"
        "for _p in [str(_THIS_DIR), str(_PARENT_DIR)]:\n"
        "    if _p not in sys.path:\n"
        "        sys.path.insert(0, _p)\n"
    )

    if "from __future__ import annotations" in code:
        return code.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n\n" + patch + "\n",
            1,
        )

    return patch + "\n" + code


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
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        print("Template:", template_path)
    else:
        print(f"[WARN] Missing template script: {template_path}")
        print("[INFO] Using embedded template inside tune_proposed_tree_guided_moe_v4_fix_import.py")
        template = EMBEDDED_TEMPLATE_SCRIPT
    generated_dir = scripts_dir / "generated_tuning_runs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    copy_dependencies_to_generated_dir(scripts_dir, generated_dir)

    cfgs = make_grid(extra_seeds=args.extra_seeds)
    if args.start_index:
        cfgs = cfgs[args.start_index:]
    if args.max_configs is not None:
        cfgs = cfgs[:args.max_configs]

    print("=" * 100)
    print("TUNE PROPOSED TREE-GUIDED MOE V4 FIX IMPORT")
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
        patched = add_import_path_patch(patched)
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
