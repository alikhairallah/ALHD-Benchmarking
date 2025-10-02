##################### CONFIGURE PARAMETERS ######################
import argparse

parser = argparse.ArgumentParser(description="Run experiment with configurable test sources and data size.")
parser.add_argument('--test_sources', nargs='*', default=[], help='List of test sources, e.g. --test_sources HARD BRAD')
parser.add_argument('--data_size', type=str, default='10', help='Dataset size to use (e.g., "10" or "full")')
args = parser.parse_args()

TEST_SOURCES = args.test_sources
DATA_SIZE    = args.data_size
print("TEST_SOURCES:", TEST_SOURCES)
print("DATA_SIZE:", DATA_SIZE)

# ===================== CONFIGURATIONS ==========================
EXPERIMENT_TYPE   = "TRADITIONAL"
DATA_PATH         = "../../../dataset/balanced/split" # update dataset path
LOGS_FOLDER       = f"../../../logs/benchmarking/{EXPERIMENT_TYPE}/{DATA_SIZE}" # update logs folder path
SEED              = 42

# Standardize on CPU for fairness/simplicity
LGBM_DEVICE_TYPE  = "cpu"

# ---- Compute / Parallelism ----
N_JOBS            = 8
CV_N_SPLITS       = 3
USE_THREADING_CV  = True

# ---- TF-IDF ----
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE  = (1, 2)

# ---- Model selection style ----
USE_GRID = True

# ---- Hyperparameter grids ----
GRIDS = {
    "LogisticRegression": {
        "C": [0.1, 1.0, 2.0],
        "solver": ["liblinear", "saga"],
        "max_iter": [500, 1000]
    },
    "LinearSVC": {
        "C": [0.1, 1.0, 2.0],
        "max_iter": [3000, 5000]
    },
    "ComplementNB": {
        "alpha": [0.01, 0.1, 0.5, 1.0],
        "norm": [True, False]
    },
    "LightGBM": {
        "learning_rate": [0.1],
        "num_leaves": [31, 63],
        "max_depth": [10]
    },
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2]
    }
}
# ================================================================

# ================= LOG FILE/FOLDER PATH CONFIGURATION ===========
if not TEST_SOURCES or TEST_SOURCES == 0:
    sources_str = "all"
elif len(TEST_SOURCES) == 1:
    sources_str = TEST_SOURCES[0].lower()
else:
    sources_str = "_".join([s.lower() for s in TEST_SOURCES])

from datetime import datetime
log_filename = f"{LOGS_FOLDER}/{datetime.now():%Y%m%d_%H%M%S}_ALHD_{EXPERIMENT_TYPE}_{sources_str}_{DATA_SIZE}_benchmarking_log.txt"

#####################################################################
#################### TRAIN / VAL / TEST PATHS #######################
#####################################################################
TRAIN_FILE = f"{DATA_PATH}/{DATA_SIZE}/ALHD_train_{sources_str}_{DATA_SIZE}.csv"
VAL_FILE   = f"{DATA_PATH}/{DATA_SIZE}/ALHD_val_{sources_str}_{DATA_SIZE}.csv"
TEST_FILE  = f"{DATA_PATH}/{DATA_SIZE}/ALHD_test_{sources_str}_{DATA_SIZE}.csv"

#####################################################################
# ================== LOGGING SETUP ==================
import logging, os, sys, warnings
os.makedirs(LOGS_FOLDER, exist_ok=True)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
logging.info(f"Log file created: {log_filename}")
logging.info(f"Absolute log path: {os.path.abspath(log_filename)}")

def custom_warning_to_log(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} ({filename}:{lineno})")
warnings.showwarning = custom_warning_to_log

def log_uncaught_exceptions(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb); return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
sys.excepthook = log_uncaught_exceptions

# ==================== IMPORTS ====================
logging.info("=============== 1. Importing libraries Started ===============")
import gc, random, numpy as np, pandas as pd
from tqdm import tqdm
import lightgbm
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)
import scipy.sparse as sp

# Repro
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["WANDB_DISABLED"]  = "true"

logging.info("========== LIBRARY VERSIONS ==========")
logging.info(f"Python:        {sys.version}")
logging.info(f"Pandas:        {pd.__version__}")
logging.info(f"Numpy:         {np.__version__}")
logging.info(f"Sklearn:       {__import__('sklearn').__version__}")
logging.info(f"LightGBM:      {lightgbm.__version__}")
logging.info("======================================")
logging.info("=============== Importing libraries Ended ===============")

# ==================== ECHO CONFIG ====================
logging.info("# ==================== EXPERIMENT CONFIGURATIONS ====================")
for k, v in [
    ("EXPERIMENT_TYPE", EXPERIMENT_TYPE),
    ("TEST_SOURCES", TEST_SOURCES),
    ("DATA_SIZE", DATA_SIZE),
    ("DATA_PATH", DATA_PATH),
    ("LOGS_FOLDER", LOGS_FOLDER),
    ("TRAIN_FILE", TRAIN_FILE),
    ("VAL_FILE", VAL_FILE),
    ("TEST_FILE", TEST_FILE),
    ("SEED", SEED),
    ("N_JOBS", N_JOBS),
    ("CV_N_SPLITS", CV_N_SPLITS),
    ("USE_THREADING_CV", USE_THREADING_CV),
    ("TFIDF_MAX_FEATURES", TFIDF_MAX_FEATURES),
    ("TFIDF_NGRAM_RANGE", TFIDF_NGRAM_RANGE),
    ("USE_GRID", USE_GRID),
    ("LGBM_DEVICE_TYPE", LGBM_DEVICE_TYPE),
]:
    logging.info(f"{k}: {v}")
from pprint import pformat
logging.info("Hyperparameter Grids:\n" + pformat(GRIDS))
logging.info("# ===================================================================")

# ==================== HELPERS ====================
def log_df_stats(df, name):
    n_rows = len(df)
    n_labels = df['label'].nunique() if 'label' in df else 'N/A'
    n_sources = df['source'].nunique() if 'source' in df else 'N/A'
    n_docs = df['document_id'].nunique() if 'document_id' in df else 'N/A'
    nulls = df.isnull().sum().sum()
    empties = (df.astype(str).apply(lambda x: x.str.strip() == '')).sum().sum()
    logging.info(f"{name} shape: {df.shape}, rows: {n_rows}")
    logging.info(f"{name} unique labels: {n_labels}, unique sources: {n_sources}, unique document_ids: {n_docs}")
    logging.info(f"{name} null values: {nulls}, empty values: {empties}")

def log_classification_and_cm(y_true, y_pred, target_names=None, model_name=""):
    if target_names is None:
        classes = np.unique(y_true)
        target_names = [str(c) for c in classes]
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report_txt = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f"True {t}" for t in target_names],
                         columns=[f"Pred {t}" for t in target_names])
    logging.info("\n" + "="*60)
    logging.info(f"Results for {model_name}")
    logging.info("="*60)
    logging.info(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
    logging.info("\nClassification Report:\n" + report_txt)
    logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

def _densify_if_needed(X, model_name: str, stage: str):
    """
    Try to return X as-is. If the downstream estimator rejects sparse matrices,
    convert to dense with a clear log line. This keeps features and values identical.
    """
    try:
        # Cheap probe: many sklearn checks will touch .dtype or shape only.
        # We leave actual fitting to the estimator; densify in the except block there if needed.
        return X
    except Exception as e:
        logging.warning(f"{model_name} ({stage}): unexpected issue with sparse input: {e}; converting to dense.")
        return X.toarray() if sp.issparse(X) else X

# ==================== LOAD DATA ====================
logging.info("=============== 3 Data Preparation: Loading preprocessed files Started ===============")
df_train = pd.read_csv(TRAIN_FILE); logging.info("TRAIN_FILE LOADED.")
df_val   = pd.read_csv(VAL_FILE);   logging.info("VAL_FILE LOADED.")
df_test  = pd.read_csv(TEST_FILE);  logging.info("TEST_FILE LOADED.")

log_df_stats(df_train, "Train")
log_df_stats(df_val, "Val")
log_df_stats(df_test, "Test")

# --- Sanity: no document_id leakage across splits ---
train_ids = set(df_train.get("document_id", []))
val_ids   = set(df_val.get("document_id", []))
test_ids  = set(df_test.get("document_id", []))
leak_train_val  = train_ids & val_ids
leak_train_test = train_ids & test_ids
leak_val_test   = val_ids & test_ids
if leak_train_val or leak_train_test or leak_val_test:
    msg = []
    if leak_train_val:  msg.append(f"Train/Val leakage: {len(leak_train_val)} (e.g. {list(leak_train_val)[:5]})")
    if leak_train_test: msg.append(f"Train/Test leakage: {len(leak_train_test)} (e.g. {list(leak_train_test)[:5]})")
    if leak_val_test:   msg.append(f"Val/Test leakage: {len(leak_val_test)} (e.g. {list(leak_val_test)[:5]})")
    full_msg = " | ".join(msg)
    logging.critical(f"DATA LEAKAGE DETECTED! {full_msg}")
    raise ValueError(f"DATA LEAKAGE DETECTED! {full_msg}")
logging.info("Sanity check passed: No document_id leakage between train/val/test splits.")
logging.info("=============== 3 Data Preparation: Loading preprocessed files Ended ===============")

# ==================== TF-IDF with one-shot duplicate coalescing ====================
logging.info("=============== 3.1 TF-IDF Vectorization Started ===============")

tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    analyzer="word",
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=True,
    dtype=np.float32
)

Xtr_raw = tfidf.fit_transform(df_train["text"])
Xva_raw = tfidf.transform(df_val["text"])
Xte_raw = tfidf.transform(df_test["text"])

ytr = df_train["label"].values
yva = df_val["label"].values
yte = df_test["label"].values

feature_names_raw = np.asarray(tfidf.get_feature_names_out(), dtype=object)

# Build stable mapping: name -> list of column indices
name_to_idx = {}
for j, name in enumerate(feature_names_raw):
    name_to_idx.setdefault(name, []).append(j)

dup_count = sum(len(ixs) - 1 for ixs in name_to_idx.values() if len(ixs) > 1)
if dup_count > 0:
    logging.warning(f"Duplicate TF-IDF feature names detected: {dup_count}. Coalescing by summation.")

# Unique names in insertion order (stability)
unique_names = np.fromiter(name_to_idx.keys(), dtype=object)

# Projection matrix A: (orig_dim -> unique_dim) sums duplicates
rows, cols, data = [], [], []
for ucol, ixs in enumerate(name_to_idx.values()):
    rows.extend(ixs)
    cols.extend([ucol] * len(ixs))
    data.extend([1.0] * len(ixs))
A = sp.csr_matrix(
    (np.asarray(data, dtype=np.float32),
     (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
    shape=(feature_names_raw.size, unique_names.size)
)

# Apply to all splits -> final coalesced CSR matrices
Xtr_csr = (sp.csr_matrix(Xtr_raw) @ A).tocsr()
Xva_csr = (sp.csr_matrix(Xva_raw) @ A).tocsr()
Xte_csr = (sp.csr_matrix(Xte_raw) @ A).tocsr()
feature_names = unique_names

# Sanity: names now unique (for logging only; models use CSR)
assert feature_names.size == np.unique(feature_names).size, "Feature names still not unique after coalescing!"

logging.info(f"TF-IDF vocab size (raw): {feature_names_raw.size:,} | after coalesce: {feature_names.size:,}")
logging.info(f"Shapes (sparse CSR) — Train: {Xtr_csr.shape}, Val: {Xva_csr.shape}, Test: {Xte_csr.shape}")
logging.info("=============== TF-IDF Vectorization Ended ===============")

# ==================== MODELS ====================
logging.info("=============== 3.2 Model Setup Started ===============")
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", n_jobs=None, random_state=SEED),
    "LinearSVC":          LinearSVC(class_weight="balanced", dual=False, random_state=SEED),
    "ComplementNB":       ComplementNB(),
    "LightGBM":           LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=1,
                                         device_type=LGBM_DEVICE_TYPE, verbosity=-1, boosting_type="goss"),
    "RandomForest":       RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)
}
logging.info("=============== Model Setup Ended ===============")

# ==================== TRAIN + EVAL ====================
cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)

# joblib control (optional)
try:
    from threadpoolctl import threadpool_limits
    _HAVE_THREADPOOLCTL = True
except Exception:
    _HAVE_THREADPOOLCTL = False
from joblib import parallel_backend

def adjusted_grid(model_name: str):
    if model_name not in GRIDS:
        return None
    return {k: list(v) for k, v in GRIDS[model_name].items()}

def _fit_with_possible_densify(estimator, X, y, model_name: str, stage: str):
    """
    Try to fit with the given X (CSR). If the estimator rejects sparse,
    densify once with a clear log and retry.
    """
    try:
        estimator.fit(X, y)
        return estimator
    except TypeError as e:
        if sp.issparse(X):
            logging.info(f"{model_name} ({stage}): estimator rejected sparse input ({e}); converting to dense and retrying.")
            X_dense = X.toarray()
            estimator.fit(X_dense, y)
            return estimator
        raise
    except ValueError as e:
        # Some builds error with ValueError on sparse; handle similarly
        if sp.issparse(X) and "sparse" in str(e).lower():
            logging.info(f"{model_name} ({stage}): estimator rejected sparse input ({e}); converting to dense and retrying.")
            X_dense = X.toarray()
            estimator.fit(X_dense, y)
            return estimator
        raise

def fit_with_grid(name, estimator, X, y):
    """GridSearchCV with accuracy scoring on (possibly sparse) X."""
    if not (USE_GRID and name in GRIDS):
        return _fit_with_possible_densify(estimator, X, y, name, stage="final-no-grid"), None

    param_grid = adjusted_grid(name)
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
        error_score="raise"
    )
    if USE_THREADING_CV and _HAVE_THREADPOOLCTL:
        with threadpool_limits(limits=1):
            with parallel_backend("threading", n_jobs=N_JOBS):
                # GridSearchCV will call .fit() internally; handle sparse rejection via try/except
                try:
                    gs.fit(X, y)
                except (TypeError, ValueError) as e:
                    if sp.issparse(X):
                        logging.info(f"{name} (grid): estimator/grid rejected sparse input ({e}); converting to dense for grid.")
                        X_dense = X.toarray()
                        gs.fit(X_dense, y)
                    else:
                        raise
    elif USE_THREADING_CV:
        with parallel_backend("threading", n_jobs=N_JOBS):
            try:
                gs.fit(X, y)
            except (TypeError, ValueError) as e:
                if sp.issparse(X):
                    logging.info(f"{name} (grid): estimator/grid rejected sparse input ({e}); converting to dense for grid.")
                    X_dense = X.toarray()
                    gs.fit(X_dense, y)
                else:
                    raise
    else:
        try:
            gs.fit(X, y)
        except (TypeError, ValueError) as e:
            if sp.issparse(X):
                logging.info(f"{name} (grid): estimator/grid rejected sparse input ({e}); converting to dense for grid.")
                X_dense = X.toarray()
                gs.fit(X_dense, y)
            else:
                raise

    logging.info(f"{name} - GridSearch best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_

def _maybe_dense_for_predict(estimator, X, model_name: str):
    """
    Use X as-is; if estimator errors on sparse at predict time, densify once and reuse.
    """
    try:
        # dry-run access; real call happens in caller
        return X
    except Exception:
        return X.toarray() if sp.issparse(X) else X

def train_and_evaluate(name, estimator):
    """CV (for selection), final fit on Train, then eval on Val/Test."""
    # All models see the same coalesced CSR inputs
    Xtr_in, Xva_in, Xte_in = Xtr_csr, Xva_csr, Xte_csr

    # 1) Grid/CV for param selection (with sparse→dense fallback if needed)
    est_to_fit, best_params = fit_with_grid(name, estimator, Xtr_in, ytr)

    # 2) Final fit on full Train (explicit for LightGBM to use early stopping)
    if name == "LightGBM":
        est_to_fit.set_params(n_estimators=est_to_fit.get_params().get("n_estimators", 800))
        try:
            est_to_fit.fit(
                Xtr_in, ytr,
                eval_set=[(Xva_in, yva)],
                eval_metric="auc",
                callbacks=[lightgbm.early_stopping(stopping_rounds=100, verbose=False)]
            )
        except (TypeError, ValueError) as e:
            # Fallback: densify only if your LightGBM build rejects CSR (rare).
            logging.info(f"LightGBM (final): rejected sparse input ({e}); converting to dense.")
            est_to_fit.fit(
                Xtr_in.toarray(), ytr,
                eval_set=[(Xva_in.toarray(), yva)],
                eval_metric="auc",
                callbacks=[lightgbm.early_stopping(stopping_rounds=100, verbose=False)]
            )
        if hasattr(est_to_fit, "best_iteration_"):
            logging.info(f"LightGBM best_iteration_ after ES: {est_to_fit.best_iteration_}")
    else:
        # Non-LGBM: already handled sparse→dense fallback inside helper
        est_to_fit = _fit_with_possible_densify(est_to_fit, Xtr_in, ytr, name, stage="final")

    # 3) Calibrate LinearSVC for probabilities (ROC-AUC)
    if name == "LinearSVC":
        calibrator = CalibratedClassifierCV(est_to_fit, method="sigmoid", cv=3)
        # Fit calibrator (sparse allowed; fallback inside helper if needed)
        try:
            calibrator.fit(Xtr_in, ytr)
        except (TypeError, ValueError):
            calibrator.fit(Xtr_in.toarray(), ytr)
        est_to_fit = calibrator

    # 4) Evaluate
    Xva_eval, Xte_eval = Xva_in, Xte_in
    try:
        yva_pred = est_to_fit.predict(Xva_eval)
    except (TypeError, ValueError):
        yva_pred = est_to_fit.predict(Xva_eval.toarray())
    try:
        yte_pred = est_to_fit.predict(Xte_eval)
    except (TypeError, ValueError):
        yte_pred = est_to_fit.predict(Xte_eval.toarray())

    if hasattr(est_to_fit, "predict_proba"):
        try:
            yte_proba = est_to_fit.predict_proba(Xte_eval)[:, 1]
        except (TypeError, ValueError):
            yte_proba = est_to_fit.predict_proba(Xte_eval.toarray())[:, 1]
    else:
        yte_proba = None

    val_acc  = accuracy_score(yva, yva_pred)
    val_f1   = f1_score(yva, yva_pred)
    test_acc = accuracy_score(yte, yte_pred)
    test_f1  = f1_score(yte, yte_pred)
    test_auc = roc_auc_score(yte, yte_proba) if yte_proba is not None else None

    logging.info(f"{name} — Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    logging.info(f"{name} — Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test ROC-AUC: {test_auc if test_auc is not None else 'N/A'}")
    log_classification_and_cm(yte, yte_pred, target_names=["Human","LLM"], model_name=name)

    metrics = {
        "best_params": best_params,
        "val_acc": val_acc, "val_f1": val_f1,
        "test_acc": test_acc, "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "input_kind": "sparse"
    }
    return est_to_fit, metrics

# ==================== RUN ALL MODELS ====================
logging.info("=============== 3.3 Training & Evaluation Started ===============")
fitted = {}
results = {}
for name, est in tqdm(models.items(), desc="Fitting models", total=len(models)):
    fitted_est, metrics = train_and_evaluate(name, est)
    fitted[name] = fitted_est
    results[name] = metrics
logging.info("=============== Training & Evaluation Finished ===============")

# ==================== SUMMARY ====================
summary = pd.DataFrame([
    {"Model": k,
     "Input": v["input_kind"],
     "Val Acc": v["val_acc"], "Val F1": v["val_f1"],
     "Test Acc": v["test_acc"], "Test F1": v["test_f1"],
     "Test ROC-AUC": v["test_roc_auc"],
     "Best Params": v["best_params"]}
    for k, v in results.items()
]).sort_values("Test Acc", ascending=False).reset_index(drop=True)

print(summary)
logging.info(f"Summary table (sorted by Test Acc desc):\n{summary}")
if not summary.empty:
    best_acc_row = summary.iloc[0]
    logging.info(f"Best by Test Accuracy: {best_acc_row['Model']} (Acc={best_acc_row['Test Acc']:.4f})")

# ==================== ERROR ANALYSIS (summaries only) ====================
logging.info("=============== 3.4 Error Analysis Started ===============")
if not summary.empty:
    best_model_name = summary.iloc[0]["Model"]
    best_model = fitted[best_model_name]
    Xte_in_best = Xte_csr
    df_test = df_test.copy()
    try:
        df_test["pred"] = best_model.predict(Xte_in_best)
    except (TypeError, ValueError):
        df_test["pred"] = best_model.predict(Xte_in_best.toarray())
    df_test["correct"] = df_test["pred"] == df_test["label"]
    df_errors = df_test[~df_test["correct"]]

    error_label_counts = df_errors["label"].value_counts(dropna=False)
    error_generator_counts = df_errors["generator"].value_counts(dropna=False) if "generator" in df_errors.columns else None
    error_source_counts = df_errors["source"].value_counts(dropna=False) if "source" in df_errors.columns else None

    logging.info(f"Error counts by label:\n{error_label_counts}")
    if error_generator_counts is not None:
        logging.info(f"Error counts by generator:\n{error_generator_counts}")
    if error_source_counts is not None:
        logging.info(f"Error counts by source:\n{error_source_counts}")

    cm = confusion_matrix(df_test["label"], df_test["pred"])
    logging.info(f"Confusion matrix (rows: true, cols: pred):\n{cm}")
logging.info("=============== 3.4 Error Analysis Ended ===============")
