##################### CONFIGURE PARAMETERS ######################
import argparse

parser = argparse.ArgumentParser(description="Run experiment with configurable test sources and data size.")
parser.add_argument('--test_sources', nargs='*', default=[], help='List of test sources, e.g. --test_sources HARD BRAD')
parser.add_argument('--data_size', type=str, default='10', help='Dataset size to use (e.g., "10" or "full")')
args = parser.parse_args()

TEST_SOURCES = args.test_sources
DATA_SIZE = args.data_size

print("TEST_SOURCES:", TEST_SOURCES)
print("DATA_SIZE:", DATA_SIZE)

# ==============================================================
# ===================== CONFIGURATIONS ==========================
# ==============================================================
EXPERIMENT_TYPE= 'TRANSFORMER' 
#TEST_SOURCES = []  # e.g. ['BRAD', 'HARD'] or [] for random split
#DATA_SIZE = '10' #'10' or 'full'
DATA_PATH= '../../../dataset/balanced/split' # update dataset path
TEMP_DIRECTORY= "./tmp"
LOGS_FOLDER = f"../../../logs/benchmarking/{EXPERIMENT_TYPE}/{DATA_SIZE}" # update log file path
SEED = 42
# ================================================================
# ================= LOG FILE/FOLDER PATH CONFIGURATION ===========
# ================================================================

# Convert TEST_SOURCES to string

if not TEST_SOURCES or TEST_SOURCES == 0:
    sources_str = "all"
elif len(TEST_SOURCES) == 1:
    sources_str = TEST_SOURCES[0].lower()
else:
    sources_str = "_".join([s.lower() for s in TEST_SOURCES])

# Build log filename
from datetime import datetime

log_filename = f"{LOGS_FOLDER}/{datetime.now():%Y%m%d_%H%M%S}_ALHD_{EXPERIMENT_TYPE}_{sources_str}_{DATA_SIZE}_benchmarking_log.txt"

# ==============================================================
# =============== GENERATED FILES PATH CONFIGURATION ===========
# ==============================================================

TRAIN_FILE = f"{DATA_PATH}/{DATA_SIZE}/ALHD_train_{sources_str}_{DATA_SIZE}.csv" 
VAL_FILE   = f"{DATA_PATH}/{DATA_SIZE}/ALHD_val_{sources_str}_{DATA_SIZE}.csv" 
TEST_FILE  = f"{DATA_PATH}/{DATA_SIZE}/ALHD_test_{sources_str}_{DATA_SIZE}.csv" 

# ==============================================================
# ==============================================================
# ==============================================================

import logging
import os
import sys
import warnings

# Ensure the 'logs' folder exists
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Log file created: {log_filename}")
logging.info(f"Absolute log path: {os.path.abspath(log_filename)}")

# Redirect warnings to logging
def custom_warning_to_log(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} ({filename}:{lineno})")
warnings.showwarning = custom_warning_to_log

# Log uncaught exceptions
def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = log_uncaught_exceptions

logging.info("# ==================== EXPERIMENT CONFIGURATIONS ====================")
logging.info(f"EXPERIMENT_TYPE: {EXPERIMENT_TYPE}")
logging.info(f"TEST_SOURCES: {TEST_SOURCES}")
logging.info(f"DATA_SIZE: {DATA_SIZE}")
logging.info(f"DATA_PATH: {DATA_PATH}")
logging.info(f"LOGS_FOLDER: {LOGS_FOLDER}")
logging.info(f"TRAIN_FILE: {TRAIN_FILE}")
logging.info(f"VAL_FILE: {VAL_FILE}")
logging.info(f"TEST_FILE: {TEST_FILE}")
logging.info(f"SEED: {SEED}")
logging.info(f"Log file created: {log_filename}")
logging.info("# ===================================================================")

logging.info("=============== 1. Importing libraries Started ===============")

# ==================== SYSTEM & DATA ====================
import gc
import random
import numpy as np
import pandas as pd

# ==================== VISUALIZATION ====================
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== TRANSFORMERS & DEEP LEARNING ====================
import arabert
import matplotlib
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from arabert.preprocess import ArabertPreprocessor

# ==================== REPRODUCIBILITY ====================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["WANDB_DISABLED"] = "true"

# ==================== LOG LIBRARY VERSIONS ====================
logging.info("========== LIBRARY VERSIONS ==========")
logging.info(f"Python:        {sys.version}")
logging.info(f"Pandas:        {pd.__version__}")
logging.info(f"Numpy:         {np.__version__}")
logging.info(f"Sklearn:       {__import__('sklearn').__version__}")
logging.info(f"PyTorch:       {torch.__version__}")
logging.info(f"Transformers:  {transformers.__version__}")
logging.info(f"Seaborn:       {sns.__version__}")
logging.info(f"Matplotlib:    {matplotlib.__version__}")
logging.info(f"Arabert:       {arabert.__version__ if hasattr(arabert, '__version__') else 'unknown'}")
logging.info("======================================")


logging.info("=============== Importing libraries Ended ===============")
logging.info("==============================================================")


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

logging.info("=============== 3 Data Preparation: Loading preprocessed files Started ===============")

df_train = pd.read_csv(TRAIN_FILE)
logging.info("TRAIN_FILE LOADED.")
df_val   = pd.read_csv(VAL_FILE)
logging.info("VAL_FILE LOADED.")
df_test  = pd.read_csv(TEST_FILE)
logging.info("TEST_FILE LOADED.")

log_df_stats(df_train, "Train")
log_df_stats(df_val, "Val")
log_df_stats(df_test, "Test")

#logging.info("Sample head of train set:\n" + df_train.head().to_string())

# --- Sanity Check: document_id leakage between splits ---
train_ids = set(df_train['document_id']) if 'document_id' in df_train else set()
val_ids   = set(df_val['document_id'])   if 'document_id' in df_val else set()
test_ids  = set(df_test['document_id'])  if 'document_id' in df_test else set()

leak_train_val = train_ids & val_ids
leak_train_test = train_ids & test_ids
leak_val_test = val_ids & test_ids

if leak_train_val or leak_train_test or leak_val_test:
    msg = []
    if leak_train_val:
        msg.append(f"Train/Val document_id leakage: {len(leak_train_val)} overlapping IDs (e.g. {list(leak_train_val)[:5]})")
    if leak_train_test:
        msg.append(f"Train/Test document_id leakage: {len(leak_train_test)} overlapping IDs (e.g. {list(leak_train_test)[:5]})")
    if leak_val_test:
        msg.append(f"Val/Test document_id leakage: {len(leak_val_test)} overlapping IDs (e.g. {list(leak_val_test)[:5]})")
    full_msg = " | ".join(msg)
    logging.critical(f"DATA LEAKAGE DETECTED! {full_msg}")
    raise ValueError(f"DATA LEAKAGE DETECTED! {full_msg}")

logging.info("Sanity check passed: No document_id leakage between train/val/test splits.")

logging.info("=============== 3 Data Preparation: Loading preprocessed files Ended ===============")
logging.info("======================================================================================")


def check_duplicates(df, name):
    total = len(df)
    # Count duplicated rows in 'text' (keep=False counts all appearances of any duplicated value)
    n_duplicates = df.duplicated(subset='text', keep=False).sum()
    n_unique = df['text'].nunique()
    print(f"==== {name.upper()} DUPLICATE CHECK ====")
    print(f"Total rows: {total}")
    print(f"Number of unique texts: {n_unique}")
    print(f"Number of duplicated texts: {n_duplicates} ({n_duplicates/total:.2%})")
    if n_duplicates > 0:
        print("Sample duplicates:")
        print(df[df.duplicated(subset='text', keep=False)].head(5)[['text', 'label']])
    print("-" * 40)

check_duplicates(df_train, "train")
check_duplicates(df_val, "val")
check_duplicates(df_test, "test")


logging.info("=============== 3.1 Transformer models train and evaluation started ===============")

def log_classification_and_cm(y_true, y_pred, target_names, model_key):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    header = f"\n{'='*60}\nResults for {model_key}\n{'='*60}"
    summary_str = f"Test set: Accuracy={acc*100:.2f}% | F1={f1_macro:.4f}"

    # Classification report DataFrame
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, digits=4)
    df_report = pd.DataFrame(report).transpose()
    report_str = "\nClassification Report:\n" + df_report.to_string()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {name}" for name in target_names],
        columns=[f"Pred {name}" for name in target_names]
    )
    cm_str = "\nConfusion Matrix:\n" + cm_df.to_string()

    log_msg = f"{header}\n{summary_str}{report_str}{cm_str}\n"
    logging.info(log_msg)

# --- Model and column configs ---
transformer_models = {
    "AraBERTv2-Base": "aubmindlab/bert-base-arabertv2",
    "AraBERTv2-Large": "aubmindlab/bert-large-arabertv2",
    "AraElectra": "aubmindlab/araelectra-base-discriminator",
    "Google-mBERT": "google-bert/bert-base-multilingual-cased",
    "XLM-R-Base": "FacebookAI/xlm-roberta-base",
    "XLM-R-Large": "FacebookAI/xlm-roberta-large",
    "ARBERTv2": "UBC-NLP/ARBERTv2",
    "MARBERT": "UBC-NLP/MARBERT",
    "Asafaya-BERT-Base": "asafaya/bert-base-arabic",
    "Asafaya-BERT-Large": "asafaya/bert-large-arabic"
}

custom_text_columns = {
    "AraBERTv2-Base": "text_for_arabert_base",
    "AraBERTv2-Large": "text_for_arabert_large",
    "AraElectra": "text_for_araelectra"
}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, logits[:, 1]),
    }

results = {}
target_names = ["Human", "LLM"]

for model_key, model_name in transformer_models.items():
    logging.info(f"Training transformer model: {model_key} ({model_name})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Select the proper column
    col = custom_text_columns[model_key] if model_key in custom_text_columns else "text"
    if col not in df_train.columns:
        raise ValueError(f"Column '{col}' required for '{model_key}' not found in DataFrame. "
                         f"Available columns: {df_train.columns.tolist()}")

    def make_hf_dataset(df):
        return Dataset.from_pandas(df[[col, 'label']].rename(columns={col: "text"}), preserve_index=False)

    datasets_dict = {
        "train": make_hf_dataset(df_train),
        "val": make_hf_dataset(df_val),
        "test": make_hf_dataset(df_test)
    }

    def tokenize(batch):
        texts = [str(x) if x is not None else "" for x in batch["text"]]
        return tokenizer(texts, truncation=True, padding='max_length', max_length=512)

    hf_datasets = DatasetDict(datasets_dict).map(tokenize, batched=True, num_proc=8).remove_columns(['text'])
    hf_datasets.set_format("torch")

    for split in hf_datasets:
        if 'label' in hf_datasets[split].column_names:
            hf_datasets[split] = hf_datasets[split].rename_column('label', 'labels')
        
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        metric_for_best_model="accuracy",
        greater_is_better=True,
        output_dir=f"{TEMP_DIRECTORY}/results_{sources_str}_{DATA_SIZE}_{model_key}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=f"{TEMP_DIRECTORY}/logs_{sources_str}_{DATA_SIZE}_{model_key}",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        seed=SEED,
        report_to="none",
        bf16=True,
        dataloader_num_workers=8
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=hf_datasets['train'],
        eval_dataset=hf_datasets['val'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logging.info(f"Starting training for {model_key}...")
    trainer.train()
    logging.info(f"Training complete for {model_key}.")

    outputs = trainer.predict(hf_datasets["test"])
    logits = outputs.predictions
    preds = np.argmax(logits, axis=-1)
    labels = outputs.label_ids

    log_classification_and_cm(labels, preds, target_names, model_key)

    test_metrics = compute_metrics((logits, labels))
    test_metrics['loss'] = outputs.metrics['test_loss']
    results[model_key] = {
        **test_metrics, 
        "confusion_matrix": confusion_matrix(labels, preds),
        "test_labels": labels,
        "test_preds": preds
    }

# === Summary Table ===
summary = pd.DataFrame([{
    "Model": name,
    "Test Accuracy": round(m["accuracy"], 4),
    "Test F1": round(m["f1"], 4),
    "Test ROC-AUC": round(m["roc_auc"], 4),
    "Test Loss": round(m["loss"], 4)
} for name, m in results.items()])

summary_msg = "\n==== Summary Table ====\n"
summary_msg += summary.to_string(index=False)
summary_msg += f"\n\nBest Accuracy Model: {summary.loc[summary['Test Accuracy'].idxmax()]['Model']}"
summary_msg += f"\nWorst Accuracy Model: {summary.loc[summary['Test Accuracy'].idxmin()]['Model']}"
summary_msg += f"\nBest F1 Model: {summary.loc[summary['Test F1'].idxmax()]['Model']}"
summary_msg += f"\nWorst F1 Model: {summary.loc[summary['Test F1'].idxmin()]['Model']}"
summary_msg += f"\nBest ROC-AUC Model: {summary.loc[summary['Test ROC-AUC'].idxmax()]['Model']}"
summary_msg += f"\nWorst ROC-AUC Model: {summary.loc[summary['Test ROC-AUC'].idxmin()]['Model']}"

logging.info(summary_msg)

# === Show metrics and confusion matrix for best accuracy model ===
best_acc_idx = summary['Test Accuracy'].idxmax()
best_model_key = summary.loc[best_acc_idx, "Model"]
best_result = results[best_model_key]

logging.info(f"\n=== Metrics and Confusion Matrix for Best Test Accuracy Model: {best_model_key} ===")
log_classification_and_cm(
    y_true=best_result.get("test_labels", None),
    y_pred=best_result.get("test_preds", None),
    target_names=target_names,
    model_key=best_model_key
)

logging.info("=============== Transformer models train and evaluation Ended ===============")
logging.info("==============================================================")


logging.info("=============== 3.2 Transformer models misclassification analysis started ===============")

# Select the best model by Test F1 (change to another metric if needed)
best_model_key = summary.loc[summary['Test F1'].idxmax(), 'Model']
logging.info(f"Starting misclassification analysis for: {best_model_key}")

# Use the same tokenizer and column for the best model
best_model_name = transformer_models[best_model_key]
col = custom_text_columns.get(best_model_key, "text_for_arabert_base")  # default to a column you know exists

tokenizer = AutoTokenizer.from_pretrained(best_model_name)

# Prepare test data for misclassification analysis
df_test_analyze = df_test.copy()
test_texts = df_test_analyze[col].tolist()
test_labels = df_test_analyze['label'].tolist()

outputs = trainer.predict(hf_datasets["test"])
preds = np.argmax(outputs.predictions, axis=-1)
labels = outputs.label_ids

df_test_analyze['pred'] = preds
df_test_analyze['correct'] = df_test_analyze['pred'] == df_test_analyze['label']
df_errors = df_test_analyze[~df_test_analyze['correct']]

# Error counts
logging.info(f"Misclassification count: {len(df_errors)} out of {len(df_test_analyze)}")

error_label_counts = df_errors['label'].value_counts()
error_generator_counts = df_errors['generator'].value_counts() if 'generator' in df_errors else "N/A"
error_source_counts = df_errors['source'].value_counts() if 'source' in df_errors else "N/A"

logging.info(f"Error counts by label:\n{error_label_counts}")
logging.info(f"Error counts by generator:\n{error_generator_counts}")
logging.info(f"Error counts by source:\n{error_source_counts}")

# Sample misclassified examples (up to 5)
num_samples = min(5, len(df_errors))
if num_samples > 0:
    # Use col instead of 'text_input'
    cols_to_show = [col, 'label', 'pred']
    if 'generator' in df_errors: cols_to_show.append('generator')
    if 'source' in df_errors: cols_to_show.append('source')
    sample_errors = df_errors[cols_to_show].sample(num_samples, random_state=SEED)
    logging.info(f"Sample misclassified examples:\n{sample_errors.to_string(index=False)}")
else:
    logging.info("No misclassified examples to display.")

logging.info("=============== Transformer models misclassification analysis ended ===============")
logging.info("==============================================================")
