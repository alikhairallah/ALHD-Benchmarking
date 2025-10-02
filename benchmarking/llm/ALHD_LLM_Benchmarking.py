##################### CONFIGURE PARAMETERS ######################
import argparse

parser = argparse.ArgumentParser(description="Run experiment with configurable test sources and data size.")
parser.add_argument('--test_sources', nargs='*', default=[], help='List of test sources, e.g. --test_sources HARD BRAD')
parser.add_argument('--data_size', type=str, default='10', help='Dataset size to use (e.g., "10" or "full")')
parser.add_argument('--hf_token', type=str, default=None,
                    help='Hugging Face access token. If omitted, will read HF_TOKEN or HUGGINGFACE_HUB_TOKEN from env.')
parser.add_argument('--cpu_threads', type=int, default=8,
                    help='Hard cap on CPU threads (OpenMP/MKL/NumExpr & torch).')
args = parser.parse_args()

TEST_SOURCES = args.test_sources
DATA_SIZE    = args.data_size

print("TEST_SOURCES:", TEST_SOURCES)
print("DATA_SIZE:", DATA_SIZE)

# ===================== CONFIGURATIONS ==========================
EXPERIMENT_TYPE     = "LLM"
DATA_PATH           = "../../../dataset/balanced/split" #update dataset path
LOGS_FOLDER         = f"../../../logs/benchmarking/{EXPERIMENT_TYPE}/{DATA_SIZE}" #update log folder path
SEED                = 42

# ---- HF cache & scratch locations ----
SCRATCH_DIR         = "/data/scratch/" #update scratch path
HF_CACHE_ROOT       = f"{SCRATCH_DIR}/hf_cache"
HF_HUB_CACHE        = f"{HF_CACHE_ROOT}/hub"
TRANSFORMERS_CACHE  = f"{HF_CACHE_ROOT}/transformers"

# ---------------------------------------------------------------
ZS_BATCH_SIZE       = 32
FS_BATCH_SIZE       = 16
MAX_NEW_TOKENS      = 1      # force exactly one token
FS_NUM_EXAMPLES     = 8
MAX_INPUT_TOKENS    = 512
CPU_THREADS         = max(1, int(args.cpu_threads))
# ==============================================================

# ================= LOG FILE/FOLDER PATH CONFIG =================
if not TEST_SOURCES or TEST_SOURCES == 0:
    sources_str = "all"
elif len(TEST_SOURCES) == 1:
    sources_str = TEST_SOURCES[0].lower()
else:
    sources_str = "_".join([s.lower() for s in TEST_SOURCES])

from datetime import datetime
log_filename = f"{LOGS_FOLDER}/{datetime.now():%Y%m%d_%H%M%S}_ALHD_{EXPERIMENT_TYPE}_{sources_str}_{DATA_SIZE}_benchmarking_log.txt"

# ================== LOGGING SETUP ==================
import logging, os, sys, warnings, shutil

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

# ================= HF TOKEN & CACHE ENVS =================
HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    logging.critical("No Hugging Face token provided! Please provide --hf_token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN in the environment.")
    sys.exit(1)
masked = (HF_TOKEN[:6] + "..." + HF_TOKEN[-4:])
logging.info(f"HF_TOKEN provided: True | token={masked}")

os.environ.setdefault("HF_HOME", HF_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", TRANSFORMERS_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# hard cap CPU usage
os.environ["OMP_NUM_THREADS"]       = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]  = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"]       = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"]= str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]   = str(CPU_THREADS)
for p in (HF_CACHE_ROOT, HF_HUB_CACHE, TRANSFORMERS_CACHE):
    os.makedirs(p, exist_ok=True)

def _free_gb(path):
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return float("inf")

# =============== GENERATED FILES PATH CONFIGURATION ===========
TRAIN_FILE = f"{DATA_PATH}/{DATA_SIZE}/ALHD_train_{sources_str}_{DATA_SIZE}.csv"
VAL_FILE   = f"{DATA_PATH}/{DATA_SIZE}/ALHD_val_{sources_str}_{DATA_SIZE}.csv"
TEST_FILE  = f"{DATA_PATH}/{DATA_SIZE}/ALHD_test_{sources_str}_{DATA_SIZE}.csv"

logging.info("=============== 1. Importing libraries Started ===============")

# ==================== SYSTEM & DATA ====================
import gc
import json
import tempfile
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)

# ==================== LLM & DEEP LEARNING ====================
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PretrainedConfig,
    LogitsProcessorList
)
from huggingface_hub import HfApi, hf_hub_download

# Global HF API handle (permanent fix for access checks)
_HF_API = HfApi()

# quiet down Transformers
transformers.utils.logging.set_verbosity_error()

# ==================== HF HUB AUTH (programmatic) ====================
try:
    from huggingface_hub import login as hf_login, hf_hub_enable_hf_transfer
    if HF_TOKEN:
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
        who = _HF_API.whoami(token=HF_TOKEN)
        logging.info(f"HuggingFace login OK: user={who.get('name') or who.get('email') or 'unknown'}")
    try:
        hf_hub_enable_hf_transfer()
        logging.info("Enabled hf_transfer for faster model downloads.")
    except Exception:
        pass
except Exception as e:
    logging.warning(f"Could not initialize HuggingFace login: {e}")

# ==================== SPEED / THREADS ====================
try:
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, CPU_THREADS // 2))
    except Exception:
        pass
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ==================== REPRODUCIBILITY ====================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["WANDB_DISABLED"] = "true"

# ==================== LOG VERSIONS & PATHS ====================
logging.info(f"Python: {sys.version}")
logging.info(f"Pandas: {pd.__version__}")
logging.info(f"Numpy: {np.__version__}")
logging.info(f"Sklearn: {__import__('sklearn').__version__}")
logging.info(f"PyTorch: {torch.__version__}")
logging.info(f"Transformers: {transformers.__version__}")
try:
    import tokenizers as _tk
    logging.info(f"Tokenizers: {_tk.__version__}")
except Exception:
    logging.info("Tokenizers: (not importable)")
logging.info("# ==================== EXPERIMENT CONFIGURATIONS ====================")
for k, v in [
    ("EXPERIMENT_TYPE",  EXPERIMENT_TYPE),
    ("TEST_SOURCES",     TEST_SOURCES),
    ("DATA_SIZE",        DATA_SIZE),
    ("DATA_PATH",        DATA_PATH),
    ("LOGS_FOLDER",      LOGS_FOLDER),
    ("SCRATCH_DIR",      SCRATCH_DIR),
    ("HF_CACHE_ROOT",    HF_CACHE_ROOT),
    ("HF_HUB_CACHE",     HF_HUB_CACHE),
    ("TRANSFORMERS_CACHE", TRANSFORMERS_CACHE),
    ("TRAIN_FILE",       TRAIN_FILE),
    ("VAL_FILE",         VAL_FILE),
    ("TEST_FILE",        TEST_FILE),
    ("SEED",             SEED),
    ("CPU_THREADS",      CPU_THREADS),
    ("ZS_BATCH_SIZE",    ZS_BATCH_SIZE),
    ("FS_BATCH_SIZE",    FS_BATCH_SIZE),
    ("MAX_NEW_TOKENS",   MAX_NEW_TOKENS),
    ("FS_NUM_EXAMPLES",  FS_NUM_EXAMPLES),
    ("MAX_INPUT_TOKENS", MAX_INPUT_TOKENS),
]:
    logging.info(f"{k}: {v}")

if _free_gb(HF_HUB_CACHE) < 40:
    logging.warning(f"Low free space in HF cache: {_free_gb(HF_HUB_CACHE):.1f} GB. Consider pruning old snapshots.")

logging.info("=============== Importing libraries Ended ===============")
logging.info("==============================================================")

# ===================== 2. Data loading =====================
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
df_train = pd.read_csv(TRAIN_FILE); logging.info("TRAIN_FILE LOADED.")
df_val   = pd.read_csv(VAL_FILE);   logging.info("VAL_FILE LOADED.")
df_test  = pd.read_csv(TEST_FILE);  logging.info("TEST_FILE LOADED.")
log_df_stats(df_train, "Train")
log_df_stats(df_val, "Val")
log_df_stats(df_test, "Test")

# --- Sanity Check: document_id leakage ---
train_ids = set(df_train['document_id']) if 'document_id' in df_train else set()
val_ids   = set(df_val['document_id'])   if 'document_id' in df_val else set()
test_ids  = set(df_test['document_id'])  if 'document_id' in df_test else set()
if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
    msg = []
    if train_ids & val_ids:  msg.append(f"Train/Val overlap: {len(train_ids & val_ids)}")
    if train_ids & test_ids: msg.append(f"Train/Test overlap: {len(train_ids & test_ids)}")
    if val_ids & test_ids:   msg.append(f"Val/Test overlap: {len(val_ids & test_ids)}")
    full_msg = " | ".join(msg)
    logging.critical(f"DATA LEAKAGE DETECTED! {full_msg}")
    raise ValueError(f"DATA LEAKAGE DETECTED! {full_msg}")
logging.info("Sanity check passed: No document_id leakage between train/val/test splits.")
logging.info("=============== 3 Data Preparation: Loading preprocessed files Ended ===============")
logging.info("======================================================================================")

# ===================== 3. LLM Benchmarking (ZS/FS only) =====================
# ---- Optional quantization presence probe ----
try:
    import bitsandbytes as _bnb  # noqa: F401
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False
    logging.warning("bitsandbytes not available. Proceeding without 8-bit quantization.")

from tqdm import tqdm

# ------------------------- System helpers -------------------------
def _approx_device_vram_gb() -> float:
    try:
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(0).total_memory
        return round(total / (1024 ** 3), 1)
    except Exception:
        return 0.0

def _bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

# ------------------------- Model registry -------------------------
MODELS = {
    "Llama-3.1-8B-Instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "family": "llama",
        "chat_template": "auto",
        "quantize_8bit": False,
        "force_no_cache": False,
        "min_vram_gb": 14
    },
    "Qwen2.5-7B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "chat_template": "auto",
        "quantize_8bit": False,
        "force_no_cache": False,
        "min_vram_gb": 12
    },
    "JAIS-13B-Chat": {
        "hf_id": "inceptionai/jais-13b-chat",
        "family": "jais",
        "chat_template": "none",
        "quantize_8bit": False,
        "force_no_cache": True,
        "min_vram_gb": 18
    },
    "gpt-oss-20b": {
        "hf_id": "openai/gpt-oss-20b",
        "family": "generic",
        "chat_template": "auto",
        "quantize_8bit": False,
        "force_no_cache": False,
        "min_vram_gb": 20
    },
}

MODEL_BS = {
    "Llama-3.1-8B-Instruct":   {"zs": 32, "fs": 16},
    "Qwen2.5-7B-Instruct":     {"zs": 32, "fs": 16},
    "JAIS-13B-Chat":           {"zs": 24, "fs": 12},
    "gpt-oss-20b":             {"zs": 24, "fs": 12},
}

# ------------------------- Access & capacity checks -------------------------
SKIP_ACCESS_CHECK = bool(int(os.environ.get("SKIP_HF_ACCESS_CHECK", "0")))

def _has_hf_access(repo_id: str) -> bool:
    if SKIP_ACCESS_CHECK:
        return True
    try:
        _HF_API.model_info(repo_id, token=HF_TOKEN)
        return True
    except Exception as e:
        logging.warning(f"No access to repo {repo_id}: {e}")
        return False

def _free_cache_ok_for(_hf_id: str) -> bool:
    return _free_gb(HF_HUB_CACHE) >= 20

def _can_host(model_cfg) -> bool:
    sys_vram = _approx_device_vram_gb()
    need = model_cfg.get("min_vram_gb", 0)
    if sys_vram < need:
        logging.warning(f"Skipping {model_cfg.get('hf_id')} — VRAM {sys_vram}GB < ~{need}GB required.")
        return False
    if not _has_hf_access(model_cfg.get("hf_id")):
        logging.warning(f"Skipping {model_cfg.get('hf_id')} — no access or gated.")
        return False
    if not _free_cache_ok_for(model_cfg.get("hf_id")):
        logging.warning(f"Skipping {model_cfg.get('hf_id')} — insufficient free cache ({_free_gb(HF_HUB_CACHE):.1f} GB).")
        return False
    return True

# ------------------------- Prompt helpers -------------------------
def _apply_chat(tokenizer, content: str, template_kind: str):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        else:
            try:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token = '[PAD]'
            except Exception:
                pass

    if template_kind == "none":
        return content

    try:
        if template_kind == "auto" and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            msgs = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logging.warning(f"Chat template unavailable/failed; using raw prompt. Reason: {e}")

    return content

def _prompt_ar_core(instruction_prefix: str, text: str, fewshot_block: str="") -> str:
    return (
        f"{instruction_prefix}\n"
        "اكتب الرقم فقط بدون أي شرح.\n"
        + (fewshot_block if fewshot_block else "")
        + "النص:\n"
        f"{text}\n"
        "التصنيف:"
    )

ZS_INSTRUCTION = "مهمتك تصنيف النص إذا كان بشرياً (0) أو من إنتاج نموذج لغوي ضخم LLM (1)."
FS_INSTRUCTION = ZS_INSTRUCTION

def _encode_batch(tokenizer, texts, add_special_tokens=True):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt",
        add_special_tokens=add_special_tokens
    )

# ------------------------- Config: load & sanitize -------------------------
def _load_raw_config_dict(hf_id: str) -> dict:
    try:
        cfg_dict, _ = PretrainedConfig.get_config_dict(hf_id, token=HF_TOKEN, trust_remote_code=True)
        return cfg_dict
    except Exception:
        path = hf_hub_download(repo_id=hf_id, filename="config.json", token=HF_TOKEN)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def _sanitize_rope_scaling_in_dict(cfg_dict: dict) -> dict:
    rs = cfg_dict.get("rope_scaling", None)
    if rs is None:
        return cfg_dict
    if not isinstance(rs, dict):
        cfg_dict["rope_scaling"] = None
        return cfg_dict
    typ = rs.get("type")
    fac = rs.get("factor")
    try:
        fac = float(fac) if fac is not None else None
    except Exception:
        fac = None
    if fac is None or fac == 1.0:
        cfg_dict["rope_scaling"] = None
        return cfg_dict
    if typ not in ("linear", "dynamic"):
        typ = "linear"
    cfg_dict["rope_scaling"] = {"type": typ, "factor": fac}
    return cfg_dict

def _apply_env_rope_override(cfg_dict: dict) -> dict:
    fac_env = os.environ.get("LLM_ROPE_FACTOR", "").strip()
    if not fac_env:
        return cfg_dict
    try:
        fac = float(fac_env)
        if fac == 1.0:
            cfg_dict["rope_scaling"] = None
        elif fac > 0:
            typ = os.environ.get("LLM_ROPE_TYPE", "linear")
            if typ not in ("linear", "dynamic"):
                typ = "linear"
            cfg_dict["rope_scaling"] = {"type": typ, "factor": fac}
    except Exception as e:
        logging.warning(f"Ignoring invalid LLM_ROPE_FACTOR='{fac_env}': {e}")
    return cfg_dict

def _make_autoconfig_safely(hf_id: str, family: str) -> AutoConfig:
    if family == "jais":
        return AutoConfig.from_pretrained(hf_id, trust_remote_code=True, token=HF_TOKEN)
    cfg_dict = _load_raw_config_dict(hf_id)
    if family == "llama":
        cfg_dict = _sanitize_rope_scaling_in_dict(cfg_dict)
        cfg_dict = _apply_env_rope_override(cfg_dict)
    tmp = tempfile.TemporaryDirectory(prefix="sanitized_cfg_")
    tmp_cfg = os.path.join(tmp.name, "config.json")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False)
    if not hasattr(_make_autoconfig_safely, "_tmp_dirs"):
        _make_autoconfig_safely._tmp_dirs = []
    _make_autoconfig_safely._tmp_dirs.append(tmp)
    return AutoConfig.from_pretrained(tmp.name, trust_remote_code=True)

# ------------------------- Repo-specific post init -------------------------
def _post_init_repo_tweaks(model, tokenizer, cfg_entry):
    family = cfg_entry.get("family", "generic")

    # Deterministic generation, silence sampling-driven warnings
    try:
        gcfg = getattr(model, "generation_config", None)
        if gcfg is not None:
            gcfg.do_sample = False
            gcfg.temperature = 1.0
            if hasattr(gcfg, "top_p"): gcfg.top_p = None
            if hasattr(gcfg, "top_k"): gcfg.top_k = None
    except Exception:
        pass

    if family == "jais":
        try:
            model.config.use_cache = False
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.use_cache = False
            logging.info("[LOAD][JAIS] Disabled KV cache (use_cache=False).")
        except Exception:
            pass

    if family == "llama":
        try:
            logging.info(f"[LOAD][Llama] rope_scaling effective: {getattr(model.config, 'rope_scaling', None)}")
        except Exception:
            pass

    if family == "qwen":
        if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("[LOAD][Qwen] Set pad_token = eos_token.")

# ------------------------- Tokenizer loader -------------------------
def _load_tokenizer(hf_id: str, family: str):
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=True, token=HF_TOKEN)
        return tok, True
    except Exception as e_fast:
        logging.warning(f"[LOAD][{hf_id}] Fast tokenizer failed: {e_fast}. Retrying with slow tokenizer.")
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False, token=HF_TOKEN)
        if tok.pad_token is None:
            if getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            else:
                try:
                    tok.add_special_tokens({'pad_token': '[PAD]'})
                    tok.pad_token = '[PAD]'
                except Exception:
                    pass
        return tok, False

# ------------------------- Model loader -------------------------
def _load_llm_for_textgen(cfg_entry):
    hf_id  = cfg_entry["hf_id"]
    family = cfg_entry.get("family", "generic")

    logging.info(f"[LOAD] text-generation tokenizer: {hf_id}")
    tok, used_fast = _load_tokenizer(hf_id, family)
    tok.padding_side = "left"
    if tok.pad_token is None:
        if getattr(tok, "eos_token", None):
            tok.pad_token = tok.eos_token
        else:
            try:
                tok.add_special_tokens({'pad_token':'[PAD]'})
                tok.pad_token = '[PAD]'
            except Exception:
                pass

    want_8bit = bool(cfg_entry.get("quantize_8bit", False))
    quant_cfg = None
    used_8bit = False
    if want_8bit and '_HAS_BNB' in globals() and _HAS_BNB:
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            used_8bit = True
            logging.info(f"[LOAD] 8-bit quantization enabled for {hf_id}")
        except Exception as e:
            logging.warning(f"[LOAD] bitsandbytes not usable; loading {hf_id} without 8-bit. Reason: {e}")
            quant_cfg = None
            used_8bit = False
    elif want_8bit:
        logging.warning(f"[LOAD] bitsandbytes not available; loading {hf_id} without 8-bit.")

    dtype = torch.bfloat16 if _bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    logging.info(f"[LOAD] dtype selected: {dtype}")

    base_cfg = _make_autoconfig_safely(hf_id, family)

    def _try_load(qcfg):
        return AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=qcfg,
            trust_remote_code=True,
            token=HF_TOKEN,
            config=base_cfg
        )

    try:
        mdl = _try_load(quant_cfg)
    except Exception as e:
        logging.warning(f"[LOAD] Initial load failed for {hf_id} (8-bit={used_8bit}). Retrying without quantization. Reason: {e}")
        mdl = _try_load(None)
        used_8bit = False

    _post_init_repo_tweaks(mdl, tok, cfg_entry)

    try:
        mdl.resize_token_embeddings(len(tok))
    except Exception:
        pass
    mdl.eval()

    if used_8bit:
        logging.info(f"[LOAD] {hf_id} loaded with 8-bit quantization.")
    else:
        logging.info(f"[LOAD] {hf_id} loaded without 8-bit quantization.")
    if not used_fast:
        logging.info(f"[LOAD] {hf_id} tokenizer loaded with use_fast=False.")

    return tok, mdl

# ------------------------- Device utils -------------------------
def move_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        try:
            moved[k] = v.to(device, non_blocking=True)
        except AttributeError:
            moved[k] = v
    return moved

# ------------------------- Logits processor (force {0,1}) -------------------------
class OnlyIdsProcessor(torch.nn.Module):
    def __init__(self, allowed_ids):
        super().__init__()
        self.register_buffer("allowed", torch.tensor(sorted(list(allowed_ids)), dtype=torch.long))

    def forward(self, input_ids: torch.Tensor, scores: torch.Tensor):
        mask = torch.full_like(scores, float('-inf'))
        idx = self.allowed.to(scores.device)
        mask.index_copy_(1, idx, scores.index_select(1, idx))
        return mask

# ------------------------- Generation -------------------------
@torch.inference_mode()
def _generate_block_forced01(model, tokenizer, prompts_batch, return_probs: bool = True, disable_cache: bool = False):
    device = model.device
    zero_id = tokenizer("0", add_special_tokens=False).input_ids[-1]
    one_id  = tokenizer("1", add_special_tokens=False).input_ids[-1]
    allowed = {zero_id, one_id}
    processor = LogitsProcessorList([OnlyIdsProcessor(allowed)])

    def _run(prompts):
        enc = _encode_batch(tokenizer, prompts)
        enc = move_to_device(enc, device)
        gen = model.generate(
            **enc,
            max_new_tokens=1,
            do_sample=False,
            temperature=1.0,
            logits_processor=processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=(False if disable_cache else True),
            output_scores=return_probs,
            return_dict_in_generate=return_probs,
        )
        if return_probs:
            last_scores = gen.scores[-1]
            logits01 = torch.stack([last_scores[:, zero_id], last_scores[:, one_id]], dim=1)
            probs01 = torch.softmax(logits01, dim=1)[:, 1]
            seq = gen.sequences
            tail = seq[:, enc["input_ids"].shape[1]:].squeeze(1)
            preds = [0 if int(t.item()) == zero_id else 1 for t in tail]
            return preds, probs01.detach().float().cpu().tolist()
        else:
            seq = gen
            tail = seq[:, enc["input_ids"].shape[1]:].squeeze(1)
            return [0 if int(t.item()) == zero_id else 1 for t in tail]

    try:
        return _run(prompts_batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    except Exception:
        pass

    if len(prompts_batch) > 1:
        mid = len(prompts_batch)//2
        left  = _generate_block_forced01(model, tokenizer, prompts_batch[:mid], return_probs=return_probs, disable_cache=disable_cache)
        right = _generate_block_forced01(model, tokenizer, prompts_batch[mid:], return_probs=return_probs, disable_cache=disable_cache)
        if return_probs:
            lp, lq = left
            rp, rq = right
            return lp + rp, lq + rq
        else:
            return left + right

    try:
        return _run(prompts_batch)
    except Exception as e:
        logging.exception(f"Per-sample generation failed: {e}")
        if return_probs:
            return [1], [0.5]
        return [1]

# ------------------------- Few-shot examples (from train) -------------------------
def _build_fewshot_block_from_train(df_train: pd.DataFrame, k: int, seed: int) -> str:
    assert 'text' in df_train.columns and 'label' in df_train.columns, "Train set must have 'text' and 'label'."
    if k <= 0:
        return ""
    if k % 2 != 0:
        logging.warning(f"FS_NUM_EXAMPLES={k} is odd; reducing to {k-1} for balance.")
        k -= 1
    half = k // 2
    df0 = df_train[df_train['label'] == 0]
    df1 = df_train[df_train['label'] == 1]
    n0 = min(half, len(df0)); n1 = min(half, len(df1))
    ex0 = df0.sample(n=n0, random_state=seed) if n0 > 0 else df0
    ex1 = df1.sample(n=n1, random_state=seed) if n1 > 0 else df1
    rows = [f"النص: {t}\nالتصنيف: 0\n" for t in ex0['text'].astype(str).tolist()]
    rows += [f"النص: {t}\nالتصنيف: 1\n" for t in ex1['text'].astype(str).tolist()]
    np.random.default_rng(seed).shuffle(rows)
    block = "أمثلة:\n" + "\n".join(rows) + "\n"
    logging.info(f"Few-shot block built from train: {len(rows)} examples (label0={n0}, label1={n1}).")
    return block

def _build_prompts(tokenizer, instruction, texts, fewshot_block="", chat_template="auto"):
    return [_apply_chat(tokenizer, _prompt_ar_core(instruction, t, fewshot_block), chat_template) for t in texts]

# ------------------------- Metrics & logging -------------------------
def _format_cm_labeled(cm, row_labels, col_labels, colw=12):
    header = " " * (colw) + "".join(f"{c:>{colw}}" for c in col_labels)
    lines = [header]
    for i, rlab in enumerate(row_labels):
        line = f"{rlab:<{colw}}" + "".join(f"{int(cm[i, j]):>{colw}d}" for j in range(cm.shape[1]))
        lines.append(line)
    return "Confusion Matrix:\n" + "\n".join(lines)

def _log_summary(title, y_true, y_pred, y_prob1=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    logging.info("="*68)
    logging.info(f"Results for {title}")
    logging.info("="*68)
    logging.info(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
    if y_prob1 is not None:
        try:
            roc = roc_auc_score(y_true, y_prob1)
            logging.info(f"ROC-AUC: {roc:.4f}")
        except Exception as e:
            logging.warning(f"ROC-AUC could not be computed: {e}")
    report = classification_report(y_true, y_pred, target_names=["Human","LLM"], digits=4)
    logging.info("Classification Report:\n" + report)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_txt = _format_cm_labeled(cm, ["True Human", "True LLM"], ["Pred Human", "Pred LLM"], colw=12)
    logging.info(cm_txt)

# ------------------------- Unified runner -------------------------
def _run_inference(model_name, cfg, texts, y_true, fewshot_block="", zs_mode=True):
    tokenizer, model = _load_llm_for_textgen(cfg)
    bs_map = MODEL_BS.get(model_name, {})
    batch_size = bs_map.get("zs" if zs_mode else "fs", ZS_BATCH_SIZE if zs_mode else FS_BATCH_SIZE)

    prompts = _build_prompts(
        tokenizer,
        ZS_INSTRUCTION if zs_mode else FS_INSTRUCTION,
        texts,
        fewshot_block=fewshot_block,
        chat_template=cfg.get("chat_template", "auto")
    )

    preds, probs = [], []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} {'ZS' if zs_mode else 'FS'} Batches", leave=False):
        batch = prompts[i:i+batch_size]
        bpreds, bprobs = _generate_block_forced01(
            model, tokenizer, batch, return_probs=True, disable_cache=cfg.get("force_no_cache", False)
        )
        preds.extend(bpreds); probs.extend(bprobs)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert len(preds) == len(texts) == len(probs), f"Length mismatch: preds={len(preds)}, probs={len(probs)}, texts={len(texts)}"
    _log_summary(("ZeroShot · " if zs_mode else "FewShot · ") + model_name, y_true, preds, y_prob1=probs)
    return preds, probs

# ------------------------- Standard pipeline -------------------------
logging.info("=========== ZERO-SHOT BENCHMARKING ===========")
logging.info(f"Approx VRAM available: {_approx_device_vram_gb()} GB")

texts = df_test["text"].astype(str).tolist()
y_true = df_test["label"].tolist()

for model_name, cfg in tqdm(list(MODELS.items()), desc="Models (Zero-shot)", leave=True):
    logging.info(f"Zero-shot start: {model_name} ({cfg['hf_id']})")
    if not _can_host(cfg):
        logging.warning(f"Skipping zero-shot for {model_name} (resource/access check failed).")
        continue
    try:
        preds, probs = _run_inference(model_name, cfg, texts, y_true, fewshot_block="", zs_mode=True)
        df_test[f'zeroshot_pred_{model_name}']  = preds
        df_test[f'zeroshot_prob1_{model_name}'] = probs
    except Exception as e:
        logging.exception(f"Zero-shot failure for {model_name}: {e}")

logging.info("=========== FEW-SHOT BENCHMARKING ===========")
fewshot_block = _build_fewshot_block_from_train(df_train, FS_NUM_EXAMPLES, SEED)

for model_name, cfg in tqdm(list(MODELS.items()), desc="Models (Few-shot)", leave=True):
    logging.info(f"Few-shot start: {model_name} ({cfg['hf_id']})")
    if not _can_host(cfg):
        logging.warning(f"Skipping few-shot for {model_name} (resource/access check failed).")
        continue
    try:
        preds, probs = _run_inference(model_name, cfg, texts, y_true, fewshot_block=fewshot_block, zs_mode=False)
        df_test[f'fewshot_pred_{model_name}']  = preds
        df_test[f'fewshot_prob1_{model_name}'] = probs
    except Exception as e:
        logging.exception(f"Few-shot failure for {model_name}: {e}")

logging.info("=========== Benchmarking completed (ZS/FS) ===========")
