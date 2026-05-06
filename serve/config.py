import os
from pathlib import Path

import torch
from dotenv import load_dotenv

_SERVE_DIR = Path(__file__).resolve().parent
load_dotenv(_SERVE_DIR / ".env")

CFG_MODEL_PATH = os.environ.get("CFG_MODEL", "k2-fsa/OmniVoice")
CFG_DEVICE = os.environ.get(
    "CFG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
_dtype_name = os.environ.get("CFG_DTYPE", "float16")
CFG_DTYPE = (
    getattr(torch, _dtype_name)
    if _dtype_name in ("float16", "bfloat16", "float32")
    else torch.float16
)
CFG_MAX_BATCH = int(os.environ.get("CFG_MAX_BATCH", "8"))
CFG_MAX_BATCH_WAIT_MS = float(os.environ.get("CFG_MAX_BATCH_WAIT_MS","200"))
CFG_MAX_PENDING_QUEUE = int(os.environ.get("CFG_MAX_PENDING_QUEUE","64"))
CFG_COMPILE_MODEL = os.environ.get("CFG_COMPILE", "0") == "1"
CFG_LOAD_ASR_MODEL = os.environ.get("CFG_LOAD_ASR_MODEL", "openai/whisper-large-v3-turbo")




