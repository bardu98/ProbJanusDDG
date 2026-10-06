"""Percorsi e costanti. Unico punto in cui si tocca il filesystem."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PJANUS_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("PJANUS_RESULTS", os.path.join(ROOT, "results"))
CKPT = os.environ.get("PJANUS_CKPT", os.path.join(ROOT, "checkpoints"))

# --- embedding delle sequenze di S2450 / S669L / S461L (2,8 GB) ------------------------------
# Il default locale rende la pipeline spostabile insieme a questa cartella.
EMB = os.environ.get("PJANUS_EMB", os.path.join(ROOT, "embeddings"))

ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
DIM = 1280          # dimensione dell'embedding ESM-2 650M per residuo
EPS = 1e-3          # pavimento su sigma
NFOLD = 5
