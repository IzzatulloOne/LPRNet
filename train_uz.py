import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# ПУТИ
# =========================
REPO_DIR   = r"C:\Users\user\Pictures\anpr_system\LPRNet_Pytorch"
TRAIN_DIR  = r"C:\Users\user\Pictures\anpr_system\dataset\final\train"
VAL_DIR    = r"C:\Users\user\Pictures\anpr_system\dataset\final\val"
PRETRAINED = r"C:\Users\user\Pictures\anpr_system\LPRNet_Pytorch\weights\Final_LPRNet_model.pth"
SAVE_PATH  = r"C:\Users\user\Pictures\anpr_system\LPRNet_Pytorch\model\lprnet_uz_best.pth"

sys.path.insert(0, REPO_DIR)
from model.LPRNet import build_lprnet

# =========================
# КОНФИГ
# =========================
# Под твой случай: фиксированная длина номера.
# Поставь сюда реальную длину твоих label.
EXPECTED_LEN = 8

# Символы, которые реально встречаются в узбекских номерах.
# Без I и O, как обычно и бывает в таких схемах.
CHARS = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K',
    'L','M','N','P','Q','R','S','T','U',
    'V','W','X','Y','Z'
]

BLANK_IDX   = 0
CHARS_DICT  = {c: i + 1 for i, c in enumerate(CHARS)}   # символы с 1
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}   # обратно
NUM_CLASSES  = len(CHARS) + 1                            # + blank

IMG_SIZE   = (94, 24)
BATCH_SIZE = 64
EPOCHS     = 60
LR         = 5e-4
WEIGHT_DECAY = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

print(f"Device      : {DEVICE}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"Classes     : {NUM_CLASSES} (blank=0)")
print(f"Plate length: {EXPECTED_LEN}")

# =========================
# ПРОВЕРКА РАЗМЕТКИ
# =========================
def scan_folder(folder: str):
    folder = Path(folder)
    bad = []
    wrong_len = []
    ok = 0

    for fpath in folder.glob("*.jpg"):
        label = fpath.stem.split("_")[0]

        if len(label) != EXPECTED_LEN:
            wrong_len.append((fpath.name, label, len(label)))
            continue

        bad_chars = [c for c in label if c not in CHARS_DICT]
        if bad_chars:
            bad.append((fpath.name, label, bad_chars))
            continue

        ok += 1

    return ok, wrong_len, bad

for split, folder in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
    ok, wrong_len, bad = scan_folder(folder)
    print(f"\n=== {split.upper()} ===")
    print(f"OK labels: {ok}")

    if wrong_len:
        print(f"Wrong length: {len(wrong_len)}")
        for fname, label, ln in wrong_len[:10]:
            print(f"  {fname} -> '{label}' (len={ln})")

    if bad:
        print(f"Bad chars: {len(bad)}")
        for fname, label, chars in bad[:10]:
            print(f"  {fname} -> '{label}' bad={chars}")

# =========================
# DATASET
# =========================
class UZPlateDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        data_dir = Path(data_dir)

        for fpath in sorted(data_dir.glob("*.jpg")):
            label = fpath.stem.split("_")[0]

            if len(label) != EXPECTED_LEN:
                continue

            if not all(c in CHARS_DICT for c in label):
                continue

            self.samples.append((str(fpath), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in: {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]

        with open(fpath, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)

        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image: {fpath}")

        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32)
        img -= 127.5
        img *= 0.0078125  # /128

        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)

train_ds = UZPlateDataset(TRAIN_DIR)
val_ds   = UZPlateDataset(VAL_DIR)

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=PIN_MEMORY,
    persistent_workers=(True if 4 > 0 else False),
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=PIN_MEMORY,
    persistent_workers=(True if 2 > 0 else False),
)

print(f"\nTrain samples: {len(train_ds)}")
print(f"Val samples  : {len(val_ds)}")

# =========================
# МОДЕЛЬ
# =========================
model = build_lprnet(
    lpr_max_len=EXPECTED_LEN,
    phase=False,
    class_num=NUM_CLASSES,
    dropout_rate=0.5
).to(DEVICE)

print("Model device:", next(model.parameters()).device)

# =========================
# ЗАГРУЗКА PRETRAINED
# =========================
state = torch.load(PRETRAINED, map_location=DEVICE)
model_state = model.state_dict()

filtered = {}
skipped = []

for k, v in state.items():
    if k in model_state and v.shape == model_state[k].shape:
        filtered[k] = v
    else:
        skipped.append(k)

model_state.update(filtered)
model.load_state_dict(model_state)

print(f"Loaded layers : {len(filtered)}")
print(f"Skipped layers : {len(skipped)}")

# =========================
# LOSS / OPTIMIZER
# =========================
ctc_loss = torch.nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# =========================
# ENCODE / DECODE
# =========================
def encode(labels):
    targets = []
    lengths = []

    for label in labels:
        enc = [CHARS_DICT[c] for c in label]
        targets.extend(enc)
        lengths.append(len(enc))

    targets = torch.tensor(targets, dtype=torch.long, device=DEVICE)
    lengths = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
    return targets, lengths

def decode(logits):
    """
    logits: [B, C, T]
    """
    preds = logits.argmax(dim=1).cpu().numpy()  # [B, T]
    results = []

    for pred in preds:
        out = []
        prev = -1
        for p in pred:
            if p != prev and p != BLANK_IDX:
                out.append(IDX_TO_CHAR.get(int(p), ""))
            prev = p
        results.append("".join(out))

    return results

def exact_match_acc(logits, labels):
    pred_text = decode(logits)
    correct = sum(p == g for p, g in zip(pred_text, labels))
    return correct, len(labels), pred_text

# =========================
# TRAIN
# =========================
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for imgs, labels in train_dl:
        imgs = imgs.to(DEVICE, non_blocking=True)

        logits = model(imgs)              # [B, C, T]
        logits = logits.permute(2, 0, 1)  # [T, B, C]
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        targets, t_lens = encode(labels)
        i_lens = torch.full(
            size=(imgs.size(0),),
            fill_value=logits.size(0),
            dtype=torch.long,
            device=DEVICE
        )

        loss = ctc_loss(log_probs, targets, i_lens, t_lens)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    # ===== VAL =====
    model.eval()
    correct_total = 0
    total_total = 0

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)  # [B, C, T]

            c, t, _ = exact_match_acc(logits, labels)
            correct_total += c
            total_total += t

    avg_loss = total_loss / max(1, len(train_dl))
    val_acc = correct_total / max(1, total_total)

    scheduler.step()

    print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | acc={val_acc:.3f} ({correct_total}/{total_total})")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  Saved best model: acc={best_acc:.3f}")

print(f"\nDone. Best acc: {best_acc:.3f}")
print(f"Model saved to: {SAVE_PATH}")