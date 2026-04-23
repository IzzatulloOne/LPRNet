import os
import sys
import cv2
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Путь к репозиторию
sys.path.insert(0, r'C:\Users\user\Pictures\anpr_system\LPRNet_Pytorch')
from model.LPRNet import build_lprnet

# ===== КОНФИГ =====
CHARS = ['0','1','2','3','4','5','6','7','8','9',
         'A','B','C','D','E','F','G','H','J','K',
         'L','M','N','O','P','Q','R','S','T','U',
         'V','W','X','Y','Z',
         '-']  # blank для CTC — всегда последний!

CHARS_DICT   = {c: i for i, c in enumerate(CHARS)}
IMG_SIZE     = (94, 24)
BATCH_SIZE   = 64
EPOCHS       = 60
LR           = 1e-4
PRETRAINED   = r'C:\Users\user\Pictures\anpr_system\LPRNet_Pytorch\weights\Final_LPRNet_model.pth'
TRAIN_DIR    = r'C:\Users\user\Pictures\anpr_system\dataset\final\train'
VAL_DIR      = r'C:\Users\user\Pictures\anpr_system\dataset\final\val'
SAVE_PATH    = r'C:\Users\user\Pictures\anpr_system\LPRNet_Pytorch\model\lprnet_uz_best.pth'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device   : {DEVICE}")
print(f"Num chars: {len(CHARS)} (включая blank)")

# ===== DATASET =====
class UZPlateDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for fpath in Path(data_dir).glob("*.jpg"):
            label = fpath.stem.split('_')[0]
            if all(c in CHARS_DICT for c in label):
                self.samples.append((str(fpath), label))
            else:
                bad = [c for c in label if c not in CHARS_DICT]
                print(f"⚠️  Пропущен {fpath.name} — неизвестные символы: {bad}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        with open(fpath, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32)
        img -= 127.5
        img *= 0.0078125
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)

train_ds = UZPlateDataset(TRAIN_DIR)
val_ds   = UZPlateDataset(VAL_DIR)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ===== МОДЕЛЬ =====
model = build_lprnet(
    lpr_max_len  = 8,
    phase        = False,   # False = inference/finetune режим
    class_num    = len(CHARS),
    dropout_rate = 0.5
)
model = model.to(DEVICE)

state = torch.load(PRETRAINED, map_location=DEVICE)

# Смотрим размер последнего слоя в весах
for k, v in state.items():
    if 'container' in k or 'classifier' in k or 'fc' in k:
        print(f"  {k}: {v.shape}")

# Фильтруем — пропускаем слои где размер не совпадает
model_state = model.state_dict()
filtered = {}
skipped  = []

for k, v in state.items():
    if k in model_state and v.shape == model_state[k].shape:
        filtered[k] = v
    else:
        skipped.append(f"{k}: {v.shape} → ожидается {model_state.get(k, '???')}")

model_state.update(filtered)
model.load_state_dict(model_state)

print(f"✅ Загружено слоёв : {len(filtered)}")
print(f"⚠️  Пропущено слоёв: {len(skipped)}")
for s in skipped:
    print(f"   {s}")

# ===== ОПТИМИЗАТОР =====
ctc_loss  = torch.nn.CTCLoss(blank=len(CHARS)-1, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def encode(labels):
    targets, lengths = [], []
    for label in labels:
        enc = [CHARS_DICT[c] for c in label]
        targets.extend(enc)
        lengths.append(len(enc))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

def decode(logits):
    preds = logits.permute(1, 0, 2).argmax(-1).cpu().numpy()
    results = []
    for pred in preds:
        chars, prev = [], -1
        for p in pred:
            if p != prev and p != len(CHARS) - 1:
                chars.append(CHARS[p])
            prev = p
        results.append(''.join(chars))
    return results

def calc_accuracy(logits, labels):
    decoded = decode(logits)
    correct = sum(p == l for p, l in zip(decoded, labels))
    return correct, len(labels)

# ===== TRAIN LOOP =====
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    total_loss = 0.0

    for imgs, labels in train_dl:
        imgs     = imgs.to(DEVICE)
        logits   = model(imgs)
        log_prob = torch.nn.functional.log_softmax(logits, dim=2)
        targets, t_lens = encode(labels)
        i_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long)

        loss = ctc_loss(log_prob, targets, i_lens, t_lens)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    # --- Val ---
    model.eval()
    correct_total, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            c, t = calc_accuracy(logits, labels)
            correct_total += c
            total         += t

    val_acc  = correct_total / total
    avg_loss = total_loss / len(train_dl)
    scheduler.step()

    print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.3f} ({correct_total}/{total})")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  💾 Лучшая модель сохранена (acc={best_acc:.3f})")

print(f"\n🏁 Готово! Лучшая точность: {best_acc:.3f}")
print(f"   Модель: {SAVE_PATH}")