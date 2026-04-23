"""
crop_plates.py — вырезает номера из сырых фото через YOLOv10

Использование:
    python crop_plates.py --input папка_с_фото --out dataset/final/train
                          --weights C:/Users/user/Pictures/ANPR/models/license_plate_detector.pt

Имена файлов:
    Uzbekistan_95-G-419-RA_2.jpg  →  crop сохраняется как  95G419RA_<hash>.jpg
    Если имя не парсится           →  crop сохраняется как  UNKNOWN_<hash>.jpg (для ручной разметки)

Флаги:
    --conf      минимальный confidence детектора (default: 0.4)
    --pad       отступ вокруг bbox в пикселях   (default: 4)
    --skip-unk  не сохранять UNKNOWN cropы
    --show      показывать окно с результатом (для отладки)
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import cv2
import numpy as np

EXPECTED_LEN = 8
VALID_CHARS = set("0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ")


# ─── парсер имени файла ───────────────────────────────────────

def parse_label(stem: str) -> str | None:
    """
    Uzbekistan_95-G-419-RA_2  →  95G419RA
    01A123BC_abc123            →  01A123BC
    """
    # стандартный формат
    plain = stem.split("_")[0].upper()
    if len(plain) == EXPECTED_LEN and all(c in VALID_CHARS for c in plain):
        return plain

    # формат Uzbekistan_DD-L-DDD-LL_N
    for part in stem.split("_"):
        if "-" in part:
            label = part.replace("-", "").upper()
            if len(label) == EXPECTED_LEN and all(c in VALID_CHARS for c in label):
                return label

    return None


def file_hash(path: Path) -> str:
    return hashlib.md5(path.name.encode()).hexdigest()[:6]


# ─── детектор ────────────────────────────────────────────────

def load_model(weights: str, device: str):
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
        model.to(device)
        print(f"[OK] Модель загружена: {weights}")
        print(f"[OK] Устройство: {device.upper()}")
        return model
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модель: {e}")
        sys.exit(1)


def detect_plates(model, image: np.ndarray, conf: float) -> list[dict]:
    """Возвращает список bbox с confidence, отсортированных по убыванию conf."""
    results = model(image, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            c = float(box.conf[0])
            detections.append({"bbox": [x1, y1, x2, y2], "conf": c})
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def crop_with_pad(image: np.ndarray, bbox: list, pad: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]


# ─── main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crop license plates via YOLOv10")
    parser.add_argument("--input",    required=True,  help="Папка с сырыми фото")
    parser.add_argument("--out",      required=True,  help="Куда сохранять cropы")
    parser.add_argument("--weights",  default=r"C:\Users\user\Pictures\ANPR\models\license_plate_detector.pt")
    parser.add_argument("--conf",     type=float, default=0.4,   help="Min confidence (default: 0.4)")
    parser.add_argument("--pad",      type=int,   default=4,     help="Отступ bbox px (default: 4)")
    parser.add_argument("--device",   default="cuda",            help="cuda / cpu")
    parser.add_argument("--skip-unk", action="store_true",       help="Не сохранять UNKNOWN")
    parser.add_argument("--show",     action="store_true",       help="Показывать окно (debug)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # папка для нераспознанных имён — ручная разметка
    unk_dir = out_dir.parent / "unknown_labels"
    if not args.skip_unk:
        unk_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_dir.glob("*.jpg")) +
                    list(input_dir.glob("*.jpeg")) +
                    list(input_dir.glob("*.png")))

    if not images:
        print(f"[ERROR] Нет изображений в {input_dir}")
        sys.exit(1)

    print(f"\n{'─'*55}")
    print(f"  Входная папка : {input_dir}")
    print(f"  Выходная папка: {out_dir}")
    print(f"  Изображений   : {len(images)}")
    print(f"  Confidence    : {args.conf}")
    print(f"  Padding       : {args.pad}px")
    print(f"{'─'*55}\n")

    model = load_model(args.weights, args.device)

    stats = {
        "total": len(images),
        "detected": 0,
        "no_detection": 0,
        "label_ok": 0,
        "label_unknown": 0,
        "saved": 0,
    }
    no_detect_list = []

    for i, fpath in enumerate(images, 1):
        label = parse_label(fpath.stem)
        h = file_hash(fpath)

        # читаем изображение
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"[SKIP] Не удалось прочитать: {fpath.name}")
            continue

        # детектируем
        detections = detect_plates(model, img, args.conf)

        if not detections:
            stats["no_detection"] += 1
            no_detect_list.append(fpath.name)
            print(f"[{i:4d}/{stats['total']}] MISS  conf<{args.conf}  {fpath.name}")
            continue

        stats["detected"] += 1
        best = detections[0]
        crop = crop_with_pad(img, best["bbox"], args.pad)

        if crop.size == 0:
            print(f"[{i:4d}/{stats['total']}] EMPTY crop  {fpath.name}")
            continue

        # определяем куда сохранять
        if label:
            stats["label_ok"] += 1
            save_name = f"{label}_{h}.jpg"
            save_path = out_dir / save_name
        else:
            stats["label_unknown"] += 1
            if args.skip_unk:
                print(f"[{i:4d}/{stats['total']}] SKIP  (unknown label)  {fpath.name}")
                continue
            save_name = f"UNKNOWN_{h}.jpg"
            save_path = unk_dir / save_name

        cv2.imwrite(str(save_path), crop)
        stats["saved"] += 1

        marker = "OK  " if label else "UNK "
        print(f"[{i:4d}/{stats['total']}] {marker} conf={best['conf']:.2f}  {label or '?':8s}  →  {save_name}")

        # отладка
        if args.show:
            x1, y1, x2, y2 = best["bbox"]
            debug = img.copy()
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, f"{label or 'UNK'} {best['conf']:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("crop_plates debug", debug)
            if cv2.waitKey(300) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    # ─── итоговый отчёт ───
    print(f"\n{'═'*55}")
    print(f"  ИТОГ")
    print(f"{'═'*55}")
    print(f"  Всего фото        : {stats['total']}")
    print(f"  Номер найден      : {stats['detected']}")
    print(f"  Не найдено        : {stats['no_detection']}")
    print(f"  Метка из имени    : {stats['label_ok']}")
    print(f"  Метка неизвестна  : {stats['label_unknown']}")
    print(f"  Сохранено cropов  : {stats['saved']}")

    detect_rate = stats["detected"] / stats["total"] * 100 if stats["total"] else 0
    print(f"\n  Detection rate    : {detect_rate:.1f}%")

    if stats["no_detection"] > 0:
        print(f"\n  Не задетектировано ({stats['no_detection']} шт) — попробуй:")
        print(f"    --conf 0.25   (снизить порог)")
        print(f"    --pad 8       (увеличить отступ)")
        print(f"  Примеры проблемных файлов:")
        for name in no_detect_list[:10]:
            print(f"    {name}")

    if stats["label_unknown"] > 0 and not args.skip_unk:
        print(f"\n  UNKNOWN cropы сохранены в: {unk_dir}")
        print(f"  Переименуй их вручную в формат LABEL_hash.jpg")
        print(f"  и перемести в {out_dir}")

    print(f"\n  Cropы готовы: {out_dir.resolve()}")
    print(f"  Следующий шаг: запусти dataset_audit.py для пересборки датасета\n")


if __name__ == "__main__":
    main()
