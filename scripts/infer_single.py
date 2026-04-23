#!/usr/bin/env python
"""Быстрая проверка одного изображения из командной строки.

Использование:
  python scripts/infer_single.py path/to/image.jpg --region UZB
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from anpr_system.core import ANPRPipeline
from anpr_system.core.detection import PlateDetector
from anpr_system.core.recognition import PlateRecognizer
from anpr_system.configs import settings

parser = argparse.ArgumentParser(description="ANPR single image inference")
parser.add_argument("image", help="Путь к изображению")
parser.add_argument("--region", default="UZB")
args = parser.parse_args()

detector = PlateDetector(settings.yolo_weights, device=settings.device)
detector.load()
recognizer = PlateRecognizer(settings.lprnet_weights, device=settings.device)
pipeline = ANPRPipeline(detector, recognizer)

image = cv2.imread(args.image)
if image is None:
    print(f"Ошибка: не удалось открыть {args.image}")
    sys.exit(1)

result = pipeline.run(image, region=args.region)
if result:
    print(f"Номер:    {result.plate}")
    print(f"Регион:   {result.region}")
    print(f"Валиден:  {result.valid}")
    print(f"Conf:     {result.confidence:.4f}")
    print(f"Время:    {result.processing_time_ms:.1f} ms")
    print(f"BBox:     {result.bbox}")
else:
    print("Номерной знак не обнаружен.")
