"""
Быстрая проверка пайплайна на одном изображении.
Использование: python scripts/test_image.py --image path/to/plate.jpg
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from core.detection import PlateDetector
from core.preprocessing import Preprocessor
from core.recognition import PlateRecognizer
from core.postprocessing import Postprocessor
from core.regions.registry import RegionRegistry
from core.pipeline import ANPRPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--region", default="UZB")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    pipeline = ANPRPipeline(
        detector=PlateDetector(
            config["model"]["detector"]["weights"],
            config["model"]["detector"]["confidence_threshold"],
            config["model"]["detector"]["device"],
        ),
        preprocessor=Preprocessor(),
        recognizer=PlateRecognizer(
            config["model"]["recognizer"]["weights"],
            config["model"]["recognizer"]["device"],
        ),
        postprocessor=Postprocessor(RegionRegistry.default()),
    )

    result = pipeline.process_file(args.image, args.region)
    print(f"\n{'='*40}")
    print(f"  Plate   : {result.plate}")
    print(f"  Conf    : {result.confidence:.3f}")
    print(f"  Valid   : {result.valid}")
    print(f"  BBox    : {result.bbox}")
    print(f"  Time    : {result.processing_time_ms:.1f} ms")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
