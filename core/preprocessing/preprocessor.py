"""
Preprocessing Module
Crop → Deskew → Normalize
"""

import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class PreprocessResult:
    image: np.ndarray    # готовое изображение для OCR
    was_deskewed: bool


class Preprocessor:
    """Подготавливает кроп номера к распознаванию."""

    TARGET_SIZE = (94, 24)  # ширина x высота для LPRNet

    def process(self, crop: np.ndarray) -> PreprocessResult:
        """
        Args:
            crop: BGR numpy array — вырезанный номер
        Returns:
            PreprocessResult с нормализованным изображением
        """
        if crop is None or crop.size == 0:
            raise ValueError("Empty crop received")

        img = crop.copy()
        was_deskewed = False

        # 1. Deskew
        deskewed = self._deskew(img)
        if deskewed is not None:
            img = deskewed
            was_deskewed = True

        # 2. Resize
        img = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # 3. Grayscale + CLAHE (устойчивость к засвету и грязи)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # 4. Обратно в 3-канальный для модели
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return PreprocessResult(image=result, was_deskewed=was_deskewed)

    def _deskew(self, image: np.ndarray) -> np.ndarray | None:
        """Перспективное выравнивание через контур."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            angle = rect[2]

            # Поворачиваем только если угол значимый
            if abs(angle) < 1.0:
                return None

            if angle < -45:
                angle += 90

            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception as e:
            logger.debug(f"Deskew failed: {e}")
            return None
