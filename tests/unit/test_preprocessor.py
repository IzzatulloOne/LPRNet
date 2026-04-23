"""Unit-тест предобработчика."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from anpr_system.core.preprocessing import Preprocessor


def test_output_shape():
    pp = Preprocessor(target_size=(94, 24))
    dummy = np.random.randint(0, 255, (60, 200, 3), dtype=np.uint8)
    out = pp.process(dummy)
    assert out.shape == (24, 94, 3), f"Ожидали (24, 94, 3), получили {out.shape}"
