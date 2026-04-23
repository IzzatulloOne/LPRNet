#!/usr/bin/env python
"""Запуск FastAPI сервера."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import uvicorn
from anpr_system.configs import settings
from anpr_system.configs.logging import setup_logging

if __name__ == "__main__":
    setup_logging(settings.log_level)
    uvicorn.run(
        "anpr_system.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False,
    )
