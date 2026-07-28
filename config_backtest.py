"""
Config centralizada para la versión liviana del backtest (sin DB de
precios locales). Lee de variables de entorno -- usa python-dotenv si
está instalado (carga .env automáticamente); si no, sigue funcionando
con las variables de entorno del sistema tal cual.

Copia .env.example a .env y ajusta los valores -- .env queda AFUERA
del repo (ver .gitignore), .env.example sí se sube (sin datos reales,
solo la plantilla).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # sin python-dotenv instalado, sigue con el entorno del sistema

UNIVERSO_DB_PATH = os.environ.get("UNIVERSO_DB_PATH", str(Path(__file__).parent / "universo_sp500.db"))
TICKER_RF = os.environ.get("TICKER_RF", "BIL")
