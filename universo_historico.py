"""
Reconstruye qué tickers estaban en el S&P 500 en una fecha pasada,
deshaciendo desde la lista ACTUAL de Wikipedia hacia atrás usando la
tabla "Selected changes to the list of S&P 500 companies" (misma
página que ya usa poblar_sectores.py).

NO resuelve sesgo de supervivencia por sí solo -- solo dice QUÉ
tickers deberían estar en el universo candidato en una fecha dada.
Los precios de esos tickers hay que conseguirlos aparte (sp500.db del
usuario + relleno puntual vía yfinance para los que ya no estén ahí).

Limitaciones conocidas (documentadas a propósito, no ocultas):
- El registro de cambios de Wikipedia es confiable desde ~2000 en
  adelante, más disperso más atrás.
- Cambios de ticker (ej. FISV -> FI) pueden no calzar 1:1 con el
  histórico de precios.
- Eventos corporativos ambiguos (spin-offs, fusiones) no siempre se
  registran como un agregar/quitar limpio.
- No es la fuente "gold standard" (CRSP/Compustat point-in-time) --
  es una aproximación razonable para un backtest propio, no para un
  paper publicado.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from io import StringIO

import pandas as pd
import requests

URL_WIKIPEDIA_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def obtener_lista_actual(html: str | None = None) -> list[str]:
    """Tabla 0 -- constituyentes de hoy. Mismo patrón que poblar_sectores.py."""
    if html is None:
        html = requests.get(URL_WIKIPEDIA_SP500, headers=HEADERS, timeout=30).text
    tabla = pd.read_html(StringIO(html))[0]
    return tabla["Symbol"].tolist()


def obtener_tabla_cambios(html: str | None = None) -> pd.DataFrame:
    """Tabla 1 -- 'Selected changes to the list of S&P 500 companies'.
    Devuelve columnas normalizadas: fecha (date), ticker_agregado,
    ticker_removido (ambos pueden ser None -- algunos eventos son solo
    reclasificación de índice, agregan a un sub-índice sin agregar al
    S&P 500 mismo, o viceversa).

    La tabla de Wikipedia trae encabezado de 2 filas (Date / Added
    (Ticker,Security) / Removed (Ticker,Security) / Reason) -- pandas
    lo aplana a columnas MultiIndex. Se identifican por posición
    (0=fecha, 1=ticker agregado, 3=ticker removido), no por nombre
    exacto del texto -- más robusto a que Wikipedia cambie mayúsculas
    o wording del encabezado, tal como ya pasó una vez con esta misma
    página (ver Talk:List_of_S&P_500_companies)."""
    if html is None:
        html = requests.get(URL_WIKIPEDIA_SP500, headers=HEADERS, timeout=30).text
    tabla = pd.read_html(StringIO(html))[1]

    if isinstance(tabla.columns, pd.MultiIndex):
        tabla.columns = [" ".join(str(c) for c in col if "Unnamed" not in str(c)).strip() for col in tabla.columns]

    cols = tabla.columns.tolist()
    if len(cols) < 4:
        raise ValueError(f"Tabla de cambios con estructura inesperada -- columnas: {cols}")

    col_fecha, col_add, col_rem = cols[0], cols[1], cols[3]

    def _limpiar(serie: pd.Series) -> pd.Series:
        # pandas 3.x activa "future.infer_string" por defecto -- una
        # columna de texto con huecos usa su StringDtype nueva, que NO
        # preserva un None de Python tal cual (lo re-convierte a su
        # propio NA, que termina comportándose como float NaN al
        # comparar/ordenar -- rompe sorted()/discard() más abajo).
        # dtype=object explícito evita esa inferencia por completo.
        valores = [None if pd.isna(v) else str(v).strip() for v in serie]
        return pd.Series(valores, index=serie.index, dtype=object)

    df = pd.DataFrame({
        "fecha": pd.to_datetime(tabla[col_fecha], errors="coerce"),
        "ticker_agregado": _limpiar(tabla[col_add]),
        "ticker_removido": _limpiar(tabla[col_rem]),
    })
    df = df.dropna(subset=["fecha"])
    return df.sort_values("fecha", ascending=False).reset_index(drop=True)


def obtener_constituyentes_en_fecha(
    fecha: date, lista_actual: list[str] | None = None, tabla_cambios: pd.DataFrame | None = None,
) -> set[str]:
    """Reconstruye el universo del S&P 500 en `fecha`, deshaciendo desde
    la lista actual hacia atrás: cada cambio POSTERIOR a `fecha` se
    revierte (agregado -> se saca del set; removido -> se agrega de
    vuelta). No hace red si ya le pasas `lista_actual`/`tabla_cambios`
    (para no re-descargar Wikipedia en cada ventana del backtest)."""
    if lista_actual is None:
        lista_actual = obtener_lista_actual()
    if tabla_cambios is None:
        tabla_cambios = obtener_tabla_cambios()

    fecha_ts = pd.Timestamp(fecha)
    constituyentes = set(lista_actual)

    posteriores = tabla_cambios[tabla_cambios["fecha"] > fecha_ts]
    for _, fila in posteriores.iterrows():
        if fila["ticker_agregado"]:
            constituyentes.discard(fila["ticker_agregado"])
        if fila["ticker_removido"]:
            constituyentes.add(fila["ticker_removido"])

    return constituyentes


# ============================================================
# Guardar/cargar en una DB liviana -- SOLO tickers y fechas de
# cambios, nada de precios. Pesa casi nada (cientos de filas), así que
# a diferencia de sp500.db/resultados_backtest.db, esta SÍ se puede
# subir a GitHub -- scrapea Wikipedia UNA vez y queda reproducible sin
# depender de la red cada vez que se corre el backtest.
# ============================================================

def guardar_universo_en_db(db_path: str, html: str | None = None):
    """Scrapea Wikipedia una vez (o usa el html ya descargado, para
    tests) y guarda lista_actual + tabla_cambios en db_path."""
    lista_actual = obtener_lista_actual(html=html)
    tabla_cambios = obtener_tabla_cambios(html=html)

    con = sqlite3.connect(db_path)
    pd.DataFrame({"ticker": lista_actual}).to_sql("lista_actual", con, if_exists="replace", index=False)
    tabla_cambios.to_sql("tabla_cambios", con, if_exists="replace", index=False)
    con.close()
    print(f"Guardado en {db_path} -- lista_actual: {len(lista_actual)} tickers, tabla_cambios: {len(tabla_cambios)} filas.")


def cargar_universo_desde_db(db_path: str) -> tuple[list[str], pd.DataFrame]:
    """Lee lista_actual/tabla_cambios ya guardados -- no toca Wikipedia."""
    con = sqlite3.connect(db_path)
    lista_actual = pd.read_sql("SELECT ticker FROM lista_actual", con)["ticker"].tolist()
    tabla_cambios = pd.read_sql("SELECT * FROM tabla_cambios", con)
    con.close()
    tabla_cambios["fecha"] = pd.to_datetime(tabla_cambios["fecha"])
    return lista_actual, tabla_cambios
