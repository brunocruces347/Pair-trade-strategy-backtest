"""
Descarga precios directo de yfinance para TODO el universo de una vez
(batch), con el máximo real de historia por duración de vela:
  - Diario (1d): 5 años (yfinance no tiene tope real más corto que eso
    para acciones ya listadas -- limitado solo por cuándo salió cada
    empresa a bolsa).
  - 1h: ~730 días (2 años) -- el tope real y confirmado de yfinance
    para intradía, sin importar qué rango le pidas (ver Opción A,
    discutida al diseñar el backtest).

Devuelve el MISMO formato largo que cargar_daily/cargar_intraday en
backtest_walk_forward_v2.py (columnas Fecha/Ticker/Close/Volume, y
además Dia/Hora para intradía) -- correr_backtest_secuencial no
necesita ningún cambio para usar estos datos en vez de los de la DB.

LIMITACIÓN CONOCIDA, a propósito (ver conversación): esto usa la lista
de tickers de HOY (universo_historico.obtener_lista_actual) para TODO
el histórico -- reintroduce sesgo de supervivencia, porque no se
puede bajar en bloque un ticker que ya salió del índice. Si se quiere
cerrar esto del todo, hay que agregar un paso de relleno: comparar
universo_historico.obtener_constituyentes_en_fecha(fecha) contra los
tickers efectivamente descargados, y bajar aparte (ticker por ticker)
los que falten.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def descargar_precios_diarios_yf(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Batch -- una sola llamada a yfinance para todos los tickers,
    no una por ticker. Devuelve Fecha/Ticker/Close/Volume, igual que
    cargar_daily."""
    datos = yf.download(tickers, period=period, interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=True)

    filas = []
    for ticker in tickers:
        try:
            df_t = datos[ticker].reset_index().dropna(subset=["Close"])
        except KeyError:
            continue  # ticker sin datos (recién listado, error de red puntual, etc.)
        if df_t.empty:
            continue
        col_fecha = "Date" if "Date" in df_t.columns else df_t.columns[0]
        df_t = df_t.rename(columns={col_fecha: "Fecha"})
        df_t["Ticker"] = ticker
        filas.append(df_t[["Fecha", "Ticker", "Close", "Volume"]])

    if not filas:
        raise ValueError("No se pudo descargar NINGÚN ticker -- revisa la lista o la conexión.")
    return pd.concat(filas, ignore_index=True).sort_values(["Fecha", "Ticker"]).reset_index(drop=True)


def descargar_precios_intraday_yf(tickers: list[str], period: str = "730d") -> pd.DataFrame:
    """Igual que arriba, pero 1h -- period="730d" ES el máximo real,
    pedir más no trae más historia (yfinance lo recorta igual, sin
    avisar). Devuelve Fecha/Dia/Hora/Ticker/Close/Volume, filtrado a
    horario de mercado (09:30-16:30), igual que cargar_intraday."""
    datos = yf.download(tickers, period=period, interval="1h", auto_adjust=True, group_by="ticker", progress=False, threads=True)

    filas = []
    for ticker in tickers:
        try:
            df_t = datos[ticker].reset_index().dropna(subset=["Close"])
        except KeyError:
            continue
        if df_t.empty:
            continue
        col_fecha = "Datetime" if "Datetime" in df_t.columns else df_t.columns[0]
        df_t = df_t.rename(columns={col_fecha: "Fecha"})
        # yfinance intradía viene con tz -- mismo fix que datos_frescos.py en producción
        df_t["Fecha"] = pd.to_datetime(df_t["Fecha"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        df_t["Dia"] = df_t["Fecha"].dt.date
        df_t["Hora"] = df_t["Fecha"].dt.strftime("%H:%M:%S")
        df_t["Ticker"] = ticker
        df_t = df_t[(df_t["Hora"] >= "09:30:00") & (df_t["Hora"] <= "16:30:00")]
        filas.append(df_t[["Fecha", "Dia", "Hora", "Ticker", "Close", "Volume"]])

    if not filas:
        raise ValueError("No se pudo descargar NINGÚN ticker -- revisa la lista o la conexión.")
    return pd.concat(filas, ignore_index=True).sort_values(["Fecha", "Ticker"]).reset_index(drop=True)
