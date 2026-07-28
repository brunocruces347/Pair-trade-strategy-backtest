"""
Nivel 3 — control diario de ADF y estabilidad de β de pares CON POSICIÓN
ABIERTA (ADF) o de TODOS los activos (estabilidad de β). A diferencia de
los niveles semanal/trimestral, este SÍ puede forzar el cierre de una
posición real (control de riesgo, no curaduría de universo).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from statsmodels.tsa.stattools import adfuller, coint


from cusum_estabilidad_backtest import calcular_estabilidad_beta  # noqa: E402 -- misma carpeta (controles/)




def calcular_ln_spread_serie(precios_x: pd.Series, precios_y: pd.Series, beta: float) -> pd.Series:
    """Spread diario, alineado por fecha. No usa precio_x0/precio_y0 (ese
    ancla es para la ventana operativa) — acá el ADF necesita la serie del
    spread en sí, cualquier normalización constante no cambia el resultado
    del test de estacionariedad."""
    idx = precios_x.index.intersection(precios_y.index)
    x, y = precios_x.loc[idx], precios_y.loc[idx]
    validos = x.notna() & y.notna() & (x > 0) & (y > 0)  # yfinance a veces trae NaN puntuales en series largas
    if not validos.all():
        x, y = x[validos], y[validos]
    return np.log(x) - beta * np.log(y)


def calcular_adf_par(precios_x: pd.Series, precios_y: pd.Series, beta: float) -> float:
    """p-value del ADF sobre el spread diario reciente."""
    spread = calcular_ln_spread_serie(precios_x, precios_y, beta)
    if len(spread) < 10:
        raise ValueError(f"Muy pocos datos para ADF ({len(spread)} observaciones)")
    resultado = adfuller(spread.to_numpy())
    return float(resultado[1])


def calcular_cointegracion_par(precios_x: pd.Series, precios_y: pd.Series) -> float:
    """p-value de cointegración (Engle-Granger) entre los precios diarios.
    SOLO DIAGNÓSTICO: re-estima un β nuevo cada vez que corre, así que no
    decide ninguna acción (violaría 'no reoptimizar durante la ventana').
    Se guarda para que veas cuánto se alejaría β si se recalculara desde
    cero, a modo de precaución -- lo que sí decide la racha/cierre es
    calcular_estabilidad_beta (CUSUM/CUSUMSQ/one-step forecast) en
    cusum_estabilidad.py, sobre el β CONGELADO."""
    idx = precios_x.index.intersection(precios_y.index)
    x, y = precios_x.loc[idx], precios_y.loc[idx]
    validos = x.notna() & y.notna() & (x > 0) & (y > 0)
    if not validos.all():
        x, y = x[validos], y[validos]
    if len(x) < 10:
        raise ValueError(f"Muy pocos datos para cointegración ({len(x)} observaciones)")
    _, pvalue, _ = coint(x.to_numpy(), y.to_numpy())
    return float(pvalue)


