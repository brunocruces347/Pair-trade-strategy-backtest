"""
Monitor de Chu-Stinchcombe-White (Chu, Stinchcombe & White, 1996,
"Monitoring Structural Change", Econometrica 64(5), 1045-1065) --
CUSUM sobre residuos de un modelo FIJO (mu/beta congelados, calibrados
UNA vez en el in-sample, nunca re-estimados), pensado específicamente
para "monitorear datos nuevos contra un modelo ya entrenado" -- a
diferencia de RecursiveLS (cusum_estabilidad.py), que re-estima el
modelo cada día.

Fórmula (confirmada cruzando el paper -- via el resumen de Zeileis et
al. -- con el código fuente real del paquete `strucchange` en R,
archivo monitoring.R, función `border`):

    S_d = promedio diario de (spread_hora - mu), agrupado por fecha real
    sigma_dia = desvío estándar de {S_d} durante el in-sample
    W(x) = (1/√n) * Σ_{d=1}^{i} (S_d / sigma_dia)     -- acumulado
    banda(x) = √( x(x-1) * [λ² + log(x/(x-1))] )
    x = (n + i) / n     -- n=días de in-sample, i=días operativos transcurridos

λ=2.45 para alpha=0.05 (CALIBRADO POR NOSOTROS vía Monte Carlo -- ver
calibrar_lambda_csw.py y calibracion_lambda_csw_resultados.csv -- no es
un valor de la literatura: el "4.6" que se había encontrado antes
pertenecía a una fórmula DISTINTA, "CUSUM sobre niveles" de
mizarlabs/López de Prado, no a esta. Falsos positivos medidos: 5.0% con
N=1000 simulaciones de un caso estable. Potencia de detección: 79%-100%
según qué tan grande/tardío sea el quiebre real).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _residuo_diario(
    precios_x_1h: pd.Series, precios_y_1h: pd.Series, beta: float, mu: float,
    precio_x0: float, precio_y0: float,
) -> pd.Series:
    """S_d: promedio diario de (spread_hora - mu), agrupado por fecha
    calendario real (no bloques fijos de 7 -- feriados y medios días no
    tienen 7 barras). El spread es el ANCLADO (ln(x/x0) - beta*ln(y/y0)),
    el mismo que usa calcular_mu_sigma_ventana para calibrar mu/sigma --
    tiene que ser exactamente esa fórmula, si no mu queda desalineado
    (mu se calibró con precios anclados, no con ln(x)-beta*ln(y) suelto)."""
    idx = precios_x_1h.index.intersection(precios_y_1h.index)
    x, y = precios_x_1h.loc[idx], precios_y_1h.loc[idx]
    validos = x.notna() & y.notna() & (x > 0) & (y > 0)
    if not validos.all():
        x, y = x[validos], y[validos]
    if len(x) == 0:
        return pd.Series(dtype=float)
    spread = (np.log(x) - np.log(precio_x0)) - beta * (np.log(y) - np.log(precio_y0))
    residuo = spread - mu
    return residuo.groupby(residuo.index.date).mean()


def calibrar_sigma_dia(
    precios_x_1h_insample: pd.Series, precios_y_1h_insample: pd.Series, beta: float, mu: float,
    precio_x0: float, precio_y0: float,
) -> float:
    """sigma_dia: desvío estándar de los PROMEDIOS diarios de
    (spread-mu), medido directo sobre el in-sample -- no derivado
    analíticamente (tipo sigma_1h*sqrt(7)), porque eso asume
    independencia entre barras horarias que este sistema no tiene por
    diseño (el spread es mean-reverting a propósito). Se llama UNA vez,
    al crear la ventana (mismo momento que se calibran mu/sigma, con el
    mismo precio_x0/precio_y0), y queda guardado en ESTADO_PAR.sigma_dia."""
    residuos_diarios = _residuo_diario(precios_x_1h_insample, precios_y_1h_insample, beta, mu, precio_x0, precio_y0)
    if len(residuos_diarios) < 2:
        raise ValueError(f"Muy pocos días de in-sample para calibrar sigma_dia ({len(residuos_diarios)} días)")
    sigma_dia = float(residuos_diarios.std())
    if sigma_dia == 0:
        raise ValueError("sigma_dia salió 0 -- no se puede normalizar (revisar los datos del in-sample)")
    return sigma_dia


def _banda(x: float, lam: float) -> float:
    return math.sqrt(x * (x - 1) * (lam ** 2 + math.log(x / (x - 1))))


def calcular_camino_csw(residuos_diarios_operativos: list[float], sigma_dia: float, n_dias_insample: int, lam: float = 2.45) -> dict:
    """Camino completo de W(x) día a día desde que arrancó lo operativo,
    y si cruzó la banda en algún punto (evaluación sobre TODO el camino,
    no solo hoy -- ver docstring del módulo, es correcto acá a
    propósito, a diferencia de CUSUM)."""
    n = n_dias_insample
    acumulado = 0.0
    w_path, banda_path = [], []
    alerta_activa = False
    for i, S_d in enumerate(residuos_diarios_operativos, start=1):
        acumulado += S_d / sigma_dia
        x = (n + i) / n
        W_x = acumulado / math.sqrt(n)
        banda_x = _banda(x, lam)
        w_path.append(W_x)
        banda_path.append(banda_x)
        if abs(W_x) > banda_x:
            alerta_activa = True
    return {
        "w_path": w_path, "banda_path": banda_path,
        "alerta_activa": alerta_activa,
        "w_hoy": w_path[-1] if w_path else None,
        "banda_hoy": banda_path[-1] if banda_path else None,
    }


def evaluar_csw(
    precios_x_1h_operativo: pd.Series, precios_y_1h_operativo: pd.Series,
    beta: float, mu: float, sigma_dia: float, n_dias_insample: int,
    precio_x0: float, precio_y0: float, lam: float = 2.45,
) -> dict:
    """Punto de entrada: de precios de 1h del período operativo (desde
    VENTANA.fecha_inicio hasta hoy) a la evaluación completa del monitor."""
    residuos_diarios = _residuo_diario(precios_x_1h_operativo, precios_y_1h_operativo, beta, mu, precio_x0, precio_y0)
    if len(residuos_diarios) == 0:
        return {"w_path": [], "banda_path": [], "alerta_activa": False, "w_hoy": None, "banda_hoy": None}
    return calcular_camino_csw(list(residuos_diarios.to_numpy()), sigma_dia, n_dias_insample, lam)
