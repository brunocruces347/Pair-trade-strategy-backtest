"""
Nivel 3 — estabilidad del β congelado, vía el marco de Brown, Durbin y
Evans (1975): CUSUM (¿el promedio de los residuos recursivos se corrió?),
CUSUMSQ (¿la varianza cambió?) y el one-step forecast test (¿el dato de
HOY es compatible con el modelo estimado con todo lo anterior?).

Reemplaza por completo el enfoque anterior (t_drift/t_significancia sobre
una ventana rolling) -- ese tenía un problema real: días consecutivos
comparten casi todos los mismos datos (ventana de 90 días desplazándose
de a 1), así que una racha de "fallas" no es evidencia independiente día
a día. CUSUM/CUSUMSQ están construidos justamente para acumular evidencia
de forma honesta sobre TODA la historia desde que se congeló β, sin ese
problema.

Corre RecursiveLS sobre TODA la serie desde el inicio de la ventana
operativa hasta hoy (no una ventana rolling) -- así CUSUM gana potencia
con cada día que pasa, tal como está pensado el método.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.recursive_ls import RecursiveLS


def _fuera_de_banda_cusum(res, alpha: float = 0.05) -> bool:
    """CUSUM: ¿el punto de HOY está fuera de su banda? Antes se revisaba
    np.any() sobre TODO el camino histórico -- eso es un trinquete de un
    solo sentido: en cuanto el camino cruzaba la banda una sola vez
    (incluso un salto de un día que se corrigió solo), quedaba en True
    para siempre, sin importar qué tan estable estuviera todo después.
    Evaluar solo el punto de hoy contra la banda de hoy es la práctica
    correcta para monitoreo secuencial -- si el camino salió y volvió a
    entrar, hoy está adentro y no debe disparar."""
    punto_hoy = np.array([res.nobs - 1])
    banda_inf, banda_sup = res._cusum_significance_bounds(alpha, points=punto_hoy)
    cusum_hoy = res.cusum[-1]
    return bool(cusum_hoy > banda_sup[0] or cusum_hoy < banda_inf[0])


def _fuera_de_banda_cusumsq(res, alpha: float = 0.05) -> bool:
    """CUSUMSQ: ¿la varianza se salió de banda en ALGÚN punto del camino
    que se le pasa? OJO: el estadístico SIEMPRE termina en exactamente
    1.0 por construcción (es la razón entre la varianza acumulada hasta
    hoy y sí misma), así que "solo hoy" nunca dispararía. Por eso se
    revisa todo el camino -- PERO quien llama a esta función debe
    pasarle un `res` ajustado solo sobre una ventana RECIENTE (ver
    calcular_estabilidad_beta), no toda la ventana operativa: cada
    cusum_squares[i] es una razón acumulada DESDE EL INICIO de la serie
    que se le dio a RecursiveLS -- si esa serie es toda la ventana
    operativa, un quiebre de varianza viejo (aunque ya esté superado)
    sigue empujando el acumulado durante mucho tiempo después, porque no
    es una ventana móvil, es acumulado real desde el día 0 de la serie."""
    d = max(res.nobs_diffuse, res.loglikelihood_burn)
    puntos = np.arange(d, res.nobs)
    banda_inf, banda_sup = res._cusum_squares_significance_bounds(alpha, points=puntos)
    cusumsq_vals = np.asarray(res.cusum_squares)
    return bool(np.any(cusumsq_vals > banda_sup) or np.any(cusumsq_vals < banda_inf))


def _limpiar_y_alinear(precios_x: pd.Series, precios_y: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = precios_x.index.intersection(precios_y.index)
    x, y = precios_x.loc[idx], precios_y.loc[idx]

    # yfinance a veces trae NaN puntuales en series largas (huecos de
    # datos, feriados no alineados entre mercados, etc.) -- alinear
    # fechas no los saca solo, hay que filtrarlos explícitamente. También
    # se descartan precios <=0 (romperían el log).
    validos = x.notna() & y.notna() & (x > 0) & (y > 0)
    if not validos.all():
        x, y = x[validos], y[validos]
    return x, y


def _ajustar_recursive_ls(precios_x: pd.Series, precios_y: pd.Series):
    """Pieza compartida entre calcular_estabilidad_beta (resumen del día,
    para CONTROL_DIARIO) y calcular_series_estabilidad_beta (camino
    completo, para graficar) -- un solo ajuste de RecursiveLS, no
    duplicado."""
    x, y = _limpiar_y_alinear(precios_x, precios_y)
    if len(x) < 30:
        raise ValueError(f"Muy pocos datos para CUSUM ({len(x)} observaciones)")

    modelo = RecursiveLS(np.log(x.to_numpy()), sm.add_constant(np.log(y.to_numpy())))
    res = modelo.fit()
    return res, x.index


def calcular_estabilidad_beta(precios_x: pd.Series, precios_y: pd.Series, alpha: float = 0.05, ventana_cusumsq_dias: int = 90) -> dict:
    """Corre RecursiveLS sobre ln(x) ~ const + ln(y), toda la historia
    disponible (desde el inicio de la ventana hasta hoy), y evalúa CUSUM
    y el one-step forecast sobre ese ajuste. Devuelve un dict con lo
    necesario para decidir la racha y para guardar en CONTROL_DIARIO --
    solo el resumen del día de HOY, no el camino completo (para eso ver
    calcular_series_estabilidad_beta).

    CUSUMSQ es un caso aparte: usa una corrida SEPARADA de RecursiveLS,
    solo sobre los últimos `ventana_cusumsq_dias` -- no reutiliza el
    ajuste de arriba. La razón: cada valor de cusum_squares es un
    acumulado DESDE EL INICIO de la serie que recibió RecursiveLS: si le
    pasáramos toda la ventana operativa, un quiebre de varianza viejo
    (aunque ya esté superado) seguiría empujando el acumulado durante
    mucho tiempo después -- no es una ventana móvil. Con un ajuste propio
    sobre solo los últimos ~90 días, el acumulado arranca de cero ahí,
    sin cargar nada de antes."""
    res, _ = _ajustar_recursive_ls(precios_x, precios_y)

    cusum_fuera_banda = _fuera_de_banda_cusum(res, alpha)

    x_reciente = precios_x.iloc[-ventana_cusumsq_dias:]
    y_reciente = precios_y.iloc[-ventana_cusumsq_dias:]
    try:
        res_reciente, _ = _ajustar_recursive_ls(x_reciente, y_reciente)
        cusumsq_fuera_banda = _fuera_de_banda_cusumsq(res_reciente, alpha)
    except ValueError:
        # Muy pocos datos recientes (par recién armado) -- no hay
        # suficiente evidencia todavía, no se marca como falla.
        cusumsq_fuera_banda = False

    residuo_hoy = float(res.resid_recursive[-1])
    one_step_pvalue = float(2 * (1 - stats.norm.cdf(abs(residuo_hoy))))

    beta_recursivo_hoy = float(res.recursive_coefficients.filtered[1][-1])

    return {
        "cusum_fuera_banda": cusum_fuera_banda,
        "cusumsq_fuera_banda": cusumsq_fuera_banda,
        "one_step_forecast_pvalue": one_step_pvalue,
        "beta_recursivo_hoy": beta_recursivo_hoy,
    }


def calcular_series_estabilidad_beta(precios_x: pd.Series, precios_y: pd.Series, alpha: float = 0.05, ventana_cusumsq_dias: int = 90) -> dict:
    """Igual que calcular_estabilidad_beta, pero devuelve el CAMINO
    COMPLETO (una serie por día, no solo el valor de hoy) -- para
    graficar.

    CUSUM/β recursivo/residuos: mismo ajuste de RecursiveLS sobre toda
    la ventana, sin volver a correrlo -- se muestra el camino completo
    como contexto histórico (la decisión de HOY solo mira el último
    punto, pero ver todo el camino ayuda a distinguir un quiebre viejo
    de uno reciente).

    CUSUMSQ: acá se devuelven las DOS versiones, para que el gráfico no
    quede desincronizado de lo que realmente decide el control diario:
      - `cusumsq`/`cusumsq_banda_*` (toda la ventana) -- contexto, no es
        lo que decide.
      - `cusumsq_reciente`/`cusumsq_reciente_banda_*` (ajuste aparte,
        solo últimos `ventana_cusumsq_dias`) -- ESTO es lo que
        calcular_estabilidad_beta usa para la racha/cierre forzado."""
    res, idx_completo = _ajustar_recursive_ls(precios_x, precios_y)
    d = max(res.nobs_diffuse, res.loglikelihood_burn)

    idx_cusum = idx_completo[d:]  # cusum/cusum_squares arrancan después del burn-in
    puntos = np.arange(d, res.nobs)
    banda_cusum_inf, banda_cusum_sup = res._cusum_significance_bounds(alpha, points=puntos)
    banda_cusumsq_inf, banda_cusumsq_sup = res._cusum_squares_significance_bounds(alpha, points=puntos)

    residuos = np.asarray(res.resid_recursive)
    residuos[:d] = np.nan  # antes del burn-in no son residuos reales, son ceros de relleno

    resultado = {
        "fechas": idx_completo,
        "beta_recursivo": np.asarray(res.recursive_coefficients.filtered[1]),
        "residuos_one_step": residuos,
        "fechas_cusum": idx_cusum,
        "cusum": np.asarray(res.cusum),
        "cusum_banda_inf": banda_cusum_inf,
        "cusum_banda_sup": banda_cusum_sup,
        "cusumsq": np.asarray(res.cusum_squares),
        "cusumsq_banda_inf": banda_cusumsq_inf,
        "cusumsq_banda_sup": banda_cusumsq_sup,
    }

    # CUSUMSQ reciente: mismo criterio que calcular_estabilidad_beta --
    # ajuste APARTE, solo sobre los últimos ventana_cusumsq_dias.
    x_reciente = precios_x.iloc[-ventana_cusumsq_dias:]
    y_reciente = precios_y.iloc[-ventana_cusumsq_dias:]
    try:
        res_reciente, idx_reciente = _ajustar_recursive_ls(x_reciente, y_reciente)
        d_reciente = max(res_reciente.nobs_diffuse, res_reciente.loglikelihood_burn)
        puntos_reciente = np.arange(d_reciente, res_reciente.nobs)
        banda_inf_r, banda_sup_r = res_reciente._cusum_squares_significance_bounds(alpha, points=puntos_reciente)
        resultado["fechas_cusumsq_reciente"] = idx_reciente[d_reciente:]
        resultado["cusumsq_reciente"] = np.asarray(res_reciente.cusum_squares)
        resultado["cusumsq_reciente_banda_inf"] = banda_inf_r
        resultado["cusumsq_reciente_banda_sup"] = banda_sup_r
    except ValueError:
        # Muy pocos datos recientes (par recién armado) -- sin serie que mostrar.
        resultado["fechas_cusumsq_reciente"] = np.array([])
        resultado["cusumsq_reciente"] = np.array([])
        resultado["cusumsq_reciente_banda_inf"] = np.array([])
        resultado["cusumsq_reciente_banda_sup"] = np.array([])

    return resultado
