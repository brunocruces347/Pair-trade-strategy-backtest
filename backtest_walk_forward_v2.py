"""
Backtest walk-forward "con las reglas nuevas" -- construido ENCIMA de
pairtrade_wfo_v3.ipynb (el notebook original, sin política), no desde
cero. Lo que cambia respecto al original:

1. VENTANAS SECUENCIALES, NO SOLAPADAS, CON CAPITAL RECICLADO -- el
   original usaba step=1 mes (ventanas solapadas, útil para robustez
   estadística) y un monto fijo por ventana. Acá cada ventana arranca
   donde terminó la anterior, y el capital de la ventana N es el
   capital_inicial + PNL acumulado de todas las ventanas 1..N-1 -- tal
   como se opera de verdad (reconstruccion_trimestral_main.py).

2. CONTROL DIARIO INYECTADO DENTRO DE LA SIMULACIÓN -- el original
   corría pair_trading_signals mecánicamente todo el OOS, con el stop
   VaR-tau como único corte. Acá, ANTES de simular, se evalúa día por
   día (sin mirar al futuro) si CUSUM+CSW+ADF -- las mismas funciones
   PURAS de producción (cusum_estabilidad.py, chu_stinchcombe_white.py,
   adf_cointegracion.py), sin pasar por la base de datos -- habrían
   forzado un cierre, y se lo pasa como corte adicional al motor de
   señales.

Todo lo demás (selección de pares, Chow test, half-life, el motor de
señales en sí, y la sección de análisis de robustez -- Rank IC,
permutación, leave-one-out, bootstrap) se reutiliza tal cual del
notebook original, sin reescribir.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

# Si cusum_estabilidad.py/etc. están co-ubicados en esta misma carpeta
# (repo liviano, sin DB), Python ya los encuentra solo. Si no (proyecto
# completo, con estos módulos en una carpeta de controles aparte),
# recién ahí se agrega esa carpeta al path -- así este mismo archivo
# sirve para las 2 versiones sin bifurcar el código.
  # ajusta si tu carpeta de controles tiene otro nombre
from cusum_estabilidad_backtest import calcular_estabilidad_beta  # noqa: E402
from chu_stinchcombe_white_backtest import evaluar_csw  # noqa: E402
from adf_cointegracion_backtest import calcular_adf_par, calcular_ln_spread_serie  # noqa: E402


import sqlite3


# ============================================================
# 1f. PORTADO TAL CUAL de pairtrade_wfo_v3.ipynb -- carga desde tu
#     sp500.db (prices_daily / prices_intraday_1h).
# ============================================================

def cargar_daily(con, start_date, end_date):
    query = "SELECT * FROM prices_daily WHERE Fecha BETWEEN ? AND ? ORDER BY Fecha, Ticker;"
    df = pd.read_sql(query, con, params=[start_date, end_date])
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def cargar_intraday(con, start_date, end_date):
    query = "SELECT * FROM prices_intraday_1h WHERE Dia BETWEEN ? AND ? ORDER BY Fecha, Ticker;"
    df = pd.read_sql(query, con, params=[start_date, end_date])
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df["Hora"] = pd.to_datetime(df["Hora"], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%H:%M:%S")
    df["Dia"] = pd.to_datetime(df["Dia"], errors="coerce", utc=True).dt.date
    df = df[(df["Hora"] >= "09:30:00") & (df["Hora"] <= "16:30:00")]
    return df


# ============================================================
# 1a-1d. PORTADO TAL CUAL de pairtrade_wfo_v3.ipynb (sin cambios)
# ============================================================

def filtrar_por_liquidez(df, col_precio, window=60, q=0.7):
    df = df.copy()
    df["DollarVolume"] = df[col_precio] * df["Volume"]
    df["ADV"] = df.groupby("Ticker")["DollarVolume"].transform(lambda x: x.rolling(window).mean())
    adv_final = df.groupby("Ticker")["ADV"].last()
    tickers_liquidos = adv_final[adv_final > adv_final.quantile(q)].index
    return df[df["Ticker"].isin(tickers_liquidos)]


def get_price_matrix(df, col_precio="Precio_Cierre"):
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.tz_localize(None)
    return df.pivot(index="Fecha", columns="Ticker", values=col_precio)


def hurst_from_beta(x, y, beta, q=1):
    x, y = np.asarray(x), np.asarray(y)
    s = x - beta * y
    taus = np.arange(1, len(s) // 4 + 1)
    Kq = [np.mean(np.abs(s[tau:] - s[:-tau]) ** q) / np.mean(np.abs(s) ** q) for tau in taus]
    reg = LinearRegression().fit(np.log(taus).reshape(-1, 1), np.log(Kq).reshape(-1, 1))
    return reg.coef_[0, 0] / q


def test_cointegracion_par(df_precios, tickerA, tickerB):
    data = df_precios[[tickerA, tickerB]].dropna()
    modelo = sm.OLS(data[tickerA], sm.add_constant(data[tickerB])).fit()
    adf = adfuller(modelo.resid)
    return {
        "tickerA": tickerA, "tickerB": tickerB,
        "alpha": modelo.params["const"], "beta": modelo.params[tickerB],
        "sigma": modelo.mse_resid ** 0.5, "rss": modelo.ssr,
        "ADF": adf[0], "pvalue": adf[1],
    }


def buscar_pares_cointegrados(df_precios, pvalue_max=0.05, hurst_max=0.5):
    resultados = []
    for tickerA, tickerB in combinations(df_precios.columns, 2):
        try:
            res = test_cointegracion_par(df_precios, tickerA, tickerB)
            res["Hurst"] = hurst_from_beta(df_precios[tickerA], df_precios[tickerB], res["beta"])
            if res["pvalue"] < pvalue_max and res["beta"] > 0 and res["Hurst"] < hurst_max:
                resultados.append(res)
        except Exception:
            continue
    return pd.DataFrame(resultados)


def calcular_half_life(df_diario, tickerA, tickerB, beta, hl_min=5, hl_max=20):
    data = df_diario[[tickerA, tickerB]].dropna()
    spread = (data[tickerA] - beta * data[tickerB]).values
    if len(spread) < 30:
        return {"half_life_dias": np.nan, "coef_ar1": np.nan, "pasa_half_life": False}
    spread_lag = spread[:-1].reshape(-1, 1)
    d_spread = np.diff(spread).reshape(-1, 1)
    modelo = LinearRegression().fit(spread_lag, d_spread)
    rho = modelo.coef_[0, 0]
    if rho >= 0 or rho <= -1:
        return {"half_life_dias": np.inf, "coef_ar1": round(rho, 4), "pasa_half_life": False}
    hl = -np.log(2) / np.log(1 + rho)
    return {"half_life_dias": round(hl, 2), "coef_ar1": round(rho, 4), "pasa_half_life": hl_min <= hl <= hl_max}


def chow_test(df_diario_70, df_diario_30, tickerA, tickerB, rss_pooled, rss_70, rss_30):
    k = 2
    n1, n2 = df_diario_70.shape[0], df_diario_30.shape[0]
    numerator = (rss_pooled - (rss_70 + rss_30)) / k
    denominator = (rss_70 + rss_30) / (n1 + n2 - 2 * k)
    F_stat = numerator / denominator if denominator != 0 else np.inf
    p_value = 1 - stats.f.cdf(F_stat, k, n1 + n2 - 2 * k)
    return {"F_stat": round(F_stat, 4), "p_value": round(p_value, 4), "pasa_retest": (0.001 < p_value < 0.05)}


def encontrar_pares_validos_periodo(fecha, colprecio, q_liquidez, df, p_value_max, hurst_max,
                                     hl_min=5, hl_max=40, diferencia_beta_max=25):
    """Idéntico a _encontrar_pares_validos_periodo del notebook original."""
    train = df[df["Fecha"].between(pd.Timestamp(fecha["train_start"]), pd.Timestamp(fecha["train_end"]))]
    precios_train = get_price_matrix(filtrar_por_liquidez(train, colprecio, q=q_liquidez), col_precio=colprecio)
    precios_train = np.log(precios_train / precios_train.iloc[0])

    split_idx = int(len(precios_train) * 0.70)
    precios_70, precios_30 = precios_train.iloc[:split_idx], precios_train.iloc[split_idx:]

    pares_train = buscar_pares_cointegrados(precios_train, pvalue_max=p_value_max, hurst_max=hurst_max)
    pares_validos = []
    rechazados = {"retest": 0, "half_life": 0, "chow_test": 0, "diferencia_beta": 0, "datos": 0}

    for _, row in pares_train.iterrows():
        t1, t2, beta_all = row["tickerA"], row["tickerB"], row["beta"]
        try:
            data_70, data_30 = precios_70[[t1, t2]].dropna(), precios_30[[t1, t2]].dropna()
        except KeyError:
            rechazados["datos"] += 1
            continue
        if len(data_30) < 50:
            rechazados["datos"] += 1
            continue

        res_70 = test_cointegracion_par(data_70, t1, t2)
        res = test_cointegracion_par(data_30, t1, t2)
        if not (res["pvalue"] < p_value_max and res["beta"] > 0):
            rechazados["retest"] += 1
            continue
        if not (100 * abs(res["beta"] - res_70["beta"]) / abs(res_70["beta"]) < diferencia_beta_max):
            rechazados["diferencia_beta"] += 1
            continue
        f1 = chow_test(data_70, data_30, t1, t2, row["rss"], res_70["rss"], res["rss"])
        if not f1["pasa_retest"]:
            rechazados["chow_test"] += 1
            continue
        f2 = calcular_half_life(data_30, t1, t2, beta_all, hl_min=hl_min, hl_max=hl_max)
        if not f2["pasa_half_life"]:
            rechazados["half_life"] += 1
            continue

        row["beta_70"], row["beta_30"] = res_70["beta"], res["beta"]
        row["p_value_chow_test"], row["half_life_dias"] = f1["p_value"], f2["half_life_dias"]
        row["score_selector"] = res["pvalue"] * (f2["half_life_dias"] / hl_max)
        row["diferencia_beta_70_30"] = 100 * abs(res["beta"] - res_70["beta"]) / abs(res_70["beta"])
        row["pvalue_30"] = res["pvalue"]
        pares_validos.append(row)

    return pd.DataFrame(pares_validos), pares_train, rechazados


# ============================================================
# 2. NUEVO -- evaluación diaria de controles (CUSUM+CSW+ADF),
#    reutilizando las funciones PURAS reales de producción, sin DB.
# ============================================================

def evaluar_cierre_forzado_por_dia(
    precios_x_diario: pd.Series, precios_y_diario: pd.Series,
    precios_x_1h: pd.Series, precios_y_1h: pd.Series,
    dias_operativos: list, fecha_inicio_ventana,
    beta: float, mu: float, sigma_dia: float | None,
    n_dias_insample: int, precio_x0: float, precio_y0: float,
    estabilidad_beta_dias_consecutivos: int = 7,
    adf_dias_consecutivos: int = 5,
    adf_pvalue_umbral: float = 0.10,
    csw_lambda: float = 2.45,
    devolver_diagnostico: bool = False,
    requerir_cusum: bool = True,
):
    """Recorre día por día el período operativo -- SIN mirar al futuro,
    cada evaluación usa solo datos hasta ese día inclusive -- y
    devuelve la primera fecha en la que las condiciones coinciden
    (o None si nunca coincide, en cuyo caso la ventana termina por
    vencimiento natural, no por control diario).

    Asume que el par TIENE posición abierta durante el período
    (por eso ADF corre todos los días, sin esperar el "gatillo" que
    usa un par plano en producción -- acá estamos decidiendo
    justamente cuándo se habría forzado el cierre de una posición).

    requerir_cusum=False -- saca la racha de CUSUM del gatillo, deja
    el cierre forzado en manos de CSW (modelo fijo, sticky) + ADF
    solamente. CUSUM se sigue calculando igual (queda en el
    diagnóstico), solo deja de ser condición necesaria para cerrar --
    para aislar si CUSUM está aportando o solo agregando ruido/demora.

    Reutiliza calcular_estabilidad_beta / evaluar_csw / calcular_adf_par
    de producción tal cual -- el tracking de racha se hace acá con
    variables de Python en vez de filas de CONTROL_DIARIO, porque un
    backtest no tiene sentido escribiéndolo a una DB fila por fila.

    Con devolver_diagnostico=True devuelve además un dict con el
    máximo que alcanzó cada racha y cuántas veces cada chequeo lanzó
    una excepción -- para distinguir "el par fue genuinamente estable"
    de "el chequeo nunca corrió por un error silencioso"."""
    racha_cusum = 0
    racha_adf = 0
    csw_alerta_activa = False

    diag = {
        "racha_cusum_max": 0, "racha_adf_max": 0, "csw_llego_a_activarse": False,
        "dias_evaluados": 0, "errores_cusum": 0, "errores_csw": 0, "errores_adf": 0,
        "ultimo_error_cusum": None, "ultimo_error_csw": None, "ultimo_error_adf": None,
    }

    for fecha_hoy in dias_operativos:
        x_diario_hoy = precios_x_diario.loc[:fecha_hoy]
        y_diario_hoy = precios_y_diario.loc[:fecha_hoy]
        if len(x_diario_hoy) < 30 or len(y_diario_hoy) < 30:
            continue  # no hay suficiente historia todavía para correr los tests

        diag["dias_evaluados"] += 1

        try:
            resultado_cusum = calcular_estabilidad_beta(x_diario_hoy, y_diario_hoy)
            racha_cusum = racha_cusum + 1 if resultado_cusum["cusum_fuera_banda"] else 0
        except Exception as e:
            diag["errores_cusum"] += 1
            diag["ultimo_error_cusum"] = f"{type(e).__name__}: {e}"

        if sigma_dia is not None:
            x_1h_hoy = precios_x_1h.loc[fecha_inicio_ventana:fecha_hoy]
            y_1h_hoy = precios_y_1h.loc[fecha_inicio_ventana:fecha_hoy]
            if len(x_1h_hoy) > 0 and len(y_1h_hoy) > 0:
                try:
                    resultado_csw = evaluar_csw(
                        x_1h_hoy, y_1h_hoy, beta, mu, sigma_dia, n_dias_insample,
                        precio_x0, precio_y0, lam=csw_lambda,
                    )
                    if resultado_csw["alerta_activa"]:
                        csw_alerta_activa = True  # sticky, nunca se apaga sola
                except Exception as e:
                    diag["errores_csw"] += 1
                    diag["ultimo_error_csw"] = f"{type(e).__name__}: {e}"

        try:
            adf_pvalue = calcular_adf_par(x_diario_hoy, y_diario_hoy, beta)
            racha_adf = racha_adf + 1 if adf_pvalue > adf_pvalue_umbral else 0
        except Exception as e:
            diag["errores_adf"] += 1
            diag["ultimo_error_adf"] = f"{type(e).__name__}: {e}"

        diag["racha_cusum_max"] = max(diag["racha_cusum_max"], racha_cusum)
        diag["racha_adf_max"] = max(diag["racha_adf_max"], racha_adf)
        diag["csw_llego_a_activarse"] = diag["csw_llego_a_activarse"] or csw_alerta_activa

        condicion_cusum = (racha_cusum >= estabilidad_beta_dias_consecutivos) if requerir_cusum else True
        if condicion_cusum and csw_alerta_activa and racha_adf >= adf_dias_consecutivos:
            return (fecha_hoy, diag) if devolver_diagnostico else fecha_hoy

    return (None, diag) if devolver_diagnostico else None


# ============================================================
# 3. pair_trading_signals -- IDÉNTICO al original, con UN agregado:
#    forzar_cierre_desde. Si se llega a esa fecha con posición
#    abierta, se cierra ahí con exit_reason="Control diario",
#    sin esperar a que el spread vuelva a la media ni al stop VaR-tau.
# ============================================================

def pair_trading_signals(
    x, y, b, media_historica, std_historica,
    Monto, VaR_lvl_sup, VaR_lvl_inf,
    take_profit_th, stop_loss_th,
    k=1.0, tau=175,
    forzar_cierre_desde: pd.Timestamp | None = None,
    x0: float | None = None, y0: float | None = None,
):
    """Igual que pair_trading_signals del notebook original -- con 2
    agregados:

    forzar_cierre_desde -- si no es None y llegamos a esa fecha (o
    después) con posición abierta, se cierra ahí con exit_reason
    "Control diario", independiente de dónde esté el spread.

    x0/y0 -- el ANCLA del spread normalizado (ln_s_arr), que es lo que
    se compara contra media_historica/std_historica y las bandas VaR.
    Si no se pasan, cae al comportamiento original (x.iloc[0]/y.iloc[0]
    -- el primer precio de la serie que se le pase). Bug real
    encontrado: si x/y son el OOS pero media_historica/std_historica
    se calibraron con el in-sample, x0/y0 TIENEN que ser el primer
    precio del in-sample también (el mismo ancla que mu/sigma) -- si
    no, el spread queda corrido por una constante respecto a las
    bandas contra las que se compara. Solo afecta ln_s_arr -- x_arr/
    y_arr (valorización, PNL, flujo) siguen siendo el precio real de
    entrada, sin tocar."""
    costo_total = 10 / 10_000

    s_arr = (x - b * y).to_numpy(float)
    x_arr = x.to_numpy(float)
    y_arr = y.to_numpy(float)
    x0_efectivo = x0 if x0 is not None else x.iloc[0]
    y0_efectivo = y0 if y0 is not None else y.iloc[0]
    ln_s_arr = ((np.log(x) - np.log(x0_efectivo)) - b * (np.log(y) - np.log(y0_efectivo))).to_numpy(float)

    idx = x.index
    n = len(s_arr)

    peso_x = Monto / (b + 1)
    peso_y = Monto * b / (b + 1)

    sig_arr = np.full(n, np.nan, dtype=object)
    pos_x = np.ones(n); pos_y = np.ones(n)
    dpos_x = np.zeros(n); dpos_y = np.zeros(n)
    val_x = np.zeros(n); val_y = np.zeros(n)
    val_par = np.zeros(n); capital = np.zeros(n)
    flujo = np.zeros(n); costo = np.zeros(n)
    pnl = np.zeros(n); pnl_acc = np.zeros(n)
    ret = np.zeros(n); log_ret = np.zeros(n)
    ret_acc = np.zeros(n)
    exit_r = np.full(n, np.nan, dtype=object)

    deal_id_arr = np.full(n, np.nan)
    deal_counter = 0

    val_x[0] = peso_x * x_arr[0]
    val_y[0] = peso_y * y_arr[0]
    val_par[0] = val_x[0] + val_y[0]

    def _recalc(t):
        val_x[t] = peso_x * x_arr[t] * pos_x[t]
        val_y[t] = peso_y * y_arr[t] * pos_y[t]
        val_par[t] = val_x[t] + val_y[t]
        flujo[t] = -(dpos_x[t] * peso_x * x_arr[t] + dpos_y[t] * peso_y * y_arr[t])
        costo[t] = costo_total * (abs(dpos_x[t] * peso_x * x_arr[t]) + abs(dpos_y[t] * peso_y * y_arr[t]))
        capital[t] = val_par[t - 1]
        pnl[t] = val_par[t] - val_par[t - 1] - costo[t] + flujo[t]
        pnl_acc[t] = pnl_acc[t - 1] + pnl[t]
        ret[t] = pnl[t] / capital[t] if capital[t] != 0 else 0.0
        log_ret[t] = np.log1p(ret[t])
        ret_acc[t] = np.expm1(log_ret[:t + 1].sum())

    position = None; hubo_trade = False; end_idx = n - 1; velas_var = 0

    for t in range(1, n):
        st, mt, sd = ln_s_arr[t], media_historica, std_historica
        pos_x[t] = pos_x[t - 1]; pos_y[t] = pos_y[t - 1]
        dpos_x[t] = 0.0; dpos_y[t] = 0.0

        if np.isnan(mt) or np.isnan(sd):
            _recalc(t); continue

        if position is not None:
            close = False; reason = None
            sobre_var = (st > VaR_lvl_sup) if position == "Sell" else (st < VaR_lvl_inf)
            velas_var = velas_var + 1 if sobre_var else 0

            # NUEVO -- corte por control diario (CUSUM+CSW+ADF)
            if forzar_cierre_desde is not None and idx[t] >= forzar_cierre_desde:
                close, reason = True, "Control diario"
            elif position == "Sell":
                if st < mt:
                    close, reason = True, "Mean Reversion"
                elif velas_var >= tau:
                    close, reason = True, f"Stop VaR τ={tau}"
            elif position == "Buy":
                if st > mt:
                    close, reason = True, "Mean Reversion"
                elif velas_var >= tau:
                    close, reason = True, f"Stop VaR τ={tau}"

            if close:
                sig_arr[t] = f"Close {position}"
                exit_r[t] = reason
                if position == "Buy":
                    pos_x[t] -= 1; pos_y[t] += 1
                else:
                    pos_x[t] += 1; pos_y[t] -= 1
                dpos_x[t] = pos_x[t] - pos_x[t - 1]
                dpos_y[t] = pos_y[t] - pos_y[t - 1]
                _recalc(t)
                deal_id_arr[t] = deal_counter
                position = None; hubo_trade = True; velas_var = 0
                continue

        if position is None and hubo_trade:
            if ret_acc[t - 1] > take_profit_th:
                end_idx = t; _recalc(t); exit_r[t] = "TP"; break
            if ret_acc[t - 1] < stop_loss_th:
                end_idx = t; _recalc(t); exit_r[t] = "SL"; break

        if position is None:
            # No abrir posiciones nuevas si ya llegamos al día de
            # control diario -- estaríamos abriendo algo que el
            # sistema real ya habría marcado para cerrar.
            puede_abrir = forzar_cierre_desde is None or idx[t] < forzar_cierre_desde
            if puede_abrir and st > mt + k * sd and st < VaR_lvl_sup:
                sig_arr[t] = "Sell"
                pos_x[t] -= 1; pos_y[t] += 1
                position = "Sell"; deal_counter += 1
                deal_id_arr[t] = deal_counter; velas_var = 0
            elif puede_abrir and st < mt - k * sd and st > VaR_lvl_inf:
                sig_arr[t] = "Buy"
                pos_x[t] += 1; pos_y[t] -= 1
                position = "Buy"; deal_counter += 1
                deal_id_arr[t] = deal_counter; velas_var = 0
            if position is not None:
                dpos_x[t] = pos_x[t] - pos_x[t - 1]
                dpos_y[t] = pos_y[t] - pos_y[t - 1]
                _recalc(t); continue

        if position is not None:
            deal_id_arr[t] = deal_counter
        _recalc(t)

    estado_arr = np.empty(n, dtype=object)
    estado_counter = 1; en_trade = False
    for t in range(n):
        estado_arr[t] = f"D{estado_counter}" if en_trade else f"E{estado_counter}"
        sig = sig_arr[t]
        if sig in ("Buy", "Sell"):
            en_trade = True; estado_arr[t] = f"D{estado_counter}"
        elif isinstance(sig, str) and sig.startswith("Close"):
            en_trade = False; estado_counter += 1

    pnl_estado = np.zeros(n); pnl_acum_estado = np.zeros(n)
    ret_estado = np.zeros(n); ret_acum_estado = np.zeros(n)
    estado_actual = estado_arr[0]; acum = acum_log = 0.0
    for t in range(n):
        if estado_arr[t] != estado_actual:
            estado_actual = estado_arr[t]; acum = acum_log = 0.0
        pnl_estado[t] = pnl[t]; acum += pnl[t]
        pnl_acum_estado[t] = acum; ret_estado[t] = ret[t]
        acum_log += log_ret[t]
        ret_acum_estado[t] = np.expm1(acum_log)

    return pd.DataFrame({
        "s": s_arr, "beta": b, "x": x_arr, "y": y_arr,
        "m": media_historica, "std": std_historica,
        "Var_Sup": VaR_lvl_sup, "Var_Inf": VaR_lvl_inf,
        "signal": sig_arr,
        "Monto": float(Monto), "Peso_X": peso_x, "Peso_Y": peso_y,
        "Posicion_X": pos_x, "Posicion_Y": pos_y,
        "Delta_pos_X": dpos_x, "Delta_pos_Y": dpos_y,
        "Valorizacion_X": val_x, "Valorizacion_Y": val_y,
        "Valorizacion_Par": val_par, "Capital": capital,
        "Flujo": flujo, "Costo": costo,
        "PNL": pnl, "PNL_Acumulado": pnl_acc,
        "Retorno": ret, "Log_Retorno": log_ret,
        "Retorno_Acumulado": ret_acc,
        "Equity_Curve": (ret + 1).cumprod(),
        "Exit_Reason": exit_r, "Deal_ID": deal_id_arr,
        "Estado": estado_arr,
        "PNL_Estado": pnl_estado, "PNL_Acumulado_Estado": pnl_acum_estado,
        "Retorno_Estado": ret_estado, "Retorno_Acumulado_Estado": ret_acum_estado,
    }, index=idx).iloc[:end_idx + 1]


# ============================================================
# 4. NUEVO -- ventanas SECUENCIALES (no solapadas), a diferencia de
#    generar_ventanas() del original (step=1 mes, solapadas).
# ============================================================

def generar_ventanas_secuenciales(fecha_inicio, fecha_fin, train_meses=24, is_meses=6, oos_dias=90):
    """Solo el período OPERATIVO (OOS) es estrictamente secuencial, sin
    solape -- es la propiedad que necesitas para reciclar capital (no
    puedes tener 2 ventanas operando el mismo capital al mismo tiempo).

    El TRAIN y el IS, en cambio, SÍ se solapan entre ventanas
    consecutivas -- a propósito, porque así es como funciona
    producción de verdad: reconstruccion_trimestral_main.py siempre
    mira los últimos `train_meses` desde HOY, no arranca de cero desde
    donde terminó el ciclo anterior. Si esto encadenara train también,
    cada ventana consumiría train_meses+oos_dias de calendario nuevo en
    vez de solo oos_dias, y con pocos años de historia rendiría muchas
    menos ventanas de las que realmente deberían caber (bug real que
    tuvo esta función en una versión anterior)."""
    ventanas = []
    fecha_fin = pd.Timestamp(fecha_fin)
    oos_start = pd.Timestamp(fecha_inicio) + pd.DateOffset(months=train_meses)
    i = 0

    while True:
        train_end = oos_start - pd.DateOffset(days=1)
        train_start = oos_start - pd.DateOffset(months=train_meses)
        is_start = train_end - pd.DateOffset(months=is_meses) + pd.DateOffset(days=1)
        is_end = train_end
        oos_end = oos_start + pd.DateOffset(days=oos_dias) - pd.DateOffset(days=1)

        if oos_end > fecha_fin:
            break

        ventanas.append({
            "etiqueta": f"V{i+1}",
            "train_start": train_start.strftime("%Y-%m-%d"), "train_end": train_end.strftime("%Y-%m-%d"),
            "is_start": is_start.strftime("%Y-%m-%d"), "is_end": is_end.strftime("%Y-%m-%d"),
            "oos_start": oos_start.strftime("%Y-%m-%d"), "oos_end": oos_end.strftime("%Y-%m-%d"),
        })
        oos_start = oos_end + pd.DateOffset(days=1)
        i += 1

    return ventanas


def generar_ventanas_solapadas(fecha_inicio, fecha_fin, train_meses=24, is_meses=6, oos_dias=90, step_meses=1):
    """SOLO para el análisis de robustez (Rank IC, cuartiles) -- NUNCA
    para la comparación causal con capital reciclado (2 ventanas que se
    solapan en el tiempo no pueden estar operando el mismo capital a la
    vez, no tiene sentido financiero). El OOS avanza cada `step_meses`
    en vez de saltar oos_dias completos -- con oos_dias=90 y
    step_meses=1, en el mismo rango de calendario rendís ~3 veces más
    puntos que generar_ventanas_secuenciales, porque los OOS se
    solapan entre sí (igual que pairtrade_wfo_v3.ipynb original).

    Correr esto con correr_backtest_secuencial(..., capital_fijo=True)
    -- si no, el capital "reciclado" entre ventanas que se solapan en
    el tiempo no significa nada real, solo ruido en el número final."""
    ventanas = []
    fecha_fin = pd.Timestamp(fecha_fin)
    oos_start = pd.Timestamp(fecha_inicio) + pd.DateOffset(months=train_meses)
    i = 0

    while True:
        train_end = oos_start - pd.DateOffset(days=1)
        train_start = oos_start - pd.DateOffset(months=train_meses)
        is_start = train_end - pd.DateOffset(months=is_meses) + pd.DateOffset(days=1)
        is_end = train_end
        oos_end = oos_start + pd.DateOffset(days=oos_dias) - pd.DateOffset(days=1)

        if oos_end > fecha_fin:
            break

        ventanas.append({
            "etiqueta": f"V{i+1}",
            "train_start": train_start.strftime("%Y-%m-%d"), "train_end": train_end.strftime("%Y-%m-%d"),
            "is_start": is_start.strftime("%Y-%m-%d"), "is_end": is_end.strftime("%Y-%m-%d"),
            "oos_start": oos_start.strftime("%Y-%m-%d"), "oos_end": oos_end.strftime("%Y-%m-%d"),
        })
        oos_start = oos_start + pd.DateOffset(months=step_meses)  # -- única diferencia con la secuencial
        i += 1

    return ventanas


def calcular_fecha_inicio_para_ventanas(fecha_minima_1h_disponible, train_meses=24, is_meses=6):
    """El train (selección de pares) es diario -- puede arrancar mucho
    antes. Pero el in-sample (mu/sigma) y el OOS (señales) son 1h -- no
    pueden pedir datos de antes de que tu 1h realmente empiece. Esta
    función calcula el fecha_inicio correcto para
    generar_ventanas_secuenciales de modo que el IS de la PRIMERA
    ventana arranque justo en fecha_minima_1h_disponible, no antes."""
    fecha_minima = pd.Timestamp(fecha_minima_1h_disponible)
    return (fecha_minima - pd.DateOffset(months=train_meses) + pd.DateOffset(months=is_meses)).strftime("%Y-%m-%d")


# ============================================================
# 5. NUEVO -- orquestador: ventanas secuenciales + capital reciclado
#    + control diario inyectado. Reemplaza al loop walk-forward del
#    notebook original (celda 22), que era solapado y sin reciclar.
# ============================================================

def correr_backtest_secuencial(
    ventanas: list[dict],
    df_daily_full: pd.DataFrame, df_intraday_full: pd.DataFrame,
    capital_inicial: float,
    col_precio="Close", q_liquidez=0.80, p_value_max=0.05, hurst_max=0.35,
    hl_min=4.45, hl_max=20, diferencia_beta=25,
    k=1.0, csw_lambda=2.45,
    estabilidad_beta_dias_consecutivos=7, adf_dias_consecutivos=5, adf_pvalue_umbral=0.10,
    periodo_in_sample_meses=6,
    usar_control_diario: bool = True,
    requerir_cusum: bool = True,
    capital_fijo: bool = False,
    universo_por_ventana: dict[str, set[str]] | None = None,
):
    """El loop principal. Por cada ventana: selecciona pares (train
    diario), calibra mu/sigma/sigma_dia (in-sample 1h), evalúa día a
    día si el control diario habría forzado un cierre, corre el motor
    de señales con ese corte, y pasa el PNL de la ventana como capital
    de la ventana siguiente -- tal como se opera de verdad.

    universo_por_ventana -- dict {etiqueta_ventana: set(tickers)}, ej.
    {"V1": {"AAPL", "MSFT", ...}, "V2": {...}, ...}. Si se pasa,
    filtra el TRAIN de cada ventana a solo esos tickers antes de
    buscar pares -- pensado para usar con
    universo_historico.obtener_constituyentes_en_fecha() y así
    reconstruir qué tickers estaban REALMENTE en el S&P 500 en cada
    ventana histórica, en vez de la lista de HOY para todo el
    backtest (que reintroduce sesgo de supervivencia cuando los datos
    de precios vienen de una descarga masiva de yfinance, que solo
    trae tickers vigentes). Opción (a) simple, sin relleno: si un
    ticker histórico no está entre los precios descargados, queda
    afuera de esa ventana sin más -- no se sale a buscarlo aparte.
    Si no se pasa nada (None), no filtra -- mismo comportamiento de
    siempre.

    usar_control_diario=False -- corre exactamente la misma selección
    de pares y el mismo motor de señales, pero SIN el corte de
    CUSUM+CSW+ADF (cada par corre hasta el vencimiento natural del OOS
    o el stop VaR-tau, como el notebook original sin política). Es la
    comparación causal directa para responder "¿el control diario
    agrega valor?" -- misma selección, mismos pares, mismos datos,
    única diferencia es ese corte.

    requerir_cusum=False -- con el control diario activo, saca la
    racha de CUSUM del gatillo -- el cierre queda en manos de CSW
    (modelo fijo, sticky) + ADF solamente. Para aislar si CUSUM
    específicamente está aportando o solo agregando ruido/demora al
    disparo (sin tener que sacar el control diario entero).

    capital_fijo=True -- NO reciclar capital entre ventanas, usar
    siempre capital_inicial. Obligatorio si `ventanas` viene de
    generar_ventanas_solapadas (OOS que se solapan en el tiempo no
    pueden estar "reciclando" el mismo capital sin que el número
    final quede sin sentido financiero) -- solo para el análisis de
    robustez (Rank IC, cuartiles), nunca para la comparación causal."""
    capital_actual = capital_inicial
    resultados_por_ventana = []
    lista_res_oos, lista_pares_validos = [], []

    for v in ventanas:
        print(f"\n{'='*60}\nVENTANA {v['etiqueta']}  (capital: {capital_actual:,.2f})\n{'='*60}")
        print(f"  Train: {v['train_start']} -> {v['train_end']}  |  IS: {v['is_start']} -> {v['is_end']}  |  OOS: {v['oos_start']} -> {v['oos_end']}")

        df_diaria = df_daily_full[df_daily_full["Fecha"].between(pd.Timestamp(v["train_start"]), pd.Timestamp(v["train_end"]))].copy()

        if universo_por_ventana is not None:
            universo_ventana = universo_por_ventana.get(v["etiqueta"], set())
            tickers_antes = df_diaria["Ticker"].nunique()
            df_diaria = df_diaria[df_diaria["Ticker"].isin(universo_ventana)]
            print(f"  Universo histórico: {len(universo_ventana)} tickers en el índice esa fecha -- "
                  f"{tickers_antes} con precio descargado -> {df_diaria['Ticker'].nunique()} tras cruzar ambos")

        ins_1h = df_intraday_full[df_intraday_full["Dia"].between(pd.Timestamp(v["is_start"]).date(), pd.Timestamp(v["is_end"]).date())].copy()
        oos_1h = df_intraday_full[df_intraday_full["Dia"].between(pd.Timestamp(v["oos_start"]).date(), pd.Timestamp(v["oos_end"]).date())].copy()

        pares_validos, pares_train, rechazados = encontrar_pares_validos_periodo(
            v, col_precio, q_liquidez, df_diaria, p_value_max=p_value_max, hurst_max=hurst_max,
            hl_min=hl_min, hl_max=hl_max, diferencia_beta_max=diferencia_beta,
        )
        print(f"  Pares: {len(pares_train)} descubiertos -> {len(pares_validos)} válidos. Rechazos: {rechazados}")

        if len(pares_validos) == 0:
            print("  Sin pares válidos -- ventana vacía, no consume tiempo del backtest pero no aporta capital nuevo.")
            resultados_por_ventana.append({"Ventana": v["etiqueta"], "PNL": 0.0, "capital_inicio": capital_actual, "capital_fin": capital_actual, "n_pares": 0})
            continue

        precios_ins = get_price_matrix(ins_1h, col_precio=col_precio).apply(pd.to_numeric, errors="coerce")
        precios_oos = get_price_matrix(oos_1h, col_precio=col_precio).apply(pd.to_numeric, errors="coerce")
        precios_diarios_mat = get_price_matrix(df_diaria, col_precio=col_precio).apply(pd.to_numeric, errors="coerce")

        tickers_disponibles = set(precios_oos.columns)
        pares_validos = pares_validos[pares_validos["tickerA"].isin(tickers_disponibles) & pares_validos["tickerB"].isin(tickers_disponibles)].copy()
        if len(pares_validos) == 0:
            print("  Sin pares con datos OOS.")
            resultados_por_ventana.append({"Ventana": v["etiqueta"], "PNL": 0.0, "capital_inicio": capital_actual, "capital_fin": capital_actual, "n_pares": 0})
            continue

        pares_validos["Ventana"] = v["etiqueta"]
        pares_validos["pair"] = pares_validos["tickerA"] + "_" + pares_validos["tickerB"]
        lista_pares_validos.append(pares_validos.copy())

        # Reparto de capital proporcional al valor inicial de cada par
        # -- mismo criterio que crear_ventana_nueva en producción.
        val_iniciales = {}
        for _, row in pares_validos.iterrows():
            tA, tB, beta = row["tickerA"], row["tickerB"], row["beta"]
            try:
                px = precios_oos[[tA, tB]].dropna().iloc[0]
                val_iniciales[f"{tA}_{tB}"] = (1 / (beta + 1)) * (px[tA] + px[tB] * beta)
            except Exception:
                val_iniciales[f"{tA}_{tB}"] = None
        suma_val = sum(v_ for v_ in val_iniciales.values() if v_ is not None)

        dias_oos = sorted(pd.to_datetime(oos_1h["Dia"].unique()))
        n_dias_insample = int(periodo_in_sample_meses * 21)

        resultados_pares = []
        for _, row in pares_validos.iterrows():
            t1, t2, beta = row["tickerA"], row["tickerB"], row["beta"]
            pair = f"{t1}_{t2}"
            monto_par = (capital_actual / suma_val) if (val_iniciales.get(pair) and suma_val != 0) else 1.0

            data_pair_ins = precios_ins[[t1, t2]].dropna()
            spread_1h_ins = (np.log(data_pair_ins[t1] / data_pair_ins[t1].iloc[0]) - beta * np.log(data_pair_ins[t2] / data_pair_ins[t2].iloc[0]))
            mu, sigma = spread_1h_ins.mean(), spread_1h_ins.std()

            data_pair_oos = precios_oos[[t1, t2]].dropna()
            if len(data_pair_oos) < 30:
                continue

            # Mismo ancla que mu/sigma (in-sample) -- antes esto salía
            # del OOS (data_pair_oos), un ancla distinta a la de
            # mu/sigma, dejando el spread operativo corrido por una
            # constante respecto a las bandas contra las que se
            # compara. Bug real, no algo que viniera de producción
            # (ESTADO_PAR.precio_x0/y0 en producción siempre son el
            # ancla del in-sample -- este desajuste era propio de esta
            # simulación del backtest).
            precio_x0, precio_y0 = float(data_pair_ins[t1].iloc[0]), float(data_pair_ins[t2].iloc[0])

            try:
                sigma_dia = float((spread_1h_ins.groupby(spread_1h_ins.index.date).mean() - mu).std())
            except Exception:
                sigma_dia = None

            fecha_inicio_ventana_ts = pd.Timestamp(v["oos_start"])
            # Igual que calcular_dias_desde_inicio_ventana en producción:
            # in-sample (periodo_in_sample_meses) + lo operativo -- NO el
            # train completo (24 meses). Pasarle el train entero hacía que
            # CUSUM quedara fuera de banda TODOS los días desde el arranque
            # (mal alimentado con 2 años de historia) y que ADF nunca
            # pudiera fallar (el ruido reciente se diluye en 24 meses de
            # cointegración genuina) -- bug real encontrado con la primera
            # corrida contra datos reales, no algo que hubiéramos visto en
            # los tests sintéticos (que sí usaban el rango correcto).
            fecha_inicio_recorte = pd.Timestamp(v["is_start"])
            precios_x_diario_full = (
                precios_diarios_mat[t1].dropna().loc[fecha_inicio_recorte:]
                if t1 in precios_diarios_mat.columns else pd.Series(dtype=float)
            )
            precios_y_diario_full = (
                precios_diarios_mat[t2].dropna().loc[fecha_inicio_recorte:]
                if t2 in precios_diarios_mat.columns else pd.Series(dtype=float)
            )

            fecha_forzada = None
            diag = None
            if not usar_control_diario:
                print(f"    [{pair}] control diario desactivado -- corre hasta vencimiento natural o stop VaR-τ")
            else:
                try:
                    fecha_forzada, diag = evaluar_cierre_forzado_por_dia(
                        precios_x_diario_full, precios_y_diario_full,
                        precios_ins[t1].dropna().combine_first(precios_oos[t1].dropna()) if t1 in precios_ins.columns else precios_oos[t1].dropna(),
                        precios_ins[t2].dropna().combine_first(precios_oos[t2].dropna()) if t2 in precios_ins.columns else precios_oos[t2].dropna(),
                        dias_oos, fecha_inicio_ventana_ts,
                        beta, mu, sigma_dia, n_dias_insample, precio_x0, precio_y0,
                        estabilidad_beta_dias_consecutivos, adf_dias_consecutivos, adf_pvalue_umbral, csw_lambda,
                        devolver_diagnostico=True, requerir_cusum=requerir_cusum,
                    )
                    if fecha_forzada is None:
                        print(f"    [{pair}] sin cierre forzado -- racha_cusum llegó a {diag['racha_cusum_max']}/{estabilidad_beta_dias_consecutivos}, "
                              f"csw se activó: {diag['csw_llego_a_activarse']}, racha_adf llegó a {diag['racha_adf_max']}/{adf_dias_consecutivos} "
                              f"(días evaluados: {diag['dias_evaluados']}, errores CUSUM/CSW/ADF: {diag['errores_cusum']}/{diag['errores_csw']}/{diag['errores_adf']})")
                    else:
                        print(f"    [{pair}] ⚠️  CIERRE FORZADO el {fecha_forzada} -- (evaluados {diag['dias_evaluados']} días de OOS antes de disparar)")
                        if diag["errores_cusum"] > 0 or diag["errores_csw"] > 0 or diag["errores_adf"] > 0:
                            print(f"    [{pair}] ⚠️  hubo errores durante la evaluación -- último CUSUM: {diag['ultimo_error_cusum']} | CSW: {diag['ultimo_error_csw']} | ADF: {diag['ultimo_error_adf']}")
                except Exception as e:
                    print(f"    [{pair}] control diario no se pudo evaluar ({e}) -- corre solo con stop VaR-tau")

            hl_media = pares_validos["half_life_dias"].mean()
            tau_calc = int(7 * hl_media) * 3

            sig = pair_trading_signals(
                x=data_pair_oos[t1], y=data_pair_oos[t2], b=beta,
                media_historica=mu, std_historica=sigma, Monto=monto_par,
                VaR_lvl_sup=mu + 2 * k * sigma, VaR_lvl_inf=mu - 2 * k * sigma,
                take_profit_th=199.9, stop_loss_th=-199.9,
                k=k, tau=tau_calc,
                forzar_cierre_desde=pd.Timestamp(fecha_forzada) if fecha_forzada is not None else None,
                x0=precio_x0, y0=precio_y0,
            )
            sig["pair"] = pair
            sig["Ventana"] = v["etiqueta"]
            sig["cierre_forzado_en"] = str(fecha_forzada) if fecha_forzada is not None else ""
            resultados_pares.append(sig)

        if not resultados_pares:
            print("  Ningún par corrió señales en esta ventana.")
            resultados_por_ventana.append({"Ventana": v["etiqueta"], "PNL": 0.0, "capital_inicio": capital_actual, "capital_fin": capital_actual, "n_pares": 0})
            continue

        df_ventana = pd.concat(resultados_pares)
        lista_res_oos.append(df_ventana)

        pnl_ventana = df_ventana["PNL"].sum()
        capital_nuevo = capital_actual + pnl_ventana
        n_forzados = df_ventana[df_ventana["Exit_Reason"] == "Control diario"]["pair"].nunique()
        print(f"  PNL ventana: {pnl_ventana:,.2f}  |  Capital {capital_actual:,.2f} -> {capital_nuevo:,.2f}  |  Pares con cierre forzado: {n_forzados}/{len(resultados_pares)}")

        resultados_por_ventana.append({
            "Ventana": v["etiqueta"], "PNL": pnl_ventana,
            "capital_inicio": capital_actual, "capital_fin": capital_nuevo,
            "n_pares": len(resultados_pares), "n_cierres_forzados": n_forzados,
        })
        capital_actual = capital_inicial if capital_fijo else capital_nuevo

    df_resumen_ventanas = pd.DataFrame(resultados_por_ventana)
    df_res_oos_total = pd.concat(lista_res_oos) if lista_res_oos else pd.DataFrame()
    df_pares_validos_total = pd.concat(lista_pares_validos, ignore_index=True) if lista_pares_validos else pd.DataFrame()

    return df_resumen_ventanas, df_res_oos_total, df_pares_validos_total
