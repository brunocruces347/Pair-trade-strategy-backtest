"""
CAPM por ventana -- descarga el mercado (^GSPC, el S&P 500 real, NO el
ticker SH que es su inverso -1x) y la tasa libre de riesgo UNA sola vez
(afuera del loop de ventanas), y recorta por fecha en memoria para
cada ventana -- mismo principio que cargar_daily/cargar_intraday en
backtest_walk_forward_v2.py: una sola descarga, muchos usos, en vez de
golpear a yfinance una vez por cada ventana (redundante -- los rangos
se solapan casi por completo entre ventanas consecutivas).

Uso típico:
    df_mercado, df_rf = descargar_datos_capm(fecha_inicio_total, fecha_fin_total)
    capm_por_ventana, datos_regresion = calcular_capm_por_ventana(df_resumen_ventanas_dia, df_mercado, df_rf)
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import yfinance as yf
import statsmodels.api as sm

TICKER_MERCADO_DEFAULT = "^GSPC"  # el S&P 500 real -- NO "SH" (ese es el inverso -1x, ProShares Short S&P500)
TICKER_RF_DEFAULT = "BIL"  # TODO: ajusta al ticker exacto del ETF treasury 3 meses que ya usas


def _extraer_cierre(df: pd.DataFrame) -> pd.Series:
    """yfinance a veces devuelve columnas MultiIndex incluso para un
    solo ticker (según versión) -- esto lo maneja sin asumir un
    formato fijo, para no repetir el tipo de sorpresa que ya tuvimos
    con otras partes de este proyecto (tz-aware/naive, date/datetime64)."""
    cierre = df["Close"] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
    if isinstance(cierre, pd.DataFrame):
        cierre = cierre.iloc[:, 0]
    return cierre.rename("Cierre")


def descargar_datos_capm(
    fecha_inicio: str, fecha_fin: str,
    ticker_mercado: str = TICKER_MERCADO_DEFAULT, ticker_rf: str = TICKER_RF_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UNA sola descarga para todo el rango que cubren tus ventanas --
    llamar esto UNA vez, antes del loop de ventanas, no adentro."""
    df_mercado_raw = yf.download(ticker_mercado, start=fecha_inicio, end=fecha_fin, interval="1d", auto_adjust=True, progress=False)
    df_rf_raw = yf.download(ticker_rf, start=fecha_inicio, end=fecha_fin, interval="1d", auto_adjust=True, progress=False)

    df_mercado = _extraer_cierre(df_mercado_raw).to_frame()
    df_rf = _extraer_cierre(df_rf_raw).to_frame()
    return df_mercado, df_rf


def calcular_capm(resumen: pd.DataFrame, df_mercado: pd.DataFrame, df_rf: pd.DataFrame):
    """Idéntico a tu calcular_capm original -- la regresión no cambia,
    solo de dónde vienen df_mercado/df_rf (ya vienen recortados por el
    .join contra el índice de `resumen`, no hace falta filtrar antes)."""
    mercado = pd.DataFrame({"Rm": df_mercado["Cierre"].pct_change()}, index=df_mercado.index)
    rf = pd.DataFrame({"Rf": df_rf["Cierre"].pct_change()}, index=df_rf.index)
    datos = resumen[["Retorno"]].rename(columns={"Retorno": "Ri"}).join(mercado).join(rf).dropna()
    datos["Exceso_i"] = datos["Ri"] - datos["Rf"]
    datos["Exceso_m"] = datos["Rm"] - datos["Rf"]
    modelo = sm.OLS(datos["Exceso_i"], sm.add_constant(datos["Exceso_m"])).fit()
    return modelo, datos


def calcular_capm_por_ventana(df_resumen_ventanas_dia: pd.DataFrame, df_mercado: pd.DataFrame, df_rf: pd.DataFrame):
    """Corre calcular_capm ventana por ventana, reusando los MISMOS
    df_mercado/df_rf ya descargados una sola vez -- ninguna llamada de
    red nueva acá adentro."""
    resultados_capm = []
    datos_regresion = {}

    for ventana, df_v in df_resumen_ventanas_dia.groupby("Ventana"):
        modelo, df_merged = calcular_capm(df_v, df_mercado, df_rf)
        resultados_capm.append({
            "Ventana": ventana,
            "Alpha": modelo.params["const"],
            "Beta": modelo.params["Exceso_m"],
            "R2": modelo.rsquared,
            "p_beta": modelo.pvalues["Exceso_m"],
        })
        datos_regresion[ventana] = {"x": df_merged["Exceso_m"].values, "y": df_merged["Exceso_i"].values, "modelo": modelo}

    capm_por_ventana = (
        pd.DataFrame(resultados_capm).set_index("Ventana")
        .sort_index(key=lambda idx: idx.str.extract(r"(\d+)")[0].astype(int))
    )
    return capm_por_ventana, datos_regresion


def construir_retorno_portafolio_por_dia(res_oos_subset: pd.DataFrame, resumen_fuente: pd.DataFrame) -> pd.DataFrame:
    """Agrega el PNL por DÍA CALENDARIO (no por barra horaria) -- res_
    oos_total está indexado por barra de 1h (ej. 09:30, 10:30...), así
    que unir directo contra mercado/RF (diarios, índice a las 00:00:00)
    por timestamp exacto nunca calzaba: el .join() daba 0 filas y el
    OLS explotaba con "zero-size array to reduction operation maximum"
    -- bug real, encontrado en la primera corrida contra datos reales.

    Retorno_t = PNL_t / Denominador_t, donde Denominador_t es la
    valorización TOTAL (suma de Valorizacion_Par de todas las filas,
    pares + cualquier overlay como una cobertura) del día anterior --
    NO capital_inicio fijo de toda la ventana (eso subestima el
    denominador real a medida que el capital se mueve dentro de la
    ventana, e infla el retorno estimado -- bug real, encontrado al
    calcular el CAPM con cobertura: el beta salía sistemáticamente más
    alto de lo que debía).

    EXCEPCIÓN -- el primer día de CADA ventana usa capital_inicio como
    ancla (no hay "valorización de ayer" disponible ahí -- ayer
    todavía era la ventana anterior, o no había operación). Esto NO es
    una regla distinta pegada a la lógica -- capital_inicio de la
    ventana N es EXACTAMENTE capital_fin de la ventana N-1 (así
    recicla capital correr_backtest_secuencial), así que anclar ahí es
    la continuación natural de "valorización t-1", solo que usando el
    número ya oficial de cierre en vez de encadenar series diarias que
    podrían tener un salto de calendario entre el fin de una ventana y
    el inicio de la siguiente.

    Sirve tanto para la CARTERA completa (pasarle todos los pares) como
    para UN par puntual (pasarle solo ese par ya filtrado) -- la
    agregación es la misma, solo cambia qué filas le pasas. Requiere
    que res_oos_subset tenga el índice llamado "Fecha" (así queda al
    cargar con guardar_cargar_resultados.cargar_resultados_backtest)."""
    df = res_oos_subset.reset_index()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()  # saca la hora, deja solo la fecha

    diario = df.groupby(["Fecha", "Ventana"]).agg(PNL=("PNL", "sum"), Valorizacion_Total=("Valorizacion_Par", "sum")).reset_index()
    diario = diario.merge(resumen_fuente[["Ventana", "capital_inicio"]], on="Ventana", how="left")
    diario = diario.sort_values(["Ventana", "Fecha"])

    diario["Valorizacion_t1"] = diario.groupby("Ventana")["Valorizacion_Total"].shift(1)
    diario["Denominador"] = diario["Valorizacion_t1"].fillna(diario["capital_inicio"])
    diario["Retorno"] = diario["PNL"] / diario["Denominador"]

    return diario.set_index("Fecha")[["Ventana", "Retorno"]]


# ============================================================
# Guardar/cargar -- para computar el CAPM UNA sola vez (con la única
# llamada de red a yfinance) y que el dashboard después solo lea de la
# DB, sin volver a golpear la red ni recalcular nada -- mismo
# principio que guardar_cargar_resultados.py para el resto del
# backtest.
# ============================================================

def guardar_datos_mercado_capm(db_path: str, df_mercado: pd.DataFrame, df_rf: pd.DataFrame):
    """Mercado/RF NO se etiquetan por fuente -- son los mismos datos
    de mercado sin importar qué variante del control diario se esté
    mirando, así que se guardan una sola vez para las 3."""
    con = sqlite3.connect(db_path)
    df_mercado.reset_index().rename(columns={df_mercado.index.name or "index": "Fecha"}).to_sql("mercado_capm", con, if_exists="replace", index=False)
    df_rf.reset_index().rename(columns={df_rf.index.name or "index": "Fecha"}).to_sql("rf_capm", con, if_exists="replace", index=False)
    con.close()
    print(f"Guardado en {db_path} -- mercado_capm: {len(df_mercado)} filas, rf_capm: {len(df_rf)} filas.")


def cargar_datos_mercado_capm(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    df_mercado = pd.read_sql("SELECT * FROM mercado_capm", con)
    df_rf = pd.read_sql("SELECT * FROM rf_capm", con)
    con.close()
    for df in (df_mercado, df_rf):
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df.set_index("Fecha", inplace=True)
    return df_mercado, df_rf


def guardar_capm_por_ventana(db_path: str, fuente: str, capm_por_ventana: pd.DataFrame):
    """Idempotente por fuente -- volver a guardar la misma fuente pisa,
    no acumula (mismo criterio que guardar_resultados_backtest)."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tablas_existentes = {fila[0] for fila in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "capm_por_ventana" in tablas_existentes:
        cur.execute("DELETE FROM capm_por_ventana WHERE fuente = ?", (fuente,))
        con.commit()

    df = capm_por_ventana.reset_index()
    df["fuente"] = fuente
    df.to_sql("capm_por_ventana", con, if_exists="append", index=False)
    con.close()


def guardar_capm_por_par(db_path: str, fuente: str, capm_por_par: pd.DataFrame):
    """capm_por_par -- una fila por combinación par+ventana, columnas
    Alpha/Beta/R2/p_beta/N -- idempotente por fuente, igual que arriba."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tablas_existentes = {fila[0] for fila in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "capm_por_par" in tablas_existentes:
        cur.execute("DELETE FROM capm_por_par WHERE fuente = ?", (fuente,))
        con.commit()

    df = capm_por_par.copy()
    df["fuente"] = fuente
    df.to_sql("capm_por_par", con, if_exists="append", index=False)
    con.close()


def cargar_capm_por_ventana(db_path: str, fuente: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM capm_por_ventana WHERE fuente = ?", con, params=[fuente])
    con.close()
    return df.set_index("Ventana").sort_index(key=lambda idx: idx.str.extract(r"(\d+)")[0].astype(int))


def cargar_capm_por_par(db_path: str, fuente: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM capm_por_par WHERE fuente = ?", con, params=[fuente])
    con.close()
    return df
