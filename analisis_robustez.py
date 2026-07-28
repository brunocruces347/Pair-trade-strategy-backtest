"""
Sección de análisis de robustez -- portada de pairtrade_wfo_v3.ipynb
(secciones 4 y 5), adaptada para trabajar con la salida de
correr_backtest_secuencial() en vez de la del loop walk-forward
solapado original. La pregunta que responde esta sección, tal como
la planteaba el notebook original:

    ¿Por qué debería creer que este proceso seguirá funcionando?

No alcanza con ver que el backtest dio positivo -- hace falta
evidencia de que el MECANISMO DE SELECCIÓN tiene poder predictivo
genuino, y de que el resultado no depende de una sola ventana con
suerte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# numpy >= 2.0 renombró trapz -> trapezoid (y en versiones más nuevas
# quitó trapz por completo) -- esto cubre ambos casos, sin asumir qué
# versión tiene tu entorno.
_trapecio = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# 4. Resumen por par y ventana -- portado de la celda 31 del
#    notebook original, adaptado a las columnas que ya trae
#    df_res_oos_total/df_pares_validos_total desde
#    correr_backtest_secuencial (mismos nombres, sin cambios).
# ============================================================

def construir_resumen_oos(df_res_oos_total: pd.DataFrame, df_pares_validos_total: pd.DataFrame) -> pd.DataFrame:
    """Una fila por par-ventana, con el resultado OOS y los atributos
    del selector (score_selector, pvalue, half_life, etc.) -- lo que
    necesita el resto de esta sección para relacionar "qué tan bueno
    parecía el par en el train" con "qué tan bien le fue en el OOS"."""
    resumen = (
        df_res_oos_total
        .assign(Fecha=df_res_oos_total.index)
        .groupby(["pair", "Ventana"])
        .agg(
            Fecha_Inicio=("Fecha", "first"), Fecha_Fin=("Fecha", "last"),
            beta=("beta", "first"),
            Valorizacion_Par_Inicial=("Valorizacion_Par", "first"),
            Valorizacion_Par_Final=("Valorizacion_Par", "last"),
            PNL_Acumulado=("PNL_Acumulado", "last"),
            Retorno_Acumulado=("Retorno_Acumulado", "last"),
        )
        .sort_values(["Ventana", "pair"])
        .reset_index()
    )

    cols_selector = [c for c in [
        "pair", "Ventana", "pvalue", "pvalue_30", "Hurst", "half_life_dias",
        "diferencia_beta_70_30", "p_value_chow_test", "beta_70", "beta_30", "score_selector",
    ] if c in df_pares_validos_total.columns]

    resumen = resumen.merge(
        df_pares_validos_total[cols_selector].drop_duplicates(["pair", "Ventana"]),
        on=["pair", "Ventana"], how="left",
    )
    return resumen.assign(
        Ventana_num=lambda x: x["Ventana"].str.extract(r"(\d+)")[0].astype(int)
    ).sort_values("Ventana_num").drop(columns="Ventana_num").reset_index(drop=True)


# ============================================================
# 5.1 Rank IC -- ¿el score_selector (calculado SOLO con
#     información de train) predice el PNL OOS?
# ============================================================

def calcular_rank_ic(resumen_oos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Spearman(-score_selector, PNL_OOS) por ventana -- score MENOR
    es mejor candidato (más cointegrado, revierte más rápido), por eso
    se invierte el signo antes de correlacionar. IC > 0 consistente a
    través de ventanas es la evidencia que buscamos; un IC que cambia
    de signo ventana a ventana sugiere que el selector no tiene poder
    predictivo genuino, aunque el backtest global haya dado positivo."""
    filas = []
    for ventana, df_v in resumen_oos.groupby("Ventana"):
        df_v = df_v.dropna(subset=["score_selector", "PNL_Acumulado"])
        if len(df_v) < 3:
            filas.append({"Ventana": ventana, "RankIC": np.nan, "p_value": np.nan, "N": len(df_v)})
            continue
        ic, pval = spearmanr(-df_v["score_selector"], df_v["PNL_Acumulado"])
        filas.append({"Ventana": ventana, "RankIC": ic, "p_value": pval, "N": len(df_v), "IC_sig": pval < 0.05})

    df_rank_ic = (
        pd.DataFrame(filas)
        .assign(Ventana_num=lambda x: x["Ventana"].str.extract(r"(\d+)")[0].astype(int))
        .sort_values("Ventana_num").drop(columns="Ventana_num").set_index("Ventana")
    )

    ic_vals = df_rank_ic["RankIC"].dropna()
    ic_medio = float(ic_vals.mean()) if len(ic_vals) else np.nan
    ic_t_stat = (ic_medio / (ic_vals.std() / np.sqrt(len(ic_vals)))) if len(ic_vals) > 1 else np.nan

    resumen_stats = {
        "ic_medio": ic_medio, "ic_t_stat": ic_t_stat,
        "pct_ventanas_ic_positivo": float((ic_vals > 0).mean()) if len(ic_vals) else np.nan,
        "pct_ventanas_ic_significativo": float(df_rank_ic["IC_sig"].mean()) if "IC_sig" in df_rank_ic else np.nan,
    }
    return df_rank_ic, resumen_stats


def imprimir_rank_ic(df_rank_ic: pd.DataFrame, stats: dict):
    print(df_rank_ic.round(3))
    print(f"\nIC medio:              {stats['ic_medio']:.3f}")
    print(f"t-stat IC medio:       {stats['ic_t_stat']:.2f}  (> 1.65 -> significativo al 10%)")
    print(f"% ventanas con IC > 0: {stats['pct_ventanas_ic_positivo']:.1%}")
    print(f"% ventanas IC signif.: {stats['pct_ventanas_ic_significativo']:.1%}")
    print("\nReferencia: IC > 0.02 bajo pero presente | > 0.05 útil | > 0.10 notable")


# ============================================================
# 5.2 Monotonicidad por cuartiles -- si el selector funciona,
#     el cuartil de MEJOR score (Q1) debería ganarle sistemáticamente
#     al de PEOR score (Q4), no solo en promedio pooled.
# ============================================================

def analizar_cuartiles(resumen_oos: pd.DataFrame) -> pd.DataFrame:
    df_pool = resumen_oos.dropna(subset=["score_selector", "PNL_Acumulado"]).copy()
    if df_pool.empty:
        return pd.DataFrame()

    df_pool["score_rank_pct"] = df_pool.groupby("Ventana")["score_selector"].transform(lambda x: x.rank(pct=True))
    df_pool["Cuartil"] = df_pool.groupby("Ventana")["score_rank_pct"].transform(
        lambda x: pd.cut(x, bins=[0.0, 0.25, 0.5, 0.75, 1.0], labels=["Q1 (mejor)", "Q2", "Q3", "Q4 (peor)"], include_lowest=True)
    )
    return (
        df_pool.groupby("Cuartil", observed=True)["PNL_Acumulado"]
        .agg(N="count", PNL_Medio="mean", PNL_Mediana="median", PNL_Total="sum", HitRate=lambda x: (x > 0).mean())
        .reset_index()
    )


def imprimir_cuartiles(stats_cuartil: pd.DataFrame):
    if stats_cuartil.empty:
        print("No hay suficientes datos con score_selector para el análisis de cuartiles.")
        return
    print(stats_cuartil.round(3))
    pnl_q = stats_cuartil["PNL_Medio"].values
    es_monot = all(pnl_q[i] > pnl_q[i + 1] for i in range(len(pnl_q) - 1))
    print(f"\n¿PNL monótonamente decreciente Q1 -> Q4? {'Sí' if es_monot else 'No'}")
    print(f"PNL medio por cuartil: {[round(v, 3) for v in pnl_q]}")


# ============================================================
# 5.3 Concentración del PNL -- ¿el resultado depende de 1-2
#     ventanas con suerte, o está repartido?
# ============================================================

def analizar_concentracion_pnl(resumen_oos: pd.DataFrame) -> dict:
    pnl_v = (
        resumen_oos.groupby("Ventana")["PNL_Acumulado"].sum()
        .sort_index(key=lambda idx: idx.str.extract(r"(\d+)")[0].astype(int))
    )
    pnl_total = float(pnl_v.sum())
    pnl_pos = pnl_v[pnl_v > 0].sort_values(ascending=False)

    if len(pnl_pos) > 0:
        lorenz_y = pnl_pos.cumsum() / pnl_pos.sum()
        lorenz_x = np.arange(1, len(pnl_pos) + 1) / len(pnl_pos)
        gini = float(1 - 2 * _trapecio(lorenz_y, lorenz_x))
        n_80pct = int((lorenz_y < 0.80).sum() + 1)
    else:
        gini, n_80pct = np.nan, 0

    return {
        "pnl_por_ventana": pnl_v, "pnl_total": pnl_total,
        "n_ventanas_positivas": int((pnl_v > 0).sum()), "n_ventanas": len(pnl_v),
        "gini_pnl_positivo": gini, "n_ventanas_80pct_del_pnl": n_80pct,
    }


def leave_one_out_por_ventana(resumen_oos: pd.DataFrame) -> pd.DataFrame:
    """¿El PNL total sigue siendo positivo si sacamos CUALQUIER
    ventana individual? Si no, el resultado depende de una ventana
    puntual -- frágil, no importa cuán bueno se vea el total."""
    pnl_v = resumen_oos.groupby("Ventana")["PNL_Acumulado"].sum()
    pnl_total = pnl_v.sum()

    filas = []
    for ventana in sorted(pnl_v.index, key=lambda v: int(v.replace("V", ""))):
        pnl_sin = pnl_v.drop(ventana).sum()
        impacto = pnl_total - pnl_sin
        filas.append({
            "Ventana_eliminada": ventana, "PNL_sin": pnl_sin, "Impacto": impacto,
            "Impacto_pct": impacto / abs(pnl_total) if pnl_total != 0 else np.nan,
            "Sigue_positivo": pnl_sin > 0,
        })
    return pd.DataFrame(filas).set_index("Ventana_eliminada")


def imprimir_leave_one_out(df_loo: pd.DataFrame):
    print(df_loo.round(2))
    n_pos_sin = df_loo["Sigue_positivo"].sum()
    total = len(df_loo)
    print(f"\nPNL positivo al eliminar cualquier ventana: {n_pos_sin}/{total}")
    if n_pos_sin == total:
        print("Robusto: no depende de ninguna ventana individual.")
    elif n_pos_sin >= total * 0.75:
        print("Mayoritariamente robusto, con pocas ventanas de alto impacto.")
    else:
        print("Frágil: una o más ventanas son críticas para el PNL positivo.")


# ============================================================
# 5.4 Jaccard -- ROTACIÓN de la cartera entre ventanas
#     consecutivas. Puramente DESCRIPTIVO -- no es evidencia a
#     favor ni en contra de invertir (si la política es reconstruir
#     cada ~3 meses, un Jaccard bajo es adaptación al mercado, no un
#     problema).
# ============================================================

def calcular_jaccard_ventanas(df_pares_validos_total: pd.DataFrame) -> pd.DataFrame:
    ventanas_j = (
        df_pares_validos_total[["Ventana", "pair"]]
        .assign(Ventana_num=lambda x: x["Ventana"].str.extract(r"(\d+)")[0].astype(int))
        .sort_values("Ventana_num")
    )
    vids = sorted(ventanas_j["Ventana_num"].unique())
    filas = []
    for i in range(len(vids) - 1):
        vn, vnp1 = f"V{vids[i]}", f"V{vids[i+1]}"
        A = set(ventanas_j[ventanas_j["Ventana"] == vn]["pair"])
        B = set(ventanas_j[ventanas_j["Ventana"] == vnp1]["pair"])
        j = len(A & B) / len(A | B) if (A | B) else np.nan
        filas.append({"Transición": f"{vn}->{vnp1}", "N_Vn": len(A), "N_Vnp1": len(B), "Comunes": len(A & B), "Jaccard": j})
    return pd.DataFrame(filas)


# ============================================================
# 9. Bootstrap global -- ¿el PNL total es distinguible de ruido?
# ============================================================

def bootstrap_pnl_total(df_res_oos_total: pd.DataFrame, n_iter: int = 10_000, semilla: int = 0) -> dict:
    """Remuestrea los trades (no las barras -- Deal_ID+pair agrupado)
    con reemplazo n_iter veces, para estimar la distribución del PNL
    total bajo la hipótesis de que el orden/composición de trades es
    intercambiable -- y ver si el PNL real cae en una zona plausible
    de esa distribución o se distingue claramente de ella."""
    rng = np.random.default_rng(semilla)
    trades = (
        # dropna=False -- por defecto pandas DESCARTA silenciosamente
        # las filas donde la clave de agrupación es NaN (las barras sin
        # trade activo, Deal_ID=NaN, que igual acumulan PNL real por la
        # posición base) -- eso hacía que el PNL del bootstrap no
        # coincidiera con el PNL real de la ventana. Bug real, encontrado
        # comparando el "PNL real" del bootstrap contra el PNL de
        # correr_backtest_secuencial en la primera corrida completa.
        df_res_oos_total.groupby(["Deal_ID", "pair", "Ventana"], dropna=False)["PNL"].sum().reset_index()
    )
    pnl_trades = trades["PNL"].to_numpy()
    if len(pnl_trades) == 0:
        raise ValueError("No hay trades para bootstrapear -- revisa df_res_oos_total.")

    pnl_total_real = float(df_res_oos_total["PNL"].sum())
    if not np.isclose(pnl_trades.sum(), pnl_total_real, atol=0.01):
        raise AssertionError(
            f"El PNL agrupado ({pnl_trades.sum():.2f}) no coincide con el PNL real total "
            f"({pnl_total_real:.2f}) -- revisa si groupby está descartando filas (ej. NaN en la clave)."
        )

    resultados = np.array([rng.choice(pnl_trades, size=len(pnl_trades), replace=True).sum() for _ in range(n_iter)])
    pnl_real = float(pnl_trades.sum())

    return {
        "pnl_real": pnl_real, "n_trades": len(pnl_trades),
        "media_bootstrap": float(resultados.mean()),
        "prob_pnl_negativo": float((resultados < 0).mean()),
        "ic_95": tuple(np.percentile(resultados, [2.5, 97.5])),
        "percentiles": dict(zip(["P1", "P5", "P50", "P95", "P99"], np.percentile(resultados, [1, 5, 50, 95, 99]))),
        "distribucion": resultados,
    }


def imprimir_bootstrap(resultado: dict):
    print(f"N° de trades:          {resultado['n_trades']}")
    print(f"PNL real:              {resultado['pnl_real']:.2f}")
    print(f"Media bootstrap:       {resultado['media_bootstrap']:.2f}")
    print(f"Prob(PNL < 0):         {resultado['prob_pnl_negativo']:.2%}")
    print(f"IC 95%:                [{resultado['ic_95'][0]:.2f}, {resultado['ic_95'][1]:.2f}]")
    for p, v in resultado["percentiles"].items():
        print(f"  {p}: {v:.2f}")
    if resultado["n_trades"] < 30:
        print("\n⚠️  Menos de 30 trades -- el bootstrap con tan pocas observaciones da un IC")
        print("    poco confiable (el propio remuestreo reusa las mismas pocas observaciones).")


# ============================================================
# PNL esperado POR VENTANA -- distinto de bootstrap_pnl_total (que
# remuestrea TRADES para dar un total). Acá la unidad de remuestreo es
# la VENTANA -- responde "¿cuánto esperaría ganar/perder, en promedio,
# en una ventana típica?", no "¿el total acumulado es ruido?".
# ============================================================

def bootstrap_pnl_esperado_por_ventana(resumen_ventanas: pd.DataFrame, n_iter: int = 10_000, semilla: int = 0) -> dict:
    """Remuestrea con reemplazo la lista de PNL por ventana (una
    observación = una ventana completa, no un trade) -- para estimar
    un intervalo de confianza no paramétrico alrededor del PNL
    promedio esperado por ventana.

    OJO -- usar solo con ventanas SECUENCIALES (generar_ventanas_
    secuenciales), nunca con las solapadas (generar_ventanas_
    solapadas): ventanas que comparten calendario entre sí no son
    observaciones independientes, y meterlas acá daría un intervalo
    de confianza falsamente angosto (más confianza de la que
    corresponde)."""
    pnl_ventanas = resumen_ventanas["PNL"].to_numpy()
    n = len(pnl_ventanas)
    if n == 0:
        raise ValueError("resumen_ventanas está vacío -- no hay nada que bootstrapear.")

    rng = np.random.default_rng(semilla)
    medias_bootstrap = np.array([rng.choice(pnl_ventanas, size=n, replace=True).mean() for _ in range(n_iter)])

    return {
        "n_ventanas": n,
        "pnl_medio_real": float(pnl_ventanas.mean()),
        "pnl_mediana_real": float(np.median(pnl_ventanas)),
        "media_bootstrap": float(medias_bootstrap.mean()),
        "prob_pnl_medio_negativo": float((medias_bootstrap < 0).mean()),
        "ic_95": tuple(np.percentile(medias_bootstrap, [2.5, 97.5])),
        "percentiles": dict(zip(["P5", "P25", "P50", "P75", "P95"], np.percentile(medias_bootstrap, [5, 25, 50, 75, 95]))),
        "distribucion": medias_bootstrap,
    }


def imprimir_bootstrap_por_ventana(resultado: dict):
    print(f"N° de ventanas:              {resultado['n_ventanas']}")
    print(f"PNL medio real (las {resultado['n_ventanas']} ventanas): {resultado['pnl_medio_real']:.2f}")
    print(f"PNL mediana real:            {resultado['pnl_mediana_real']:.2f}")
    print(f"Media bootstrap:              {resultado['media_bootstrap']:.2f}")
    print(f"Prob(PNL medio < 0):          {resultado['prob_pnl_medio_negativo']:.2%}")
    print(f"IC 95% del PNL esperado:      [{resultado['ic_95'][0]:.2f}, {resultado['ic_95'][1]:.2f}]")
    for p, v in resultado["percentiles"].items():
        print(f"  {p}: {v:.2f}")
    if resultado["n_ventanas"] < 10:
        print(f"\n⚠️  Solo {resultado['n_ventanas']} ventanas -- el bootstrap remuestrea de un conjunto muy chico,")
        print("    el intervalo va a ser ancho y algo escalonado (combinaciones limitadas), no una")
        print("    distribución suave. Mejor que nada, pero no esperes precisión con esta muestra.")


# ============================================================
# Comparación visual simple de las 3 variantes del control diario --
# UN gráfico, 3 líneas, sin dashboard/dropdown. fig.show() abre en el
# navegador directo, no necesita levantar ningún server.
# ============================================================

def graficar_comparacion_capital(
    resumenes: dict[str, "pd.DataFrame"], capital_inicial: float,
) -> "go.Figure":
    """resumenes -- dict {nombre_variante: df_resumen_ventanas}, ej.:
        {"CUSUM+CSW+ADF": resumen_completo, "Solo CSW+ADF": resumen_sin_cusum, "Sin control": resumen_sin}
    Cada df_resumen_ventanas es el que devuelve correr_backtest_secuencial
    -- necesita las columnas "Ventana" y "capital_fin", en orden
    cronológico (ya vienen así)."""
    import plotly.graph_objects as go

    colores = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = go.Figure()

    for i, (nombre, resumen) in enumerate(resumenes.items()):
        x = ["Inicio"] + list(resumen["Ventana"])
        y = [capital_inicial] + list(resumen["capital_fin"])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=nombre,
            line=dict(color=colores[i % len(colores)], width=2),
        ))

    fig.add_hline(y=capital_inicial, line=dict(color="gray", dash="dot"), annotation_text="capital inicial")
    fig.update_layout(
        title="Capital acumulado por ventana -- comparación de variantes del control diario",
        xaxis_title="Ventana", yaxis_title="Capital ($)",
        height=550, hovermode="x unified",
    )
    return fig


# ============================================================
# Bootstrap PAREADO -- ¿la diferencia entre 2 fuentes es distinguible
# de cero? Mucho más potente que comparar 2 IC independientes (como
# bootstrap_pnl_esperado_por_ventana por separado en cada fuente),
# porque cancela la varianza COMPARTIDA entre corridas -- mismas
# ventanas, mismos pares subyacentes (solo cambia si/cuándo se cierra
# forzado), así que gran parte del ruido de mercado es común a las 2
# y se resta al hacer la diferencia ventana a ventana.
# ============================================================

def bootstrap_diferencia_entre_fuentes(
    resumen_a: pd.DataFrame, resumen_b: pd.DataFrame,
    nombre_a: str = "A", nombre_b: str = "B", n_iter: int = 10_000, semilla: int = 0,
) -> dict:
    """Requiere que ambas fuentes compartan ventanas (mismo texto en
    "Ventana") para poder emparejar -- si alguna ventana existe en una
    fuente y no en la otra, se descarta de la comparación."""
    a = resumen_a.set_index("Ventana")["PNL"]
    b = resumen_b.set_index("Ventana")["PNL"]

    ventanas_comunes = sorted(set(a.index) & set(b.index), key=lambda v: int("".join(c for c in v if c.isdigit()) or 0))
    if not ventanas_comunes:
        raise ValueError("Las 2 fuentes no comparten ninguna ventana -- no se puede emparejar (¿vienen de rangos de fecha distintos?).")

    diferencias = (a.loc[ventanas_comunes] - b.loc[ventanas_comunes]).to_numpy()
    n = len(diferencias)

    rng = np.random.default_rng(semilla)
    medias_bootstrap = np.array([rng.choice(diferencias, size=n, replace=True).mean() for _ in range(n_iter)])

    return {
        "nombre_a": nombre_a, "nombre_b": nombre_b,
        "n_ventanas": n, "ventanas": ventanas_comunes, "diferencias": diferencias,
        "diferencia_media_real": float(diferencias.mean()),
        "media_bootstrap": float(medias_bootstrap.mean()),
        "prob_diferencia_menor_o_igual_a_cero": float((medias_bootstrap <= 0).mean()),
        "ic_95": tuple(np.percentile(medias_bootstrap, [2.5, 97.5])),
        "percentiles": dict(zip(["P5", "P25", "P50", "P75", "P95"], np.percentile(medias_bootstrap, [5, 25, 50, 75, 95]))),
        "distribucion": medias_bootstrap,
    }


def imprimir_bootstrap_diferencia(resultado: dict):
    print(f"Comparando: {resultado['nombre_a']}  vs.  {resultado['nombre_b']}")
    print(f"N° de ventanas emparejadas: {resultado['n_ventanas']}")
    print(f"\nDiferencia por ventana ({resultado['nombre_a']} - {resultado['nombre_b']}):")
    for v, d in zip(resultado["ventanas"], resultado["diferencias"]):
        print(f"  {v}: {d:+.2f}")

    print(f"\nDiferencia media real:     {resultado['diferencia_media_real']:+.2f}")
    print(f"Media bootstrap:           {resultado['media_bootstrap']:+.2f}")
    print(f"Prob(diferencia <= 0):     {resultado['prob_diferencia_menor_o_igual_a_cero']:.2%}")
    print(f"IC 95% de la diferencia:   [{resultado['ic_95'][0]:+.2f}, {resultado['ic_95'][1]:+.2f}]")
    for p, v in resultado["percentiles"].items():
        print(f"  {p}: {v:+.2f}")

    if resultado["ic_95"][0] > 0:
        print(f"\n-> El IC 95% NO cruza cero -- {resultado['nombre_a']} es significativamente MEJOR que {resultado['nombre_b']} (al 95%).")
    elif resultado["ic_95"][1] < 0:
        print(f"\n-> El IC 95% NO cruza cero -- {resultado['nombre_a']} es significativamente PEOR que {resultado['nombre_b']} (al 95%).")
    else:
        print("\n-> El IC 95% cruza cero -- la diferencia NO es estadísticamente significativa con esta muestra.")
    if resultado["n_ventanas"] < 10:
        print(f"\n⚠️  Solo {resultado['n_ventanas']} ventanas emparejadas -- interpretar con cautela,")
        print("    igual que el resto de los bootstraps con muestra chica de este proyecto.")
