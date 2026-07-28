"""
Test de las piezas NUEVAS de backtest_walk_forward_v2.py (lo portado
tal cual del notebook original no se vuelve a probar acá -- ya estaba
validado en pairtrade_wfo_v3.ipynb). Cubre:
  1. generar_ventanas_secuenciales -- sin solape, sin huecos
  2. pair_trading_signals con forzar_cierre_desde -- corta exacto en
     la fecha, no abre posiciones nuevas después
  3. evaluar_cierre_forzado_por_dia -- dispara con un quiebre real de
     beta, no dispara sin quiebre (reusa CUSUM/CSW/ADF de producción)
"""
import numpy as np
import pandas as pd

from backtest_walk_forward_v2 import (
    generar_ventanas_secuenciales, pair_trading_signals, evaluar_cierre_forzado_por_dia,
)


def test_ventanas_secuenciales_sin_solape():
    ventanas = generar_ventanas_secuenciales("2020-01-01", "2026-07-01", train_meses=24, is_meses=6, oos_dias=90)
    assert len(ventanas) >= 2, "con 6+ años de rango debería generar al menos 2 ventanas"
    for i in range(len(ventanas) - 1):
        fin_actual = pd.Timestamp(ventanas[i]["oos_end"])
        inicio_siguiente = pd.Timestamp(ventanas[i + 1]["train_start"])
        assert inicio_siguiente == fin_actual + pd.DateOffset(days=1), \
            f"hueco o solape entre {ventanas[i]['etiqueta']} y {ventanas[i+1]['etiqueta']}"
    print(f"OK  test_ventanas_secuenciales_sin_solape ({len(ventanas)} ventanas)")


def test_ventana_no_cabe_no_se_genera():
    """Si el rango es más corto que train+is+oos, no debe generar nada
    (no una ventana parcial/incompleta)."""
    ventanas = generar_ventanas_secuenciales("2026-01-01", "2026-06-01", train_meses=24, is_meses=6, oos_dias=90)
    assert ventanas == []
    print("OK  test_ventana_no_cabe_no_se_genera")


def test_forzar_cierre_desde_corta_exacto_y_bloquea_aperturas():
    n = 100
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    mu, sigma = 0.0, 0.01
    x = pd.Series(100 * np.exp(np.cumsum(np.full(n, 0.002))), index=idx)  # nunca revierte solo
    y = pd.Series(np.full(n, 50.0), index=idx)

    sin_corte = pair_trading_signals(
        x=x, y=y, b=1.0, media_historica=mu, std_historica=sigma, Monto=1000,
        VaR_lvl_sup=mu + 2 * sigma, VaR_lvl_inf=mu - 2 * sigma,
        take_profit_th=199.9, stop_loss_th=-199.9, k=1.0, tau=1000,
    )
    assert len(sin_corte) == n, "sin corte forzado y sin reversión, debería correr toda la serie"

    fecha_corte = idx[30]
    con_corte = pair_trading_signals(
        x=x, y=y, b=1.0, media_historica=mu, std_historica=sigma, Monto=1000,
        VaR_lvl_sup=mu + 2 * sigma, VaR_lvl_inf=mu - 2 * sigma,
        take_profit_th=199.9, stop_loss_th=-199.9, k=1.0, tau=1000,
        forzar_cierre_desde=fecha_corte,
    )
    cierres = con_corte[con_corte["Exit_Reason"] == "Control diario"]
    assert len(cierres) == 1
    assert cierres.index[0] == fecha_corte

    aperturas_despues = con_corte.loc[fecha_corte:][con_corte.loc[fecha_corte:]["signal"].isin(["Buy", "Sell"])]
    assert len(aperturas_despues) == 0, "no debe abrir posiciones nuevas después del corte"
    print("OK  test_forzar_cierre_desde_corta_exacto_y_bloquea_aperturas")


def _generar_precios_sinteticos(n_dias, semilla, quiebre_desde_dia=None, beta_quiebre=None, beta=0.8):
    """Mismo generador usado para calibrar lambda de CSW -- 7 barras/día
    hábil, con o sin quiebre real de beta a mitad de camino."""
    rng = np.random.default_rng(semilla)
    dias = pd.bdate_range("2026-01-01", periods=n_dias)
    horas = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    idx = pd.DatetimeIndex(sorted(pd.Timestamp(f"{d.date()} {h}") for d in dias for h in horas))
    n = len(idx)
    log_y = np.cumsum(rng.normal(0, 0.005, n))
    beta_usado = np.full(n, beta)
    if quiebre_desde_dia is not None:
        beta_usado[quiebre_desde_dia * 7:] = beta_quiebre
    ruido = rng.normal(0, 0.01, n)
    log_x = beta_usado * log_y + ruido
    precio_x = pd.Series(np.exp(log_x) * 100, index=idx)
    precio_y = pd.Series(np.exp(log_y) * 50, index=idx)
    return precio_x, precio_y


def test_control_diario_dispara_con_quiebre_real():
    precio_x, precio_y = _generar_precios_sinteticos(120, semilla=1, quiebre_desde_dia=60, beta_quiebre=1.4)
    precio_x_diario = precio_x.resample("D").last().dropna()
    precio_y_diario = precio_y.resample("D").last().dropna()

    fecha_inicio_ventana = precio_x_diario.index[40]
    dias_operativos = list(precio_x_diario.index[40:])
    primer_bar = precio_x.loc[precio_x.index >= fecha_inicio_ventana].index[0]
    mu = float((np.log(precio_x.loc[:fecha_inicio_ventana] / precio_x.iloc[0])
                - 0.8 * np.log(precio_y.loc[:fecha_inicio_ventana] / precio_y.iloc[0])).mean())

    resultado = evaluar_cierre_forzado_por_dia(
        precio_x_diario, precio_y_diario, precio_x, precio_y,
        dias_operativos, fecha_inicio_ventana,
        beta=0.8, mu=mu, sigma_dia=0.01, n_dias_insample=40,
        precio_x0=float(precio_x.loc[primer_bar]), precio_y0=float(precio_y.loc[primer_bar]),
    )
    assert resultado is not None, "con quiebre real de beta, debería forzar el cierre en algún punto"
    print(f"OK  test_control_diario_dispara_con_quiebre_real (cierre en {resultado})")


def test_control_diario_no_dispara_sin_quiebre():
    precio_x, precio_y = _generar_precios_sinteticos(120, semilla=2, quiebre_desde_dia=None)
    precio_x_diario = precio_x.resample("D").last().dropna()
    precio_y_diario = precio_y.resample("D").last().dropna()

    fecha_inicio_ventana = precio_x_diario.index[40]
    dias_operativos = list(precio_x_diario.index[40:])
    primer_bar = precio_x.loc[precio_x.index >= fecha_inicio_ventana].index[0]
    mu = float((np.log(precio_x.loc[:fecha_inicio_ventana] / precio_x.iloc[0])
                - 0.8 * np.log(precio_y.loc[:fecha_inicio_ventana] / precio_y.iloc[0])).mean())

    resultado = evaluar_cierre_forzado_por_dia(
        precio_x_diario, precio_y_diario, precio_x, precio_y,
        dias_operativos, fecha_inicio_ventana,
        beta=0.8, mu=mu, sigma_dia=0.01, n_dias_insample=40,
        precio_x0=float(precio_x.loc[primer_bar]), precio_y0=float(precio_y.loc[primer_bar]),
    )
    assert resultado is None, "sin quiebre, no debería forzar el cierre"
    print("OK  test_control_diario_no_dispara_sin_quiebre")


if __name__ == "__main__":
    test_ventanas_secuenciales_sin_solape()
    test_ventana_no_cabe_no_se_genera()
    test_forzar_cierre_desde_corta_exacto_y_bloquea_aperturas()
    test_control_diario_dispara_con_quiebre_real()
    test_control_diario_no_dispara_sin_quiebre()
    print("\nTodos los tests pasaron correctamente.")
