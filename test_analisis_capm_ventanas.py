"""Test de analisis_capm_ventanas.py -- yf.download mockeado (el
sandbox no tiene acceso a la red de Yahoo Finance), con alpha/beta
VERDADEROS conocidos para confirmar que el modelo los recupera."""
import numpy as np
import pandas as pd
from unittest.mock import patch

import analisis_capm_ventanas as capm_mod
from analisis_capm_ventanas import descargar_datos_capm, calcular_capm, calcular_capm_por_ventana


def _armar_mercado_y_rf_sinteticos(semilla=0, n_dias=200):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2025-01-01", periods=n_dias)

    precio_mercado = 4000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_dias)))
    df_mercado_yf = pd.DataFrame({("Close", "^GSPC"): precio_mercado}, index=idx)
    df_mercado_yf.columns = pd.MultiIndex.from_tuples(df_mercado_yf.columns)  # MultiIndex -- un formato posible de yfinance

    precio_rf = 100 * np.exp(np.cumsum(rng.normal(0.00005, 0.0002, n_dias)))
    df_rf_yf = pd.DataFrame({"Close": precio_rf}, index=idx)  # columnas planas -- el otro formato posible

    return df_mercado_yf, df_rf_yf


def test_descargar_datos_capm_maneja_multiindex_y_columnas_planas():
    df_mercado_yf, df_rf_yf = _armar_mercado_y_rf_sinteticos()

    with patch.object(capm_mod.yf, "download", side_effect=lambda ticker, **kw: df_mercado_yf if ticker == "^GSPC" else df_rf_yf):
        df_mercado, df_rf = descargar_datos_capm("2025-01-01", "2025-12-01")

    assert list(df_mercado.columns) == ["Cierre"]
    assert list(df_rf.columns) == ["Cierre"]
    assert len(df_mercado) == 200 and len(df_rf) == 200
    print("OK  test_descargar_datos_capm_maneja_multiindex_y_columnas_planas")


def test_calcular_capm_por_ventana_recupera_beta_conocido():
    df_mercado_yf, df_rf_yf = _armar_mercado_y_rf_sinteticos(semilla=1)
    with patch.object(capm_mod.yf, "download", side_effect=lambda ticker, **kw: df_mercado_yf if ticker == "^GSPC" else df_rf_yf):
        df_mercado, df_rf = descargar_datos_capm("2025-01-01", "2025-12-01")

    rng = np.random.default_rng(1)
    alpha_real, beta_real = 0.0003, 1.2
    Rm = df_mercado["Cierre"].pct_change().dropna()
    Rf = df_rf["Cierre"].pct_change().dropna().reindex(Rm.index).fillna(0)
    ruido = rng.normal(0, 0.002, len(Rm))
    Ri = alpha_real + Rf + beta_real * (Rm - Rf) + ruido

    resumen = pd.concat([
        pd.DataFrame({"Retorno": Ri.iloc[:100], "Ventana": "V1"}, index=Ri.index[:100]),
        pd.DataFrame({"Retorno": Ri.iloc[100:], "Ventana": "V2"}, index=Ri.index[100:]),
    ])

    capm_por_ventana, datos_regresion = calcular_capm_por_ventana(resumen, df_mercado, df_rf)

    assert list(capm_por_ventana.index) == ["V1", "V2"]
    for v in ["V1", "V2"]:
        assert abs(capm_por_ventana.loc[v, "Beta"] - beta_real) < 0.15, f"{v}: beta recuperado lejos del real"
        assert capm_por_ventana.loc[v, "R2"] > 0.8, f"{v}: R2 muy bajo para datos casi sin ruido"
    assert set(datos_regresion.keys()) == {"V1", "V2"}
    print(f"OK  test_calcular_capm_por_ventana_recupera_beta_conocido (Beta V1={capm_por_ventana.loc['V1','Beta']:.3f}, V2={capm_por_ventana.loc['V2','Beta']:.3f})")


def test_yf_download_se_llama_una_sola_vez_por_ticker_no_por_ventana():
    """El punto central del diseño: descargar_datos_capm se llama UNA
    vez (afuera del loop de ventanas) -- confirma que solo hace 2
    llamadas a yf.download en total (mercado + rf), sin importar
    cuántas ventanas procese calcular_capm_por_ventana después."""
    df_mercado_yf, df_rf_yf = _armar_mercado_y_rf_sinteticos(semilla=2)
    llamadas = []

    def _mock_download(ticker, **kw):
        llamadas.append(ticker)
        return df_mercado_yf if ticker == "^GSPC" else df_rf_yf

    with patch.object(capm_mod.yf, "download", side_effect=_mock_download):
        df_mercado, df_rf = descargar_datos_capm("2025-01-01", "2025-12-01")

    rng = np.random.default_rng(2)
    Rm = df_mercado["Cierre"].pct_change().dropna()
    resumen = pd.concat([
        pd.DataFrame({"Retorno": rng.normal(0.0005, 0.01, 50), "Ventana": v}, index=Rm.index[i * 50:(i + 1) * 50])
        for i, v in enumerate(["V1", "V2", "V3"])
    ])
    # calcular_capm_por_ventana NO debe volver a llamar a yf.download -- ya tiene todo lo que necesita
    calcular_capm_por_ventana(resumen, df_mercado, df_rf)

    assert len(llamadas) == 2, f"debería haber exactamente 2 llamadas a yf.download (mercado + rf), hubo {len(llamadas)}"
    print("OK  test_yf_download_se_llama_una_sola_vez_por_ticker_no_por_ventana")


if __name__ == "__main__":
    test_descargar_datos_capm_maneja_multiindex_y_columnas_planas()
    test_calcular_capm_por_ventana_recupera_beta_conocido()
    test_yf_download_se_llama_una_sola_vez_por_ticker_no_por_ventana()
    print("\nTodos los tests pasaron correctamente.")
