"""Test de descargar_precios_yf.py -- yf.download mockeado (el sandbox
no tiene acceso a la red de Yahoo Finance), con estructura MultiIndex
real (group_by='ticker')."""
import numpy as np
import pandas as pd
from unittest.mock import patch

import descargar_precios_yf as dpy


def _armar_mock_diario(tickers, semilla=1, n_dias=100):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2021-07-01", periods=n_dias)
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    datos = pd.DataFrame(rng.normal(100, 5, (n_dias, len(cols))), index=idx, columns=cols)
    datos.index.name = "Date"
    return datos


def _armar_mock_intraday(tickers, semilla=2, n_barras=50):
    rng = np.random.default_rng(semilla)
    idx = pd.date_range("2025-01-02 09:30", periods=n_barras, freq="h", tz="America/New_York")
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    datos = pd.DataFrame(rng.normal(100, 5, (n_barras, len(cols))), index=idx, columns=cols)
    datos.index.name = "Datetime"
    return datos


def test_descargar_precios_diarios_yf_formato_correcto():
    tickers = ["AAA", "BBB", "CCC"]
    mock_diario = _armar_mock_diario(tickers)
    with patch.object(dpy.yf, "download", return_value=mock_diario):
        df = dpy.descargar_precios_diarios_yf(tickers, period="5y")

    assert list(df.columns) == ["Fecha", "Ticker", "Close", "Volume"]
    assert set(df["Ticker"].unique()) == set(tickers)
    assert len(df) == 100 * 3
    print("OK  test_descargar_precios_diarios_yf_formato_correcto")


def test_descargar_precios_intraday_yf_sin_timezone_y_filtrado_a_horario_mercado():
    tickers = ["AAA", "BBB"]
    mock_intraday = _armar_mock_intraday(tickers)
    with patch.object(dpy.yf, "download", return_value=mock_intraday):
        df = dpy.descargar_precios_intraday_yf(tickers, period="730d")

    assert list(df.columns) == ["Fecha", "Dia", "Hora", "Ticker", "Close", "Volume"]
    assert df["Fecha"].dt.tz is None, "la Fecha debe quedar sin timezone (naive), igual que cargar_intraday"
    assert df["Hora"].between("09:30:00", "16:30:00").all()
    print("OK  test_descargar_precios_intraday_yf_sin_timezone_y_filtrado_a_horario_mercado")


def test_ticker_faltante_en_la_descarga_se_salta_sin_romper():
    """Simula un ticker delistado/con error puntual -- yf.download a
    veces simplemente no trae esa columna en el resultado. No debe
    romper el resto de la descarga."""
    tickers_pedidos = ["AAA", "BBB", "FALTA"]
    mock_diario = _armar_mock_diario(["AAA", "BBB"])  # "FALTA" nunca aparece en el resultado

    with patch.object(dpy.yf, "download", return_value=mock_diario):
        df = dpy.descargar_precios_diarios_yf(tickers_pedidos, period="5y")

    assert set(df["Ticker"].unique()) == {"AAA", "BBB"}, "debe traer los que sí están, sin romper por el que falta"
    print("OK  test_ticker_faltante_en_la_descarga_se_salta_sin_romper")


def test_ningun_ticker_disponible_lanza_error_claro():
    with patch.object(dpy.yf, "download", return_value=pd.DataFrame()):
        try:
            dpy.descargar_precios_diarios_yf(["AAA", "BBB"], period="5y")
            assert False, "debería haber lanzado ValueError -- no hay ningún dato"
        except ValueError:
            pass
    print("OK  test_ningun_ticker_disponible_lanza_error_claro")


if __name__ == "__main__":
    test_descargar_precios_diarios_yf_formato_correcto()
    test_descargar_precios_intraday_yf_sin_timezone_y_filtrado_a_horario_mercado()
    test_ticker_faltante_en_la_descarga_se_salta_sin_romper()
    test_ningun_ticker_disponible_lanza_error_claro()
    print("\nTodos los tests pasaron correctamente.")
