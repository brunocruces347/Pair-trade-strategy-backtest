"""
Test de universo_historico.py contra un fixture HTML construido con
datos REALES de la tabla de cambios de Wikipedia (verificados por
búsqueda web, no inventados) -- no depende de red.
"""
from pathlib import Path
from datetime import date

import universo_historico as uh

FIXTURE_HTML = Path(__file__).parent / "fixture_sp500_test.html"


def _escribir_fixture():
    """Mismo HTML usado para validar el módulo durante el desarrollo --
    5 cambios reales, incluidos los 2 casos límite (agregado sin
    removido explícito -- spinoff; y removido sin agregado explícito)."""
    FIXTURE_HTML.write_text("""
<html><body>
<table class="wikitable">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
<tr><td>VEEV</td><td>Veeva Systems</td><td>Health Care</td></tr>
<tr><td>CASY</td><td>Casey's</td><td>Consumer Staples</td></tr>
<tr><td>Q</td><td>Qnity Electronics</td><td>Information Technology</td></tr>
<tr><td>SOLV</td><td>Solventum</td><td>Health Care</td></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
</table>
<table class="wikitable">
<tr><th rowspan="2">Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>May 7, 2026</td><td>VEEV</td><td>Veeva Systems</td><td>CTRA</td><td>Coterra Energy</td><td>Devon Energy acquiring Coterra Energy.</td></tr>
<tr><td>April 9, 2026</td><td>CASY</td><td>Casey's</td><td>HOLX</td><td>Hologic</td><td>Blackstone and TPG acquired Hologic.</td></tr>
<tr><td>November 3, 2025</td><td>Q</td><td>Qnity Electronics</td><td></td><td></td><td>DuPont spun off Qnity Electronics.</td></tr>
<tr><td>October 31, 2025</td><td></td><td></td><td>KMX</td><td>CarMax</td><td>Market capitalization change.</td></tr>
<tr><td>April 1, 2024</td><td>SOLV</td><td>Solventum</td><td></td><td></td><td>3M Co. spun off Solventum.</td></tr>
</table>
</body></html>
""", encoding="utf-8")


def test_obtener_tabla_cambios_estructura_correcta():
    _escribir_fixture()
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    tabla = uh.obtener_tabla_cambios(html=html)
    assert len(tabla) == 5
    assert list(tabla["fecha"])[0] == pd_ts("2026-05-07")
    print("OK  test_obtener_tabla_cambios_estructura_correcta")


def test_maneja_agregado_o_removido_vacio_sin_romper():
    """Caso límite real: un spinoff (Q, SOLV) agrega sin remover; una
    reclasificación (KMX) remueve sin agregar explícito ese día."""
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    tabla = uh.obtener_tabla_cambios(html=html)
    fila_q = tabla[tabla["ticker_agregado"] == "Q"].iloc[0]
    assert fila_q["ticker_removido"] is None
    fila_kmx = tabla[tabla["ticker_removido"] == "KMX"].iloc[0]
    assert fila_kmx["ticker_agregado"] is None
    print("OK  test_maneja_agregado_o_removido_vacio_sin_romper")


def test_reconstruye_antes_de_un_cambio_reciente():
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    lista_actual = uh.obtener_lista_actual(html=html)
    tabla = uh.obtener_tabla_cambios(html=html)

    universo = uh.obtener_constituyentes_en_fecha(date(2026, 5, 1), lista_actual, tabla)
    assert "CTRA" in universo, "antes del cambio del 7-may, CTRA debía seguir en el índice"
    assert "VEEV" not in universo, "VEEV todavía no existía en el índice antes de esa fecha"
    print("OK  test_reconstruye_antes_de_un_cambio_reciente")


def test_reconstruye_antes_de_un_spinoff():
    """SOLV (Solventum) se separó de 3M el 1-abr-2024 -- antes de esa
    fecha, no debería existir como constituyente."""
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    lista_actual = uh.obtener_lista_actual(html=html)
    tabla = uh.obtener_tabla_cambios(html=html)

    universo = uh.obtener_constituyentes_en_fecha(date(2024, 1, 1), lista_actual, tabla)
    assert "SOLV" not in universo
    assert {"CTRA", "HOLX", "KMX"} <= universo, "los que salieron después deben seguir estando"
    print("OK  test_reconstruye_antes_de_un_spinoff")


def test_reconstruye_hoy_calza_con_la_lista_actual():
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    lista_actual = uh.obtener_lista_actual(html=html)
    tabla = uh.obtener_tabla_cambios(html=html)

    universo = uh.obtener_constituyentes_en_fecha(date(2026, 7, 23), lista_actual, tabla)
    assert universo == set(lista_actual)
    print("OK  test_reconstruye_hoy_calza_con_la_lista_actual")


def pd_ts(s):
    import pandas as pd
    return pd.Timestamp(s)


if __name__ == "__main__":
    test_obtener_tabla_cambios_estructura_correcta()
    test_maneja_agregado_o_removido_vacio_sin_romper()
    test_reconstruye_antes_de_un_cambio_reciente()
    test_reconstruye_antes_de_un_spinoff()
    test_reconstruye_hoy_calza_con_la_lista_actual()
    print("\nTodos los tests pasaron correctamente.")
