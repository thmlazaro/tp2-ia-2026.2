"""Testes do avaliador e das convenções. Rodar na raiz:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avaliacao import avaliar, juntar, metricas  # noqa: E402
from utils import COMPETENCIAS as C, arredondar  # noqa: E402


def gabarito():
    rng = np.random.default_rng(0)
    g = pd.DataFrame(rng.choice([0, 40, 80, 120, 160, 200], size=(50, 5)), columns=C)
    g.insert(0, "id", [f"e{i}" for i in range(50)])
    return g


def como_preds(df):
    p = df[["id"] + C].copy()
    p["total"] = p[C].sum(axis=1)
    return p


def test_previsao_perfeita():
    g = gabarito()
    tab, _ = avaliar(g, como_preds(g))
    assert (tab["MAE"] == 0).all() and np.allclose(tab["QWK"], 1) and (tab["Acc_exata"] == 1).all()


def test_previsao_constante_tem_qwk_zero():
    g = gabarito()
    p = g.copy()
    p[C] = 120
    tab, _ = avaliar(g, como_preds(p))
    assert np.allclose(tab.set_index("alvo").loc[C, "QWK"], 0)
    assert (tab["MAE"] > 0).all()


def test_valores_conhecidos():
    m = metricas([0, 40, 80, 120], [0, 80, 80, 200], tol=40)
    assert m["MAE"] == pytest.approx((0 + 40 + 0 + 80) / 4)
    assert m["Acc_exata"] == pytest.approx(0.5)
    assert m["Acc_tol"] == pytest.approx(0.75)


def test_qwk_penaliza_erro_grande_mais_que_pequeno():
    y = [0, 40, 80, 120, 160, 200] * 5
    perto = [min(v + 40, 200) for v in y]
    longe = [200 - v for v in y]
    assert metricas(y, perto, 80)["QWK"] > metricas(y, longe, 80)["QWK"]


def test_tolerancia_total_e_100():
    g = gabarito()
    p = g.copy()
    p.loc[:, "c1"] = np.where(p["c1"] <= 160, p["c1"] + 40, p["c1"] - 40)   # erro de 40 no total
    tab, _ = avaliar(g, como_preds(p))
    assert tab.set_index("alvo").loc["total", "Acc_tol"] == 1.0


def test_falta_previsao_interrompe():
    g = gabarito()
    with pytest.raises(ValueError, match="sem previsão"):
        juntar(g, como_preds(g).iloc[:-1])


def test_subconjunto_por_ids():
    g = gabarito()
    tab, m = avaliar(g, como_preds(g).iloc[:10], ids=g["id"].iloc[:10])
    assert len(m) == 10 and (tab["n"] == 10).all()


def test_rejeita_nota_invalida_e_total_errado():
    g = gabarito()
    p = como_preds(g)
    p.loc[0, "c1"] = 100
    with pytest.raises(ValueError, match="fora de"):
        juntar(g, p)
    p = como_preds(g)
    p.loc[0, "total"] += 40
    with pytest.raises(ValueError, match="total diferente"):
        juntar(g, p)


def test_arredondar():
    assert list(arredondar([-15, 19, 21, 61, 135, 199, 260])) == [0, 0, 40, 80, 120, 200, 200]
