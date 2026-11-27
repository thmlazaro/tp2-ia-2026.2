"""Testes da interface Streamlit (sem rede e sem modelo: não chamam a API)."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("BERTIMBAU_DIR", "modelos/pasta_que_nao_existe")
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_abre_sem_erros(app):
    assert not app.exception
    assert app.title[0].value.endswith("Corretor automático de redações do ENEM")
    assert [r.value for r in app.radio] == ["Ambos"]
    assert "❌ Modelo BERTimbau" in [m.value for m in app.sidebar.markdown]


def test_texto_curto_gera_aviso(app):
    app.text_area(key="texto").set_value("curto demais").run()
    app.button[1].click().run()
    assert any("pelo menos 30 palavras" in w.value for w in app.warning)


def test_sem_modelo_mostra_erro_e_nao_trava(app):
    app.radio[0].set_value("BERTimbau").run()
    app.text_area(key="texto").set_value("palavra " * 60).run()
    app.button[1].click().run()
    assert not app.exception
    assert any("modelo não encontrado" in e.value for e in app.error)


def test_exemplo_preenche_texto_e_tema(app):
    app.button[0].click().run()
    assert len(app.text_area(key="texto").value.split()) > 30
    assert app.text_input(key="tema").value
