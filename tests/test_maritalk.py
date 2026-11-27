"""Testes do módulo MariTalk com um cliente FALSO (nenhuma chamada real, custo zero)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import maritalk  # noqa: E402

BOA = {"c1": 120, "c2": 160, "c3": 120, "c4": 160, "c5": 80, "total": 999,
       "justificativas": {"c1": "a", "c2": "b", "c3": "c", "c4": "d", "c5": "e"},
       "sugestoes": ["s1", "s2"]}


def resposta(conteudo, tin=900, tout=300):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=conteudo))],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


def erro_http(classe, status):
    import httpx2
    req = httpx2.Request("POST", "https://chat.maritaca.ai/api/chat/completions")
    return classe("erro simulado", response=httpx2.Response(status, request=req), body=None)


class ClienteFalso:
    """Devolve, em ordem, os itens de `roteiro` (string = conteúdo; Exception = lança)."""

    def __init__(self, roteiro):
        self.roteiro, self.chamadas = list(roteiro), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.chamadas.append(kw)
        item = self.roteiro.pop(0)
        if isinstance(item, Exception):
            raise item
        return resposta(item)


@pytest.fixture(autouse=True)
def cache_temporario(tmp_path, monkeypatch):
    monkeypatch.setenv("MARITALK_CACHE", str(tmp_path / "cache"))


def corrigir(cliente, texto="redação de teste", **kw):
    return maritalk.corrigir(texto, "tema", cliente=cliente, dormir=lambda s: None, **kw)


def test_resposta_valida_e_total_recalculado():
    out = corrigir(ClienteFalso([json.dumps(BOA)]))
    assert out["total"] == 640                     # ignora o "total": 999 do modelo
    assert out["justificativas"]["c5"] == "e" and out["sugestoes"] == ["s1", "s2"]
    assert out["tokens_entrada"] == 900 and not out["do_cache"]


def test_json_com_cercas_e_texto_em_volta():
    out = corrigir(ClienteFalso(["Claro! Segue:\n```json\n" + json.dumps(BOA) + "\n```"]))
    assert out["c2"] == 160


def test_nota_invalida_repete_e_depois_aceita():
    ruim = {**BOA, "c3": 100}
    cli = ClienteFalso([json.dumps(ruim), "sem json aqui", json.dumps(BOA)])
    out = corrigir(cli)
    assert out["tentativas"] == 3 and len(cli.chamadas) == 3


def test_desiste_apos_3_novas_tentativas():
    cli = ClienteFalso(["lixo"] * 4)
    with pytest.raises(maritalk.FalhaCorrecao, match="4 tentativas"):
        corrigir(cli)
    assert len(cli.chamadas) == 4


def test_429_espera_crescente():
    from openai import RateLimitError
    esperas = []
    cli = ClienteFalso([erro_http(RateLimitError, 429), erro_http(RateLimitError, 429), json.dumps(BOA)])
    out = maritalk.corrigir("x", cliente=cli, dormir=esperas.append, espera_base=5)
    assert out["c1"] == 120 and esperas == [5, 10]


def test_chave_invalida_nao_repete():
    from openai import AuthenticationError
    cli = ClienteFalso([erro_http(AuthenticationError, 401)])
    with pytest.raises(maritalk.FalhaCorrecao, match="autenticação"):
        corrigir(cli)
    assert len(cli.chamadas) == 1


def test_json_mode_sem_suporte_cai_para_modo_normal():
    from openai import BadRequestError
    cli = ClienteFalso([erro_http(BadRequestError, 400), json.dumps(BOA)])
    out = corrigir(cli, json_mode=True)
    assert out["c1"] == 120
    assert "response_format" in cli.chamadas[0] and "response_format" not in cli.chamadas[1]


def test_cache_evita_segunda_chamada():
    cli = ClienteFalso([json.dumps(BOA)])
    corrigir(cli, texto="mesma redação")
    out = corrigir(ClienteFalso([]), texto="mesma redação")      # cliente vazio: falharia se chamasse
    assert out["do_cache"] and out["c4"] == 160 and len(cli.chamadas) == 1


def test_lote_registra_falhas_e_custo(monkeypatch):
    cli = ClienteFalso([json.dumps(BOA), "lixo", "lixo", "lixo", "lixo"])
    monkeypatch.setattr(maritalk, "criar_cliente", lambda api_key=None: cli)
    monkeypatch.setattr(maritalk.time, "sleep", lambda s: None)
    df = pd.DataFrame({"id": ["a", "b"], "texto": ["t1", "t2"], "tema": ["x", "x"]})
    preds, just, falhas, resumo = maritalk.corrigir_lote(df, progresso=False, espera_base=0)
    assert list(preds["id"]) == ["a"] and falhas[0]["id"] == "b"
    assert resumo["taxa_falha"] == 0.5
    assert resumo["custo_estimado_rs"] == pytest.approx((900 * 5 + 300 * 20) / 1e6)


def test_limite_de_custo_interrompe_lote(monkeypatch):
    cli = ClienteFalso([json.dumps(BOA)] * 3)
    monkeypatch.setattr(maritalk, "criar_cliente", lambda api_key=None: cli)
    monkeypatch.setattr(maritalk.time, "sleep", lambda s: None)
    df = pd.DataFrame({"id": list("abc"), "texto": ["t1", "t2", "t3"], "tema": ["x"] * 3})
    preds, _, falhas, resumo = maritalk.corrigir_lote(df, progresso=False, limite_custo=0.01)
    assert list(preds["id"]) == ["a"] and len(cli.chamadas) == 1      # 1ª custa ~R$0,0105 > 0,01
    assert resumo["interrompido_por_limite"] and [f["id"] for f in falhas] == ["b", "c"]


def test_sem_chave_da_erro_claro(monkeypatch):
    monkeypatch.delenv("MARITACA_API_KEY", raising=False)
    monkeypatch.setattr(maritalk, "RAIZ", Path("/nao/existe"))
    with pytest.raises(RuntimeError, match="MARITACA_API_KEY"):
        maritalk.criar_cliente()
