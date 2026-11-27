"""Etapa 5: correção por LLM (Sabiá-4, Maritaca AI) com prompting zero-shot.

O modelo NÃO é treinado: recebe as instruções de corretor do ENEM e devolve um JSON
com as 5 notas, justificativas e sugestões. O código garante a confiabilidade:
  - valida o JSON (5 notas presentes, cada uma em {0, 40, ..., 200});
  - recalcula o total no código (nunca confia no total do modelo);
  - repete a chamada em caso de JSON inválido ou erro temporário (até 3 novas tentativas),
    com espera crescente em erro 429 (limite de requisições);
  - guarda cada resposta válida em disco (cache): a mesma redação nunca é paga duas vezes;
  - registra tokens gastos para calcular o custo real.

A chave vem da variável de ambiente MARITACA_API_KEY ou do arquivo .env na raiz do
repositório (linha MARITACA_API_KEY=...). Nunca do código.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import COMPETENCIAS as C, NOTAS_VALIDAS  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
BASE_URL = "https://chat.maritaca.ai/api"
MODELO = "sabia-4"
VERSAO_PROMPT = "v1"            # mude ao alterar o prompt: o cache antigo deixa de valer
TEMPERATURA = 0.0
MAX_TOKENS_SAIDA = 1200
# Preços do sabia-4 em R$ por 1 milhão de tokens (README do grupo, consultado em 25/11/2026)
PRECO_ENTRADA = 5.0
PRECO_SAIDA = 20.0

SISTEMA = """Você é um corretor experiente da redação do ENEM e segue a matriz de referência do INEP.
Avalie a redação dissertativo-argumentativa abaixo nas 5 competências:

C1 - Domínio da modalidade escrita formal da língua portuguesa (gramática, ortografia, pontuação, registro).
C2 - Compreensão da proposta e aplicação de conceitos de várias áreas do conhecimento para desenvolver o tema,
     dentro dos limites estruturais do texto dissertativo-argumentativo em prosa.
C3 - Seleção, relação, organização e interpretação de informações, fatos, opiniões e argumentos em defesa de
     um ponto de vista (projeto de texto).
C4 - Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação (coesão).
C5 - Elaboração de proposta de intervenção para o problema abordado, respeitando os direitos humanos
     (agente, ação, meio/modo, finalidade e detalhamento).

Níveis de cada competência:
200 = excelente/pleno; 160 = bom; 120 = mediano; 80 = insuficiente; 40 = precário; 0 = ausente ou desconhecimento.
Fuga total ao tema ou texto que não seja dissertativo-argumentativo: 0 em todas as competências.

Cada competência recebe exatamente um destes valores: 0, 40, 80, 120, 160 ou 200.
Responda APENAS com um objeto JSON válido, sem texto antes ou depois e sem blocos de código, neste formato:
{"c1": <nota>, "c2": <nota>, "c3": <nota>, "c4": <nota>, "c5": <nota>,
 "justificativas": {"c1": "<1 ou 2 frases>", "c2": "<...>", "c3": "<...>", "c4": "<...>", "c5": "<...>"},
 "sugestoes": ["<sugestão concreta 1>", "<sugestão concreta 2>", "<sugestão concreta 3>"]}"""


class FalhaCorrecao(RuntimeError):
    """Todas as tentativas falharam para uma redação."""


# ============================================================ chave e cliente
def ler_chave(api_key=None):
    """Ordem: argumento > variável de ambiente > arquivo .env na raiz do repositório."""
    if api_key:
        return api_key
    if os.getenv("MARITACA_API_KEY"):
        return os.environ["MARITACA_API_KEY"]
    env = RAIZ / ".env"
    if env.exists():
        for linha in env.read_text(encoding="utf-8").splitlines():
            if linha.strip().startswith("MARITACA_API_KEY="):
                return linha.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def criar_cliente(api_key=None, timeout=120):
    from openai import OpenAI

    chave = ler_chave(api_key)
    if not chave:
        raise RuntimeError("Chave da Maritaca não encontrada: defina MARITACA_API_KEY "
                           "(variável de ambiente, .env na raiz ou st.secrets).")
    # max_retries=0: as novas tentativas são controladas por corrigir(), com registro
    return OpenAI(api_key=chave, base_url=BASE_URL, timeout=timeout, max_retries=0)


def mensagem_usuario(texto, tema=""):
    cab = f"Tema da proposta: {tema}\n\n" if tema else ""
    return f"{cab}Redação:\n<<<\n{texto}\n>>>"


# ============================================================ validação
def extrair_json(conteudo):
    """Aceita JSON puro, com cercas ```json ou com texto em volta."""
    s = re.sub(r"```(?:json)?", "", conteudo or "").strip()
    ini, fim = s.find("{"), s.rfind("}")
    if ini < 0 or fim <= ini:
        raise ValueError("resposta sem objeto JSON")
    return json.loads(s[ini:fim + 1])


def validar(d):
    """Devolve (notas com total recalculado, justificativas, sugestões) ou lança ValueError."""
    if not isinstance(d, dict):
        raise ValueError("JSON não é um objeto")
    notas = {}
    for c in C:
        v = d.get(c, d.get(c.upper()))
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if isinstance(v, bool) or not isinstance(v, int) or v not in NOTAS_VALIDAS:
            raise ValueError(f"{c} inválida: {v!r}")
        notas[c] = v
    notas["total"] = sum(notas[c] for c in C)
    just = d.get("justificativas") or {}
    just = {c: str(just.get(c, just.get(c.upper(), ""))) for c in C} if isinstance(just, dict) else {}
    sug = d.get("sugestoes") or d.get("sugestões") or []
    sug = [str(s) for s in sug] if isinstance(sug, list) else [str(sug)]
    return notas, just, sug


# ============================================================ cache
def pasta_cache():
    return Path(os.getenv("MARITALK_CACHE", RAIZ / "results" / "cache_maritalk"))


def caminho_cache(texto, tema=""):
    h = hashlib.sha256(f"{MODELO}|{VERSAO_PROMPT}|{TEMPERATURA}|{tema}|{texto}".encode("utf-8"))
    return pasta_cache() / f"{h.hexdigest()[:24]}.json"


def _resultado(bruto, do_cache):
    notas, just, sug = validar(extrair_json(bruto["conteudo"]))
    return {**notas, "justificativas": just, "sugestoes": sug,
            "tokens_entrada": bruto.get("tokens_entrada", 0), "tokens_saida": bruto.get("tokens_saida", 0),
            "tentativas": bruto.get("tentativas", 1), "do_cache": do_cache}


# ============================================================ chamada com novas tentativas
def corrigir(texto, tema="", api_key=None, cliente=None, max_novas_tentativas=3,
             json_mode=False, espera_base=5.0, dormir=time.sleep):
    """Corrige uma redação. Devolve dict com c1..c5, total, justificativas, sugestoes,
    tokens_entrada, tokens_saida, tentativas, do_cache. Lança FalhaCorrecao se tudo falhar."""
    from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                        AuthenticationError, BadRequestError, PermissionDeniedError, RateLimitError)

    arq = caminho_cache(texto, tema)
    if arq.exists():
        return _resultado(json.loads(arq.read_text(encoding="utf-8")), do_cache=True)

    cliente = cliente or criar_cliente(api_key)
    extras = {"response_format": {"type": "json_object"}} if json_mode else {}
    erros = []
    for tentativa in range(1 + max_novas_tentativas):
        try:
            resp = cliente.chat.completions.create(
                model=MODELO,
                messages=[{"role": "system", "content": SISTEMA},
                          {"role": "user", "content": mensagem_usuario(texto, tema)}],
                temperature=TEMPERATURA if tentativa == 0 else 0.3,  # varia um pouco ao repetir
                max_tokens=MAX_TOKENS_SAIDA,
                **extras,
            )
            conteudo = resp.choices[0].message.content or ""
            validar(extrair_json(conteudo))                      # ValueError se inválido
            uso = getattr(resp, "usage", None)
            bruto = {"conteudo": conteudo, "modelo": MODELO, "versao_prompt": VERSAO_PROMPT,
                     "tokens_entrada": int(getattr(uso, "prompt_tokens", 0) or 0),
                     "tokens_saida": int(getattr(uso, "completion_tokens", 0) or 0),
                     "tentativas": tentativa + 1, "json_mode": json_mode,
                     "data": time.strftime("%Y-%m-%d %H:%M:%S")}
            arq.parent.mkdir(parents=True, exist_ok=True)
            arq.write_text(json.dumps(bruto, ensure_ascii=False, indent=1), encoding="utf-8")
            return _resultado(bruto, do_cache=False)
        except (AuthenticationError, PermissionDeniedError) as e:   # chave errada: não adianta repetir
            raise FalhaCorrecao(f"autenticação recusada pela API: {e}") from e
        except BadRequestError as e:
            if json_mode:                                          # API sem suporte a json_object
                extras, json_mode = {}, False
                erros.append(f"400 com response_format; repetindo sem ({e})")
                continue
            raise FalhaCorrecao(f"requisição inválida: {e}") from e
        except RateLimitError as e:                                # 429: espera crescente
            erros.append("429 limite de requisições")
            espera = espera_base * 2 ** tentativa
            try:
                espera = max(espera, float(e.response.headers.get("retry-after", 0)))
            except (AttributeError, TypeError, ValueError):
                pass
            dormir(espera)
        except (APIConnectionError, APITimeoutError) as e:
            erros.append(f"conexão/timeout: {e}")
            dormir(espera_base * (tentativa + 1))
        except APIStatusError as e:                                # 5xx e outros
            erros.append(f"HTTP {e.status_code}")
            dormir(espera_base * (tentativa + 1))
        except (ValueError, json.JSONDecodeError) as e:            # JSON inválido ou nota fora do conjunto
            erros.append(f"resposta inválida: {e}")
    raise FalhaCorrecao(f"falhou após {1 + max_novas_tentativas} tentativas: {' | '.join(erros)}")


# ============================================================ lote e custo
def custo(tokens_entrada, tokens_saida):
    return (tokens_entrada * PRECO_ENTRADA + tokens_saida * PRECO_SAIDA) / 1e6


def corrigir_lote(df, api_key=None, pausa=0.5, progresso=True, limite_custo=None, **kw):
    """Corrige todas as redações de df (colunas id, texto, tema).

    Devolve (preds, justificativas, falhas, resumo). Pode ser interrompido e repetido:
    o que já deu certo vem do cache, sem custo. Se limite_custo (R$) for atingido pelas
    chamadas novas, o lote para e as redações restantes ficam registradas como não corrigidas.
    """
    cliente = None
    linhas, justificativas, falhas = [], [], []
    tok_in = tok_out = novas = 0
    gasto_novo = 0.0
    interrompido = False
    it = df.itertuples(index=False)
    if progresso:
        from tqdm.auto import tqdm
        it = tqdm(it, total=len(df), desc="MariTalk")
    for r in it:
        tema = getattr(r, "tema", "") or ""
        if limite_custo is not None and gasto_novo >= limite_custo:
            interrompido = True
            falhas.append({"id": r.id, "erro": f"não executada: limite de custo R$ {limite_custo} atingido"})
            continue
        try:
            if cliente is None and not caminho_cache(r.texto, tema).exists():
                cliente = criar_cliente(api_key)
            out = corrigir(r.texto, tema, api_key=api_key, cliente=cliente, **kw)
        except FalhaCorrecao as e:
            falhas.append({"id": r.id, "erro": str(e)[:500]})
            continue
        linhas.append({"id": r.id, **{c: out[c] for c in C}, "total": out["total"]})
        justificativas.append({"id": r.id, "justificativas": out["justificativas"],
                               "sugestoes": out["sugestoes"]})
        tok_in += out["tokens_entrada"]
        tok_out += out["tokens_saida"]
        if not out["do_cache"]:
            novas += 1
            gasto_novo += custo(out["tokens_entrada"], out["tokens_saida"])
            time.sleep(pausa)
    resumo = {"n": len(df), "ok": len(linhas), "falhas": len(falhas),
              "taxa_falha": len(falhas) / max(len(df), 1), "chamadas_novas": novas,
              "custo_chamadas_novas_rs": round(gasto_novo, 4), "interrompido_por_limite": interrompido,
              "tokens_entrada": tok_in, "tokens_saida": tok_out,
              "custo_estimado_rs": round(custo(tok_in, tok_out), 4)}
    return pd.DataFrame(linhas, columns=["id"] + C + ["total"]), justificativas, falhas, resumo


def salvar_lote(preds, justificativas, falhas, resumo, pasta=RAIZ / "results"):
    pasta = Path(pasta)
    preds.to_csv(pasta / "preds_maritalk.csv", index=False)
    with open(pasta / "maritalk_justificativas.jsonl", "w", encoding="utf-8") as f:
        for j in justificativas:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    pd.DataFrame(falhas, columns=["id", "erro"]).to_csv(pasta / "falhas_maritalk.csv", index=False)
    (pasta / "maritalk_resumo.json").write_text(
        json.dumps({**resumo, "modelo": MODELO, "versao_prompt": VERSAO_PROMPT, "temperatura": TEMPERATURA,
                    "preco_rs_por_milhao": {"entrada": PRECO_ENTRADA, "saida": PRECO_SAIDA}},
                   indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================ linha de comando
def main(argv=None):
    """python src/maritalk.py --dados data/test.csv --limite-custo 16"""
    import argparse

    ap = argparse.ArgumentParser(description="Corrige um conjunto de redações com o Sabiá-4.")
    ap.add_argument("--dados", default=str(RAIZ / "data" / "test.csv"))
    ap.add_argument("--ids", default=None, help="CSV com coluna id (subconjunto)")
    ap.add_argument("--limite-custo", type=float, default=None, help="para ao gastar R$ X em chamadas novas")
    ap.add_argument("--sem-json-mode", action="store_true")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.dados)
    if a.ids:
        df = df[df["id"].isin(set(pd.read_csv(a.ids)["id"]))]
    print(f"{len(df)} redações | modelo {MODELO} | prompt {VERSAO_PROMPT} | limite R$ {a.limite_custo}", flush=True)
    partes = []
    for i in range(0, len(df), 50):               # blocos de 50 só para mostrar o progresso
        parte = corrigir_lote(df.iloc[i:i + 50], progresso=False, json_mode=not a.sem_json_mode,
                              limite_custo=None if a.limite_custo is None else
                              a.limite_custo - sum(p[3]["custo_chamadas_novas_rs"] for p in partes))
        partes.append(parte)
        feitos = sum(len(p[0]) for p in partes)
        gasto = sum(p[3]["custo_chamadas_novas_rs"] for p in partes)
        print(f"{min(i + 50, len(df))}/{len(df)} | ok {feitos} | falhas {sum(len(p[2]) for p in partes)} "
              f"| gasto novo R$ {gasto:.2f}", flush=True)
    preds = pd.concat([p[0] for p in partes], ignore_index=True)
    just = [j for p in partes for j in p[1]]
    falhas = [f for p in partes for f in p[2]]
    resumo = {"n": len(df), "ok": len(preds), "falhas": len(falhas), "taxa_falha": len(falhas) / max(len(df), 1)}
    for k in ["chamadas_novas", "tokens_entrada", "tokens_saida", "custo_chamadas_novas_rs"]:
        resumo[k] = sum(p[3][k] for p in partes)
    resumo["custo_estimado_rs"] = round(custo(resumo["tokens_entrada"], resumo["tokens_saida"]), 4)
    resumo["interrompido_por_limite"] = any(p[3]["interrompido_por_limite"] for p in partes)
    salvar_lote(preds, just, falhas, resumo)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
