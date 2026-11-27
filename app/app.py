"""Etapa 7: aplicação Streamlit para corrigir redações do ENEM.

Executar na raiz do repositório:   streamlit run app/app.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from utils import COMPETENCIAS as C  # noqa: E402

DIR_MODELO = Path(os.getenv("BERTIMBAU_DIR", RAIZ / "modelos" / "bertimbau_final"))
ROTULOS = {"c1": "C1 · Norma padrão", "c2": "C2 · Tema e estrutura", "c3": "C3 · Argumentação",
           "c4": "C4 · Coesão", "c5": "C5 · Proposta de intervenção"}
MIN_PALAVRAS = 30

st.set_page_config(page_title="Corretor de Redações ENEM", page_icon=":memo:", layout="wide")


# ============================================================ recursos
def chave_maritaca():
    """st.secrets > variável de ambiente > .env (nunca no código)."""
    try:
        if "MARITACA_API_KEY" in st.secrets:
            return st.secrets["MARITACA_API_KEY"]
    except Exception:          # sem .streamlit/secrets.toml
        pass
    import maritalk
    return maritalk.ler_chave()


@st.cache_resource(show_spinner="Carregando o BERTimbau (só na primeira vez)...")
def carregar_bertimbau(diretorio: str):
    import bertimbau
    bertimbau.carregar(diretorio)          # mantém tokenizer e modelo em memória
    return bertimbau


def corrigir_bertimbau(texto):
    return carregar_bertimbau(str(DIR_MODELO)).predict(texto, str(DIR_MODELO))


def corrigir_maritalk(texto, tema):
    import maritalk
    chave = chave_maritaca()
    if not chave:
        raise RuntimeError("chave da Maritaca não configurada (veja o README).")
    return maritalk.corrigir(texto, tema, api_key=chave, max_novas_tentativas=2, json_mode=True)


@st.cache_data
def exemplos():
    """Redações do conjunto de desenvolvimento (dev) para demonstração."""
    dev = pd.read_csv(RAIZ / "data" / "dev.csv")
    return dev.sample(20, random_state=7).reset_index(drop=True)


# ============================================================ exibição
def mostrar_notas(res):
    st.metric("Nota total", f"{res['total']} / 1000")
    for c in C:
        st.progress(res[c] / 200, text=f"**{ROTULOS[c]}**: {res[c]} / 200")
    if res.get("truncado"):
        st.caption("⚠️ Texto com mais de 512 tokens: o BERTimbau leu apenas o início.")
    if "justificativas" in res:
        with st.expander("Justificativas por competência", expanded=True):
            for c in C:
                st.markdown(f"**{ROTULOS[c]}** — {res['justificativas'].get(c) or '—'}")
        with st.expander("Sugestões de melhoria", expanded=True):
            for s in res["sugestoes"] or ["—"]:
                st.markdown(f"- {s}")
        origem = "resposta guardada em cache (sem custo)" if res.get("do_cache") else \
            f"{res['tokens_entrada']} + {res['tokens_saida']} tokens"
        st.caption(f"Sabiá-4 · {origem}")


def comparar(resultados):
    tab = pd.DataFrame({nome: [r[c] for c in C] + [r["total"]] for nome, r in resultados.items()},
                       index=[ROTULOS[c] for c in C] + ["Total"])
    if len(resultados) == 2:
        a, b = list(resultados)
        tab[f"Diferença ({b} − {a})"] = tab[b] - tab[a]
    st.subheader("Comparação")
    st.dataframe(tab, width="stretch")
    st.bar_chart(tab.loc[[ROTULOS[c] for c in C], list(resultados)], stack=False, height=280)


# ============================================================ barra lateral
with st.sidebar:
    st.header("Configuração")
    escolha = st.radio("Modelo", ["BERTimbau", "MariTalk", "Ambos"], index=2,
                       help="BERTimbau: modelo ajustado (fine-tuning) no Essay-BR, roda localmente. "
                            "MariTalk: Sabiá-4 via API, sem treino (zero-shot), com justificativas.")
    st.divider()
    st.markdown("**Situação**")
    st.markdown(("✅" if (DIR_MODELO / "config.json").exists() else "❌") + " Modelo BERTimbau")
    st.markdown(("✅" if chave_maritaca() else "❌") + " Chave da API Maritaca")
    st.divider()
    st.caption("TP2 · Inteligência Artificial 2026.2\n\nThales Lázaro, Rafael Dayvid, "
               "Pedro Peixoto, Vinícius Carvalho")

# ============================================================ página principal
st.title("📝 Corretor automático de redações do ENEM")
st.markdown("Cole uma redação dissertativo-argumentativa e compare as notas das **5 competências** "
            "dadas por um modelo ajustado (**BERTimbau**) e por um LLM via API (**MariTalk / Sabiá-4**).")

if "texto" not in st.session_state:
    st.session_state.texto, st.session_state.tema = "", ""

col_ex, _ = st.columns([1, 3])
if col_ex.button("Carregar redação de exemplo", help="Sorteia uma redação do conjunto de desenvolvimento"):
    ex = exemplos()
    i = st.session_state.get("i_exemplo", -1) + 1
    st.session_state.i_exemplo = i
    linha = ex.iloc[i % len(ex)]
    st.session_state.texto, st.session_state.tema = linha["texto"], linha["tema"]
    st.session_state.nota_real = {**{c: int(linha[c]) for c in C}, "total": int(linha["score"])}
    st.session_state.texto_exemplo = linha["texto"]

tema = st.text_input("Tema da redação (opcional, ajuda o MariTalk)", key="tema")
texto = st.text_area("Redação", key="texto", height=320, placeholder="Cole aqui o texto da redação...")
n_palavras = len(texto.split())
st.caption(f"{n_palavras} palavras")

if st.button("Corrigir", type="primary", width="stretch"):
    if n_palavras < MIN_PALAVRAS:
        st.warning(f"Cole uma redação com pelo menos {MIN_PALAVRAS} palavras.")
        st.stop()
    modelos = ["BERTimbau", "MariTalk"] if escolha == "Ambos" else [escolha]
    colunas = st.columns(len(modelos))
    resultados = {}
    for col, nome in zip(colunas, modelos):
        with col:
            st.subheader(nome)
            try:
                with st.spinner(f"Corrigindo com {nome}..."):
                    if nome == "BERTimbau":
                        if not (DIR_MODELO / "config.json").exists():
                            raise FileNotFoundError(
                                f"modelo não encontrado em '{DIR_MODELO.relative_to(RAIZ) if DIR_MODELO.is_relative_to(RAIZ) else DIR_MODELO}'. "
                                "Descompacte bertimbau_final.zip em modelos/bertimbau_final (veja o README).")
                        res = corrigir_bertimbau(texto)
                    else:
                        res = corrigir_maritalk(texto, tema)
                resultados[nome] = res
                mostrar_notas(res)
            except Exception as e:                       # o app nunca trava por causa de um modelo
                st.error(f"Não foi possível corrigir com o {nome}.\n\n**Detalhe:** {e}")
                if nome == "MariTalk":
                    st.info("Verifique a conexão com a internet e a chave da API. "
                            "O BERTimbau continua funcionando sem a API.")
    if len(resultados) == 2:
        comparar(resultados)
    real = st.session_state.get("nota_real")
    if real and st.session_state.get("texto_exemplo") == texto and resultados:
        st.info("Nota dos corretores humanos (redação de exemplo do corpus): "
                + " · ".join(f"{c.upper()} {real[c]}" for c in C) + f" · **total {real['total']}**")
