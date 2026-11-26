"""Etapa 3: preparação do corpus Extended Essay-BR.

Fonte: https://github.com/lplnufpi/essay-br (Marinho, Anchiêta e Moura, JIDM 2022),
fixada no commit abaixo para garantir reprodutibilidade.

Os splits oficiais não vêm prontos no repositório: são gerados pelo script
build_dataset.py dos autores (train_test_split duas vezes, random_state=230,
70/15/15). Reproduzimos exatamente esse procedimento em gerar_splits_oficiais().

Uso pela linha de comando (na raiz do repositório):
    python src/dados.py
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import COMPETENCIAS, NOTAS_VALIDAS  # noqa: E402

COMMIT = "7f0ea9cf854c8a24474b573ff5cdc02f0c7df4d8"
URL_BASE = f"https://raw.githubusercontent.com/lplnufpi/essay-br/{COMMIT}/extended-corpus"
ARQUIVOS = ["extended_essay-br.csv", "prompts.csv"]
MODELO_TOKENIZER = "neuralmind/bert-base-portuguese-cased"
RAIZ = Path(__file__).resolve().parents[1]
C = COMPETENCIAS


# ---------------------------------------------------------------- 1. obter dados
def baixar_corpus(pasta_raw=RAIZ / "data" / "raw"):
    """Baixa os CSVs do corpus (só se ainda não existirem) e devolve (redacoes, prompts)."""
    import requests

    pasta_raw = Path(pasta_raw)
    pasta_raw.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS:
        destino = pasta_raw / nome
        if not destino.exists():
            r = requests.get(f"{URL_BASE}/{nome}", timeout=120)
            r.raise_for_status()
            destino.write_bytes(r.content)
    corpus = pd.read_csv(pasta_raw / ARQUIVOS[0])
    prompts = pd.read_csv(pasta_raw / ARQUIVOS[1])
    return corpus, prompts


def gerar_splits_oficiais(corpus):
    """Reproduz build_dataset.py dos autores: 70% treino, 15% dev, 15% teste, random_state=230.

    Observação: apesar do nome 'stratify_colname' no script original, o parâmetro
    'stratify' não é passado ao train_test_split, então a divisão é aleatória simples.
    Reproduzimos o comportamento real, não o nome.
    """
    y = corpus[["score"]]
    treino, temp, _, y_temp = train_test_split(corpus, y, test_size=1.0 - 0.7, random_state=230)
    dev, teste, _, _ = train_test_split(temp, y_temp, test_size=0.15 / (0.15 + 0.15), random_state=230)
    assert len(treino) + len(dev) + len(teste) == len(corpus)
    return treino, dev, teste


# ---------------------------------------------------------------- 2. formato padrão
def como_lista(x):
    """Converte a coluna 'essay' (lista de parágrafos salva como texto) em lista Python."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str) and x.strip().startswith("["):
        return ast.literal_eval(x)          # literal_eval: só aceita literais, não executa código
    return [x]


def limpar_paragrafo(p):
    p = str(p).replace("\xa0", " ")         # espaço não separável aparece com frequência no corpus
    return re.sub(r"[ \t]+", " ", p).strip()


def preparar(df, prompts):
    """Formato padrão do grupo: id, texto, c1..c5, score, prompt, tema."""
    titulos = prompts.set_index("id")["title"]
    out = pd.DataFrame({
        # id estável = posição da redação no arquivo original do corpus
        "id": [f"essay_{i:05d}" for i in df.index],
        "texto": df["essay"].apply(
            lambda pars: "\n".join(q for q in (limpar_paragrafo(p) for p in como_lista(pars)) if q)),
    })
    for c in C:
        out[c] = df[c].astype(int).values
    out["score"] = df["score"].astype(int).values
    out["prompt"] = df["prompt"].astype(int).values
    out["tema"] = df["prompt"].map(titulos).fillna("").astype(str).values
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- 3. verificações
def verificar(df):
    """Contagens de problemas de qualidade (não altera os dados)."""
    n_pal = df["texto"].str.split().str.len()
    return {
        "n": int(len(df)),
        "notas_invalidas": int((~df[C].isin(NOTAS_VALIDAS).all(axis=1)).sum()),
        "soma_diferente_score": int((df[C].sum(axis=1) != df["score"]).sum()),
        "vazias": int((n_pal == 0).sum()),
        "menos_de_50_palavras": int((n_pal < 50).sum()),
        "palavras_media": float(n_pal.mean()),
    }


def limpar(df):
    """Remove notas inválidas e textos vazios; recalcula o total a partir das competências."""
    df = df[df[C].isin(NOTAS_VALIDAS).all(axis=1)]
    df = df[df["texto"].str.strip().str.len() > 0].copy()
    df["score"] = df[C].sum(axis=1)
    return df.reset_index(drop=True)


def normalizar(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def hash_texto(t):
    return hashlib.md5(normalizar(t).encode("utf-8")).hexdigest()


def remover_duplicatas(train, dev, test):
    """Remove do treino/dev textos que aparecem no teste (e do treino os que aparecem no dev).

    O teste nunca perde redações por causa de duplicatas com outros splits.
    """
    for df in (train, dev, test):
        df["hash"] = df["texto"].map(hash_texto)
    h_tr, h_dv, h_te = set(train.hash), set(dev.hash), set(test.hash)
    rel = {
        "treino_e_teste": len(h_tr & h_te),
        "treino_e_dev": len(h_tr & h_dv),
        "dev_e_teste": len(h_dv & h_te),
        "internas_treino": int(train.hash.duplicated().sum()),
        "internas_dev": int(dev.hash.duplicated().sum()),
        "internas_teste": int(test.hash.duplicated().sum()),
    }
    n0 = (len(train), len(dev), len(test))
    train = train[~train.hash.isin(h_te | h_dv)].drop_duplicates("hash")
    dev = dev[~dev.hash.isin(h_te)].drop_duplicates("hash")
    test = test.drop_duplicates("hash")
    rel["removidas"] = {"train": n0[0] - len(train), "dev": n0[1] - len(dev), "test": n0[2] - len(test)}
    return [d.drop(columns="hash").reset_index(drop=True) for d in (train, dev, test)], rel


# ---------------------------------------------------------------- 4. tokens
def contar_tokens(df, tokenizer):
    ids = tokenizer(df["texto"].tolist(), add_special_tokens=True, verbose=False)["input_ids"]
    return np.array([len(x) for x in ids])


# ---------------------------------------------------------------- 5. figura
def grafico_distribuicao(train, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
    for ax, c in zip(axes, C):
        cont = train[c].value_counts().reindex(NOTAS_VALIDAS, fill_value=0)
        ax.bar([str(v) for v in cont.index], cont.values, color="#1f3a5f")
        ax.set_title(f"Competência {c[1]}")
        ax.set_xlabel("nota")
        for x, v in enumerate(cont.values):
            ax.text(x, v, str(v), ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("nº de redações (treino)")
    fig.suptitle(f"Distribuição das notas por competência no treino (n = {len(train)})", y=1.02)
    plt.tight_layout()
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- pipeline completo
def executar(pasta_dados=RAIZ / "data", pasta_results=RAIZ / "results"):
    from transformers import AutoTokenizer

    pasta_dados, pasta_results = Path(pasta_dados), Path(pasta_results)
    est = {"fonte": f"lplnufpi/essay-br@{COMMIT}"}

    corpus, prompts = baixar_corpus(pasta_dados / "raw")
    est["corpus_total"] = int(len(corpus))
    brutos = gerar_splits_oficiais(corpus)
    est["splits_oficiais"] = {n: int(len(d)) for n, d in zip(["train", "dev", "test"], brutos)}

    train, dev, test = (preparar(d, prompts) for d in brutos)
    est["verificacao_antes"] = {n: verificar(d) for n, d in [("train", train), ("dev", dev), ("test", test)]}
    train, dev, test = limpar(train), limpar(dev), limpar(test)
    (train, dev, test), est["duplicatas"] = remover_duplicatas(train, dev, test)

    est["temas"] = {
        "treino": int(train["prompt"].nunique()),
        "teste": int(test["prompt"].nunique()),
        "frac_teste_com_tema_do_treino": float(test["prompt"].isin(set(train["prompt"])).mean()),
    }

    tok = AutoTokenizer.from_pretrained(MODELO_TOKENIZER)
    est["tokens"] = {}
    for nome, d in [("train", train), ("dev", dev), ("test", test)]:
        d["n_tokens"] = contar_tokens(d, tok)
        n = d["n_tokens"]
        est["tokens"][nome] = {"media": float(n.mean()), "mediana": float(n.median()),
                               "p90": float(n.quantile(0.9)), "max": int(n.max()),
                               "frac_maior_512": float((n > 512).mean())}

    est["distribuicao_treino"] = {
        c: {str(k): float(v) for k, v in
            train[c].value_counts(normalize=True).reindex(NOTAS_VALIDAS, fill_value=0).items()}
        for c in C}
    est["final"] = {n: int(len(d)) for n, d in [("train", train), ("dev", dev), ("test", test)]}

    cols = ["id", "texto"] + C + ["score", "prompt", "tema", "n_tokens"]
    for nome, d in [("train", train), ("dev", dev), ("test", test)]:
        d[cols].to_csv(pasta_dados / f"{nome}.csv", index=False)
    grafico_distribuicao(train, pasta_results / "figuras" / "dist_notas.png")
    (pasta_results / "estatisticas_dados.json").write_text(
        json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")
    return train, dev, test, est


if __name__ == "__main__":
    *_, est = executar()
    print(json.dumps(est, indent=2, ensure_ascii=False))
