"""Etapa 6: baselines.

1. Trivial: prevê sempre a nota mais comum do treino em cada competência.
   Serve para mostrar por que o MAE sozinho engana (MAE razoável, QWK = 0).
2. TF-IDF + Ridge: representação por n-gramas de palavras (1 e 2) com pesos TF-IDF
   e uma regressão linear com regularização L2 para as 5 competências ao mesmo tempo.
   O alpha (força da regularização) é escolhido pelo QWK médio no dev; o modelo
   final é treinado só no treino (mesmo protocolo do BERTimbau).

Uso (na raiz do repositório):  python src/baseline_tfidf.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliacao import metricas  # noqa: E402
from utils import COMPETENCIAS as C, SEED, arredondar  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
PARAMS_TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)


def para_previsoes(ids, notas):
    df = pd.DataFrame(np.asarray(notas, dtype=int), columns=C)
    df.insert(0, "id", np.asarray(ids))
    df["total"] = df[C].sum(axis=1)
    return df


def qwk_medio(y_df, p):
    return float(np.mean([metricas(y_df[c], p[:, k], 80)["QWK"] for k, c in enumerate(C)]))


def baseline_trivial(train, alvo):
    moda = [int(train[c].mode()[0]) for c in C]
    return para_previsoes(alvo["id"], np.tile(moda, (len(alvo), 1))), dict(zip(C, moda))


def ajustar_tfidf(train, dev, alphas=ALPHAS):
    """Ajusta o TF-IDF no treino e escolhe o alpha pelo QWK médio no dev."""
    vec = TfidfVectorizer(**PARAMS_TFIDF)
    x_tr, x_dv = vec.fit_transform(train["texto"]), vec.transform(dev["texto"])
    busca = []
    for a in alphas:
        p = arredondar(Ridge(alpha=a, random_state=SEED).fit(x_tr, train[C].values).predict(x_dv))
        busca.append({"alpha": a, "qwk_medio_dev": qwk_medio(dev, p),
                      "mae_medio_dev": float(np.abs(p - dev[C].values).mean())})
    melhor = max(busca, key=lambda b: b["qwk_medio_dev"])
    modelo = Ridge(alpha=melhor["alpha"], random_state=SEED).fit(x_tr, train[C].values)
    return vec, modelo, busca, melhor


def executar(pasta_dados=RAIZ / "data", pasta_saida=RAIZ / "results"):
    pasta_dados, pasta_saida = Path(pasta_dados), Path(pasta_saida)
    train = pd.read_csv(pasta_dados / "train.csv")
    dev = pd.read_csv(pasta_dados / "dev.csv")
    test = pd.read_csv(pasta_dados / "test.csv")

    preds_triv, moda = baseline_trivial(train, test)
    preds_triv.to_csv(pasta_saida / "preds_trivial.csv", index=False)

    inicio = time.time()
    vec, modelo, busca, melhor = ajustar_tfidf(train, dev)
    preds_tfidf = para_previsoes(test["id"], arredondar(modelo.predict(vec.transform(test["texto"]))))
    preds_tfidf.to_csv(pasta_saida / "preds_tfidf.csv", index=False)

    info = {"trivial": {"moda_treino": moda},
            "tfidf": {"parametros_tfidf": {**PARAMS_TFIDF, "ngram_range": list(PARAMS_TFIDF["ngram_range"])},
                      "vocabulario": len(vec.vocabulary_), "busca_alpha_dev": busca,
                      "alpha_escolhido": melhor["alpha"], "seed": SEED,
                      "duracao_s": round(time.time() - inicio, 1)}}
    (pasta_saida / "baselines_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    print(json.dumps(executar(), indent=2))
