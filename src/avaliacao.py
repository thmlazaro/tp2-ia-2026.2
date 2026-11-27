"""Etapa 6: avaliador único. Todas as métricas do relatório saem daqui.

Lê arquivos results/preds_<modelo>.csv (colunas id, c1..c5, total), cruza com o
gabarito pelo id e calcula, por competência e no total:
  MAE, QWK (kappa de Cohen com pesos quadráticos), acurácia exata e
  acurácia com tolerância (±80 por competência, ±100 no total).

Todos os modelos são avaliados no MESMO conjunto de redações: se um arquivo não
tiver previsão para alguma redação do conjunto, a avaliação é interrompida.

Uso (na raiz do repositório):
  python src/avaliacao.py --gabarito data/test.csv --preds "results/preds_*.csv"
  python src/avaliacao.py ... --ids data/subconjunto_ids.csv     # subconjunto comum
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ALVOS, COMPETENCIAS as C, NOTAS_VALIDAS  # noqa: E402

TOLERANCIA = {**{c: 80 for c in C}, "total": 100}
NOMES_MODELOS = {"trivial": "Trivial (moda)", "tfidf": "TF-IDF + Ridge",
                 "bertimbau": "BERTimbau", "maritalk": "MariTalk (Sabiá-4)"}
ORDEM_MODELOS = ["trivial", "tfidf", "bertimbau", "maritalk"]
CORES = {"trivial": "#9e9e9e", "tfidf": "#c27c0e", "bertimbau": "#1f3a5f", "maritalk": "#2e7d4f"}


# ============================================================ métricas
def metricas(y, p, tol):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    erro = np.abs(y - p)
    # kappa indefinido (ex.: gabarito e previsão constantes) -> 0, que é "concordância de acaso"
    qwk = cohen_kappa_score(y, p, weights="quadratic") if len(set(y) | set(p)) > 1 else 0.0
    return {"MAE": float(erro.mean()),
            "QWK": float(np.nan_to_num(qwk)),
            "Acc_exata": float((erro == 0).mean()),
            "Acc_tol": float((erro <= tol).mean())}


def validar_previsoes(preds, nome="previsões"):
    faltando = [c for c in ["id"] + ALVOS if c not in preds.columns]
    if faltando:
        raise ValueError(f"{nome}: colunas ausentes {faltando}")
    if preds["id"].duplicated().any():
        raise ValueError(f"{nome}: ids duplicados")
    if not preds[C].isin(NOTAS_VALIDAS).all().all():
        raise ValueError(f"{nome}: há notas fora de {NOTAS_VALIDAS}")
    if not (preds[C].sum(axis=1) == preds["total"]).all():
        raise ValueError(f"{nome}: total diferente da soma das competências")


def juntar(gabarito, preds, ids=None, nome="previsões"):
    """Une gabarito e previsões pelo id; exige previsão para todas as redações avaliadas."""
    validar_previsoes(preds, nome)
    g = gabarito[["id"] + C].copy()
    g["total"] = g[C].sum(axis=1)
    if ids is not None:
        ids = set(ids)
        nao_existem = ids - set(g["id"])
        if nao_existem:
            raise ValueError(f"{len(nao_existem)} ids do subconjunto não estão no gabarito")
        g = g[g["id"].isin(ids)]
    m = g.merge(preds[["id"] + ALVOS], on="id", how="left", suffixes=("_real", "_prev"))
    sem_prev = m[[f"{a}_prev" for a in ALVOS]].isna().any(axis=1)
    if sem_prev.any():
        raise ValueError(f"{nome}: {int(sem_prev.sum())} redações sem previsão "
                         f"(ex.: {m.loc[sem_prev, 'id'].head(3).tolist()}). "
                         "Todos os modelos devem ser avaliados no MESMO conjunto (use --ids).")
    for a in ALVOS:
        m[f"{a}_prev"] = m[f"{a}_prev"].astype(int)
    return m


def avaliar(gabarito, preds, ids=None, nome="previsões"):
    m = juntar(gabarito, preds, ids, nome)
    linhas = [{"alvo": a, "n": len(m), **metricas(m[f"{a}_real"], m[f"{a}_prev"], TOLERANCIA[a])}
              for a in ALVOS]
    return pd.DataFrame(linhas), m


def avaliar_arquivos(gabarito, arquivos, ids=None):
    """Avalia vários preds_*.csv. Devolve (tabela longa, {modelo: dataframe unido})."""
    tabelas, juntados = [], {}
    for arq in arquivos:
        nome = Path(arq).stem.replace("preds_", "")
        tab, m = avaliar(gabarito, pd.read_csv(arq), ids, nome)
        tab.insert(0, "modelo", nome)
        tabelas.append(tab)
        juntados[nome] = m
    ordem = {m: i for i, m in enumerate(ORDEM_MODELOS)}
    juntados = dict(sorted(juntados.items(), key=lambda kv: ordem.get(kv[0], 99)))
    tabela = pd.concat(tabelas, ignore_index=True)
    tabela["_o"] = tabela["modelo"].map({m: i for i, m in enumerate(juntados)}) * 10 \
        + tabela["alvo"].map({a: i for i, a in enumerate(ALVOS)})
    tabela = tabela.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return tabela, juntados


def resumo(tabela, metrica):
    """Tabela larga: modelos nas linhas, c1..c5 + média das competências + total nas colunas."""
    ordem = list(dict.fromkeys(tabela["modelo"]))
    larga = tabela.pivot(index="modelo", columns="alvo", values=metrica).loc[ordem, ALVOS]
    larga.insert(5, "média c1–c5", larga[C].mean(axis=1))
    larga.index = [NOMES_MODELOS.get(m, m) for m in larga.index]
    return larga


# ============================================================ gráficos
def graficos(tabela, juntados, pasta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    modelos = list(juntados)
    rotulo = lambda m: NOMES_MODELOS.get(m, m)  # noqa: E731

    # 1) QWK por competência e modelo
    piv = tabela[tabela.alvo.isin(C)].pivot(index="alvo", columns="modelo", values="QWK")
    piv = piv.reindex(C)[modelos]
    fig, ax = plt.subplots(figsize=(9, 4))
    largura = 0.8 / len(modelos)
    for i, mo in enumerate(modelos):
        xs = np.arange(len(C)) + (i - (len(modelos) - 1) / 2) * largura
        barras = ax.bar(xs, piv[mo].values, largura, label=rotulo(mo), color=CORES.get(mo))
        ax.bar_label(barras, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(range(len(C)), [c.upper() for c in C])
    ax.set_ylabel("QWK (teste)"); ax.set_ylim(min(0, piv.values.min() - 0.05), 1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("QWK por competência e modelo"); ax.legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(pasta / "qwk_por_competencia.png", dpi=200); plt.close(fig)

    # 2) Real × previsto no total (com leve deslocamento aleatório para ver a densidade)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, len(modelos), figsize=(3.8 * len(modelos), 3.9), squeeze=False)
    for ax, mo in zip(axes[0], modelos):
        m = juntados[mo]
        j = lambda s: s + rng.uniform(-12, 12, len(s))  # noqa: E731
        ax.scatter(j(m["total_real"]), j(m["total_prev"]), s=7, alpha=0.3, color=CORES.get(mo))
        ax.plot([0, 1000], [0, 1000], "k--", lw=0.8)
        ax.set_xlim(-20, 1020); ax.set_ylim(-20, 1020)
        ax.set_title(rotulo(mo)); ax.set_xlabel("nota total real"); ax.set_ylabel("nota total prevista")
    plt.tight_layout(); plt.savefig(pasta / "real_vs_previsto_total.png", dpi=200); plt.close(fig)

    # 3) Matriz de confusão da competência mais difícil (menor QWK médio entre os modelos)
    dificil = piv.mean(axis=1).idxmin()
    for antiga in pasta.glob("confusao_*.png"):     # a competência mais difícil pode mudar
        antiga.unlink()
    fig, axes = plt.subplots(1, len(modelos), figsize=(3.8 * len(modelos), 3.7), squeeze=False)
    for ax, mo in zip(axes[0], modelos):
        m = juntados[mo]
        cm = confusion_matrix(m[f"{dificil}_real"], m[f"{dificil}_prev"], labels=list(NOTAS_VALIDAS))
        ax.imshow(cm, cmap="Blues")
        for i in range(len(NOTAS_VALIDAS)):
            for k in range(len(NOTAS_VALIDAS)):
                ax.text(k, i, cm[i, k], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, k] > cm.max() / 2 else "black")
        ax.set_xticks(range(6), NOTAS_VALIDAS, fontsize=7); ax.set_yticks(range(6), NOTAS_VALIDAS, fontsize=7)
        ax.set_xlabel("previsto"); ax.set_ylabel("real"); ax.set_title(f"{rotulo(mo)} · {dificil.upper()}")
    plt.tight_layout(); plt.savefig(pasta / f"confusao_{dificil}.png", dpi=200); plt.close(fig)
    return dificil


# ============================================================ linha de comando
def main(argv=None):
    ap = argparse.ArgumentParser(description="Avalia previsões de notas contra o gabarito.")
    ap.add_argument("--gabarito", default="data/test.csv")
    ap.add_argument("--preds", nargs="+", default=["results/preds_*.csv"],
                    help="arquivos ou padrões glob de previsões")
    ap.add_argument("--ids", default=None, help="CSV com coluna id (subconjunto comum)")
    ap.add_argument("--saida", default="results/tabela_final.csv")
    ap.add_argument("--figuras", default="results/figuras")
    a = ap.parse_args(argv)

    arquivos = sorted({f for padrao in a.preds for f in glob.glob(padrao)})
    if not arquivos:
        raise SystemExit(f"nenhum arquivo de previsão encontrado em {a.preds}")
    gab = pd.read_csv(a.gabarito)
    ids = pd.read_csv(a.ids)["id"].tolist() if a.ids else None
    tabela, juntados = avaliar_arquivos(gab, arquivos, ids)

    saida = Path(a.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    tabela.round(4).to_csv(saida, index=False)
    with open(saida.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(f"Conjunto: {a.gabarito}" + (f" (subconjunto {a.ids})" if a.ids else "")
                + f" · n = {tabela['n'].iloc[0]} redações\n\n")
        for met, casas in [("QWK", 3), ("MAE", 1), ("Acc_exata", 3), ("Acc_tol", 3)]:
            f.write(f"## {met}\n\n{resumo(tabela, met).round(casas).to_markdown()}\n\n")
    dificil = graficos(tabela, juntados, a.figuras)

    pd.set_option("display.width", 160)
    print(f"n = {tabela['n'].iloc[0]} redações · modelos: {', '.join(juntados)}\n")
    for met, casas in [("QWK", 3), ("MAE", 1), ("Acc_tol", 3)]:
        print(f"{met}:\n{resumo(tabela, met).round(casas).to_string()}\n")
    print(f"Tabela: {saida} e {saida.with_suffix('.md')} · figuras: {a.figuras} "
          f"(competência mais difícil: {dificil.upper()})")


if __name__ == "__main__":
    main()
