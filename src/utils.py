"""Convenções compartilhadas por todos os módulos do TP2.

Todo código que lida com notas deve importar daqui, para que o arredondamento,
os nomes das competências e a semente sejam idênticos em todos os modelos.
"""
import numpy as np

COMPETENCIAS = ["c1", "c2", "c3", "c4", "c5"]
ALVOS = COMPETENCIAS + ["total"]
NOTAS_VALIDAS = (0, 40, 80, 120, 160, 200)
SEED = 42

NOMES_COMPETENCIAS = {
    "c1": "Domínio da norma padrão da língua escrita",
    "c2": "Compreensão da proposta e desenvolvimento do tema",
    "c3": "Seleção e organização de argumentos",
    "c4": "Mecanismos linguísticos de coesão",
    "c5": "Proposta de intervenção respeitando os direitos humanos",
}


def arredondar(x):
    """Converte notas contínuas para o múltiplo de 40 mais próximo, limitado a [0, 200].

    Aceita escalar, lista ou array (de qualquer formato) e devolve array de int.
    Empates (ex.: 20, 60) seguem o arredondamento do NumPy (para o par mais próximo).
    """
    x = np.asarray(x, dtype=float)
    return np.clip(np.round(x / 40.0) * 40, 0, 200).astype(int)


def total(notas):
    """Soma das 5 competências de um dict {'c1':..,...,'c5':..}."""
    return int(sum(int(notas[c]) for c in COMPETENCIAS))
