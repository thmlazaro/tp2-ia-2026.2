"""Etapa 4: BERTimbau com fine-tuning para as 5 competências (regressão multi-saída).

Treino (no Colab com GPU):   treinar(train_df, dev_df, dir_modelo, dir_checkpoints)
Previsão em lote:            prever_dataframe(df, dir_modelo)
Uso no app:                  predict(texto, dir_modelo) -> {'c1':..,...,'c5':..,'total':..,'truncado':..}

Decisões:
- Uma única cabeça com 5 saídas (num_labels=5, problem_type="regression", perda MSE).
  As notas são ordinais: errar 160 por 120 é menos grave que errar por 0, e a MSE
  penaliza mais os erros grandes, o que combina com o QWK.
- Rótulos divididos por 200 (ficam em [0, 1]); previsão multiplicada por 200 e
  arredondada ao múltiplo de 40 com utils.arredondar.
- Seleção do melhor checkpoint pelo QWK médio no dev (nunca pelo teste).
"""
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import COMPETENCIAS as C, SEED, arredondar  # noqa: E402

MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512
HIPERPARAMETROS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 4,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}


# ============================================================ métricas
def qwk_por_competencia(y, p):
    from sklearn.metrics import cohen_kappa_score

    # nan_to_num: se o modelo prevê sempre a mesma nota, o kappa pode ser indefinido -> 0
    return [float(np.nan_to_num(cohen_kappa_score(y[:, k], p[:, k], weights="quadratic")))
            for k in range(len(C))]


def compute_metrics(eval_pred):
    """Usada pelo Trainer a cada avaliação no dev."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    p = arredondar(np.asarray(preds) * 200)
    y = arredondar(np.asarray(labels) * 200)
    qwks = qwk_por_competencia(y, p)
    m = {f"qwk_{c}": q for c, q in zip(C, qwks)}
    m["qwk_medio"] = float(np.mean(qwks))
    m["mae_medio"] = float(np.abs(p - y).mean())
    return m


# ============================================================ dados
def para_dataset(df, tokenizer, max_len=MAX_LEN):
    from datasets import Dataset

    ds = Dataset.from_dict({
        "texto": df["texto"].tolist(),
        "labels": (df[C].to_numpy(dtype="float32") / 200.0).tolist(),
    })
    return ds.map(lambda b: tokenizer(b["texto"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["texto"])


# ============================================================ treino
def _argumentos(dir_checkpoints, hp, seed, max_steps):
    import torch
    from transformers import TrainingArguments

    kw = dict(
        output_dir=str(dir_checkpoints),
        **hp,
        fp16=torch.cuda.is_available(),     # precisão mista só faz sentido na GPU
        save_strategy="epoch",
        save_only_model=True,               # checkpoints de ~440 MB em vez de ~1,3 GB
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="qwk_medio",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    if max_steps:                           # só para o teste rápido do código
        kw.update(max_steps=max_steps, save_strategy="steps", save_steps=max_steps,
                  logging_steps=1)
    # Compatibilidade entre versões do transformers (o Colab pode ter a 4.x ou a 5.x):
    aceitos = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" in kw and "warmup_ratio" not in aceitos:
        kw["warmup_steps"] = kw.pop("warmup_ratio")      # 5.x: float em [0, 1) = fração dos passos
    chave_eval = "eval_strategy" if "eval_strategy" in aceitos else "evaluation_strategy"  # < 4.41
    kw[chave_eval] = "steps" if max_steps else "epoch"
    kw["eval_steps"] = max_steps
    return TrainingArguments(**kw)


def treinar(train_df, dev_df, dir_modelo, dir_checkpoints, hp=None, seed=SEED,
            max_steps=None, max_len=MAX_LEN, modelo_base=MODELO_BASE):
    """Faz o fine-tuning, salva o melhor modelo (pelo QWK médio no dev) e o histórico.

    Devolve um dict com hiperparâmetros, métricas do dev, histórico e versões.
    """
    import torch
    import transformers
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, set_seed)

    hp = {**HIPERPARAMETROS, **(hp or {})}
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(modelo_base)
    ds_train, ds_dev = para_dataset(train_df, tok, max_len), para_dataset(dev_df, tok, max_len)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_base, num_labels=len(C), problem_type="regression")

    trainer = Trainer(model=modelo, args=_argumentos(dir_checkpoints, hp, seed, max_steps),
                      train_dataset=ds_train, eval_dataset=ds_dev,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    inicio = time.time()
    trainer.train()
    duracao = time.time() - inicio

    dir_modelo = Path(dir_modelo)
    dir_modelo.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(dir_modelo))          # melhor checkpoint (load_best_model_at_end)
    tok.save_pretrained(str(dir_modelo))
    metricas_dev = trainer.evaluate()

    info = {
        "modelo_base": modelo_base,
        "hiperparametros": {**hp, "max_length": max_len, "seed": seed,
                            "fp16": bool(torch.cuda.is_available()), "max_steps": max_steps},
        "n_treino": len(train_df), "n_dev": len(dev_df),
        "melhor_checkpoint": trainer.state.best_model_checkpoint,
        "melhor_qwk_medio_dev": trainer.state.best_metric,
        "metricas_dev": metricas_dev,
        "historico": trainer.state.log_history,
        "duracao_treino_s": round(duracao, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versoes": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (dir_modelo / "treino.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return info


def grafico_treino(info, caminho):
    """Curvas por época: perda de treino, perda e QWK médio no dev."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = info["historico"]
    ev = [h for h in hist if "eval_qwk_medio" in h]
    tr = [h for h in hist if "loss" in h and "eval_loss" not in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot([h["epoch"] for h in tr], [h["loss"] for h in tr], label="treino", alpha=.7)
    a1.plot([h["epoch"] for h in ev], [h["eval_loss"] for h in ev], "o-", label="dev")
    a1.set_xlabel("época"); a1.set_ylabel("perda (MSE, escala 0–1)"); a1.legend()
    a2.plot([h["epoch"] for h in ev], [h["eval_qwk_medio"] for h in ev], "o-", color="#1f3a5f")
    a2.set_xlabel("época"); a2.set_ylabel("QWK médio (dev)"); a2.set_ylim(0, 1)
    fig.suptitle("Fine-tuning do BERTimbau")
    plt.tight_layout()
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=200, bbox_inches="tight")
    return fig


# ============================================================ previsão
_recursos = {}


def carregar(diretorio=None):
    """Carrega tokenizer e modelo uma única vez por processo."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    diretorio = str(diretorio or os.getenv("BERTIMBAU_DIR", "modelos/bertimbau_final"))
    if diretorio not in _recursos:
        if not Path(diretorio, "config.json").exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em '{diretorio}'. Baixe a pasta bertimbau_final do Drive "
                "para modelos/bertimbau_final ou defina a variável BERTIMBAU_DIR.")
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(diretorio)
        modelo = AutoModelForSequenceClassification.from_pretrained(diretorio).to(dispositivo).eval()
        _recursos[diretorio] = (tok, modelo, dispositivo)
    return _recursos[diretorio]


def prever_lote(textos, diretorio=None, tamanho_lote=16, max_len=MAX_LEN):
    """Notas (n, 5) já arredondadas a múltiplos de 40."""
    import torch

    tok, modelo, dispositivo = carregar(diretorio)
    saidas = []
    with torch.no_grad():
        for i in range(0, len(textos), tamanho_lote):
            lote = tok(list(textos[i:i + tamanho_lote]), truncation=True, max_length=max_len,
                       padding=True, return_tensors="pt").to(dispositivo)
            saidas.append(modelo(**lote).logits.float().cpu().numpy())
    return arredondar(np.vstack(saidas) * 200)


def prever_dataframe(df, diretorio=None, tamanho_lote=16):
    """DataFrame no formato padrão de previsões: id, c1..c5, total."""
    notas = prever_lote(df["texto"].tolist(), diretorio, tamanho_lote)
    out = pd.DataFrame(notas, columns=C)
    out.insert(0, "id", df["id"].values)
    out["total"] = out[C].sum(axis=1)
    return out


def predict(texto, diretorio=None):
    """Corrige uma redação. Devolve {'c1':..,...,'c5':..,'total':..,'truncado':bool}."""
    tok, _, _ = carregar(diretorio)
    notas = dict(zip(C, (int(v) for v in prever_lote([texto], diretorio)[0])))
    notas["total"] = sum(notas[c] for c in C)
    notas["truncado"] = len(tok(texto)["input_ids"]) > MAX_LEN
    return notas
