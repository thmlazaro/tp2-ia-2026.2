# Correção automática de redações do ENEM com modelos de linguagem

Trabalho Prático 2 · Inteligência Artificial 2026.2 (UFSJ) · Profa. Érica Carvalho
**Grupo:** Thales Lázaro, Rafael Dayvid, Pedro Peixoto, Vinícius Carvalho

O projeto prevê as notas das **5 competências** da redação do ENEM (0 a 200, de 40 em 40) e compara quatro abordagens no corpus **Extended Essay-BR**:

| Modelo | Abordagem |
|---|---|
| **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) | *fine-tuning* como regressão com 5 saídas |
| **MariTalk** (Sabiá-4, Maritaca AI) | LLM via API, *prompting zero-shot*, com justificativas e sugestões |
| **TF-IDF + Ridge** | baseline clássico |
| **Trivial** | sempre a nota mais comum do treino |

Inclui uma aplicação **Streamlit** para corrigir uma redação com o BERTimbau, o MariTalk ou ambos lado a lado.

---

## 1. Instalação

Requisitos: **Python 3.10+** (testado com 3.12) e Git.

```bash
git clone https://github.com/thmlazaro/tp2-ia-2026.2.git
cd tp2-ia-2026.2
python -m venv .venv
```

Ative o ambiente virtual:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
  (se der erro de permissão: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo)
- **Linux/macOS:** `source .venv/bin/activate`

Instale as dependências (sem GPU, primeiro o PyTorch de CPU, que é bem menor):

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **Windows:** se aparecer `OSError: [WinError 126] ... c10.dll`, falta o
> [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). Instale-o e tente de novo.

## 2. Modelo BERTimbau treinado

O modelo (~420 MB) não fica no Git. Baixe `bertimbau_final.zip` na página de
[**Releases**](https://github.com/thmlazaro/tp2-ia-2026.2/releases) do repositório e descompacte em:

```
modelos/bertimbau_final/      (deve conter config.json, model.safetensors, tokenizer.json ...)
```

Outra pasta pode ser usada com a variável de ambiente `BERTIMBAU_DIR`.

## 3. Chave da API Maritaca (só para o MariTalk)

Crie o arquivo **`.env`** na raiz do repositório (modelo em `.env.example`):

```
MARITACA_API_KEY=sua_chave_aqui
```

Alternativas: variável de ambiente `MARITACA_API_KEY` ou `.streamlit/secrets.toml` (modelo em `.streamlit/secrets.toml.example`).
O `.env` e o `secrets.toml` estão no `.gitignore`: **nunca** coloque a chave no código. Sem a chave, o app continua funcionando com o BERTimbau.

## 4. Executar a aplicação

Na raiz do repositório, com o ambiente virtual ativado:

```bash
streamlit run app/app.py
```

Abra **http://localhost:8501**. Na barra lateral escolha o modelo (BERTimbau, MariTalk ou Ambos) e confira se o modelo e a chave aparecem com ✅. Cole uma redação (ou use **Carregar redação de exemplo**) e clique em **Corrigir**. A primeira correção do BERTimbau demora alguns segundos (carregamento do modelo); as seguintes levam ~1 s na CPU.

## 5. Reproduzir os resultados

| Passo | Comando / arquivo | Saída |
|---|---|---|
| Dados (Etapa 3) | `python src/dados.py` · `notebooks/01_dados.ipynb` | `data/{train,dev,test}.csv`, `results/estatisticas_dados.json` |
| BERTimbau (Etapa 4) | `notebooks/02_bertimbau.ipynb` no **Google Colab com GPU T4** | modelo + `results/preds_bertimbau.csv` |
| MariTalk (Etapa 5) | `python src/maritalk.py --dados data/test.csv --limite-custo 16` · `notebooks/03_maritalk.ipynb` | `results/preds_maritalk.csv`, justificativas, custo |
| Baselines (Etapa 6) | `python src/baseline_tfidf.py` · `notebooks/04_baselines.ipynb` | `results/preds_{trivial,tfidf}.csv` |
| Avaliação (Etapa 6) | `python src/avaliacao.py` | `results/tabela_final.{csv,md}`, `results/figuras/` |

- Os splits são os oficiais do Extended Essay-BR (procedimento dos autores, semente 230). As demais sementes são 42.
- O conjunto de teste não foi usado para nenhuma escolha (hiperparâmetros e `alpha` foram escolhidos no dev).
- O MariTalk guarda cada resposta em `results/cache_maritalk/` (fora do Git): rodar de novo não gera custo.

### Testes

```bash
python -m pytest -q
```

Testam o avaliador (métricas com respostas conhecidas), o módulo MariTalk (com um cliente falso, sem chamadas pagas) e a interface do app.

## 6. Estrutura

```
data/        CSVs limpos (train, dev, test) e licença do Essay-BR
notebooks/   01_dados · 02_bertimbau (Colab) · 03_maritalk · 04_baselines
src/         utils · dados · bertimbau · maritalk · baseline_tfidf · avaliacao
app/         app.py (Streamlit)
results/     previsões, tabela final, figuras, resumos de treino e custo
tests/       testes automáticos (pytest)
```

## 7. Custos da API (Maritaca, sabia-4)

Preços anotados em 25/11/2026: **R$ 5** por milhão de tokens de entrada e **R$ 20** por milhão de tokens de saída; saldo inicial de R$ 20.
Custo real da correção das 985 redações de teste: **R$ 13,63** (R$ 0,0138 por redação), ver `results/maritalk_resumo.json`.

## Licença e créditos

Código sob a [licença MIT](LICENSE) (© 2026 Thales Lázaro, Rafael Dayvid, Pedro Peixoto, Vinícius Carvalho).
Dados derivados do **Extended Essay-BR** (Marinho, Anchiêta e Moura, *JIDM*, 2022), licença MIT — ver [`data/LICENSE-ESSAY-BR.txt`](data/LICENSE-ESSAY-BR.txt).
Modelo base: [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (Souza, Nogueira e Lotufo, 2020).
