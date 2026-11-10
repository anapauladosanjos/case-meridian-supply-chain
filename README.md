# Meridian Components — Análise de Supply Chain

Case técnico (Estágio em Análise de Dados). Base: [Supply Chain Dataset (Kaggle)](https://www.kaggle.com/datasets/nadhirmazariboufares/supply-chain-dataset), dados sintéticos de 2024–2025 (11 tabelas, 44.917 linhas).

## O que há aqui
| Pasta | Conteúdo |
|---|---|
| `notebook/Case_Meridian_Supply_Chain.ipynb` | Análise completa (pronta para o Google Colab) |
| `notebook/Case_Meridian_Supply_Chain_executado.ipynb` | Mesma análise, com as saídas de uma execução local |
| `notebook/Case_Meridian_Atrasos_Fornecedores.ipynb` | **Análise dos atrasos de fornecedores** (qualidade dos dados, quem atrasa, causas, casos extremos, risco e previsão), em linguagem simples. Pronto para o Colab; `_executado` traz as saídas |
| `notebook/atrasos_lib.py` | Biblioteca usada pelo notebook de atrasos e pelo app (o notebook a traz embutida na 1ª célula) |
| `streamlit_app/` | App interativo: abas do case + **Atrasos de fornecedores** com filtros por período, fornecedor, tier, país e urgência |
| `figs_atrasos/`, `resultados_atrasos/` | Gráficos e tabelas exportados pelo notebook de atrasos |
| `figs/`, `resultados/` | Gráficos e números exportados pelo notebook |
| `docs/` | Premissas, decisões analíticas e limitações |

## Principais achados
- **Fornecedores:** SUP-04, SUP-11 e SUP-17 atrasam 7,5–7,8 dias em média e respondem por ~59% dos dias de atraso; a nota de qualidade não prevê rejeição nem atraso.
- **Margem e frete:** Fasteners tem margem bruta de 46,3%, mas o frete equivale a 119% da receita da categoria (margem líquida −72,9%); o prejuízo está no modal Air.
- **Sazonalidade e estoque:** demanda −38% em agosto e +46% no 4º trimestre, nos dois anos; a ruptura sobe de 4,2% (Q3) para 7,2% (Q4) com ponto de reposição estático.

- **Atrasos de fornecedores:** 58% das entregas atrasam, mas o atraso grave (≥ 4 dias, 14,4%) vem só de SUP-04, SUP-17 e SUP-11 (alto risco; 59% dos dias de atraso). O atraso **não** está associado a valor, tamanho, urgência, mês, país ou tier (o efeito de país/tier some sem esses 3); a identidade do fornecedor explica 63% da variação. Não foi encontrada associação com ruptura ou nível de serviço (limitação do desenho dos dados); custo financeiro do atraso **não é mensurável** nesta base.

Simulações (ex.: troca de modal) são hipóteses; relações são associações, não causas.

## Como reproduzir
**Google Colab:** `File > Upload notebook`, escolha `notebook/Case_Meridian_Supply_Chain.ipynb` e `Ambiente de execução > Executar tudo`. A célula de carregamento baixa os CSVs via `kagglehub`; se falhar, pede upload manual.

**Colab (atrasos):** idem, com `notebook/Case_Meridian_Atrasos_Fornecedores.ipynb` (roda em ~2 min; não precisa instalar nada).

**Local:**
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn kagglehub jupyter
jupyter notebook notebook/Case_Meridian_Supply_Chain.ipynb
```

**Streamlit:**
```bash
pip install -r streamlit_app/requirements.txt
CASE_DATA_DIR=./data streamlit run streamlit_app/app.py   # pasta com os 11 CSVs
# sem a variável, o app baixa o dataset via kagglehub (requer credenciais/Internet)
```
