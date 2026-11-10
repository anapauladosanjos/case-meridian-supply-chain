# Premissas, decisões analíticas e limitações

## 1. Escopo
O case não traz uma pergunta única e diz que não é preciso usar todas as tabelas. Priorizei três frentes ligadas por impacto financeiro e de serviço: (A) fornecedores e compras, (B) margem, frete e portfólio, (C) sazonalidade e estoque. Ficaram fora produção/qualidade industrial, nível de serviço por cliente e modelagem estrela (apenas proposta no notebook).

## 2. Qualidade dos dados
- 11 tabelas, 44.917 linhas, 85 colunas; sem duplicidades; chaves primárias únicas; 0 registros órfãos em 12 relações.
- Único campo com nulos: `actual_delivery_date` (193), que são exatamente os pedidos cancelados. **Não imputei**: o nulo tem significado de negócio.
- Preço realizado = preço de lista em 100% das linhas (sem desconto): não dá para analisar margem de preço.
- `capacity_units` é incompatível com o estoque (ocupação de 81% a 187%): **não usei capacidade**.
- Trânsito real ≈ 5 dias em todos os modais, contra 1 a 14 dias na tabela `carrier`: registrado como limitação; afeta a leitura de custo/benefício do modal Air.
- Outliers (3×IQR) sinalizados e **mantidos** (distribuições assimétricas por natureza). Frete maior que a receita em parte das linhas **não é erro**: é o principal achado.
- O Apêndice A do case traz nomes de colunas "prováveis"; usei os nomes confirmados nos CSVs (ex.: `mode`, `units_on_hand`, `is_single_source`).

## 3. Definições de métricas
- **Atraso** = entrega real − prometida (dias), só pedidos entregues (2.807; 193 cancelados fora).
- **Rejeição** = Σ rejeitado / Σ recebido (agregada por soma, nunca média de taxas).
- **Margem bruta** = receita − qtd × custo padrão. **Margem líquida de frete** = margem bruta − frete. Custos de armazém/produção não estão incluídos.
- **Estoque**: foto semanal tratada como taxa; nunca somado entre semanas. **Ruptura** = estoque = 0; **abaixo do ROP** = estoque < ponto de reposição.
- **Fornecedor crônico**: atraso médio ≥ 3 dias.

## 4. Decisões técnicas
- Granularidade controlada antes de cada join (`validate='m:1'` e `assert` de linhas/receita preservadas).
- Testes: Spearman (rating × rejeição/atraso), Kruskal–Wallis (atraso entre fornecedores), qui-quadrado (rejeição entre tiers). Com n = 20 fornecedores, o poder é baixo; as conclusões são descritivas.
- Sazonalidade vs. estoque: o estoque total cai ao longo do período, o que confunde a leitura; por isso comparei Q4 com Q3 em cada ano, verifiquei que o ROP é estático e repeti as correlações com tendência removida.
- Redistribuição entre armazéns: cobertura depende do critério de excedente (k × ROP); apresentei faixa de sensibilidade (k = 2, 3, 5) em vez de um número único.
- Simulação de frete: kg × custo/kg do modal × fator de calibração (correlação 0,99 com o frete real). É hipótese; não modela prazo nem nível de serviço.

## 5. Limitações
Dados sintéticos; 24 meses (dois ciclos sazonais); 20 fornecedores; resultados são associações, não causas; moeda não informada (valores em unidades monetárias).


---

## 6. Atrasos de fornecedores (notebook `Case_Meridian_Atrasos_Fornecedores`)

**Definições.** Atraso = entrega real − prometida (dias; 2.807 entregues). Classes: antecipada (< 0), pontual (= 0), atrasada (> 0). *Atraso grave* = 4 dias ou mais (corte justificado pela quebra de patamar na distribuição; é escolha analítica). *Frequência* = % de entregas atrasadas; *severidade/gravidade* = atraso médio só entre as atrasadas; frequência ≠ gravidade.

**Qualidade dos dados.** 17 verificações (tipos, datas, ausentes, duplicidades, negativos, consistência entre datas, nomenclatura, registros incompletos). Única ação: conversão de datas (texto → datetime) na **cópia tratada**; a base original fica intacta e nada foi excluído. Pedidos cancelados (193) ficam fora das métricas de atraso, sem imputação (o nulo tem significado). Calcular o % de atraso sobre os 3.000 pedidos (em vez dos 2.807 entregues) subestimaria o problema.

**Risco.** Alto = gravidade atípica (z modificado de Iglewicz–Hoaglin > 3,5 na severidade) **e** frequência acima da mediana com significância (binomial unilateral, correção de Holm, α = 5%). Intermediário = um critério; baixo = nenhum. K-Means (k por silhueta; k = 2, silhueta 0,93) confirma os mesmos 3 fornecedores. Resultado bimodal (3 / 0 / 17): não há zona intermediária.

**Outliers.** IQR (1,5× e 3×), z, percentis, boxplots e Isolation Forest (contaminação 5% = parâmetro de sensibilidade). Os atrasos extremos são eventos reais (datas coerentes, concentrados nos 3 fornecedores); **mantidos**. Comparação com/sem outliers: a lista de fornecedores crônicos não muda.

**Estatística.** Atraso é discreto e assimétrico (Shapiro–Wilk rejeita normalidade): testes não paramétricos (Kruskal–Wallis, Mann–Whitney, Spearman, Kendall, Fisher). Regressões (logística e OLS com erros HC3) ajustadas nos 17 fornecedores normais, pois os 3 crônicos atrasam 100% (separação perfeita). País/tier parecem associados, mas o efeito desaparece sem os 3 crônicos (**confundimento**). Muitos testes: resultados isolados com p entre 0,01 e 0,05 não são tratados como achados.

**Modelos.** Regressão logística, árvore (prof. 3), Random Forest e Gradient Boosting, mais linha de base; regressão linear/RF/GB para os dias de atraso. Validação: temporal (2024 → 2025), 5-fold estratificada e *leave-one-supplier-out*. **Vazamento:** nenhuma variável pós-entrega (qtd. recebida/rejeitada, gasto realizado, classe, datas reais) entra nos modelos; atributos de cadastro do fornecedor funcionam como identificador, então o desempenho alto com eles é memorização (AUC ≈ 0,99), enquanto só com atributos do pedido o AUC é ≈ 0,5 e em fornecedor novo o F1 é 0. Uma regra simples por fornecedor iguala o ML (F1 0,95). Valor dos modelos = diagnóstico, não previsão. O AUC agrupado em *leave-one-supplier-out* foi omitido de propósito. Séries temporais (ARIMA/Prophet) **não** foram aplicadas: 24 meses não separam sazonalidade de ruído.

**Impacto.** Medido: sem prêmio de preço, volume parecido, mesma rejeição, mesma proporção de pedidos urgentes. Testado sem evidência: ruptura (painel produto × mês com efeitos fixos) e nível de serviço (limitação: os 40 produtos são comprados dos mesmos 20 fornecedores e não há ligação compra ↔ remessa; ausência de evidência não é evidência de ausência). Hipótese: exposição de estoque em atrasos de 10–14 dias vs cobertura do ROP (~27 dias em mediana). Não mensurável: custo financeiro, multas, planejamento, experiência do cliente. **Nenhum valor em R$ é atribuído aos atrasos.** Fornecedor "estratégico" não existe na base; foram comparadas três definições (fonte única, tier A, top 25% do gasto).

**Validação.** 13 verificações de coerência no notebook (reconciliação de totais, recálculo independente do atraso, treino/teste disjuntos, ausência de vazamento, robustez a outliers). O app Streamlit usa o mesmo módulo (`atrasos_lib.py`) e exibe uma tabela de conferência com o notebook.

**Limitações.** Dados sintéticos (padrão de 3 crônicos + 17 homogêneos é improvável em dados reais); n = 20 fornecedores; 24 meses; resultados são associações, não causas; o dashboard publicado como artefato (versão anterior) **não** inclui esta análise: ela está no app Streamlit.
