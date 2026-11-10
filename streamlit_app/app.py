"""Meridian Components — Análise de Supply Chain (app Streamlit).

Como rodar:
    pip install -r requirements.txt
    # opção 1: pasta com os 11 CSVs do dataset (nadhirmazariboufares/supply-chain-dataset)
    CASE_DATA_DIR=./data streamlit run app.py
    # opção 2: sem variável, o app baixa o dataset via kagglehub
"""
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pagina_atrasos

st.set_page_config(page_title="Meridian · Supply Chain", page_icon="📦", layout="wide")

AZUL, LARANJA, CINZA = "#2a78d6", "#d9531e", "#898781"
TABELAS = ["carrier", "customer", "inventory_snapshot", "product", "production_run",
           "purchase_order", "purchase_order_line", "shipment", "shipment_line",
           "supplier", "warehouse"]


def br(v, d=1):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner="Carregando dados…")
def carregar():
    pasta = os.environ.get("CASE_DATA_DIR")
    if not pasta or not Path(pasta).exists():
        import kagglehub
        pasta = kagglehub.dataset_download("nadhirmazariboufares/supply-chain-dataset")
    arquivos = {p.stem: p for p in Path(pasta).rglob("*.csv")}
    faltam = [t for t in TABELAS if t not in arquivos]
    if faltam:
        st.error(f"CSVs ausentes: {faltam}")
        st.stop()
    t = {n: pd.read_csv(arquivos[n]) for n in TABELAS}
    for tab, cols in {"purchase_order": ["order_date", "promised_date", "actual_delivery_date"],
                      "shipment": ["ship_date", "delivery_date"],
                      "inventory_snapshot": ["snapshot_date"]}.items():
        for c in cols:
            t[tab][c] = pd.to_datetime(t[tab][c])

    # Compras: atraso só em pedidos entregues
    po = t["purchase_order"].merge(t["supplier"], on="supplier_id", validate="m:1")
    ent = po[po["status"] == "Delivered"].copy()
    ent["atraso"] = (ent["actual_delivery_date"] - ent["promised_date"]).dt.days
    linhas = t["purchase_order_line"].merge(po[["po_id", "supplier_id"]], on="po_id", validate="m:1")
    rej = linhas.groupby("supplier_id")[["qty_received", "qty_rejected"]].sum()
    rej["rejeicao"] = rej["qty_rejected"] / rej["qty_received"]

    # Vendas: linha de remessa + cabeçalho + modal + produto
    v = (t["shipment_line"]
         .merge(t["shipment"], on="shipment_id", validate="m:1")
         .merge(t["carrier"][["carrier_id", "mode"]], on="carrier_id", validate="m:1")
         .merge(t["product"][["product_id", "product_code", "category", "standard_cost", "unit_weight_kg"]],
                on="product_id", validate="m:1")
         .merge(t["customer"][["customer_id", "industry_segment"]], on="customer_id", validate="m:1"))
    assert len(v) == len(t["shipment_line"])
    v["custo"] = v["qty_shipped"] * v["standard_cost"]
    v["ano"] = v["ship_date"].dt.year
    v["mes"] = v["ship_date"].dt.month

    inv = t["inventory_snapshot"].merge(t["product"][["product_id", "product_code", "category"]],
                                        on="product_id", validate="m:1")
    inv["ruptura"] = inv["units_on_hand"] == 0
    inv["abaixo_rop"] = inv["units_on_hand"] < inv["reorder_point"]
    inv["ano"] = inv["snapshot_date"].dt.year
    inv["mes"] = inv["snapshot_date"].dt.month
    return ent, rej, t["supplier"], v, inv


ent, rej, sup, vendas, inv = carregar()

# ---------------- Sidebar ----------------
st.sidebar.title("Filtros")
anos = st.sidebar.multiselect("Ano (vendas e estoque)", sorted(vendas["ano"].unique()),
                              default=sorted(vendas["ano"].unique()))
cats = st.sidebar.multiselect("Categoria", sorted(vendas["category"].unique()),
                              default=sorted(vendas["category"].unique()))
modais = st.sidebar.multiselect("Modal de frete", sorted(vendas["mode"].unique()),
                                default=sorted(vendas["mode"].unique()))
st.sidebar.caption("Atraso e rejeição (aba Fornecedores) usam todo o período.")
v = vendas[vendas["ano"].isin(anos) & vendas["category"].isin(cats) & vendas["mode"].isin(modais)]
iv = inv[inv["ano"].isin(anos) & inv["category"].isin(cats)]

st.title("📦 Meridian Components — Análise de Supply Chain")
st.caption("Dados sintéticos (Kaggle), 2024–2025. Margem bruta = receita − qtd × custo padrão; "
           "margem líquida = margem bruta − frete. Resultados são associações, não causas.")

if v.empty:
    st.warning("Nenhum dado para os filtros escolhidos.")
    st.stop()

rec, cus, fre = v["revenue"].sum(), v["custo"].sum(), v["freight_cost"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Receita (mi)", br(rec / 1e6))
c2.metric("Margem bruta", br((rec - cus) / rec * 100) + "%")
c3.metric("Frete / receita", br(fre / rec * 100) + "%")
c4.metric("Margem líquida de frete", br((rec - cus - fre) / rec * 100) + "%")

# ---------------- Números do notebook + apoio de leitura ----------------
def _carregar_k():
    arq = Path(__file__).parent / "resultados" / "resultados.json"
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None


KP = _carregar_k()
pc0 = lambda x: br(x * 100, 0) + "%"
pc1 = lambda x: br(x * 100, 1) + "%"


def leitura_topo(frase):
    """Uma frase simples no topo da aba: a conclusão antes do gráfico."""
    st.markdown(f"#### {frase}")


def leitura_base(significa, fazer, selo=None):
    """Duas caixas ao fim da aba: o que significa e o que fazer primeiro."""
    c1, c2 = st.columns(2)
    c1.info("**O que isso significa**\n\n" + significa)
    c2.success("**O que fazer primeiro**\n\n" + fazer)
    if selo:
        st.caption(selo)


# ---------------- A história em 1 minuto ----------------
def _historia():
    K = KP
    if K is None:
        return
    with st.container(border=True):
        st.markdown("### 📖 A história em 1 minuto")
        st.markdown(
            f"A Meridian **vende bem e tem boa margem** ({pc0(K['mb_pct'])} antes do frete), mas os dados mostram "
            "**quatro pontos de atenção**. Cada um é explicado em uma aba abaixo, com o gráfico que prova. "
            "*Os números desta caixa usam os dois anos completos e não mudam com os filtros.*")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"#### 1. 🚚 Poucos fornecedores concentram os atrasos\n"
                f"**{pc0(K['pct_atraso_geral'])}** das entregas chegam depois do prometido, quase sempre por poucos dias. "
                f"O problema sério está em **3 fornecedores** ({', '.join(K['cronicos'])}): atrasam em **todas** as entregas, "
                f"cerca de **{br(sum(K['cron_atraso']) / 3, 0)} dias**. São 15% dos fornecedores, mas "
                f"**{pc0(K['cron_share_dias'])} de todos os dias de atraso**. Ter nota alta de qualidade não protege "
                f"(o fornecedor com a maior nota, {K['rating_top_nome']}, é um deles).\n\n"
                f"👉 *Abas **1 · Fornecedores** e **5 · Atrasos de fornecedores**.*")
            st.markdown(
                f"#### 3. 📉 O calendário é muito marcado\n"
                f"As vendas **caem cerca de {pc0(abs(K['ago_vs']))} em agosto** e **sobem cerca de {pc0(K['q4_vs'])} de outubro a dezembro**, "
                f"em relação aos outros meses. O padrão se repete em 2024 e 2025 e em todos os segmentos de clientes, "
                f"então dá para planejar com antecedência.\n\n"
                f"👉 *Aba **3 · Calendário**.*")
        with c2:
            st.markdown(
                f"#### 2. 💸 O frete \"come\" a margem de um produto\n"
                f"Depois do frete, a margem cai de {pc1(K['mb_pct'])} para **{pc1(K['ml_pct'])}**. "
                f"A categoria **Fasteners** vale só {pc1(K['fas_share_receita'])} da receita, mas o frete dela equivale a "
                f"**{pc0(K['fas_frete_pct'])} do que ela vende**: a margem líquida fica **negativa ({pc0(K['fas_ml_pct'])})**. "
                f"O frete aéreo concentra **{pc0(K['air_share_frete'])} do gasto com frete**, e o tempo real de trânsito é "
                f"parecido em todos os modais (~5 dias).\n\n"
                f"👉 *Aba **2 · Frete e margem**.*")
            st.markdown(
                f"#### 4. 📦 Falta de estoque aumenta no fim do ano\n"
                f"**{pc0(K['abaixo_geral'])}** das posições de estoque ficam abaixo do ponto de reposição e **{pc1(K['ruptura_geral'])}** chegam a zero. "
                f"No 4º trimestre a ruptura sobe ({pc1(K['ruptura_q3'])} no 3º tri para {pc1(K['ruptura_q4'])} no 4º). "
                f"O ponto de reposição é **fixo o ano todo**, enquanto a demanda muda. Isso é uma **hipótese** a testar: "
                f"os dados mostram a coincidência, não provam a causa.\n\n"
                f"👉 *Aba **4 · Estoque**.*")
        st.caption("✅ **Fato** = medido direto nos dados · 🔗 **Associação** = aparecem juntos, sem provar causa · "
                   "💡 **Hipótese** = explicação possível que os dados não confirmam. Dados sintéticos (Kaggle).")


_historia()

aba1, aba2, aba3, aba4, aba5 = st.tabs(["1 · Fornecedores", "2 · Frete e margem", "3 · Calendário", "4 · Estoque", "5 · Atrasos de fornecedores"])

# ---------------- Fornecedores ----------------
with aba1:
    f = ent.groupby(["supplier_id", "supplier_name", "tier", "quality_rating"]).agg(
        atraso_medio=("atraso", "mean"), pct_atrasado=("atraso", lambda s: (s > 0).mean()),
        pedidos=("po_id", "count")).reset_index().merge(rej[["rejeicao"]], on="supplier_id")
    ordem = f.sort_values("atraso_medio", ascending=False)["supplier_name"].tolist()
    f["cronico"] = f["atraso_medio"] >= 3
    if KP:
        leitura_topo(f"Três fornecedores concentram os atrasos; os outros {len(f) - 3} atrasam pouco")
    st.subheader("Atraso médio por fornecedor (dias, só pedidos entregues)")
    ch = alt.Chart(f).mark_bar().encode(
        x=alt.X("supplier_name:N", sort=ordem, title=None),
        y=alt.Y("atraso_medio:Q", title="Atraso médio (dias)"),
        color=alt.condition("datum.cronico", alt.value(LARANJA), alt.value(CINZA)),
        tooltip=["supplier_name", "tier", alt.Tooltip("atraso_medio:Q", format=".1f"),
                 alt.Tooltip("pct_atrasado:Q", format=".0%"), "pedidos"]).properties(height=320)
    st.altair_chart(ch, width="stretch")
    cr = f[f["cronico"]]
    cron_dias = ent[ent["supplier_id"].isin(cr["supplier_id"])]["atraso"].clip(lower=0).sum()
    tot_dias = ent["atraso"].clip(lower=0).sum()
    st.info(f"Fornecedores com atraso médio ≥ 3 dias: **{', '.join(cr['supplier_name'])}** — "
            f"{br(cron_dias / tot_dias * 100, 0)}% dos dias de atraso acumulados.")

    if KP:
        leitura_base(
            f"Não é um problema de todos: **{', '.join(KP['cronicos'])}** atrasam em todas as entregas, cerca de "
            f"**{br(sum(KP['cron_atraso']) / 3, 0)} dias**, e somam **{pc0(KP['cron_share_dias'])}** dos dias de atraso com "
            f"{pc0(KP['cron_share_gasto'])} do gasto. Os demais atrasam em média menos de 1 dia. Nota de qualidade alta não protege: "
            f"{KP['rating_top_nome']} tem a maior nota da base e é um dos três.",
            "Abrir um plano de ação com esses três fornecedores e acompanhar o atraso deles todo mês. "
            "Começar a registrar o custo de cada atraso (multa, frete urgente, falta de produto), hoje impossível de medir.",
            "✅ Fato: medido direto nos dados. O custo em reais do atraso não foi estimado porque a base não o traz.")
    st.subheader("Nota de qualidade × taxa de rejeição")
    rho = f["quality_rating"].rank().corr(f["rejeicao"].rank())
    sc = alt.Chart(f).mark_circle(size=140).encode(
        x=alt.X("quality_rating:Q", scale=alt.Scale(zero=False), title="quality_rating"),
        y=alt.Y("rejeicao:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.04]), title="Rejeição"),
        color=alt.Color("tier:N", scale=alt.Scale(range=[AZUL, LARANJA, "#1baf7a"])),
        tooltip=["supplier_name", "tier", "quality_rating", alt.Tooltip("rejeicao:Q", format=".2%")]
    ).properties(height=320)
    st.altair_chart(sc, width="stretch")
    st.caption(f"A rejeição é parecida em todos os fornecedores (cerca de {pc1(KP['rej_geral']) if KP else '2,4%'}) e a nota de "
               f"qualidade não a explica (correlação de Spearman ρ = {br(rho, 2)}, n = {len(f)}). Correlação não prova causa.")
    st.dataframe(f.drop(columns=["cronico"]).sort_values("atraso_medio", ascending=False)
                 .style.format({"atraso_medio": "{:.1f}", "pct_atrasado": "{:.0%}", "rejeicao": "{:.2%}"}),
                 width="stretch", hide_index=True)

# ---------------- Margem e frete ----------------
with aba2:
    g = v.groupby("category").agg(receita=("revenue", "sum"), custo=("custo", "sum"),
                                  frete=("freight_cost", "sum")).reset_index()
    g["mb"] = (g.receita - g.custo) / g.receita
    g["ml"] = (g.receita - g.custo - g.frete) / g.receita
    g["frete_pct"] = g.frete / g.receita
    if KP:
        leitura_topo(f"O frete tira {br((KP['mb_pct'] - KP['ml_pct']) * 100, 1)} pontos de margem, e a perda está quase toda em Fasteners")
    st.subheader("Margem depois do frete, por categoria")
    bars = alt.Chart(g).mark_bar().encode(
        y=alt.Y("category:N", sort="-x", title=None),
        x=alt.X("ml:Q", axis=alt.Axis(format="%"), title="Margem líquida (% da receita)"),
        color=alt.condition("datum.ml < 0", alt.value(LARANJA), alt.value(AZUL)),
        tooltip=["category", alt.Tooltip("ml:Q", format=".1%"), alt.Tooltip("mb:Q", format=".1%"),
                 alt.Tooltip("frete_pct:Q", format=".1%")]).properties(height=260)
    st.altair_chart(bars, width="stretch")

    st.subheader("Margem depois do frete por categoria e transporte (azul = lucro, laranja = prejuízo)")
    m = v.groupby(["category", "mode"]).agg(receita=("revenue", "sum"), custo=("custo", "sum"),
                                            frete=("freight_cost", "sum")).reset_index()
    m["ml"] = (m.receita - m.custo - m.frete) / m.receita
    m["ml_cor"] = m["ml"].clip(-0.5, 0.5)
    m["rot"] = (m["ml"] * 100).round(0).astype(int).astype(str) + "%"
    base = alt.Chart(m).encode(x=alt.X("mode:N", title=None), y=alt.Y("category:N", title=None))
    heat = base.mark_rect().encode(color=alt.Color("ml_cor:Q", scale=alt.Scale(
        domain=[-0.5, 0, 0.5], range=[LARANJA, "#f0efec", AZUL]), legend=None),
        tooltip=["category", "mode", alt.Tooltip("ml:Q", format=".1%")])
    st.altair_chart((heat + base.mark_text(fontSize=14).encode(text="rot:N")).properties(height=300),
                    width="stretch")

    if KP:
        leitura_base(
            f"Fasteners vale só {pc1(KP['fas_share_receita'])} da receita, mas o frete dele equivale a **{pc0(KP['fas_frete_pct'])} do que ele vende**: "
            f"a margem depois do frete fica em **{pc0(KP['fas_ml_pct'])}**. O transporte aéreo concentra **{pc0(KP['air_share_frete'])}** do gasto com frete, "
            "mas o tempo real de viagem é de uns 5 dias em todos os tipos de transporte.",
            "Rever o transporte de Fasteners e Bearings. A simulação abaixo mostra o efeito de trocar o aéreo por outro modal.",
            "✅ Fato (margens) e 💡 simulação (troca de transporte): a simulação é uma conta com os dados, não um resultado observado.")
    st.subheader("Simulação: trocar o transporte de Fasteners e Bearings")
    destino = st.selectbox("Trocar o transporte aéreo por", ["Road", "Rail", "Sea"], index=1)
    alvo = ["Fasteners", "Bearings"]
    base_v = vendas[vendas["ano"].isin(anos)]
    cpk = (base_v.groupby("mode")["freight_cost"].sum() /
           (base_v["qty_shipped"] * base_v["unit_weight_kg"]).groupby(base_v["mode"]).sum())
    sel = (base_v["category"].isin(alvo)) & (base_v["mode"] == "Air")
    novo = (base_v.loc[sel, "qty_shipped"] * base_v.loc[sel, "unit_weight_kg"]) * cpk[destino]
    economia = base_v.loc[sel, "freight_cost"].sum() - novo.sum()
    r_tot = base_v["revenue"].sum()
    ml_atual = (r_tot - base_v["custo"].sum() - base_v["freight_cost"].sum()) / r_tot
    ml_novo = ml_atual + economia / r_tot
    a, b, c = st.columns(3)
    a.metric("Economia de frete simulada (mi)", br(economia / 1e6))
    b.metric("Margem líquida atual", br(ml_atual * 100) + "%")
    c.metric("Margem líquida simulada", br(ml_novo * 100) + "%")
    st.caption("Simulação simplificada: só Air → modal escolhido; frete = kg × custo médio/kg do modal. "
               "O efeito no prazo não é modelado — trate como hipótese a validar em piloto.")

# ---------------- Sazonalidade ----------------
with aba3:
    mensal = v.groupby(["ano", "mes"])["qty_shipped"].sum().reset_index()
    mensal["ano"] = mensal["ano"].astype(str)
    if KP:
        leitura_topo("As vendas seguem um calendário: caem em agosto e disparam de outubro a dezembro, nos dois anos")
    st.subheader("Unidades enviadas por mês")
    ln = alt.Chart(mensal).mark_line(point=True).encode(
        x=alt.X("mes:O", title="Mês"), y=alt.Y("qty_shipped:Q", title="Unidades"),
        color=alt.Color("ano:N", scale=alt.Scale(range=[CINZA, AZUL])),
        tooltip=["ano", "mes", alt.Tooltip("qty_shipped:Q", format=",.0f")]).properties(height=340)
    st.altair_chart(ln, width="stretch")
    mm = v.groupby(["ano", "mes"])["qty_shipped"].sum().groupby("mes").mean()
    ago = mm.loc[8] / mm.drop([8]).mean() - 1
    q4 = mm.loc[[10, 11, 12]].mean() / mm.drop([10, 11, 12]).mean() - 1
    st.caption(f"Com os filtros atuais: agosto {br(ago * 100, 0)}% e Q4 {br(q4 * 100, 0)}% vs. a média dos demais meses.")
    if KP:
        leitura_base(
            f"Em agosto vende-se cerca de **{pc0(abs(KP['ago_vs']))} menos** e de outubro a dezembro cerca de **{pc0(KP['q4_vs'])} mais**, "
            "comparado aos outros meses. O padrão se repete em 2024 e 2025 e em todos os segmentos de clientes, então é previsível.",
            "Planejar compras e produção pelo calendário: reforçar o estoque antes de outubro e aproveitar agosto para manutenção.",
            "✅ Fato. Os números desta caixa usam os dois anos completos; a linha logo acima do gráfico muda com os filtros.")
    seg = v.groupby(["industry_segment", "mes"])["qty_shipped"].sum().reset_index()
    seg["indice"] = seg["qty_shipped"] / seg.groupby("industry_segment")["qty_shipped"].transform("mean")
    st.subheader("O padrão do ano por segmento de cliente (1 = média do segmento)")
    st.altair_chart(alt.Chart(seg).mark_line().encode(
        x=alt.X("mes:O", title="Mês"), y=alt.Y("indice:Q", title="Índice"),
        color="industry_segment:N", tooltip=["industry_segment", "mes", alt.Tooltip("indice:Q", format=".2f")]
    ).properties(height=300), width="stretch")

# ---------------- Estoque ----------------
with aba4:
    if iv.empty:
        st.warning("Sem dados de estoque para os filtros.")
    else:
        if KP:
            leitura_topo("No fim do ano faltam mais produtos no estoque, e a categoria mais vendida sofre mais")
        st.caption("**Ruptura** = produto zerado no armazém. **Ponto de reposição (ROP)** = nível de estoque em que se pede mais produto; "
                   "'abaixo do ROP' = estoque menor que esse nível. As fotos semanais são tratadas como taxas, nunca somadas.")
        em = iv.groupby(["ano", "mes"]).agg(ruptura=("ruptura", "mean"), abaixo=("abaixo_rop", "mean")).reset_index()
        em["ano"] = em["ano"].astype(str)
        met = st.radio("Indicador", ["ruptura", "abaixo"], horizontal=True,
                       format_func=lambda x: {"ruptura": "% em ruptura", "abaixo": "% abaixo do ROP"}[x])
        st.altair_chart(alt.Chart(em).mark_line(point=True).encode(
            x=alt.X("mes:O", title="Mês"), y=alt.Y(f"{met}:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("ano:N", scale=alt.Scale(range=[CINZA, LARANJA])),
            tooltip=["ano", "mes", alt.Tooltip(f"{met}:Q", format=".1%")]).properties(height=320),
            width="stretch")
        q3 = em[em["mes"].isin([7, 8, 9])]["ruptura"].mean()
        q4i = em[em["mes"].isin([10, 11, 12])]["ruptura"].mean()
        st.caption(f"Ruptura média: Q3 {br(q3 * 100)}% → Q4 {br(q4i * 100)}% (com os filtros atuais).")
        cat = iv.groupby("category").agg(abaixo=("abaixo_rop", "mean"), ruptura=("ruptura", "mean")).reset_index()
        st.subheader("% das posições de estoque abaixo do ponto de reposição, por categoria")
        st.altair_chart(alt.Chart(cat).mark_bar().encode(
            y=alt.Y("category:N", sort="-x", title=None), x=alt.X("abaixo:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.condition(alt.datum.abaixo > 0.25, alt.value(LARANJA), alt.value(AZUL)),
            tooltip=["category", alt.Tooltip("abaixo:Q", format=".1%"), alt.Tooltip("ruptura:Q", format=".1%")]
        ).properties(height=240), width="stretch")
        sku = (iv.groupby("product_code").agg(abaixo=("abaixo_rop", "mean"), ruptura=("ruptura", "mean"))
               .sort_values("abaixo", ascending=False).head(10).reset_index())
        st.subheader("10 produtos que mais ficam abaixo do ponto de reposição")
        st.dataframe(sku.rename(columns={"product_code": "produto", "abaixo": "% abaixo do ROP", "ruptura": "% zerado"})
                     .style.format({"% abaixo do ROP": "{:.0%}", "% zerado": "{:.0%}"}), hide_index=True,
                     width="stretch")
        if KP:
            leitura_base(
                f"**{pc0(KP['abaixo_geral'])}** das posições ficam abaixo do ponto de reposição e **{pc1(KP['ruptura_geral'])}** chegam a zero. "
                f"No 4º trimestre as faltas sobem de {pc1(KP['ruptura_q3'])} para {pc1(KP['ruptura_q4'])}. O ponto de reposição é o mesmo o ano todo, "
                f"enquanto a demanda muda. Hydraulics ({pc0(KP['hyd_receita_share'])} da receita) tem {pc0(KP['abaixo_cat']['Hydraulics'])} das posições abaixo do ponto.",
                "Testar pontos de reposição que mudem com a época do ano, começando por Hydraulics e pelos produtos da tabela acima.",
                "💡 Hipótese: a ligação entre ponto de reposição fixo e faltas é uma explicação possível; os dados mostram que os dois coincidem, não que um causa o outro. "
                "Nesta base também não achamos ligação entre atrasos de fornecedores e faltas de estoque, mas o desenho dos dados dificilmente a mostraria.")

# ---------------- Atrasos de fornecedores ----------------
with aba5:
    pagina_atrasos.render()
