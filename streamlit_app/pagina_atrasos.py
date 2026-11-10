"""Aba 'Atrasos de fornecedores' do app Meridian.

Usa a MESMA biblioteca do notebook (atrasos_lib.py), então os números do app e do notebook vêm do mesmo código.
Os filtros (período, fornecedor, tier, país, urgência) valem para todos os painéis; as tabelas de modelos são
resultados fixos do notebook (base completa) e NÃO respondem aos filtros, o que está dito na própria aba.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import atrasos_lib as L

AZUL, LARANJA, AMARELO, CINZA, VERDE = "#2a78d6", "#eb6834", "#e8b923", "#b9b8ae", "#1baf7a"
ESC_RISCO = alt.Scale(domain=["Alto", "Intermediário", "Baixo"], range=[LARANJA, AMARELO, AZUL])
ESC_CLASSE = alt.Scale(domain=["Antecipada", "Pontual", "Atrasada"], range=[VERDE, CINZA, LARANJA])
PASTA_RES = Path(__file__).parent / "resultados"


def br(v, d=1):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pc(v, d=0):
    return br(v * 100, d) + "%"


# ------------------------------------------------------------------------------------------------ dados
@st.cache_data(show_spinner="Preparando a base de atrasos…")
def base():
    brutos = L.carregar_brutos(L.localizar_dados())
    T = L.tratar(brutos)
    ped = L.base_pedidos(T)
    return ped, L.auditoria(brutos, T), L.antes_depois(brutos, T, ped)


@st.cache_data(show_spinner="Calculando classificação de risco e segmentos…")
def analise(ped_filtrado):
    f = L.metricas_fornecedor(ped_filtrado)
    r = L.classificar_risco(f)
    try:
        g, tab_k = L.agrupar(r)
    except Exception:  # base filtrada pequena demais para agrupar
        g, tab_k = r.assign(grupo="—"), pd.DataFrame(columns=["k", "inércia", "silhueta"])
    return r, g, tab_k, float(r.attrs.get("pct_base_mediana", np.nan))


@st.cache_data(show_spinner="Ajustando modelos de exemplo (2024 → 2025)…")
def curvas(ped):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    em = L.matriz_modelo(ped)
    y = em["grave"].astype(int).values
    tr, te = (em.ano == 2024).values, (em.ano == 2025).values
    out = {}
    for rot, (num, cat) in L.conjuntos_de_atributos().items():
        pipe = L._pipe(GradientBoostingClassifier(random_state=L.SEED), num, cat).fit(em.loc[tr, num + cat], y[tr])
        p = pipe.predict_proba(em.loc[te, num + cat])[:, 1]
        fpr, tpr, _ = roc_curve(y[te], p)
        out[rot] = dict(fpr=fpr, tpr=tpr, auc=float(roc_auc_score(y[te], p)),
                        cm=confusion_matrix(y[te], (p >= .5).astype(int)).tolist())
    imp = L.importancia_permutacao(em, "grave", n_rep=3)
    return out, imp


def carregar_resultados():
    res = {}
    for nome in ["modelos_classificacao.csv", "modelos_regressao.csv", "insights.csv"]:
        p = PASTA_RES / nome
        res[nome] = pd.read_csv(p) if p.exists() else None
    p = PASTA_RES / "resultados_atrasos.json"
    res["K"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return res


@st.cache_data(show_spinner="Rodando os modelos (pode levar ~1 min)…")
def modelos_na_hora(ped):
    em = L.matriz_modelo(ped)
    g, _ = L.avaliar_classificacao(em, "grave")
    a, _ = L.avaliar_classificacao(em, "atrasado")
    g.insert(0, "alvo", "grave (≥ 4 dias)")
    a.insert(0, "alvo", "atrasado (> 0)")
    return pd.concat([g, a], ignore_index=True), L.avaliar_regressao(em)


# ------------------------------------------------------------------------------------------------ página
def render():
    ped, aud, antes_depois = base()
    res = carregar_resultados()
    K = res["K"]

    st.subheader("Atrasos de fornecedores")
    st.markdown(
        "#### Resumindo\n"
        f"- **{pc(K['pct_atr'], 0)} das entregas chegam depois do prometido**, mas quase sempre por poucos dias "
        f"(metade atrasa até {br(K['atraso_mediano'], 0)} dia).\n"
        f"- **O atraso grave (4 dias ou mais) está concentrado em 3 fornecedores**: {', '.join(K['cron_nomes'])}. "
        f"Eles atrasam em todas as entregas e respondem por {pc(K['cron_share_dias'], 0)} de todos os dias de atraso.\n"
        "- **Não é o tamanho do pedido, o país nem o tipo de fornecedor**: o que explica o atraso é *quem entrega*.\n"
        "- **Quanto o atraso custa em dinheiro não dá para medir** com estes dados, então nenhum valor foi estimado.")
    st.caption("**Como medimos o atraso:** dia em que a entrega chegou menos o dia prometido (só pedidos entregues). "
               "Prometeu na segunda e chegou na quinta? São 3 dias. Número negativo = chegou antes; zero = chegou no dia. "
               "**Atraso grave = 4 dias ou mais.** *Frequência* = em quantas entregas atrasa. *Gravidade* = quantos dias atrasa quando atrasa. "
               "Os resultados mostram relações entre as coisas, não causas.")

    # ---------------- filtros (valem para todos os painéis desta aba) ----------------
    meses = sorted(ped["ano_mes"].unique())
    forn_all = sorted(ped["fornecedor"].unique())
    with st.expander("🔎 Filtros desta aba", expanded=True):
        c1, c2, c3, c4 = st.columns([1.4, 1.2, 1, 1])
        periodo = c1.select_slider("Período (mês do pedido)", options=meses, value=(meses[0], meses[-1]))
        sel_forn = c2.multiselect("Fornecedor (vazio = todos)", forn_all)
        tiers = c3.multiselect("Tier", sorted(ped["tier"].unique()), help="Nível do fornecedor (A, B ou C).")
        paises = c4.multiselect("País", sorted(ped["pais"].unique()))
        urg = st.radio("Urgência do pedido", ["Todos", "Só urgentes", "Só normais"], horizontal=True)
        st.caption("**Período** e **urgência** mudam a base do cálculo (inclusive o risco de cada fornecedor). "
                   "**Fornecedor, tier e país** só escolhem o que aparece na tela.")

    d = ped[(ped.ano_mes >= periodo[0]) & (ped.ano_mes <= periodo[1])]
    if urg.startswith("Só urgentes"):
        d = d[d.expedited == 1]
    elif urg.startswith("Só normais"):
        d = d[d.expedited == 0]
    ent = d[d.entregue].copy()
    if len(ent) < 100:
        st.warning("Poucos pedidos entregues no filtro atual (< 100): amplie o período ou a urgência.")
        return

    r, g, tab_k, base_mediana = analise(d)
    visiveis = set(r.fornecedor)
    if sel_forn:
        visiveis &= set(sel_forn)
    if tiers:
        visiveis &= set(r[r.tier.isin(tiers)].fornecedor)
    if paises:
        visiveis &= set(r[r.pais.isin(paises)].fornecedor)
    if not visiveis:
        st.warning("Nenhum fornecedor atende aos filtros escolhidos.")
        return
    risco_de = r.set_index("fornecedor")["risco"].to_dict()
    ent["risco"] = ent.fornecedor.map(risco_de)
    ev = ent[ent.fornecedor.isin(visiveis)].copy()   # entregas dos fornecedores exibidos
    rv = r[r.fornecedor.isin(visiveis)].copy()

    abas = st.tabs(["Indicadores gerais", "Ranking", "Evolução", "Matriz de risco", "Conclusões e recomendações"])

    # ================================================================ 1. Indicadores gerais
    with abas[0]:
        k = st.columns(6)
        k[0].metric("Entregas analisadas", br(len(ev), 0))
        k[1].metric("% atrasadas", pc(ev.atrasado.mean(), 1))
        k[2].metric("% pontuais", pc((ev.atraso == 0).mean(), 1))
        k[3].metric("% antecipadas", pc((ev.atraso < 0).mean(), 1))
        k[4].metric("Atraso mediano", f"{br(ev.atraso.median(), 0)} dia(s)")
        k[5].metric("% atraso grave (4+ dias)", pc(ev.grave.mean(), 1))
        k = st.columns(4)
        k[0].metric("Atraso médio", f"{br(ev.atraso.mean(), 2)} dias")
        k[1].metric("Atraso nos 10% piores casos", f"{br(ev.atraso.quantile(.9), 0)} dias")
        atr = ev[ev.atraso > 0]
        k[2].metric("Atraso médio, só das atrasadas", f"{br(atr.atraso.mean(), 1)} dias" if len(atr) else "—")
        k[3].metric("Total de dias de atraso", br(atr.atraso.sum(), 0))
        cl = ev.assign(classe=lambda x: np.select([x.atraso < 0, x.atraso == 0], ["Antecipada", "Pontual"], "Atrasada"))
        cc = cl.groupby("classe").size().rename("n").reset_index()
        cc["pct"] = cc.n / cc.n.sum()
        st.altair_chart(alt.Chart(cc).mark_bar().encode(
            y=alt.Y("classe:N", sort=["Antecipada", "Pontual", "Atrasada"], title=None),
            x=alt.X("pct:Q", axis=alt.Axis(format="%"), title="% das entregas"),
            color=alt.Color("classe:N", scale=ESC_CLASSE, legend=None),
            tooltip=["classe", "n", alt.Tooltip("pct:Q", format=".1%")]).properties(height=140), width="stretch")
        n_alto = int((rv.risco == "Alto").sum())
        st.info(f"Entre os fornecedores exibidos, **{n_alto}** estão classificados como **alto risco** no período. "
                f"Atraso grave: **{int(ev.grave.sum())}** entregas ({pc(ev.grave.mean(), 1)}).")
        with st.expander("🧪 Dá para confiar nos dados? (3.000 pedidos)"):
            st.markdown("Nenhum dado foi descartado: só converti as datas de texto para data de verdade. "
                        "Os pedidos cancelados ficam de fora do cálculo de atraso, porque não têm data de entrega.")
            st.dataframe(aud.drop(columns=["#"]), hide_index=True, width="stretch")
            st.markdown("**Antes × depois do tratamento**")
            st.dataframe(antes_depois.astype(str), hide_index=True, width="stretch")

    # ================================================================ 2. Ranking
    with abas[1]:
        st.markdown("**Quem atrasa mais?** À esquerda, quantos dias atrasa quando atrasa; à direita, em quantas entregas atrasa.")
        ordem = rv.sort_values("severidade", ascending=False)
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(ordem).mark_bar().encode(
            y=alt.Y("fornecedor:N", sort=alt.EncodingSortField("severidade", order="descending"), title=None),
            x=alt.X("severidade:Q", title="Dias de atraso, quando atrasa"),
            color=alt.Color("risco:N", scale=ESC_RISCO, legend=alt.Legend(title="Risco")),
            tooltip=["fornecedor", "tier", "pais", alt.Tooltip("severidade:Q", format=".1f"),
                     alt.Tooltip("pct_atrasado:Q", format=".0%"), "pedidos"]).properties(height=22 * len(ordem) + 40,
                                                                                  title="Gravidade"), width="stretch")
        c2.altair_chart(alt.Chart(ordem).mark_bar().encode(
            y=alt.Y("fornecedor:N", sort=alt.EncodingSortField("severidade", order="descending"), title=None),
            x=alt.X("pct_atrasado:Q", axis=alt.Axis(format="%"), title="% das entregas atrasadas", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("risco:N", scale=ESC_RISCO, legend=None),
            tooltip=["fornecedor", alt.Tooltip("pct_atrasado:Q", format=".0%"), "pedidos"]).properties(
            height=22 * len(ordem) + 40, title="Frequência"), width="stretch")
        tab = ordem[["fornecedor", "risco", "tier", "pais", "pedidos", "pct_antecipado", "pct_pontual", "pct_atrasado",
                     "atraso_medio", "atraso_mediano", "severidade", "desvio", "p90", "maximo", "pct_grave",
                     "share_dias_atraso", "share_gasto"]]
        tab.columns = ["fornecedor", "risco", "tier", "país", "pedidos", "% antecip.", "% pontual", "% atrasado", "atraso médio",
                       "mediana", "gravidade", "desvio padrão", "P90", "máx.", "% grave", "% dos dias de atraso", "% do gasto"]
        st.dataframe(tab.style.format({"% antecip.": "{:.0%}", "% pontual": "{:.0%}", "% atrasado": "{:.0%}", "atraso médio": "{:.2f}", "mediana": "{:.1f}",
                                       "gravidade": "{:.1f}", "desvio padrão": "{:.1f}", "P90": "{:.1f}", "% grave": "{:.0%}",
                                       "% dos dias de atraso": "{:.1%}", "% do gasto": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption("Quanto maiores o desvio padrão e o P90, **menos previsível** é o prazo do fornecedor. "
                   "(P90 = o atraso que só 10% das entregas ultrapassam.)")

    # ================================================================ 3. Evolução
    with abas[2]:
        m = ev.groupby("ano_mes").agg(pct_atrasado=("atrasado", "mean"), atraso_medio=("atraso", "mean"), n=("po_id", "count")).reset_index()
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(m).mark_line(point=True, color=AZUL).encode(
            x=alt.X("ano_mes:O", title="Mês do pedido", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("pct_atrasado:Q", axis=alt.Axis(format="%"), title="% atrasadas", scale=alt.Scale(zero=False)),
            tooltip=["ano_mes", alt.Tooltip("pct_atrasado:Q", format=".1%"), "n"]).properties(height=300, title="% de entregas atrasadas por mês"),
            width="stretch")
        gp = ev.assign(grupo=np.where(ev.risco == "Alto", "Alto risco", "Demais")).groupby(["ano_mes", "grupo"]).atraso.mean().reset_index()
        c2.altair_chart(alt.Chart(gp).mark_line(point=True).encode(
            x=alt.X("ano_mes:O", title="Mês do pedido", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("atraso:Q", title="Atraso médio (dias)"),
            color=alt.Color("grupo:N", scale=alt.Scale(domain=["Alto risco", "Demais"], range=[LARANJA, AZUL]), legend=alt.Legend(title=None)),
            tooltip=["ano_mes", "grupo", alt.Tooltip("atraso:Q", format=".2f")]).properties(height=300, title="Atraso médio: alto risco × demais"),
            width="stretch")
        if len(m) >= 8:
            from scipy import stats as sp
            tau = sp.kendalltau(np.arange(len(m)), m.pct_atrasado)
            st.caption(f"{'Não há tendência clara de piora nem de melhora ao longo dos meses.' if tau.pvalue > .05 else 'Há uma tendência clara ao longo dos meses.'} "
                       f"(Teste de Kendall: τ = {br(tau.statistic, 2)}, p = {br(tau.pvalue, 2)}.) Com 24 meses, a leitura é descritiva.")
        st.markdown("**Por fornecedor** (até 6 escolhidos)")
        esc = sel_forn[:6] if sel_forn else rv.sort_values("severidade", ascending=False).fornecedor.head(3).tolist() + \
            rv.sort_values("severidade").fornecedor.head(1).tolist()
        pf = ev[ev.fornecedor.isin(esc)].groupby(["ano_mes", "fornecedor"]).atraso.mean().reset_index()
        st.altair_chart(alt.Chart(pf).mark_line(point=True).encode(
            x=alt.X("ano_mes:O", title="Mês do pedido", axis=alt.Axis(labelAngle=-45)), y=alt.Y("atraso:Q", title="Atraso médio (dias)"),
            color=alt.Color("fornecedor:N", legend=alt.Legend(title=None)),
            tooltip=["ano_mes", "fornecedor", alt.Tooltip("atraso:Q", format=".2f")]).properties(height=300), width="stretch")

    # ================================================================ 7. Matriz de risco
    with abas[3]:
        sev_lim = float(r.severidade.median() + 3.5 * np.median(np.abs(r.severidade - r.severidade.median())) / 0.6745)
        mm = r.assign(destaque=r.fornecedor.isin(visiveis))
        pontos = alt.Chart(mm).mark_circle(stroke="white", strokeWidth=1.2).encode(
            x=alt.X("pct_atrasado:Q", axis=alt.Axis(format="%"), title="Frequência: % das entregas atrasadas", scale=alt.Scale(domain=[.3, 1.05])),
            y=alt.Y("severidade:Q", title="Gravidade: atraso médio quando atrasa (dias)", scale=alt.Scale(domain=[0, 10])),
            size=alt.Size("share_gasto:Q", legend=alt.Legend(title="% do gasto", format=".0%"), scale=alt.Scale(range=[150, 1100])),
            color=alt.Color("risco:N", scale=ESC_RISCO, legend=alt.Legend(title="Risco")),
            opacity=alt.condition("datum.destaque", alt.value(.9), alt.value(.18)),
            tooltip=["fornecedor", "risco", "tier", "pais", alt.Tooltip("pct_atrasado:Q", format=".0%"), alt.Tooltip("severidade:Q", format=".1f"),
                     alt.Tooltip("share_gasto:Q", format=".1%")])
        linhas = alt.Chart(pd.DataFrame({"y": [sev_lim]})).mark_rule(color="#e34948", strokeDash=[5, 4]).encode(y="y:Q") + \
            alt.Chart(pd.DataFrame({"x": [base_mediana]})).mark_rule(color=CINZA, strokeDash=[2, 3]).encode(x="x:Q")
        rot = alt.Chart(mm[(mm.risco == "Alto")]).mark_text(align="right", dx=-12, fontSize=11).encode(
            x="pct_atrasado:Q", y="severidade:Q", text="fornecedor:N")
        st.altair_chart((linhas + pontos + rot).properties(height=460, title="Matriz de risco: frequência × gravidade (bolha = % do gasto)"), width="stretch")
        st.caption(f"Acima da linha vermelha (≈ {br(sev_lim, 1)} dias), a gravidade foge do normal. A linha cinza é a frequência típica entre os 20 fornecedores. "
                   "Quem está fora dos filtros aparece esmaecido.")
        top = r.sort_values("dias_atraso_total", ascending=False).head(8)
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            y=alt.Y("fornecedor:N", sort="-x", title=None), x=alt.X("dias_atraso_total:Q", title="Dias de atraso acumulados"),
            color=alt.Color("risco:N", scale=ESC_RISCO, legend=None),
            tooltip=["fornecedor", "risco", alt.Tooltip("dias_atraso_total:Q", format=",.0f"), alt.Tooltip("share_gasto:Q", format=".1%")]
        ).properties(height=260, title="Exposição: onde estão os dias de atraso"), width="stretch")
        st.caption("O risco mede só o atraso. A base não traz o custo do atraso (multas, frete urgente, falta de produto), "
                   "então **nenhum valor em dinheiro foi estimado**.")

    # ================================================================ 10. Insights e recomendações
    with abas[4]:
        alto = r[r.risco == "Alto"].sort_values("severidade", ascending=False)
        st.markdown("### O que os filtros atuais mostram")
        if len(alto):
            st.success(f"**{', '.join(alto.fornecedor)}** estão em alto risco no período: atrasam em média **{br(alto.severidade.mean(), 1)} dias** quando atrasam, "
                       f"somam **{pc(alto.share_dias_atraso.sum())}** dos dias de atraso e **{pc(alto.share_gasto.sum())}** do gasto.")
        else:
            st.info("Nenhum fornecedor classificado como alto risco no período/urgência selecionados.")
        st.markdown("### Principais descobertas")
        st.caption("Estas usam **todos os dados, sem filtros**, e são as mesmas do notebook. A caixa verde acima muda quando você muda os filtros.")
        # Números copiados de resultados/insights.csv.
        st.markdown(
            "- ✅ **Atrasar é comum, mas quase sempre por pouco.** 58% das entregas atrasam, e o atraso típico é de 1 dia. Só 14% passam de 3 dias.\n"
            "- ✅ **Três fornecedores concentram o problema.** SUP-04, SUP-17 e SUP-11 atrasam todas as entregas (cerca de 7,6 dias), "
            "somam 59% dos dias de atraso e todos os atrasos graves, com só 16% do gasto. Os outros 17 são parecidos entre si: atrasam por cerca de 2 dias e nenhum tem atraso grave.\n"
            "- ✅ **O problema é permanente.** Em 24 meses não há piora, melhora nem época do ano pior.\n"
            "- 🔗 **O que explica o atraso é quem entrega.** País e tier (nível do fornecedor) parecem pesar, mas a diferença some sem os 3 fornecedores. "
            "O fornecedor explica 63% do atraso e as características do pedido, menos de 1%.\n"
            "- ✅ **Esses 3 não cobram mais caro nem rejeitam mais.** Preço, volume e rejeição (2,4%) são iguais aos dos outros, e eles não têm mais pedidos urgentes.\n"
            "- 🔗 **Não achamos ligação com falta de estoque nem com o serviço ao cliente.** Isso não prova que não existe: a base não permite concluir.\n"
            "- 💡 **Possível risco no estoque (hipótese).** O ponto de reposição cobre cerca de 27 dias de demanda, mas atrasos de 10 a 14 dias poderiam passar disso em 13% dos casos. Não foi medido.\n"
            "- ✅ **Não precisa de modelo complicado.** Os modelos de previsão só acertam quando sabem quem é o fornecedor, e uma regra simples por fornecedor funciona igual. "
            "Os atrasos extremos são reais, não erros: tirá-los esconderia o problema.")
        st.caption("✅ **Fato** = medido diretamente. 🔗 **Associação** = relação estatística, não prova causa. 💡 **Hipótese** = plausível, mas não medida nesta base.")
        st.markdown("### Recomendações")
        st.markdown(
            "1. **Conversar com os fornecedores de alto risco**, com metas de prazo, acompanhamento e, se preciso, revisão do contrato. *Prioridade alta*\n"
            "2. **Cobrar por fornecedor, não por país ou categoria**: país, nível e fonte única não explicam o atraso. *Alta*\n"
            "3. **Passar a medir o custo do atraso** (multas, frete urgente, parada de linha, falta de produto) e ligar cada compra à sua entrega. "
            "Hoje não dá para dizer quanto o atraso custa em dinheiro. *Alta*\n"
            "4. **Rever os prazos prometidos** por esses fornecedores, ou combinar um prazo realista com penalidade. O atraso estica o prazo combinado em 21% a 108%. *Média*\n"
            "5. **Testar se o estoque de segurança aguenta** nos produtos mais ligados a esses fornecedores: a cobertura mediana é de cerca de 27 dias e os atrasos mais longos "
            "chegam a 10–14 dias (hipótese). *Média*\n"
            "6. **Acompanhar cada fornecedor todo mês** (% de entregas atrasadas e dias de atraso), com uma lista de observação, em vez de um modelo de previsão. *Média*\n"
            "7. **Avaliar fornecedores alternativos** para os de alto risco (nenhum é fonte única). *Média a baixa*")
        with st.expander("O que foi medido e o que não foi"):
            st.markdown("- **Medido (fato):** preço, volume, rejeição e pedidos urgentes são parecidos entre os fornecedores.\n"
                        "- **Testado, sem resultado:** falta de estoque e nível de serviço ao cliente. A base dificilmente mostraria esse efeito: "
                        "todos os produtos vêm dos mesmos 20 fornecedores e não há ligação entre compra e remessa. "
                        "*Não achar evidência não prova que o efeito não existe.*\n"
                        "- **Hipótese:** com atrasos de 10 a 14 dias, o estoque de segurança pode não bastar.\n"
                        "- **Não dá para medir:** custo em dinheiro, multas e impacto no planejamento e na experiência do cliente.")
