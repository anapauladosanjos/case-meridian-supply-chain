# -*- coding: utf-8 -*-
"""Biblioteca de análise de atrasos de fornecedores (Meridian Components).

Usada pelo notebook `Case_Meridian_Atrasos_Fornecedores.ipynb` (o código é copiado para uma célula)
e pelo app Streamlit (`import atrasos_lib`). Assim os dois calculam exatamente a mesma coisa.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

NOMES = ["carrier", "customer", "inventory_snapshot", "product", "production_run", "purchase_order",
         "purchase_order_line", "shipment", "shipment_line", "supplier", "warehouse"]
DATAS = {"purchase_order": ["order_date", "promised_date", "actual_delivery_date"],
         "shipment": ["ship_date", "delivery_date"], "inventory_snapshot": ["snapshot_date"],
         "production_run": ["run_date"]}
LIM_GRAVE = 4          # dias: atraso >= 4 é "grave" (quebra de patamar na distribuição, ver notebook)
SEED = 42


# ----------------------------------------------------------------------------------------------
# Carregamento
# ----------------------------------------------------------------------------------------------
def _completo(p):
    return p is not None and all(os.path.exists(os.path.join(p, n + ".csv")) for n in NOMES)


def localizar_dados():
    """CASE_DATA_DIR -> kagglehub -> pastas locais. (O upload manual no Colab é feito no notebook.)"""
    p = os.environ.get("CASE_DATA_DIR")
    if _completo(p):
        return p
    try:
        import kagglehub
        p = kagglehub.dataset_download("nadhirmazariboufares/supply-chain-dataset")
        if _completo(p):
            return p
    except Exception as e:  # noqa: BLE001
        print("kagglehub indisponível:", type(e).__name__)
    for p in ["data", ".", "/content/data", "/content", "data_upload"]:
        if _completo(p):
            return p
    raise FileNotFoundError("Não encontrei os 11 CSVs (defina CASE_DATA_DIR ou use o upload manual).")


def carregar_brutos(pasta):
    """Lê os CSVs SEM nenhuma conversão (a base original fica preservada)."""
    return {n: pd.read_csv(os.path.join(pasta, n + ".csv")) for n in NOMES}


def tratar(brutos):
    """Cópia tratada: só converte tipos (datas). Nenhum registro é removido aqui."""
    T = {n: df.copy() for n, df in brutos.items()}
    for n, cols in DATAS.items():
        for c in cols:
            T[n][c] = pd.to_datetime(T[n][c], errors="coerce")
    for n in ("supplier", "product", "purchase_order"):
        for c in T[n].select_dtypes(include=["object", "string"]).columns:
            T[n][c] = T[n][c].astype(str).str.strip()
    return T


# ----------------------------------------------------------------------------------------------
# Base analítica de pedidos (1 linha por pedido)
# ----------------------------------------------------------------------------------------------
def classe_entrega(a):
    return pd.Series(np.where(a < 0, "Antecipada", np.where(a == 0, "Pontual", "Atrasada")), index=a.index)


def base_pedidos(T):
    po, pol, sup = T["purchase_order"], T["purchase_order_line"], T["supplier"]
    pol = pol.assign(valor=pol.qty_ordered * pol.unit_cost, gasto=pol.qty_received * pol.unit_cost)
    agg = pol.groupby("po_id").agg(valor=("valor", "sum"), gasto=("gasto", "sum"), itens=("qty_ordered", "sum"),
                                   linhas=("po_line_id", "count"), qtd_rec=("qty_received", "sum"),
                                   qtd_rej=("qty_rejected", "sum")).reset_index()
    p = (po.merge(sup.rename(columns={"supplier_name": "fornecedor", "country": "pais"}), on="supplier_id",
                  how="left", validate="m:1").merge(agg, on="po_id", how="left", validate="1:1"))
    p["entregue"] = p.status.eq("Delivered")
    p["atraso"] = (p.actual_delivery_date - p.promised_date).dt.days.where(p.entregue)
    p["prazo_prometido"] = (p.promised_date - p.order_date).dt.days
    p["classe"] = np.where(p.entregue, classe_entrega(p.atraso.fillna(0)), "Sem entrega")
    p["atrasado"] = (p.atraso > 0).astype(float).where(p.entregue)
    p["grave"] = (p.atraso >= LIM_GRAVE).astype(float).where(p.entregue)
    p["atraso_rel"] = p.atraso / p.prazo_prometido
    p["ano_mes"] = p.order_date.dt.to_period("M").astype(str)
    p["ano"] = p.order_date.dt.year
    p["mes"] = p.order_date.dt.month
    p["dow"] = p.order_date.dt.dayofweek
    return p


# ----------------------------------------------------------------------------------------------
# Auditoria de qualidade
# ----------------------------------------------------------------------------------------------
def auditoria(brutos, T):
    """Tabela do 'Diagnóstico e Qualidade dos Dados'. Tudo calculado nos dados brutos/tratados."""
    bpo, bpol, bsup, bprod = (brutos[k] for k in ("purchase_order", "purchase_order_line", "supplier", "product"))
    po, pol, sup, prod = (T[k] for k in ("purchase_order", "purchase_order_line", "supplier", "product"))
    N = len(po)
    ent = po[po.status.eq("Delivered")]
    atr = (ent.actual_delivery_date - ent.promised_date).dt.days
    lead = po.merge(sup[["supplier_id", "lead_time_days"]], on="supplier_id")
    prazo = (lead.promised_date - lead.order_date).dt.days
    q1, q3 = atr.quantile([.25, .75]); lim_sup = q3 + 1.5 * (q3 - q1)
    cols_data = [c for c in DATAS["purchase_order"]]
    texto = sum(int(bpo[c].map(lambda v: isinstance(v, str)).sum()) for c in cols_data)
    invalidas = sum(int((pd.to_datetime(bpo[c], errors="coerce").isna() & bpo[c].notna()).sum()) for c in cols_data)
    var_nome = 0
    for df, c in [(bsup, "country"), (bsup, "tier"), (bprod, "category"), (bpo, "status")]:
        norm = df[c].astype(str).str.strip().str.casefold()
        var_nome += int((df[c].astype(str) != df[c].astype(str).str.strip()).sum())
        var_nome += int(df.groupby(norm)[c].nunique().gt(1).sum())
    neg = int((pol[["qty_ordered", "qty_received", "qty_rejected", "unit_cost"]] < 0).sum().sum()
              + (sup[["lead_time_days", "payment_terms_days", "min_order_value"]] < 0).sum().sum())
    canc = po[po.status.eq("Cancelled")]
    rows = [
        ("Tipos", "Datas lidas como texto no CSV (3 colunas de purchase_order)", texto, 3 * N,
         "O CSV traz datas como texto; sem conversão, não dá para calcular prazos.",
         "Tratar: converter para datetime na cópia tratada (o original continua intacto).",
         "Habilita todos os cálculos de atraso. Sem efeito sobre os números."),
        ("Datas", "Datas impossíveis de converter", invalidas, 3 * N, "Datas com formato inválido.",
         "Nenhuma ação necessária." if invalidas == 0 else "Tratar separadamente.",
         "Nenhum registro perdido." if invalidas == 0 else "Registros ficariam sem atraso calculável."),
        ("Ausentes", "actual_delivery_date vazio", int(bpo.actual_delivery_date.isna().sum()), N,
         "Pedidos sem data de entrega. Coincidem com os pedidos cancelados.",
         "Tratar separadamente: manter na base, fora das métricas de atraso. Não imputar.",
         "Se contados como 'não atrasados', o % de atraso cairia (ver comparação antes/depois)."),
        ("Ausentes", "Outras colunas de purchase_order / purchase_order_line / supplier com nulos",
         int(bpo.drop(columns="actual_delivery_date").isna().sum().sum() + bpol.isna().sum().sum() + bsup.isna().sum().sum()),
         N, "Campos obrigatórios vazios.", "Nenhuma ação necessária.", "Nenhum."),
        ("Duplicidade", "Linhas duplicadas ou po_id repetido", int(bpo.duplicated().sum() + bpo.po_id.duplicated().sum()), N,
         "Pedidos repetidos inflariam contagens.", "Nenhuma ação necessária.", "Nenhum."),
        ("Datas", "Prazo prometido anterior à data do pedido", int((po.promised_date < po.order_date).sum()), N,
         "Promessa antes do pedido é impossível.", "Nenhuma ação necessária.", "Nenhum."),
        ("Datas", "Entrega real anterior à data do pedido", int((ent.actual_delivery_date < ent.order_date).sum()),
         len(ent), "Entrega antes de pedir é impossível.", "Nenhuma ação necessária.", "Nenhum."),
        ("Datas", "Pedido 'Delivered' sem data de entrega", int((po.status.eq("Delivered") & po.actual_delivery_date.isna()).sum()),
         N, "Entregue sem data impediria medir o atraso.", "Nenhuma ação necessária.", "Nenhum."),
        ("Datas", "Pedido 'Cancelled' com data de entrega", int((canc.actual_delivery_date.notna()).sum()), len(canc),
         "Cancelado que aparece como entregue.", "Nenhuma ação necessária.", "Nenhum."),
        ("Valores", "Valores negativos (quantidades, custo unitário, prazos, mínimos)", neg, len(pol),
         "Quantidades ou custos negativos seriam erro.", "Nenhuma ação necessária.", "Nenhum."),
        ("Valores", "Qtd. recebida > pedida, ou rejeitada > recebida", int((pol.qty_received > pol.qty_ordered).sum()
                                                                          + (pol.qty_rejected > pol.qty_received).sum()),
         len(pol), "Inconsistência de quantidades.", "Nenhuma ação necessária.", "Nenhum."),
        ("Consistência", "Prazo prometido diferente do lead time cadastrado do fornecedor", int((prazo != lead.lead_time_days).sum()),
         N, "A promessa deveria seguir o lead time do cadastro.", "Nenhuma ação necessária (promessa = pedido + lead time).",
         "Confirma que 'prometido' é um prazo contratual, não negociado caso a caso."),
        ("Nomenclatura", "Categorias com grafias diferentes para o mesmo valor (país, tier, categoria, status)", var_nome,
         len(sup) + len(prod) + N, "Variações de caixa/espaços fragmentariam agrupamentos.",
         "Nenhuma ação necessária (espaços removidos por segurança na cópia tratada).", "Nenhum."),
        ("Incompletos", "Pedidos sem nenhuma linha de item", int((~po.po_id.isin(pol.po_id)).sum()), N,
         "Pedido sem itens não tem valor nem quantidade.", "Nenhuma ação necessária.", "Nenhum."),
        ("Incompletos", "Pedidos cancelados com itens recebidos", int(pol[pol.po_id.isin(canc.po_id)].qty_received.gt(0).sum()),
         len(pol), "Cancelado deveria ter recebido zero.", "Nenhuma ação necessária.", "Nenhum."),
        ("Extremos", f"Atraso acima do limite IQR 1,5x (> {lim_sup:.0f} dias)", int((atr > lim_sup).sum()), len(ent),
         "Atrasos longos. São eventos reais concentrados em poucos fornecedores (ver seção de outliers).",
         "Manter e tratar separadamente (comparar com e sem).", "Remover esconderia exatamente o problema que se quer medir."),
        ("Extremos", "Entregas antecipadas (antes do prometido)", int((atr < 0).sum()), len(ent),
         "Entrega adiantada não é erro; é um resultado diferente de pontual.",
         "Manter; classificar à parte (Antecipada / Pontual / Atrasada).",
         "Tratar adiantada como pontual esconderia a variabilidade dos prazos."),
    ]
    a = pd.DataFrame(rows, columns=["categoria", "verificação", "afetados", "base", "problema", "decisão", "impacto da decisão"])
    a["% da base"] = (a.afetados / a.base * 100).round(2)
    a.insert(0, "#", range(1, len(a) + 1))
    return a


def antes_depois(brutos, T, ped):
    """Resumo antes/depois do tratamento: contagens e efeito sobre o indicador principal."""
    bpo = brutos["purchase_order"]
    ent = ped[ped.entregue]
    naive = float((ent.atrasado == 1).sum() / len(bpo))
    return pd.DataFrame([
        ("Pedidos na base", len(bpo), len(ped), "Nenhum pedido removido"),
        ("Colunas de purchase_order", bpo.shape[1], T["purchase_order"].shape[1], "Só conversão de tipo (datas)"),
        ("Datas como texto (colunas)", 3, 0, "Convertidas para datetime"),
        ("Pedidos com atraso calculável", 0, len(ent), "Entregues (cancelados ficam à parte)"),
        ("% de pedidos atrasados (denominador = todos os 3.000)", f"{naive*100:.1f}%", f"{ent.atrasado.mean()*100:.1f}%",
         "Denominador correto = só entregues"),
        ("Atraso médio (dias)", "—", f"{ent.atraso.mean():.2f}", "Só entregues"),
    ], columns=["indicador", "base original", "base tratada", "observação"])


# ----------------------------------------------------------------------------------------------
# Métricas por fornecedor
# ----------------------------------------------------------------------------------------------
def metricas_fornecedor(ped):
    e = ped[ped.entregue].copy()
    g = e.groupby("fornecedor")

    def cond(s):  # severidade condicional: atraso médio SÓ entre as entregas atrasadas
        return s[s > 0].mean() if (s > 0).any() else 0.0

    f = pd.DataFrame({
        "tier": g.tier.first(), "pais": g.pais.first(), "rating": g.quality_rating.first(),
        "single_source": g.is_single_source.first(), "prazo_prometido": g.prazo_prometido.mean(),
        "pedidos": g.size(),
        "pct_antecipado": g.atraso.apply(lambda s: (s < 0).mean()), "pct_pontual": g.atraso.apply(lambda s: (s == 0).mean()),
        "pct_atrasado": g.atraso.apply(lambda s: (s > 0).mean()),
        "atraso_medio": g.atraso.mean(), "atraso_mediano": g.atraso.median(), "desvio": g.atraso.std(),
        "p90": g.atraso.quantile(.9), "maximo": g.atraso.max(), "severidade": g.atraso.apply(cond),
        "dias_atraso_total": g.atraso.apply(lambda s: s.clip(lower=0).sum()),
        "pct_grave": g.atraso.apply(lambda s: (s >= LIM_GRAVE).mean()),
        "gasto": g.gasto.sum(), "pct_expedite": g.expedited.mean(),
    })
    f["cv"] = f.desvio / f.atraso_medio.abs().replace(0, np.nan)
    f["atraso_rel"] = f.atraso_medio / f.prazo_prometido
    f["share_gasto"] = f.gasto / f.gasto.sum()
    f["share_dias_atraso"] = f.dias_atraso_total / f.dias_atraso_total.sum()
    return f.reset_index()


def z_modificado(x):
    """Escore z robusto (Iglewicz & Hoaglin): 0,6745·(x − mediana)/MAD. Limite usual: |z| > 3,5."""
    x = np.asarray(x, float)
    mad = np.median(np.abs(x - np.median(x)))
    return 0.6745 * (x - np.median(x)) / (mad if mad > 0 else np.nan)


def classificar_risco(f, alfa=0.05):
    """Risco de atraso com dois critérios estatísticos (sem cortes arbitrários):
    - GRAVIDADE: severidade (atraso médio entre as entregas atrasadas) com z modificado > 3,5.
    - FREQUÊNCIA: % atrasado acima da mediana dos fornecedores, teste binomial unilateral com correção de Holm.
    Alto = os dois; Intermediário = um dos dois; Baixo = nenhum."""
    f = f.copy()
    f["z_severidade"] = z_modificado(f.severidade)
    f["crit_gravidade"] = f.z_severidade > 3.5
    base = float(f.pct_atrasado.median())
    p = np.array([stats.binomtest(int(round(r.pct_atrasado * r.pedidos)), int(r.pedidos), base, alternative="greater").pvalue
                  for r in f.itertuples()])
    ordem = np.argsort(p); m = len(p); ajust = np.empty(m); run = 0.0
    for rank, i in enumerate(ordem):
        run = max(run, min(1.0, (m - rank) * p[i])); ajust[i] = run
    f["p_frequencia"] = ajust
    f["crit_frequencia"] = f.p_frequencia < alfa
    n = f.crit_gravidade.astype(int) + f.crit_frequencia.astype(int)
    f["risco"] = np.select([n == 2, n == 1], ["Alto", "Intermediário"], "Baixo")
    f.attrs["pct_base_mediana"] = base
    return f


def agrupar(f, k_max=5):
    """K-Means com atributos padronizados. Escolhe k pela silhueta (k=2..k_max). Retorna (f_com_grupo, tabela_k)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    cols = ["pct_atrasado", "severidade", "desvio", "p90"]
    X = StandardScaler().fit_transform(f[cols])
    res = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        res.append((k, km.inertia_, silhouette_score(X, km.labels_)))
    tab = pd.DataFrame(res, columns=["k", "inércia", "silhueta"])
    k = int(tab.loc[tab.silhueta.idxmax(), "k"])
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    f = f.copy(); f["grupo"] = km.labels_
    ordem = f.groupby("grupo").severidade.mean().sort_values(ascending=False).index
    nomes = {g: f"Grupo {i + 1}" for i, g in enumerate(ordem)}
    f["grupo"] = f.grupo.map(nomes)
    return f, tab


# ----------------------------------------------------------------------------------------------
# Outliers
# ----------------------------------------------------------------------------------------------
def limites_iqr(s, k=1.5):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - k * i, q3 + k * i


def resumo_outliers(df, colunas):
    out = []
    for nome, col in colunas.items():
        s = df[col].dropna()
        lo, hi = limites_iqr(s); lo3, hi3 = limites_iqr(s, 3)
        z = (s - s.mean()) / s.std()
        out.append({"variável": nome, "n": len(s), "limite inferior (1,5×IQR)": round(lo, 2), "limite superior (1,5×IQR)": round(hi, 2),
                    "outliers 1,5×IQR": int(((s < lo) | (s > hi)).sum()),
                    "% 1,5×IQR": round(((s < lo) | (s > hi)).mean() * 100, 2),
                    "outliers 3×IQR": int(((s < lo3) | (s > hi3)).sum()), "|z| > 3": int((z.abs() > 3).sum()),
                    "P1": round(s.quantile(.01), 2), "P99": round(s.quantile(.99), 2), "máximo": round(s.max(), 2)})
    return pd.DataFrame(out)


def isolation(ped, contaminacao=0.05):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    e = ped[ped.entregue].copy()
    X = StandardScaler().fit_transform(np.c_[e.atraso, e.prazo_prometido, np.log1p(e.valor), np.log1p(e.itens)])
    m = IsolationForest(n_estimators=300, contamination=contaminacao, random_state=SEED).fit(X)
    e["iso_outlier"] = m.predict(X) == -1
    e["iso_score"] = -m.score_samples(X)
    return e


# ----------------------------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------------------------
NUM_PED = ["expedited", "log_valor", "log_itens", "linhas", "mes", "dow"]
NUM_FORN = ["quality_rating", "payment_terms_days", "min_order_value", "is_single_source", "lead_time_days"]
CAT_FORN = ["pais", "tier"]


def matriz_modelo(ped):
    e = ped[ped.entregue].copy()
    e["log_valor"] = np.log1p(e.valor); e["log_itens"] = np.log1p(e.itens)
    return e.reset_index(drop=True)


def _pipe(modelo, num, cat):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    pre = ColumnTransformer([("n", StandardScaler(), num)] + ([("c", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else []))
    return Pipeline([("pre", pre), ("m", modelo)])


def modelos_classificacao():
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    return {"Linha de base (classe majoritária)": DummyClassifier(strategy="prior"),
            "Regressão logística": LogisticRegression(max_iter=2000, C=1.0),
            "Árvore de decisão (prof. 3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=SEED)}


def modelos_regressao():
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    return {"Linha de base (média do treino)": DummyRegressor(),
            "Regressão linear": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=SEED)}


def conjuntos_de_atributos():
    return {"Só atributos do pedido": (NUM_PED, []),
            "Pedido + atributos do fornecedor": (NUM_PED + NUM_FORN, CAT_FORN)}


def _metricas_clf(y, p, prob):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    return {"precisão": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "F1": f1_score(y, p, zero_division=0), "ROC-AUC": auc}


def avaliar_classificacao(e, alvo="grave", cv_rep=None):
    """Três esquemas de validação:
    - 'Temporal': treina em 2024, testa em 2025 (cenário real de uso: prever pedidos futuros de fornecedores conhecidos).
    - 'Aleatória 5-fold': mistura períodos e fornecedores (pode inflar o resultado se o fornecedor 'vaza' pelos atributos).
    - 'Leave-one-supplier-out': testa em fornecedor que o modelo nunca viu (generalização a novos fornecedores)."""
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
    y = e[alvo].astype(int).values
    tr, te = (e.ano == 2024).values, (e.ano == 2025).values
    linhas, guarda = [], {}
    for fs, (num, cat) in conjuntos_de_atributos().items():
        X = e[num + cat]
        for nome, mod in modelos_classificacao().items():
            pipe = _pipe(clone(mod), num, cat)
            pipe.fit(X[tr], y[tr]); prob = pipe.predict_proba(X[te])[:, 1]
            linhas.append({"atributos": fs, "modelo": nome, "validação": "Temporal (2024→2025)", **_metricas_clf(y[te], (prob >= .5).astype(int), prob)})
            guarda[(fs, nome)] = (y[te], prob)
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
            prob = cross_val_predict(_pipe(clone(mod), num, cat), X, y, cv=skf, method="predict_proba")[:, 1]
            linhas.append({"atributos": fs, "modelo": nome, "validação": "Aleatória (5-fold)", **_metricas_clf(y, (prob >= .5).astype(int), prob)})
            prob = cross_val_predict(_pipe(clone(mod), num, cat), X, y, cv=LeaveOneGroupOut(), groups=e.fornecedor.values,
                                     method="predict_proba")[:, 1]
            m = _metricas_clf(y, (prob >= .5).astype(int), prob)
            m["ROC-AUC"] = np.nan   # AUC agrupada de 20 modelos diferentes não é comparável: omitida de propósito
            linhas.append({"atributos": fs, "modelo": nome, "validação": "Fornecedor novo (leave-one-supplier-out)", **m})
    return pd.DataFrame(linhas), guarda


def avaliar_regressao(e):
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    y = e.atraso.values.astype(float)
    tr, te = (e.ano == 2024).values, (e.ano == 2025).values
    met = lambda yt, yp: {"MAE": mean_absolute_error(yt, yp), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "R²": r2_score(yt, yp)}  # noqa: E731
    linhas = []
    for fs, (num, cat) in conjuntos_de_atributos().items():
        X = e[num + cat]
        for nome, mod in modelos_regressao().items():
            pipe = _pipe(clone(mod), num, cat).fit(X[tr], y[tr])
            linhas.append({"atributos": fs, "modelo": nome, "validação": "Temporal (2024→2025)", **met(y[te], pipe.predict(X[te]))})
            yp = cross_val_predict(_pipe(clone(mod), num, cat), X, y, cv=LeaveOneGroupOut(), groups=e.fornecedor.values)
            linhas.append({"atributos": fs, "modelo": nome, "validação": "Fornecedor novo (leave-one-supplier-out)", **met(y, yp)})
    return pd.DataFrame(linhas)


def importancia_permutacao(e, alvo="grave", n_rep=5):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    num, cat = conjuntos_de_atributos()["Pedido + atributos do fornecedor"]
    y = e[alvo].astype(int).values
    tr, te = (e.ano == 2024).values, (e.ano == 2025).values
    pipe = _pipe(GradientBoostingClassifier(random_state=SEED), num, cat).fit(e.loc[tr, num + cat], y[tr])
    r = permutation_importance(pipe, e.loc[te, num + cat], y[te], scoring="roc_auc", n_repeats=n_rep, random_state=SEED)
    return pd.DataFrame({"atributo": num + cat, "queda_AUC": r.importances_mean, "dp": r.importances_std}).sort_values("queda_AUC", ascending=False)


# ----------------------------------------------------------------------------------------------
# Impacto no estoque / serviço (testes de associação)
# ----------------------------------------------------------------------------------------------
def cobertura_rop(T):
    """ROP expresso em dias de demanda (ROP médio / demanda média diária do par armazém×produto)."""
    sl = T["shipment_line"].merge(T["shipment"][["shipment_id", "warehouse_id", "ship_date"]], on="shipment_id")
    dias = (T["shipment"].ship_date.max() - T["shipment"].ship_date.min()).days + 1
    d = sl.groupby(["warehouse_id", "product_id"]).qty_shipped.sum() / dias
    rop = T["inventory_snapshot"].groupby(["warehouse_id", "product_id"]).reorder_point.mean()
    return (rop / d).dropna()


def painel_produto_mes(T, ped):
    """Painel produto×mês: exposição a entregas graves (mês da entrega) e ruptura/abaixo do ROP no mês seguinte."""
    pol, inv = T["purchase_order_line"], T["inventory_snapshot"]
    L = pol.merge(ped[ped.entregue][["po_id", "atraso", "actual_delivery_date"]], on="po_id")
    L["mes_ent"] = L.actual_delivery_date.dt.to_period("M").astype(str)
    g = L.groupby(["product_id", "mes_ent"]).apply(lambda x: pd.Series({
        "un": x.qty_received.sum(), "un_grave": x.loc[x.atraso >= LIM_GRAVE, "qty_received"].sum(),
        "atraso_pond": np.average(x.atraso, weights=x.qty_received) if x.qty_received.sum() > 0 else np.nan})).reset_index()
    g["sh_grave"] = g.un_grave / g.un
    inv = inv.assign(mes=inv.snapshot_date.dt.to_period("M").astype(str), rup=inv.units_on_hand == 0,
                     abx=inv.units_on_hand < inv.reorder_point)
    iv = inv.groupby(["product_id", "mes"]).agg(rup=("rup", "mean"), abx=("abx", "mean")).reset_index()
    meses = sorted(iv.mes.unique()); prox = {m: meses[i + 1] for i, m in enumerate(meses[:-1])}
    g["mes_prox"] = g.mes_ent.map(prox)
    return g.merge(iv, left_on=["product_id", "mes_prox"], right_on=["product_id", "mes"], how="inner")


def regra_simples(e, alvo="grave"):
    """Referência sem aprendizado de máquina: marca como 'grave' o pedido de todo fornecedor cujo atraso médio de 2024
    é um valor atípico (z modificado > 3,5). Avaliada em 2025."""
    tr, te = e[e.ano == 2024], e[e.ano == 2025]
    m = tr.groupby("fornecedor").atraso.mean()
    z = pd.Series(z_modificado(m.values), index=m.index)
    marcados = set(z[z > 3.5].index)
    pred = te.fornecedor.isin(marcados).astype(int).values
    y = te[alvo].astype(int).values
    score = te.fornecedor.map(tr.groupby("fornecedor")[alvo].mean()).values
    r = _metricas_clf(y, pred, score)
    r["fornecedores_marcados"] = sorted(marcados)
    return r


def painel_exposicao_custo(T, ped):
    """Índice de custo unitário por fornecedor (custo da linha ÷ custo médio do mesmo produto)."""
    pol = T["purchase_order_line"].merge(ped[ped.entregue][["po_id", "fornecedor"]], on="po_id")
    pol["idx_custo"] = pol.unit_cost / pol.groupby("product_id").unit_cost.transform("mean")
    return pol.groupby("fornecedor").idx_custo.mean()
