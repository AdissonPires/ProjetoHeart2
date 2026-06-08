import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080c12; color: #e6edf3; }
.block-container { padding-top: 1.8rem; }
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2530; }
.metric-card {
    background: #0d1117; border: 1px solid #1e2530;
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #f0f6fc; font-family: 'DM Mono', monospace; }
.metric-label { font-size: 0.7rem; color: #4e5d6c; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e2530; }
.stTabs [data-baseweb="tab"] { color: #4e5d6c; background: transparent; }
.stTabs [aria-selected="true"] { color: #f0f6fc !important; border-bottom: 2px solid #3b82f6 !important; background: transparent !important; }
.stButton > button { background: #3b82f6; color: white; border: none; border-radius: 8px; font-weight: 600; width: 100%; }
h1, h2, h3 { color: #f0f6fc !important; }
hr { border-color: #1e2530 !important; }
</style>
""", unsafe_allow_html=True)

P = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
         font=dict(color="#8b949e", family="DM Sans", size=12),
         margin=dict(l=10, r=10, t=40, b=10),
         legend=dict(bgcolor="rgba(0,0,0,0)"),
         title_font=dict(color="#c9d1d9", size=13))

@st.cache_data
def load_and_train():
    df = pd.read_csv("heart.csv")
    df["Status"] = df["HeartDisease"].map({0: "Saudável", 1: "Doente"})
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0,40,50,60,70,100], labels=["<40","40-50","50-60","60-70","70+"])

    dm = df.drop(columns=["Status","AgeGroup"]).copy()
    enc = {}
    for col in dm.select_dtypes("object").columns:
        le = LabelEncoder(); dm[col] = le.fit_transform(dm[col]); enc[col] = le

    X, y = dm.drop("HeartDisease", axis=1), dm["HeartDisease"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    bayes  = GaussianNB().fit(Xtr, ytr)
    tree   = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr, ytr)
    forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    return df, enc, X, Xte, yte, bayes, tree, forest

df, enc, X, Xte, yte, bayes, tree, forest = load_and_train()

# Sidebar
st.sidebar.markdown("### Filtros")
f_sex   = st.sidebar.multiselect("Sexo",         df["Sex"].unique(),          default=df["Sex"].unique())
f_dor   = st.sidebar.multiselect("Tipo de Dor",  df["ChestPainType"].unique(), default=df["ChestPainType"].unique())
f_slope = st.sidebar.multiselect("Inclinação ST", df["ST_Slope"].unique(),     default=df["ST_Slope"].unique())
age_r   = st.sidebar.slider("Faixa Etária", int(df["Age"].min()), int(df["Age"].max()),
                             (int(df["Age"].min()), int(df["Age"].max())))

dff = df[df["Sex"].isin(f_sex) & df["ChestPainType"].isin(f_dor) &
         df["ST_Slope"].isin(f_slope) & df["Age"].between(*age_r)]

# Header
st.markdown("## Dashboard Interativo de Doença Cardíaca")
st.markdown(f"<span style='color:#4e5d6c;font-size:0.8rem;'>Projeto Heart 2.0</span>", unsafe_allow_html=True)
st.markdown(f"<span style='color:#4e5d6c;font-size:0.8rem;'>{len(dff):,} de {len(df):,} pacientes</span>", unsafe_allow_html=True)
st.divider()

# KPIs
for col, (label, val) in zip(st.columns(5), [
    ("Pacientes",        f"{len(dff):,}"),
    ("Taxa de Doença",   f"{dff['HeartDisease'].mean()*100:.1f}%"),
    ("Idade Média",      f"{dff['Age'].mean():.1f}"),
    ("Colesterol Médio", f"{dff['Cholesterol'].mean():.0f}"),
    ("MaxHR Médio",      f"{dff['MaxHR'].mean():.0f}"),
]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.divider()

aba1, aba2, aba3 = st.tabs(["Análise dos Dados", "Classificação", "Métricas"])

# ── ABA 1
with aba1:
    c1, c2, c3 = st.columns(3)
    with c1:
        cnt = dff["Status"].value_counts().reset_index(); cnt.columns = ["Status","Total"]
        fig = px.pie(cnt, names="Status", values="Total", hole=0.6,
                     color="Status", color_discrete_map={"Saudável":"#2ecc71","Doente":"#e74c3c"})
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(title="Distribuição", showlegend=False, **P)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dor = dff.groupby("ChestPainType")["HeartDisease"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(dor, x="ChestPainType", y="HeartDisease",
                     color="HeartDisease", color_continuous_scale=["#2ecc71","#f59e0b","#e74c3c"],
                     text=dor["HeartDisease"].apply(lambda x: f"{x:.0%}"),
                     labels={"HeartDisease":"Prob.","ChestPainType":"Dor"})
        fig.update_traces(textposition="outside"); fig.update_coloraxes(showscale=False)
        fig.update_layout(title="Risco por Tipo de Dor", **P)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        sr = dff.groupby("Sex")["HeartDisease"].mean().reset_index()
        sr["Sexo"] = sr["Sex"].map({"M":"Masculino","F":"Feminino"})
        fig = px.bar(sr, x="Sexo", y="HeartDisease",
                     color="HeartDisease", color_continuous_scale=["#2ecc71","#e74c3c"],
                     text=sr["HeartDisease"].apply(lambda x: f"{x:.0%}"),
                     labels={"HeartDisease":"Taxa"})
        fig.update_traces(textposition="outside"); fig.update_coloraxes(showscale=False)
        fig.update_layout(title="Taxa por Sexo", **P)
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        ar = dff.groupby("AgeGroup", observed=True)["HeartDisease"].mean().reset_index()
        fig = px.line(ar, x="AgeGroup", y="HeartDisease", markers=True, line_shape="spline",
                      labels={"AgeGroup":"Faixa Etária","HeartDisease":"Taxa"})
        fig.update_traces(line_color="#e74c3c", marker=dict(size=9, color="#e74c3c"))
        fig.update_layout(title="Prevalência por Faixa Etária", **P)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        fig = px.histogram(dff, x="Age", nbins=20, color="Status",
                           color_discrete_map={"Saudável":"#2ecc71","Doente":"#e74c3c"},
                           barmode="overlay", opacity=0.7,
                           labels={"Age":"Idade"})
        fig.update_layout(title="Distribuição de Idade", **P)
        st.plotly_chart(fig, use_container_width=True)

    corr = dff.select_dtypes("number").corr()
    fig = px.imshow(corr, text_auto=".2f",
                    color_continuous_scale=[[0,"#e74c3c"],[0.5,"#0d1117"],[1,"#3b82f6"]],
                    zmin=-1, zmax=1)
    fig.update_layout(title="Matriz de Correlação", **P)
    st.plotly_chart(fig, use_container_width=True)

# ── ABA 2
with aba2:
    c1, c2 = st.columns(2)
    with c1:
        age   = st.number_input("Age", value=50)
        sex   = st.selectbox("Sex", df["Sex"].unique())
        chest = st.selectbox("ChestPainType", df["ChestPainType"].unique())
        rbp   = st.number_input("RestingBP", value=120)
        chol  = st.number_input("Cholesterol", value=200)
        fbs   = st.selectbox("FastingBS", [0,1])
    with c2:
        ecg    = st.selectbox("RestingECG", df["RestingECG"].unique())
        maxhr  = st.number_input("MaxHR", value=150)
        angina = st.selectbox("ExerciseAngina", df["ExerciseAngina"].unique())
        old    = st.number_input("Oldpeak", value=1.0, step=0.1)
        slope  = st.selectbox("ST_Slope", df["ST_Slope"].unique())

    if st.button("Classificar Paciente"):
        inp = pd.DataFrame({"Age":[age],"Sex":[sex],"ChestPainType":[chest],"RestingBP":[rbp],
                            "Cholesterol":[chol],"FastingBS":[fbs],"RestingECG":[ecg],
                            "MaxHR":[maxhr],"ExerciseAngina":[angina],"Oldpeak":[old],"ST_Slope":[slope]})
        ie = inp.copy()
        for col in enc: ie[col] = enc[col].transform(ie[col])

        prob = bayes.predict_proba(ie)[0]
        preds = {"Naive Bayes": bayes.predict(ie)[0],
                 "Árvore":      tree.predict(ie)[0],
                 "Random Forest": forest.predict(ie)[0]}

        st.divider()
        a, b = st.columns(2)
        with a:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2ecc71">{prob[0]*100:.1f}%</div><div class="metric-label">Sem Doença</div></div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#e74c3c">{prob[1]*100:.1f}%</div><div class="metric-label">Com Doença</div></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(prob[1]*100, 1),
            number={"suffix":"%","font":{"color":"#f0f6fc","size":30,"family":"DM Mono"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#4e5d6c"},
                   "bar":{"color":"#e74c3c" if prob[1]>0.5 else "#2ecc71","thickness":0.25},
                   "bgcolor":"#0d1117","borderwidth":0,
                   "steps":[{"range":[0,50],"color":"rgba(46,204,113,0.08)"},
                             {"range":[50,100],"color":"rgba(231,76,60,0.08)"}]},
            title={"text":"Risco de Doença Cardíaca","font":{"color":"#8b949e","size":13}}
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=230, margin=dict(l=20,r=20,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

        res = pd.DataFrame({"Modelo": list(preds.keys()),
                            "Predição": ["Doença Cardíaca" if v==1 else "Saudável" for v in preds.values()]})
        st.dataframe(res, use_container_width=True, hide_index=True)

# ── ABA 3
with aba3:
    modelos = {"Naive Bayes": bayes, "Árvore": tree, "Random Forest": forest}
    rows = []
    for nome, m in modelos.items():
        p = m.predict(Xte)
        rows.append({"Modelo":nome, "Accuracy":round(accuracy_score(yte,p),4),
                     "Precision":round(precision_score(yte,p),4),
                     "Recall":round(recall_score(yte,p),4), "F1":round(f1_score(yte,p),4)})
    rdf = pd.DataFrame(rows)

    st.dataframe(rdf.style.background_gradient(
        subset=["Accuracy","Precision","Recall","F1"], cmap="RdYlGn", vmin=0.6, vmax=1.0
    ).format("{:.4f}", subset=["Accuracy","Precision","Recall","F1"]),
    use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(rdf, x="Modelo", y="Accuracy", color="Modelo",
                     color_discrete_sequence=["#3b82f6","#f59e0b","#2ecc71"],
                     text=rdf["Accuracy"].apply(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition="outside"); fig.update_layout(showlegend=False, yaxis_range=[0,1.1], title="Accuracy", **P)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ml = rdf.melt(id_vars="Modelo", value_vars=["Accuracy","Precision","Recall","F1"],
                      var_name="Métrica", value_name="Valor")
        fig = px.line(ml, x="Métrica", y="Valor", color="Modelo", markers=True, line_shape="spline",
                      color_discrete_sequence=["#3b82f6","#f59e0b","#2ecc71"])
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(title="Comparação Geral", yaxis_range=[0.6,1.05], **P)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    for idx, (nome, m) in enumerate(modelos.items()):
        cm = confusion_matrix(yte, m.predict(Xte))
        fig = px.imshow(cm, text_auto=True, x=["Saudável","Doente"], y=["Saudável","Doente"],
                        color_continuous_scale=[[0,"#0d1117"],[1,"#3b82f6"]])
        fig.update_layout(title=nome, coloraxis_showscale=False, **P)
        with cols[idx]:
            st.plotly_chart(fig, use_container_width=True)