import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Config

st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide"
)

st.title("Dashboard - Heart Disease")

# Carregamento dos dados

df = pd.read_csv("heart.csv")

# Preprocessamento

df_model = df.copy()

encoders = {}

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model.drop("HeartDisease", axis=1)
y = df_model["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modelos

bayes = GaussianNB()
bayes.fit(X_train, y_train)

tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
tree.fit(X_train, y_train)

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
forest.fit(X_train, y_train)

# ABAS

aba1, aba2, aba3 = st.tabs([
    "1 - Análise dos Dados",
    "2 - Classificação",
    "3 - Métricas"
])

# ABA 1 - Análise Exploratória Dados

with aba1:

    st.header("Análise Exploratória dos Dados")

    st.subheader(
        "Objetivo: Verificar a quantidade de pacientes com e sem doença cardíaca"
    )

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="HeartDisease",
        ax=ax
    )

    ax.set_title("Distribuição da Doença Cardíaca")

    st.pyplot(fig)

    st.subheader(
        "Objetivo: Analisar a distribuição das idades"
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df["Age"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribuição da Idade")

    st.pyplot(fig)

    st.subheader(
        "Objetivo: Comparar sexo e ocorrência da doença"
    )

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="Sex",
        hue="HeartDisease",
        ax=ax
    )

    ax.set_title("Sexo x Doença Cardíaca")

    st.pyplot(fig)

    st.subheader(
        "Objetivo: Avaliar correlações entre variáveis"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df_model.corr(),
        cmap="coolwarm",
        annot=False,
        ax=ax
    )

    ax.set_title("Matriz de Correlação")

    st.pyplot(fig)


# ABA 2 -> Classificação


with aba2:

    st.header("Classificação Probabilística")

    age = st.number_input("Age", value=50)

    sex = st.selectbox(
        "Sex",
        df["Sex"].unique()
    )

    chest = st.selectbox(
        "ChestPainType",
        df["ChestPainType"].unique()
    )

    restingbp = st.number_input(
        "RestingBP",
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        value=200
    )

    fasting = st.selectbox(
        "FastingBS",
        [0, 1]
    )

    ecg = st.selectbox(
        "RestingECG",
        df["RestingECG"].unique()
    )

    maxhr = st.number_input(
        "MaxHR",
        value=150
    )

    angina = st.selectbox(
        "ExerciseAngina",
        df["ExerciseAngina"].unique()
    )

    oldpeak = st.number_input(
        "Oldpeak",
        value=1.0
    )

    slope = st.selectbox(
        "ST_Slope",
        df["ST_Slope"].unique()
    )

    if st.button("Classificar"):

        entrada = pd.DataFrame({
            "Age":[age],
            "Sex":[sex],
            "ChestPainType":[chest],
            "RestingBP":[restingbp],
            "Cholesterol":[cholesterol],
            "FastingBS":[fasting],
            "RestingECG":[ecg],
            "MaxHR":[maxhr],
            "ExerciseAngina":[angina],
            "Oldpeak":[oldpeak],
            "ST_Slope":[slope]
        })

        entrada_model = entrada.copy()

        for col in encoders:
            entrada_model[col] = encoders[col].transform(
                entrada_model[col]
            )

        prob = bayes.predict_proba(
            entrada_model
        )[0]

        pred_bayes = bayes.predict(
            entrada_model
        )[0]

        pred_tree = tree.predict(
            entrada_model
        )[0]

        pred_forest = forest.predict(
            entrada_model
        )[0]

        st.subheader("Resultado Bayes")

        st.write(
            f"Probabilidade Sem Doença: {prob[0]*100:.2f}%"
        )

        st.write(
            f"Probabilidade Com Doença: {prob[1]*100:.2f}%"
        )

        resultado = pd.DataFrame({
            "Método": [
                "Bayes",
                "Árvore",
                "Random Forest"
            ],
            "Predição": [
                pred_bayes,
                pred_tree,
                pred_forest
            ]
        })

        st.subheader(
            "Comparação dos Métodos"
        )

        st.dataframe(resultado)

# ABA 3 -> Métricas

with aba3:

    st.header("Avaliação dos Modelos")

    modelos = {
        "Bayes": bayes,
        "Árvore": tree,
        "Random Forest": forest
    }

    resultados = []

    for nome, modelo in modelos.items():

        pred = modelo.predict(X_test)

        resultados.append({
            "Modelo": nome,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred), 4),
            "Recall": round(recall_score(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4)
        })

    st.dataframe(
        pd.DataFrame(resultados)
    )