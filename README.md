# Análise de Doenças Cardíacas com Aprendizagem Automática

## Disciplina: Estatística e Probabilidade
**Equipa:** Adisson Pires, Lucas do Carmo, Murilo Dias

---

## 💻 Sobre o Projeto

Este projeto consiste numa análise preditiva e estatística voltada para a área da saúde cardiovascular. O objetivo principal é identificar a probabilidade de um paciente apresentar uma doença cardíaca com base nos seus dados clínicos. 

Para isso, foi desenvolvido um classificador manual utilizando o **Teorema de Bayes** (calculado de raiz em escala logarítmica e com suavização de Laplace), além de modelos clássicos de *Machine Learning* via Scikit-Learn (**KNN, Árvore de Decisão e Random Forest**). Os resultados e as análises exploratórias foram integrados num *dashboard* interativo desenvolvido com **Streamlit**.

---

## 📊 O Dataset (Conjunto de Dados)

O estudo utiliza o *Heart Failure Prediction Dataset*, disponível publicamente no Kaggle. Este conjunto de dados unifica 5 bases históricas do repositório UCI Machine Learning (*Cleveland, Hungarian, Switzerland, Long Beach VA e Stalog Heart Dataset*).

* **Instâncias:** 918 pacientes
* **Atributos:** 12 (11 preditores + 1 variável alvo)
* **Variável Alvo:** `HeartDisease` (0 = Saudável | 1 = Com doença cardíaca)

### Variáveis Clínicas Analisadas:
* `Age`: Idade do paciente (anos)
* `Sex`: Sexo biológico (M/F)
* `ChestPainType`: Tipo de dor no peito (ATA, NAP, ASY, TA)
* `RestingBP`: Pressão arterial em repouso (mm Hg)
* `Cholesterol`: Colesterol sérico (mg/dL)
* `FastingBS`: Glicose em jejum > 120 mg/dL (0/1)
* `RestingECG`: Resultado do ECG em repouso (Normal, ST, LVH)
* `MaxHR`: Frequência cardíaca máxima atingida
* `ExerciseAngina`: Angina induzida por exercício (Y/N)
* `Oldpeak`: Depressão do segmento ST induzida por exercício
* `ST_Slope`: Inclinação do segmento ST (Up, Flat, Down)

---

## ⚙️ Pré-processamento e Tratamento de Dados

Para garantir a integridade das predições, o conjunto de dados passou pelas seguintes etapas de *pipeline*:

1.  **Identificação de Zeros Biologicamente Impossíveis:** Foram detetados 172 registos com colesterol a zero e 1 registo com pressão arterial a zero.
2.  **Imputação por Grupo:** Estes valores nulos mascarados foram substituídos pela **mediana agrupada por sexo biológico**, preservando particularidades fisiológicas e evitando distorções causadas por *outliers*.
3.  **Engenharia de Atributos (*Feature Engineering*):** * Aplicação de *One-Hot Encoding* para converter variáveis categóricas textuais em binárias.
    * Padronização de *features* numéricas utilizando o *StandardScaler* para o algoritmo KNN.
    * Conversão da variável alvo para o tipo Booleano para evitar inclusões indevidas em matrizes de correlação numérica.

---

## 🧠 Modelos Implementados

### 1. Naive Bayes Manual
Desenvolvido de raiz (sem o uso de bibliotecas de ML prontas para esta finalidade):
* **Variáveis Contínuas:** Abordadas via Verosimilhança Gaussiana.
* **Variáveis Categóricas:** Implementadas com **Suavização de Laplace** para solucionar o *zero-frequency problem* (probabilidades a zero em combinações não vistas no treino).
* **Evitação de Underflow:** Cálculo estruturado sob a soma de **Log-Probabilidades**, convertido no final em distribuições reais via transformação *Softmax*.

### 2. Algoritmos de Machine Learning (Scikit-Learn)
* **K-Nearest Neighbors (KNN):** Configurado com $K=5$ e distância euclidiana.
* **Árvore de Decisão:** Limitada em profundidade máxima (`max_depth`) para garantir a interpretabilidade e prevenir *overfitting*.
* **Random Forest:** Conjunto de árvores configurado para ampliar a estabilidade das métricas de predição.

---

## 📈 Resultados Comparativos

Os desempenhos obtidos pelos métodos avaliados estruturam-se da seguinte forma:

| Métrica | Naive Bayes (Manual) | KNN (K=5) | Árvore de Decisão |
| :--- | :---: | :---: | :---: |
| **Acurácia** | 84,1% | ~85,5% | ~86,2% |
| **Precisão** | 85,3% | ~85,9% | ~86,8% |
| **Recall** | 86,6% | ~87,1% | ~87,5% |
| **F1-Score** | 85,9% | ~86,5% | ~87,1% |

*Nota: O modelo Naive Bayes manual foi avaliado na base completa para validação de consistência matemática, enquanto os restantes algoritmos utilizaram a divisão estratificada de treino/teste (70/30).*

---

## 🖥️ O Dashboard Interativo (Streamlit)

A interface gráfica web do projeto divide-se em três separadores funcionais:

1.  **Análise dos Dados (EDA):** Gráficos interativos ilustrando a distribuição de diagnósticos, pirâmide de risco por idade, impacto do sexo biológico na incidência clínica e mapa térmico de correlações.
2.  **Classificação Probabilística:** Simulador prático onde o utilizador preenche os dados de um novo paciente e obtém instantaneamente as predições de diagnóstico e probabilidades associadas.
3.  **Métricas:** Painel comparativo de performance entre os classificadores implementados.

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
Certifique-se de que possui o Python 3.8+ instalado na sua máquina.

### Passos para Instalação

1. Clone o repositório para o seu ambiente local:
   ```bash
   git clone [https://github.com/o-seu-utilizador/o-seu-repositorio.git](https://github.com/o-seu-utilizador/o-seu-repositorio.git)
   cd o-seu-repositorio
