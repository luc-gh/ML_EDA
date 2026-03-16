import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import base64
from io import BytesIO
import re


# ============= FUNÇÕES DO EDA =============

def _translate_report_html_to_pt_br(html_content):
    """
    Traduz os principais textos fixos do ydata-profiling para PT-BR.
    Observacao: nao cobre 100% dos termos internos da biblioteca.
    """
    translations = {
        "Dataset": "Conjunto de dados",
        "Overview": "Visao geral",
        "Variables": "Variaveis",
        "Interactions": "Interacoes",
        "Correlations": "Correlacoes",
        "Missing values": "Valores ausentes",
        "Missing Values": "Valores ausentes",
        "Sample": "Amostra",
        "Duplicate rows": "Linhas duplicadas",
        "Reproduction": "Reproducao",
        "Warnings": "Alertas",
        "Alert": "Alerta",
        "Type": "Tipo",
        "Distinct": "Distintos",
        "Unique": "Unicos",
        "Missing": "Ausentes",
        "Memory size": "Uso de memoria",
        "Record size": "Tamanho por registro",
        "Categorical": "Categorica",
        "Numeric": "Numerica",
        "Boolean": "Booleana",
        "Date": "Data",
        "Text": "Texto",
        "Unsupported": "Nao suportado",
        "Distribution": "Distribuicao",
        "Statistics": "Estatisticas",
        "Quantile statistics": "Estatisticas de quantis",
        "Descriptive statistics": "Estatisticas descritivas",
        "Extreme values": "Valores extremos",
        "Most frequent values": "Valores mais frequentes",
        "Length": "Comprimento",
        "Characters": "Caracteres",
        "Words": "Palavras",
        "Minimum": "Minimo",
        "Maximum": "Maximo",
        "Mean": "Media",
        "Median": "Mediana",
        "Variance": "Variancia",
        "Standard deviation": "Desvio padrao",
        "Skewness": "Assimetria",
        "Kurtosis": "Curtose",
        "Zeros": "Zeros",
        "Negative": "Negativos",
        "Positive": "Positivos",
        "Histogram": "Histograma",
        "Scatter": "Dispersao",
        "No description": "Sem descricao",
        "Toggle navigation": "Alternar navegacao",
    }

    # Evita substituicoes parciais incorretas priorizando chaves mais longas.
    for source in sorted(translations.keys(), key=len, reverse=True):
        target = translations[source]
        pattern = re.compile(re.escape(source))
        html_content = pattern.sub(target, html_content)

    return html_content


def _fig_to_base64(fig):
    """Converte uma figura matplotlib para base64 para embedding em HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_base64}"


def generate_eda_report(df, title="Relatório de Análise Exploratória de Dados"):
    """
    Gera um relatório de EDA usando Jinja2 para construir o HTML completo.
    """
    # Computar dados para o relatório

    # Visão geral
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    total_missing = df.isnull().sum().sum()

    # Informações das colunas
    col_info = pd.DataFrame({
        "Coluna": df.columns,
        "Tipo_de_Dado": df.dtypes.astype(str).values,
        "Nao_Nulos": df.count().values,
        "Valores_Faltantes": df.isnull().sum().values,
        "Percent_Faltantes": (df.isnull().sum().values / len(df) * 100).round(2)
    })

    # Estatísticas descritivas
    numeric_cols = df.select_dtypes(include=['number']).columns
    desc_stats = None
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe().T

    # Valores faltantes
    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Coluna": missing_data[missing_data > 0].index,
        "Faltantes": missing_data[missing_data > 0].values,
        "Percent_Faltantes": missing_percent[missing_data > 0].values
    }) if missing_data.sum() > 0 else None

    # Gráfico de valores faltantes
    missing_plot = None
    if missing_data.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing_data[missing_data > 0].plot(kind='barh', ax=ax, color='#e74c3c')
        ax.set_xlabel("Quantidade de Valores Faltantes")
        ax.set_title("Distribuição de Valores Faltantes por Coluna")
        missing_plot = _fig_to_base64(fig)
        plt.close(fig)

    # Distribuições numéricas
    hist_plots = []
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[col].dropna(), bins=30, color='#3498db', edgecolor='black', alpha=0.7)
            ax.set_title(f"Distribuição de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequência")
            hist_plots.append(_fig_to_base64(fig))
            plt.close(fig)

    # Distribuições categóricas
    cat_plots = []
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() <= 20:
            fig, ax = plt.subplots(figsize=(10, 5))
            df[col].value_counts().head(20).plot(kind='bar', ax=ax, color='#2ecc71', edgecolor='black')
            ax.set_title(f"Frequência de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequência")
            plt.xticks(rotation=45, ha='right')
            cat_plots.append(_fig_to_base64(fig))
            plt.close(fig)

    # Correlações
    corr_matrix = None
    corr_plot = None
    corr_pairs = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Matriz de Correlação - Variáveis Numéricas")
        corr_plot = _fig_to_base64(fig)
        plt.close(fig)

        # Correlações mais fortes
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                pairs.append({
                    "Var1": corr_matrix.columns[i],
                    "Var2": corr_matrix.columns[j],
                    "Corr": corr_matrix.iloc[i, j]
                })
        corr_pairs = pd.DataFrame(pairs).sort_values("Corr", key=abs, ascending=False).head(10)

    # Duplicatas
    num_duplicates = df.duplicated().sum()

    # Template Jinja2
    template_str = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ title }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; }
            .section { margin-bottom: 40px; }
        </style>
    </head>
    <body>
        <h1>{{ title }}</h1>

        <div class="section">
            <h2>Visão Geral dos Dados</h2>
            <p>Linhas: {{ num_rows }}</p>
            <p>Colunas: {{ num_cols }}</p>
            <p>Tamanho (MB): {{ "%.2f"|format(memory_mb) }}</p>
            <p>Valores Faltantes: {{ total_missing }}</p>
        </div>

        <div class="section">
            <h2>Primeiras Linhas do Dataset</h2>
            <table>
                <thead>
                    <tr>
                        {% for col in df.columns %}
                        <th>{{ col }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in df.head(10).itertuples(index=False) %}
                    <tr>
                        {% for value in row %}
                        <td>{{ value }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Informações das Colunas</h2>
            <table>
                <thead>
                    <tr>
                        <th>Coluna</th>
                        <th>Tipo de Dado</th>
                        <th>Não Nulos</th>
                        <th>Valores Faltantes</th>
                        <th>% Faltantes</th>
                    </tr>
                </thead>
                <tbody>
                    {% for info in col_info.itertuples(index=False) %}
                    <tr>
                        <td>{{ info.Coluna }}</td>
                        <td>{{ info.Tipo_de_Dado }}</td>
                        <td>{{ info.Nao_Nulos }}</td>
                        <td>{{ info.Valores_Faltantes }}</td>
                        <td>{{ "%.2f"|format(info.Percent_Faltantes) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if desc_stats is not none %}
        <div class="section">
            <h2>Estatísticas Descritivas - Variáveis Numéricas</h2>
            <table>
                <thead>
                    <tr>
                        <th>Coluna</th>
                        {% for stat in desc_stats.columns %}
                        <th>{{ stat }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in desc_stats.itertuples() %}
                    <tr>
                        <td>{{ row.Index }}</td>
                        {% for value in row[1:] %}
                        <td>{{ "%.2f"|format(value) if value == value else 'NaN' }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% if missing_df is not none %}
        <div class="section">
            <h2>Análise de Valores Faltantes</h2>
            <table>
                <thead>
                    <tr>
                        <th>Coluna</th>
                        <th>Faltantes</th>
                        <th>% Faltantes</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in missing_df.itertuples(index=False) %}
                    <tr>
                        <td>{{ row.Coluna }}</td>
                        <td>{{ row.Faltantes }}</td>
                        <td>{{ "%.2f"|format(row.Percent_Faltantes) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% if missing_plot %}
            <img src="{{ missing_plot }}" alt="Gráfico de Valores Faltantes">
            {% endif %}
        </div>
        {% endif %}

        {% if hist_plots %}
        <div class="section">
            <h2>Distribuição das Variáveis Numéricas</h2>
            {% for plot in hist_plots %}
            <img src="{{ plot }}" alt="Histograma">
            {% endfor %}
        </div>
        {% endif %}

        {% if cat_plots %}
        <div class="section">
            <h2>Distribuição das Variáveis Categóricas</h2>
            {% for plot in cat_plots %}
            <img src="{{ plot }}" alt="Gráfico de Frequência">
            {% endfor %}
        </div>
        {% endif %}

        {% if corr_plot %}
        <div class="section">
            <h2>Análise de Correlações</h2>
            <img src="{{ corr_plot }}" alt="Matriz de Correlação">
            {% if corr_pairs is not none %}
            <h3>Correlações mais Fortes</h3>
            <table>
                <thead>
                    <tr>
                        <th>Variável 1</th>
                        <th>Variável 2</th>
                        <th>Correlação</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in corr_pairs.itertuples(index=False) %}
                    <tr>
                        <td>{{ row.Var1 }}</td>
                        <td>{{ row.Var2 }}</td>
                        <td>{{ "%.2f"|format(row.Corr) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>
        {% endif %}

        <div class="section">
            <h2>Análise de Duplicatas</h2>
            <p>Linhas Duplicadas: {{ num_duplicates }}</p>
        </div>

    </body>
    </html>
    """

    template = Template(template_str)
    html_output = template.render(
        title=title,
        num_rows=num_rows,
        num_cols=num_cols,
        memory_mb=memory_mb,
        total_missing=total_missing,
        df=df,
        col_info=col_info,
        desc_stats=desc_stats,
        missing_df=missing_df,
        missing_plot=missing_plot,
        hist_plots=hist_plots,
        cat_plots=cat_plots,
        corr_plot=corr_plot,
        corr_pairs=corr_pairs,
        num_duplicates=num_duplicates
    )

    return html_output


def save_report_to_html(df, filename="relatorio.html", title="Relatório de Análise Exploratória de Dados"):
    """
    Salva o relatório em um arquivo HTML.
    """
    html_str = generate_eda_report(df, title=title)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_str)
    return filename


# ============= INTERFACE STREAMLIT =============

# Configuração da página
st.set_page_config(
    page_title="Análise Exploratória de Dados (EDA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📊 EDA - Análise Exploratória de Dados")

# Inicializar session state para rastrear arquivo atual
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "use_default_dataset" not in st.session_state:
    st.session_state.use_default_dataset = False

# Layout - Upload do Dataset
st.subheader("📥 Upload do Dataset")
uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

# Detectar se um novo arquivo foi selecionado (para limpar análise anterior)
if uploaded_file is not None:
    file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "uploaded"
    if st.session_state.current_file_name != file_name:
        st.session_state.current_file_name = file_name
        st.session_state.use_default_dataset = False
        st.rerun()

# Abaixo: Formato CSV recomendado e Dataset Padrão
col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("💡 Formato CSV recomendado"):
        st.caption("""
        • Colunas separadas por vírgula (,)
        • Primeira linha como cabeçalho
        • Valores numéricos sem formatação
        • Codificação UTF-8
        """)

with col2:
    st.subheader("📊 Dataset Padrão")
    if st.button("📈 Usar Titanic Dataset", use_container_width=True):
        st.session_state.use_default_dataset = True
        st.session_state.current_file_name = "titanic.csv"
        st.rerun()

# Definir qual arquivo será usado
if st.session_state.use_default_dataset:
    uploaded_file = "titanic.csv"  # Usar o arquivo padrão
    # Mostrar que o dataset padrão foi selecionado
    st.info("📈 Usando Titanic Dataset padrão. Clique em 'Importar novo arquivo' para usar outro dataset.")
elif uploaded_file is None:
    st.session_state.use_default_dataset = False

if uploaded_file is not None or st.session_state.use_default_dataset:
    progress_text = st.empty()
    progress_bar = st.progress(0, text="Aguardando início do processamento...")

    try:
        progress_text.info("📖 Lendo arquivo CSV...")
        progress_bar.progress(15, text="Lendo arquivo CSV")

        # Carregar o arquivo (padrão ou importado)
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        if df.empty:
            raise ValueError("O arquivo CSV está vazio.")

        progress_text.info("✅ Arquivo lido. Iniciando análise...")
        progress_bar.progress(40, text="Gerando análises")

        # SEÇÃO 1: VISÃO GERAL
        st.subheader("📋 Visão Geral")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Linhas", f"{df.shape[0]:,}")
        col2.metric("Colunas", df.shape[1])
        col3.metric("Tamanho (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        col4.metric("Faltantes", f"{df.isnull().sum().sum()}")

        with st.expander("🔍 Primeiras 10 linhas"):
            st.dataframe(df.head(10), use_container_width=True, height=300)

        # SEÇÃO 2: INFORMAÇÕES DAS COLUNAS
        st.subheader("📊 Colunas")
        col_info = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo": df.dtypes.astype(str).values,
            "Não Nulos": df.count().values,
            "Faltantes": df.isnull().sum().values,
            "% Faltantes": (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True, height=300)

        # SEÇÃO 3: ESTATÍSTICAS DESCRITIVAS
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 0:
            st.subheader("📈 Estatísticas Descritivas")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True, height=300)

        # SEÇÃO 4: VALORES FALTANTES
        missing_data = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df) * 100).round(2)

        if missing_data.sum() > 0:
            st.subheader("⚠️ Valores Faltantes")
            col_a, col_b = st.columns([1, 1])

            with col_a:
                missing_df_view = pd.DataFrame({
                    "Coluna": missing_data[missing_data > 0].index,
                    "Qtd": missing_data[missing_data > 0].values,
                    "% Faltantes": missing_percent[missing_data > 0].values
                })
                st.dataframe(missing_df_view, use_container_width=True, height=250)

            with col_b:
                fig, ax = plt.subplots(figsize=(8, 4))
                missing_data[missing_data > 0].plot(kind='barh', ax=ax, color='#e74c3c')
                ax.set_xlabel("Quantidade")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.success("✅ Nenhum valor faltante!")

        # SEÇÃO 5: DISTRIBUIÇÃO DAS VARIÁVEIS
        st.subheader("📊 Distribuição das Variáveis")

        if len(numeric_cols) > 0:
            with st.expander("📈 Histogramas - Variáveis Numéricas"):
                num_cols_per_row = 2
                num_rows = (len(numeric_cols) + num_cols_per_row - 1) // num_cols_per_row

                fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(12, 4*num_rows))
                axes = axes.flatten()

                for idx, col in enumerate(numeric_cols):
                    axes[idx].hist(df[col].dropna(), bins=25, color='#3498db', edgecolor='black', alpha=0.7)
                    axes[idx].set_title(col, fontweight='bold', fontsize=10)
                    axes[idx].set_ylabel("Frequência", fontsize=9)

                for idx in range(len(numeric_cols), len(axes)):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)

        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            with st.expander("📊 Gráficos de Frequência - Variáveis Categóricas"):
                for col in categorical_cols:
                    if df[col].nunique() <= 20:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        df[col].value_counts().head(15).plot(kind='bar', ax=ax, color='#2ecc71', edgecolor='black')
                        ax.set_title(f"{col}", fontweight='bold')
                        ax.set_ylabel("Frequência")
                        plt.xticks(rotation=45, ha='right', fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig)

        # SEÇÃO 6: CORRELAÇÕES
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlações")
            corr_matrix = df[numeric_cols].corr()

            col_corr1, col_corr2 = st.columns([1, 1])

            with col_corr1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           square=True, ax=ax, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
                ax.set_title("Matriz de Correlação", fontweight='bold', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

            with col_corr2:
                st.caption("**Correlações mais Fortes:**")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            "Var 1": corr_matrix.columns[i],
                            "Var 2": corr_matrix.columns[j],
                            "Corr": corr_matrix.iloc[i, j]
                        })
                corr_df = pd.DataFrame(corr_pairs).sort_values("Corr", key=abs, ascending=False)
                st.dataframe(corr_df.head(10), use_container_width=True, height=250)

        # SEÇÃO 7: DUPLICATAS
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            st.warning(f"⚠️ {num_duplicates} linhas duplicadas encontradas")
            with st.expander("Ver duplicatas"):
                st.dataframe(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)), use_container_width=True, height=300)
        else:
            st.success(f"✅ Nenhuma linha duplicada")

        # GERAR RELATÓRIO HTML
        st.divider()

        with st.spinner("⏳ Gerando relatório HTML..."):
            progress_text.info("Processando dados para relatório...")
            progress_bar.progress(80, text="Gerando HTML com Jinja2")

            html_report = generate_eda_report(df, title="Relatório EDA")

            progress_text.info("Finalizando...")
            progress_bar.progress(95, text="Finalizando")

        st.success("✅ Relatório gerado com sucesso!")
        progress_bar.progress(100, text="Concluído")

        # BOTÃO DE DOWNLOAD
        st.subheader("📥 Download do Relatório")
        html_bytes = html_report.encode("utf-8")
        st.download_button(
            label="⬇️ Baixar relatório.html",
            data=html_bytes,
            file_name="relatorio.html",
            mime="text/html",
            use_container_width=True
        )

    except Exception as exc:
        progress_bar.progress(100, text="Erro no processamento")
        st.error("❌ Erro ao gerar o relatório:")
        st.exception(exc)

