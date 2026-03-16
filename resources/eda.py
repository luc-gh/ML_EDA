import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import base64
from io import BytesIO
import re


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