import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda import generate_eda_report

# Configuração da página
st.set_page_config(
    page_title="Análise Exploratória de Dados (EDA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📊 EDA - Análise Exploratória de Dados")

# Layout com colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📥 Upload do Dataset")
    uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

with col2:
    with st.expander("💡 Formato CSV recomendado"):
        st.caption("""
        • Colunas separadas por vírgula (,)
        • Primeira linha como cabeçalho
        • Valores numéricos sem formatação
        • Codificação UTF-8
        """)

if uploaded_file is not None:
    progress_text = st.empty()
    progress_bar = st.progress(0, text="Aguardando início do processamento...")

    try:
        progress_text.info("📖 Lendo arquivo CSV...")
        progress_bar.progress(15, text="Lendo arquivo CSV")
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
                missing_df = pd.DataFrame({
                    "Coluna": missing_data[missing_data > 0].index,
                    "Qtd": missing_data[missing_data > 0].values,
                    "% Faltantes": missing_percent[missing_data > 0].values
                })
                st.dataframe(missing_df, use_container_width=True, height=250)

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
