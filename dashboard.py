import streamlit as st
import basedosdados as bd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Projeto Final de COACADA",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fundo do app */
    .main {
        background-color: #f5f7fa;
    }


    /* Títulos */
    h1,h2,h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #1e3a8a;
    }


    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #101010;
        color: white;
    }


    /* Textos da sidebar */
    [data-testid="stSidebar"] * {
        color: !important;
    }


    /* Cards */
    .metric-card {
        padding: 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, #6a00f4 0%, #8a2be2 100%);
        box-shadow: 0 0 20px rgba(138, 43, 226, 0.4);
        text-align: center;
        align-items: center;
        border: 1px solid rgba(255,255,255,0.15);
        font-weight: 700;
        color: #ffffff;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
   
    /* Questionário */
    .quiz-container {
        background: linear-gradient(135deg, #F3F3F3 0%, #FFFFFF 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
   
    .big-question {
        font-size: 1.2em;
        font-weight: 600;
        color: #FFFFFF;
        margin: 20px 0 10px 0;
    }
   
    /* Boxes de alerta */
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
        color: black;
    }
   
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
        color: black;
        
    }
   
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
        color: black;
    }
   
    .danger-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
        font-size: 1.1em;
        color: black;
    }
            
    div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child > div {
        background-color: #000000 !important; /* Roxo do seu tema */
    }
        
    div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        border-color: #F3F3F3 !important;
    }
            
    /* Cor da borda quando selecionada */
    div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        border-color: #F3F3F3 !important; /* Mude esta cor para a que desejar */
    }
    


</style>
""", unsafe_allow_html=True)

# Configurações do BigQuery
BILLING_PROJECT_ID = "profound-portal-480504-a2"
DATASET_ID = "analise_cocada"

# Iniciar Aplicação com Questionário
if 'questionario_completo' not in st.session_state:
    st.session_state.questionario_completo = False
if 'pega_onibus' not in st.session_state:
    st.session_state.pega_onibus = None
if 'linha_mais_pega' not in st.session_state:
    st.session_state.linha_mais_pega = None
if 'chute_mais_atrasado' not in st.session_state:
    st.session_state.chute_mais_atrasado = None


# ======================================================================
# HEADER ESTILIZADO
# ======================================================================


st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
        <div style='text-align: center;'>
            <h1 style='color: white; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                🚌 Painel de Atrasos do Fundão
            </h1>
            <p style='color: #f0f0f0; font-size: 1.3em; margin-top: 10px;'>
                <strong>João Victor Borges Nascimento</strong> | COCADA 2025-2 - UFRJ
            </p>
            <div style='display: flex; justify-content: center; gap: 30px; margin-top: 20px; flex-wrap: wrap;'>
                <div style='background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 20px; backdrop-filter: blur(10px);'>
                    <span style='color: white; font-size: 0.9em;'>📊 Análise com K-Means + PCA</span>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 20px; backdrop-filter: blur(10px);'>
                    <span style='color: white; font-size: 0.9em;'>🗺️ Dados da Prefeitura do Rio (SMTR)</span>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 20px; backdrop-filter: blur(10px);'>
                    <span style='color: white; font-size: 0.9em;'>📈 Gráficos Interativos</span>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# Questionário Inicial

if not st.session_state.questionario_completo:
    st.markdown("""
        <div class='quiz-container'>
            <h2 style='text-align: center; color: #667eea;'>📋 Pesquisa Rápida</h2>
            <p style='text-align: center; color: #666;'>
                Antes de ver os dados, conte sua experiência com os ônibus do Fundão!
            </p>
        </div>
    """, unsafe_allow_html=True)
   
    with st.form("questionario_form"):
        st.markdown("<div class='big-question'>1️⃣ Você pega algum ônibus municipal que passe por dentro do Fundão?</div>", unsafe_allow_html=True)
        pega_onibus = st.radio(
            "",
            ["Sim", "Não"],
            key="q1",
            label_visibility="collapsed"
        )
       
        st.markdown("<div class='big-question'>2️⃣ Qual ônibus municipal você mais pegou internamente no Fundão?</div>", unsafe_allow_html=True)
        linha_mais_pega = st.text_input(
            "",
            placeholder="Ex: 917, 918, 355, 321, etc.",
            key="q2",
            label_visibility="collapsed"
        )
       
        st.markdown("<div class='big-question'>3️⃣ Qual seu chute para o ônibus mais atrasado?</div>", unsafe_allow_html=True)
        chute_mais_atrasado = st.text_input(
            "",
            placeholder="Ex: 917, 918, 355, 321, etc.",
            key="q3",
            label_visibility="collapsed"
        )
       
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Ver Resultados", use_container_width=True)
       
        if submitted:
            st.session_state.pega_onibus = pega_onibus
            st.session_state.linha_mais_pega = linha_mais_pega
            st.session_state.chute_mais_atrasado = chute_mais_atrasado
            st.session_state.questionario_completo = True
           
            # Salvar respostas
            try:
                if "respostas" not in st.session_state:
                    st.session_state.respostas = []
               
                st.session_state.respostas.append({
                    "timestamp": datetime.now().isoformat(),
                    "pega_onibus": pega_onibus,
                    "linha_mais_pega": linha_mais_pega,
                    "chute_mais_atrasado": chute_mais_atrasado
                })
            except Exception:
                pass
           
            st.rerun()
   
    st.stop()



# Funcão para carregar dados do BigQuery



@st.cache_data(show_spinner=True)
def carregar_dados():
    query = f"""
    WITH dados_kmeans AS (
        SELECT * FROM ML.PREDICT(MODEL `{BILLING_PROJECT_ID}.{DATASET_ID}.modelo_kmeans`,
            (SELECT * FROM `{BILLING_PROJECT_ID}.{DATASET_ID}.tabela_resumo` WHERE tempo_total_fundao IS NOT NULL))
    ),
    dados_pca AS (
        SELECT * FROM ML.PREDICT(MODEL `{BILLING_PROJECT_ID}.{DATASET_ID}.modelo_pca`,
            (SELECT * FROM `{BILLING_PROJECT_ID}.{DATASET_ID}.tabela_resumo` WHERE tempo_total_fundao IS NOT NULL))
    )
    SELECT
        k.linha,
        k.data,
        k.tempo_total_fundao,
        k.prop_atrasadas,
        k.velocidade_media_fundao,
        CAST(k.centroid_id AS STRING) as cluster_id,
        p.principal_component_1 as pc1,
        p.principal_component_2 as pc2
    FROM dados_kmeans k
    JOIN dados_pca p
      ON k.linha = p.linha AND k.data = p.data
    """
    return bd.read_sql(query, billing_project_id=BILLING_PROJECT_ID)


with st.spinner("🔄 Carregando dados do BigQuery..."):
    df = carregar_dados()
    st.success(f"✅ {len(df):,} registros carregados do BigQuery!")


# Sidebar


st.sidebar.header("📝 Suas Respostas")
st.sidebar.markdown(f"""
<div class='success-box'>
    <strong>🚌 Pega ônibus no Fundão:</strong> {st.session_state.pega_onibus}<br>
    <strong>🎯 Linha mais utilizada:</strong> {st.session_state.linha_mais_pega or "Não informado"}<br>
    <strong>🔮 Seu chute (mais atrasado):</strong> {st.session_state.chute_mais_atrasado or "Não informado"}
</div>
""", unsafe_allow_html=True)


if st.sidebar.button("🔄 Refazer Questionário"):
    st.session_state.questionario_completo = False
    st.rerun()


st.sidebar.markdown("---")
st.sidebar.header("⚙️ Filtros de Análise")


linhas = sorted(df["linha"].unique())
linha_sel = st.sidebar.multiselect(
    "📍 Selecione as linhas:",
    linhas,
    default=linhas,
    help="Escolha uma ou mais linhas para análise detalhada"
)


df_filtrado = df[df["linha"].isin(linha_sel)]


# Chute da linha

linha_mais_atrasada = df.groupby("linha")["prop_atrasadas"].mean().idxmax()
percentual_pior = df.groupby("linha")["prop_atrasadas"].mean().max() * 100


if st.session_state.chute_mais_atrasado:
    if st.session_state.chute_mais_atrasado.strip() == str(linha_mais_atrasada):
        st.markdown(f"""
            <div class='success-box' style='text-align: center; font-size: 1.1em;'>
                🎉 <strong>PARABÉNS!</strong> Você acertou! A linha <strong>{linha_mais_atrasada}</strong>
                é realmente a mais atrasada com {percentual_pior:.1f}% de viagens atrasadas!
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='info-box' style='text-align: center; font-size: 1.1em;'>
                🤔 Você chutou a linha <strong>{st.session_state.chute_mais_atrasado}</strong>,
                mas a mais atrasada é a <strong>{linha_mais_atrasada}</strong> com {percentual_pior:.1f}% de atrasos.
                Veja os dados abaixo!
            </div>
        """, unsafe_allow_html=True)


# Cards de Métricas Principais

col1, col2, col3 = st.columns(3)


with col1:
    st.markdown(f"""
    <div class="metric-card">
    <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">
        {df_filtrado['tempo_total_fundao'].median():.1f} min
    </div>
    <div style="font-size: 1rem; opacity: 0.85;">
        Mediana de tempo no Fundão
    </div>
</div>
""", unsafe_allow_html=True)


with col2:
    st.markdown(f"""
    <div class="metric-card">
    <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">
        {df_filtrado['prop_atrasadas'].mean()*100:.1f}%
    </div>
    <div style="font-size: 1rem; opacity: 0.85;">
        Atraso médio das viagens
    </div>
</div>
    """, unsafe_allow_html=True)


with col3:
    st.markdown(f"""
    <div class="metric-card">
    <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">
        {df_filtrado['cluster_id'].nunique()}
    </div>
    <div style="font-size: 1rem; opacity: 0.85;">
        Clusters identificados
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# Abas

aba = st.tabs([
    "📉 Causa e Efeito",
    "📌 Clusters PCA",
    "📦 Comparação Direta",
    "🗺️ Mapa Fundão",
    "🏆 O Pódio da Lerdeza"
])


# ======================================================================
# ABA 1 – CAUSA E EFEITO
# ======================================================================


with aba[0]:
    st.header("📉 A Prova da Causa e Efeito")
   
    st.markdown("""
    <div class='info-box'>
        <strong>💡 Interpretação:</strong> Este gráfico mostra a correlação entre o tempo que cada linha passa
        no Fundão (eixo X) e a proporção de viagens atrasadas (eixo Y). <br>
        <strong>Pontos no canto superior direito = Linhas problemáticas</strong> (muito tempo + muitos atrasos)
    </div>
    """, unsafe_allow_html=True)
   
    fig_scatter = px.scatter(
        df_filtrado,
        x="tempo_total_fundao",
        y="prop_atrasadas",
        color="cluster_id",
        size="velocidade_media_fundao",
        hover_data=["linha", "data"],
        title="Tempo no Fundão × Proporção de Atrasos",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=600,
        labels={
            "tempo_total_fundao": "Tempo Total no Fundão (minutos)",
            "prop_atrasadas": "Proporção de Viagens Atrasadas",
            "cluster_id": "Cluster"
        }
    )
   
    fig_scatter.update_layout(plot_bgcolor='white')
    fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
   
    st.plotly_chart(fig_scatter, use_container_width=True)
   
    st.markdown("""
    <div class='warning-box'>
        <strong>🎯 Como interpretar:</strong><br>
        • Cada ponto = um dia de operação de uma linha<br>
        • <strong>Canto superior direito</strong> = Dias ruins (muito tempo + muito atraso)<br>
        • <strong>Canto inferior esquerdo</strong> = Dias bons (pouco tempo + pouco atraso)<br>
        • A correlação é clara: <strong>mais tempo interno = mais atrasos</strong>
    </div>
    """, unsafe_allow_html=True)


    # ======================================================================
# ABA 2 – CLUSTERS PCA
# ======================================================================


with aba[1]:
    st.header("📌 Mapa de Clusters – Análise PCA")
   
    st.markdown("""
    <div class='info-box'>
        <strong>🔬 Análise de Componentes Principais (PCA):</strong>
        Este gráfico reduz múltiplas variáveis (tempo, velocidade, atrasos) em 2 dimensões principais,
        revelando padrões naturais de agrupamento entre as linhas.
    </div>
    """, unsafe_allow_html=True)
   
    fig_pca = px.scatter(
        df_filtrado,
        x="pc1",
        y="pc2",
        color="cluster_id",
        hover_data=["linha", "tempo_total_fundao", "prop_atrasadas"],
        title="Distribuição dos Clusters no Espaço PCA",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=600,
        labels={
            "pc1": "Componente Principal 1",
            "pc2": "Componente Principal 2",
            "cluster_id": "Cluster"
        }
    )
   
    fig_pca.update_layout(plot_bgcolor='white')
    fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
   
    st.plotly_chart(fig_pca, use_container_width=True)


# ======================================================================
# ABA 3 – COMPARAÇÃO DIRETA (BOXPLOT)
# ======================================================================


with aba[2]:
    st.header("📦 Comparação Direta Entre Linhas")
   
    st.markdown("""
    <div class='info-box'>
        <strong>💡 Dica de Uso:</strong> Use o filtro na barra lateral para selecionar apenas 2-3 linhas
        e ver a diferença brutal entre uma linha problemática (ex: 321) e uma linha eficiente (ex: 945 ou 635).
    </div>
    """, unsafe_allow_html=True)
   
    if len(linha_sel) == 0:
        st.warning("⚠️ Selecione pelo menos uma linha no filtro lateral!")
    else:
        ordem = df_filtrado.groupby("linha")["tempo_total_fundao"].median().sort_values(ascending=False).index
       
        fig_box = px.box(
            df_filtrado,
            x="linha",
            y="tempo_total_fundao",
            color="cluster_id",
            title="Distribuição de Tempos por Linha (Box Plot)",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=600,
            labels={
                "linha": "Linha",
                "tempo_total_fundao": "Tempo Total no Fundão (minutos)",
                "cluster_id": "Cluster"
            }
        )
       
        fig_box.update_xaxes(categoryorder='array', categoryarray=ordem)
        fig_box.update_layout(plot_bgcolor='white')
       
        st.plotly_chart(fig_box, use_container_width=True)
       
        # Estatísticas comparativas
        if len(linha_sel) >= 2:
            st.subheader("📊 Comparação Estatística")
           
            comparacao = df_filtrado.groupby("linha").agg({
                "tempo_total_fundao": ["min", "median", "max", "sum"],
                "prop_atrasadas": "mean"
            }).round(2)
           
            comparacao.columns = ['Mínimo (min)', 'Mediana (min)', 'Máximo (min)', 'Total Acumulado (min)', 'Prop. Atrasos']
            comparacao['Prop. Atrasos'] = (comparacao['Prop. Atrasos'] * 100).round(1).astype(str) + '%'
           
            st.dataframe(comparacao, use_container_width=True)


# ======================================================================
# FUNÇÃO DO MAPA
# ======================================================================


@st.cache_data(show_spinner=True)
def carregar_mapa():
    query_mapa = f"""
    WITH classificacao_do_dia AS (
        SELECT linha, data, centroid_id as cluster_id
        FROM ML.PREDICT(MODEL `{BILLING_PROJECT_ID}.{DATASET_ID}.modelo_kmeans`,
            (SELECT * FROM `{BILLING_PROJECT_ID}.{DATASET_ID}.tabela_resumo`))
        WHERE data = '2025-10-15'
    ),
    pontos_gps AS (
        SELECT
            servico as linha,
            latitude,
            longitude,
            timestamp_gps
        FROM `datario.transporte_rodoviario_municipal.gps_onibus`
        WHERE data = '2025-10-15' # Filtrar por data específica
          AND latitude BETWEEN -22.870 AND -22.838
          AND longitude BETWEEN -43.250 AND -43.198
    )
    SELECT p.*, CAST(c.cluster_id AS STRING) as cluster_id
    FROM pontos_gps p
    JOIN classificacao_do_dia c ON p.linha = c.linha
    LIMIT 5000
    """
    return bd.read_sql(query_mapa, billing_project_id=BILLING_PROJECT_ID)


# ======================================================================
# ABA 4 – MAPA
# ======================================================================


with aba[3]:
    st.header("🗺️ Mapa de Circulação por Cluster – Fundão")
   
    with st.spinner("🗺️ Carregando pontos GPS..."):
        try:
            df_mapa = carregar_mapa()
           
            if len(df_mapa) > 0:
                # Filtrar pelo que foi selecionado
                if linha_sel:
                    df_mapa = df_mapa[df_mapa["linha"].isin(linha_sel)]
               
                fig_map = px.scatter_mapbox(
                    df_mapa,
                    lat="latitude",
                    lon="longitude",
                    color="cluster_id",
                    hover_name="linha",
                    hover_data=["timestamp_gps"],
                    zoom=13,
                    height=700,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Distribuição Espacial dos Clusters"
                )
                fig_map.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig_map, use_container_width=True)
               
                st.info(f"📍 Exibindo {len(df_mapa):,} pontos GPS no mapa")
            else:
                st.warning("⚠️ Nenhum dado de GPS disponível para a data/linhas selecionadas.")
        except Exception as e:
            st.error(f"❌ Erro ao carregar mapa: {str(e)}")


# ABA 5 – O Pódio

with aba[4]:
    st.header("🏆 O Podium da Vergonha: Linhas Mais Atrasadas")
   
    st.markdown("""
    <div class='info-box'>
        <strong>📊 Metodologia:</strong> Este ranking ordena as linhas pela <strong>proporção média de viagens atrasadas</strong>.
        Consideramos apenas linhas com pelo menos 5 observações para garantir confiabilidade estatística.
    </div>
    """, unsafe_allow_html=True)
   
    # Calcular ranking
    linhas_com_dados = df.groupby("linha").size()
    linhas_validas = linhas_com_dados[linhas_com_dados >= 5].index
   
    ranking_atrasos = df[df["linha"].isin(linhas_validas)].groupby("linha").agg({
        "prop_atrasadas": ["mean", "std", "count"],
        "tempo_total_fundao": ["mean", "sum"],
        "velocidade_media_fundao": "mean"
    })
   
    ranking_atrasos.columns = ['_'.join(col).strip() for col in ranking_atrasos.columns.values]
    ranking_atrasos = ranking_atrasos.sort_values("prop_atrasadas_mean", ascending=False).head(10)
   
    # Tabela completa do ranking
    st.subheader("📋 Top 10 Linhas Mais Atrasadas")
   
    ranking_display = pd.DataFrame({
        "🏆": range(1, len(ranking_atrasos) + 1),
        "Linha": ranking_atrasos.index,
        "% Atrasos": (ranking_atrasos["prop_atrasadas_mean"] * 100).round(1).astype(str) + "%",
        "Observações": ranking_atrasos["prop_atrasadas_count"].astype(int),
        "Tempo Médio": ranking_atrasos["tempo_total_fundao_mean"].round(1).astype(str) + " min",
        "Tempo Total": ranking_atrasos["tempo_total_fundao_sum"].round(0).astype(int).apply(lambda x: f"{x:,}") + " min"
    })
   
    st.dataframe(ranking_display, hide_index=True, use_container_width=True, height=400)
   
    st.markdown("---")
   
    # ALERTA VERMELHO - Destaque da pior linha
    linha_pior = ranking_atrasos.index[0]
    pior_atraso = ranking_atrasos.iloc[0]["prop_atrasadas_mean"] * 100
    pior_tempo_medio = ranking_atrasos.iloc[0]["tempo_total_fundao_mean"]
    pior_tempo_total = ranking_atrasos.iloc[0]["tempo_total_fundao_sum"]
    pior_observacoes = int(ranking_atrasos.iloc[0]["prop_atrasadas_count"])
   
    st.markdown(f"""
    <div class='danger-box' style='text-align: center;'>
        🚨 <strong>ALERTA VERMELHO - LINHA CAMPEÃ DE ATRASOS</strong> 🚨<br><br>
        <span style='font-size: 2em; color: #dc3545; font-weight: bold;'>LINHA {linha_pior}</span><br><br>
        <strong style='color: #000;'>{pior_atraso:.1f}%</strong> de viagens atrasadas (baseado em {pior_observacoes} observações)<br>
        Tempo médio de <strong style='color: #000;'>{pior_tempo_medio:.1f} minutos</strong> no Fundão<br>
        Tempo total acumulado: <strong style='color: #000;'>{int(pior_tempo_total):,} minutos</strong><br><br>
        <em style='color: #721c24;'>⚠️ Este não é um dado isolado. É uma média consolidada de múltiplas observações.</em>
    </div>
    """, unsafe_allow_html=True)


# FOOTER


st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #EEEEEE; padding: 20px;'>
        <p><strong>COCADA 2025-2</strong> - Universidade Federal do Rio de Janeiro</p>
        <p>Dados: BigQuery</p>
        <p>Dúvidas? @jv.borges18</p>
    </div>
""", unsafe_allow_html=True)

