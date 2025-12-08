"""
Projeto COCADA 2025-2: Análise de Atrasos - Linhas do Fundão
João Victor Borges Nascimento - 121064604

FOCO: Atrasos especificamente nos pontos da Cidade Universitária
METODOLOGIA: Processamento em Nuvem (BigQuery ML) para K-Means e PCA
"""

import plotly.express as px
import basedosdados as bd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. CONFIGURAÇÃO (PROJETO NOVO)
# ============================================================================

# ID do projeto onde criamos a tabela e os modelos
BILLING_PROJECT_ID = "profound-portal-480504-a2" 
DATASET_ID = "analise_cocada"

print("=" * 80)
print(f" PROJETO COCADA 2025-2 - ANÁLISE VIA BIGQUERY ML ".center(80, "="))
print("=" * 80)
print(f"🚀 Conectando ao projeto: {BILLING_PROJECT_ID}")

# ============================================================================
# 2. QUERY DE EXTRAÇÃO (JUNTA DADOS + K-MEANS + PCA)
# ============================================================================

# Esta query faz o trabalho pesado:
# 1. Pega os dados resumidos da tabela_resumo
# 2. Aplica o modelo K-Means para descobrir o Cluster
# 3. Aplica o modelo PCA para descobrir as coordenadas X/Y
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
    k.tempo_total_fundao,     -- Minutos gastos dentro do campus
    k.prop_atrasadas,         -- % de viagens que atrasaram no dia
    k.velocidade_media_fundao,
    CAST(k.centroid_id AS STRING) as cluster_id, -- Grupo definido pela IA
    p.principal_component_1 as pc1, -- Eixo X do PCA
    p.principal_component_2 as pc2  -- Eixo Y do PCA
FROM dados_kmeans k
JOIN dados_pca p
  ON k.linha = p.linha AND k.data = p.data
"""

# ============================================================================
# 3. EXECUÇÃO E VISUALIZAÇÃO
# ============================================================================

try:
    print("⏳ Baixando dados processados da nuvem...")
    df = bd.read_sql(query, billing_project_id=BILLING_PROJECT_ID)
    print(f"✅ Sucesso! {len(df)} registros carregados.")
    
    # Configuração visual
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # --- GRÁFICO 1: Mapa de Clusters (PCA) ---
    plt.figure()
    sns.scatterplot(
        data=df, x='pc1', y='pc2', 
        hue='cluster_id', palette='viridis', s=60, alpha=0.8
    )
    plt.title('Mapa de Clusters (Projeção PCA)', fontsize=14, fontweight='bold')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.show()
    print("📈 Gráfico 1 (PCA) gerado.")

    # --- GRÁFICO 2: O Impacto Real (Tempo x Atraso) ---
    plt.figure()
    sns.scatterplot(
        data=df, x='tempo_total_fundao', y='prop_atrasadas', 
        hue='cluster_id', palette='viridis', s=80, alpha=0.7
    )
    plt.title('Impacto do Trânsito Interno na Pontualidade', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo Gasto no Fundão (minutos/dia)', fontsize=12)
    plt.ylabel('Proporção de Viagens Atrasadas (0-1)', fontsize=12)
    plt.legend(title='Grupo')
    plt.tight_layout()
    plt.show()
    print("📈 Gráfico 2 (Dispersão) gerado.")

    # --- GRÁFICO 3: Ranking de Linhas (Quem perde mais tempo?) ---
    plt.figure(figsize=(14, 6))
    # Ordena as linhas pela mediana de tempo gasto
    order = df.groupby('linha')['tempo_total_fundao'].median().sort_values(ascending=False).index
    
    sns.boxplot(x='linha', y='tempo_total_fundao', data=df, order=order, palette="Blues_r")
    plt.title('Ranking: Quais linhas perdem mais tempo no Fundão?', fontsize=14, fontweight='bold')
    plt.xlabel('Linha')
    plt.ylabel('Minutos no Fundão')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("📈 Gráfico 3 (Boxplot) gerado.")

    # Salvar CSV final para o relatório
    df.to_csv('resultados_cocada_final.csv', index=False)
    print("\n📁 Arquivo 'resultados_cocada_final.csv' salvo com sucesso.")
    print("🏁 ANÁLISE CONCLUÍDA!")

except Exception as e:
    print("\n❌ ERRO FATAL:")
    print(e)
    print("\nDICA: Verifique se você rodou os comandos 'CREATE MODEL' no BigQuery Console antes de executar este script.")

# ============================================================================
# 4. PLOTANDO O MAPA REAL (CORRIGIDO)
# ============================================================================

import plotly.express as px

print("\n🗺️ Gerando mapa geoespacial do Fundão...")

# Alteração: Trocamos 'gps_sppo' por 'gps_onibus' que funcionou antes
query_mapa = f"""
WITH classificacao_do_dia AS (
    -- Descobre qual é o cluster de cada linha neste dia
    SELECT linha, data, centroid_id as cluster_id
    FROM ML.PREDICT(MODEL `{BILLING_PROJECT_ID}.{DATASET_ID}.modelo_kmeans`, 
        (SELECT * FROM `{BILLING_PROJECT_ID}.{DATASET_ID}.tabela_resumo`))
    WHERE data = '2025-10-15' 
),
pontos_gps AS (
    -- Pega os pontos brutos da tabela CORRETA (gps_onibus)
    SELECT 
        servico as linha,
        latitude,
        longitude,
        timestamp_gps
    FROM `datario.transporte_rodoviario_municipal.gps_onibus`
    WHERE data = '2025-10-15'
      AND latitude BETWEEN -22.870 AND -22.838
      AND longitude BETWEEN -43.250 AND -43.198
)
SELECT 
    p.linha,
    p.latitude,
    p.longitude,
    p.timestamp_gps,
    CAST(c.cluster_id AS STRING) as cluster_id
FROM pontos_gps p
JOIN classificacao_do_dia c
  ON p.linha = c.linha
LIMIT 5000
"""

try:
    # Adicionei reauth=True por segurança
    df_mapa = bd.read_sql(query_mapa, billing_project_id=BILLING_PROJECT_ID)
    
    if len(df_mapa) > 0:
        print(f"✅ Carregados {len(df_mapa)} pontos de GPS.")

        fig_map = px.scatter_mapbox(
            df_mapa, 
            lat="latitude", 
            lon="longitude", 
            color="cluster_id", 
            hover_name="linha",
            hover_data=["timestamp_gps"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            zoom=13, 
            height=600,
            title="Mapa de Calor: Onde cada grupo circula (2025-10-15)"
        )

        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        
        fig_map.show()
        fig_map.write_html("mapa_fundao_clusters.html")
        print("💾 Mapa salvo como 'mapa_fundao_clusters.html'.")
        
    else:
        print("⚠️ Nenhum dado retornado para 15/10/2024. Tente outra data se necessário.")

except Exception as e:
    print(f"❌ Erro ao gerar mapa: {e}")