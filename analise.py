"""
Projeto COCADA 2025-2: Análise de Atrasos - Linhas do Fundão
João Victor Borges Nascimento - 121064604

Usando Base dos Dados (basedosdados) para acesso aos dados
"""

import basedosdados as bd
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# IMPORTANTE: Substitua pelo ID do seu projeto GCP
# Para obter: https://console.cloud.google.com/
BILLING_PROJECT_ID = "universal-helix-468201-g9"  # <<<< ALTERE AQUI

# Linhas que atendem o Fundão
LINHAS_FUNDAO = [
    '355', '371', '384', '385', '386', '387', '388', '389', '390', '391',
    '393', '394', '395', '396', '397', '398', '399', '633', '634', '635',
    '636', '638', '639', '653', '770', '771', '774', '775', '776', '777',
    '778', '779', '905', '910', '911', '917', '918'
]

# Período de análise
DATA_INICIO = '2025-03-01'
DATA_FIM = '2025-10-31'

# Threshold de atraso
ATRASO_THRESHOLD = 15  # minutos

# ============================================================================
# 1. EXTRAÇÃO DE DADOS
# ============================================================================

def extrair_dados_fundao(billing_project_id, teste=False):
    """
    Extrai dados das linhas do Fundão usando Base dos Dados
    
    Args:
        billing_project_id: ID do projeto GCP para billing
        teste: Se True, extrai apenas amostra pequena
    """
    print("=" * 80)
    print(" EXTRAÇÃO DE DADOS ".center(80, "="))
    print("=" * 80)
    
    linhas_str = "', '".join(LINHAS_FUNDAO)
    limite = "LIMIT 5000" if teste else ""
    
    query = f"""
    SELECT 
        data,
        servico as linha,
        datetime_partida,
        datetime_chegada,
        tempo_viagem,
        distancia_planejada,
        perc_conformidade_shape,
        perc_conformidade_registros,
        id_viagem,
        sentido
    FROM 
        `datario.transporte_rodoviario_municipal.viagem_onibus`
    WHERE 
        servico IN ('{linhas_str}')
        AND data BETWEEN '{DATA_INICIO}' AND '{DATA_FIM}'
        AND EXTRACT(DAYOFWEEK FROM data) NOT IN (1, 7)
        AND datetime_partida IS NOT NULL
        AND datetime_chegada IS NOT NULL
        AND tempo_viagem IS NOT NULL
        AND tempo_viagem > 0
    ORDER BY data, servico, datetime_partida
    {limite}
    """
    
    print(f"\n📊 Consultando Base dos Dados...")
    print(f"   Período: {DATA_INICIO} a {DATA_FIM}")
    print(f"   Linhas: {len(LINHAS_FUNDAO)}")
    print(f"   Modo: {'TESTE (amostra)' if teste else 'COMPLETO'}")
    
    try:
        df = bd.read_sql(query, billing_project_id=billing_project_id)
        
        print(f"\n✅ Dados carregados com sucesso!")
        print(f"   Total de viagens: {len(df):,}")
        print(f"   Linhas únicas: {df['linha'].nunique()}")
        print(f"   Período: {df['data'].min()} a {df['data'].max()}")
        print(f"   Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar dados: {e}")
        print("\n💡 Dicas:")
        print("   1. Verifique se BILLING_PROJECT_ID está correto")
        print("   2. Configure: gcloud auth application-default login")
        print("   3. Ou use: bd.list_dataset_tables('datario') para listar tabelas")
        return None

# ============================================================================
# 2. PRÉ-PROCESSAMENTO E CÁLCULO DE FEATURES
# ============================================================================

def preprocessar_dados(df):
    """
    Calcula features de atraso e velocidade
    """
    print("\n" + "=" * 80)
    print(" PRÉ-PROCESSAMENTO ".center(80, "="))
    print("=" * 80)
    
    df = df.copy()
    
    # 1. Converter tempo de viagem para minutos
    df['tempo_viagem_min'] = df['tempo_viagem'] / 60.0
    
    # 2. Extrair features temporais
    df['hora_partida'] = pd.to_datetime(df['datetime_partida']).dt.hour
    df['dia_semana'] = pd.to_datetime(df['datetime_partida']).dt.dayofweek
    df['mes'] = pd.to_datetime(df['data']).dt.month
    
    # Classificar período do dia
    def classificar_periodo(hora):
        if 6 <= hora < 9:
            return 'pico_manha'
        elif 17 <= hora < 20:
            return 'pico_tarde'
        else:
            return 'fora_pico'
    
    df['periodo'] = df['hora_partida'].apply(classificar_periodo)
    
    # 3. Calcular velocidade média (km/h)
    df['velocidade_kmh'] = np.where(
        df['tempo_viagem_min'] > 0,
        (df['distancia_planejada'] / 1000) / (df['tempo_viagem_min'] / 60),
        0
    )
    
    # 4. Calcular tempo de referência (baseline) por linha/hora/dia_semana
    print("\n📐 Calculando tempo de referência (baseline)...")
    tempo_ref = df.groupby(['linha', 'hora_partida', 'dia_semana'])['tempo_viagem_min'].agg([
        ('tempo_referencia', 'median'),
        ('tempo_q25', lambda x: x.quantile(0.25)),
        ('tempo_q75', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    df = df.merge(tempo_ref, on=['linha', 'hora_partida', 'dia_semana'], how='left')
    
    # 5. Calcular "atraso" como desvio do tempo de referência
    df['atraso_min'] = df['tempo_viagem_min'] - df['tempo_referencia']
    df['atrasada'] = df['atraso_min'] > ATRASO_THRESHOLD
    
    # 6. Calcular variabilidade (IQR)
    df['tempo_iqr'] = df['tempo_q75'] - df['tempo_q25']
    
    # Estatísticas
    print(f"\n✅ Features calculadas")
    print(f"   Tempo médio de viagem: {df['tempo_viagem_min'].mean():.1f} min")
    print(f"   Velocidade média: {df['velocidade_kmh'].mean():.1f} km/h")
    print(f"   Atraso médio: {df['atraso_min'].mean():+.1f} min")
    print(f"   % viagens atrasadas (>{ATRASO_THRESHOLD}min): {df['atrasada'].mean()*100:.1f}%")
    
    return df

def agregar_por_linha_dia(df):
    """
    Agrega dados por linha × dia para análise
    """
    print("\n" + "=" * 80)
    print(" AGREGAÇÃO LINHA × DIA ".center(80, "="))
    print("=" * 80)
    
    # Agregação
    agregado = df.groupby(['linha', 'data']).agg({
        'atraso_min': ['mean', 'std', 'max', 'min'],
        'atrasada': 'mean',
        'velocidade_kmh': ['mean', 'std'],
        'tempo_viagem_min': ['mean', 'std'],
        'tempo_iqr': 'mean',
        'perc_conformidade_shape': 'mean',
        'perc_conformidade_registros': 'mean',
        'id_viagem': 'count'
    }).reset_index()
    
    # Simplificar nomes
    agregado.columns = [
        'linha', 'data',
        'atraso_medio', 'atraso_std', 'atraso_max', 'atraso_min',
        'prop_atrasadas',
        'velocidade_media', 'velocidade_std',
        'tempo_viagem_medio', 'tempo_viagem_std',
        'variabilidade_iqr',
        'conformidade_shape', 'conformidade_registros',
        'num_viagens'
    ]
    
    # Features temporais
    agregado['dia_semana'] = pd.to_datetime(agregado['data']).dt.dayofweek
    agregado['mes'] = pd.to_datetime(agregado['data']).dt.month
    
    # Filtrar registros com poucas viagens (outliers)
    agregado = agregado[agregado['num_viagens'] >= 3]
    
    print(f"\n✅ Agregação concluída")
    print(f"   Registros linha×dia: {len(agregado):,}")
    print(f"   Linhas únicas: {agregado['linha'].nunique()}")
    print(f"   Dias únicos: {agregado['data'].nunique()}")
    
    return agregado

# ============================================================================
# 3. ANÁLISE EXPLORATÓRIA
# ============================================================================

def analise_exploratoria(df_agregado):
    """
    Análise exploratória com rankings e estatísticas
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE EXPLORATÓRIA ".center(80, "="))
    print("=" * 80)
    
    # Rankings por linha
    ranking_atraso = df_agregado.groupby('linha')['atraso_medio'].mean().sort_values(ascending=False)
    ranking_prop = df_agregado.groupby('linha')['prop_atrasadas'].mean().sort_values(ascending=False)
    ranking_velocidade = df_agregado.groupby('linha')['velocidade_media'].mean().sort_values(ascending=True)
    
    print("\n🔴 TOP 10 LINHAS - MAIOR ATRASO MÉDIO")
    print("-" * 50)
    for i, (linha, atraso) in enumerate(ranking_atraso.head(10).items(), 1):
        print(f"  {i:2d}. Linha {linha:>4s}: {atraso:+6.1f} min")
    
    print("\n🟡 TOP 10 LINHAS - MAIOR % VIAGENS ATRASADAS")
    print("-" * 50)
    for i, (linha, prop) in enumerate(ranking_prop.head(10).items(), 1):
        print(f"  {i:2d}. Linha {linha:>4s}: {prop*100:5.1f}%")
    
    print("\n🐌 TOP 10 LINHAS - MENOR VELOCIDADE MÉDIA")
    print("-" * 50)
    for i, (linha, vel) in enumerate(ranking_velocidade.head(10).items(), 1):
        print(f"  {i:2d}. Linha {linha:>4s}: {vel:5.1f} km/h")
    
    # Estatísticas gerais
    print("\n📊 ESTATÍSTICAS GERAIS")
    print("-" * 50)
    print(f"  Atraso médio geral: {df_agregado['atraso_medio'].mean():+.1f} min")
    print(f"  Desvio padrão atraso: {df_agregado['atraso_medio'].std():.1f} min")
    print(f"  Velocidade média geral: {df_agregado['velocidade_media'].mean():.1f} km/h")
    print(f"  Conformidade média: {df_agregado['conformidade_shape'].mean()*100:.1f}%")
    
    return ranking_atraso, ranking_prop, ranking_velocidade

# ============================================================================
# 4. PCA - ANÁLISE DE COMPONENTES PRINCIPAIS
# ============================================================================

def aplicar_pca(df_agregado, n_components=3):
    """
    Redução de dimensionalidade via PCA
    """
    print("\n" + "=" * 80)
    print(" PCA - ANÁLISE DE COMPONENTES PRINCIPAIS ".center(80, "="))
    print("=" * 80)
    
    # Selecionar features
    features = [
        'atraso_medio', 'atraso_std', 'atraso_max',
        'prop_atrasadas',
        'velocidade_media', 'velocidade_std',
        'tempo_viagem_medio', 'tempo_viagem_std',
        'variabilidade_iqr',
        'conformidade_shape', 'conformidade_registros'
    ]
    
    X = df_agregado[features].fillna(0)
    
    # Remover valores infinitos
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Variância explicada
    print("\n📈 VARIÂNCIA EXPLICADA")
    print("-" * 50)
    cumsum = 0
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        cumsum += var
        print(f"  PC{i}: {var*100:5.1f}% (acumulado: {cumsum*100:5.1f}%)")
    
    # Loadings
    print("\n🔍 LOADINGS (contribuição de cada feature)")
    print("-" * 50)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i}' for i in range(1, n_components+1)],
        index=features
    )
    
    for i in range(n_components):
        print(f"\n  PC{i+1} - Principais fatores:")
        top_features = loadings[f'PC{i+1}'].abs().nlargest(3)
        for feat, _ in top_features.items():
            val = loadings.loc[feat, f'PC{i+1}']
            sinal = "+" if val > 0 else ""
            print(f"    {feat:30s}: {sinal}{val:.3f}")
    
    return X_pca, pca, scaler, X_scaled, features

# ============================================================================
# 5. K-MEANS CLUSTERING
# ============================================================================

def determinar_k_otimo(X, k_range=range(2, 8)):
    """
    Método do cotovelo + silhouette para determinar k
    """
    print("\n" + "=" * 80)
    print(" DETERMINAÇÃO DO K ÓTIMO ".center(80, "="))
    print("=" * 80)
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            sil = silhouette_score(X, labels)
            silhouettes.append(sil)
        else:
            silhouettes.append(0)
        
        print(f"  k={k}: inércia={kmeans.inertia_:8.2f}, silhouette={silhouettes[-1]:.3f}")
    
    # Plotar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Método do cotovelo
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inércia', fontsize=11)
    axes[0].set_title('Método do Cotovelo', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette
    axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Coeficiente de Silhueta', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_otimo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sugerir k ótimo
    k_sugerido = k_range[np.argmax(silhouettes)]
    print(f"\n💡 K sugerido (maior silhouette): {k_sugerido}")
    
    return inertias, silhouettes

def aplicar_kmeans(X, n_clusters=4):
    """
    Clusterização via K-Means
    """
    print(f"\n⚙️  Aplicando K-Means com k={n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Silhouette score
    sil_score = silhouette_score(X, clusters)
    
    print(f"✅ Clustering concluído")
    print(f"   Silhouette Score: {sil_score:.3f}")
    
    # Distribuição
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"\n   Distribuição de clusters:")
    for cluster_id, count in zip(unique, counts):
        print(f"     Cluster {cluster_id}: {count:5d} registros ({count/len(clusters)*100:4.1f}%)")
    
    return clusters, kmeans

def analisar_clusters(df_agregado, clusters):
    """
    Caracteriza cada cluster identificando perfis
    """
    print("\n" + "=" * 80)
    print(" CARACTERIZAÇÃO DOS CLUSTERS ".center(80, "="))
    print("=" * 80)
    
    df_agregado = df_agregado.copy()
    df_agregado['cluster'] = clusters
    
    resultados_clusters = []
    
    for cluster_id in sorted(df_agregado['cluster'].unique()):
        print(f"\n{'='*80}")
        print(f" CLUSTER {cluster_id} ".center(80, "="))
        print('='*80)
        
        cluster_df = df_agregado[df_agregado['cluster'] == cluster_id]
        
        # Tamanho
        print(f"\n📊 Tamanho: {len(cluster_df):,} registros ({len(cluster_df)/len(df_agregado)*100:.1f}%)")
        
        # Linhas
        linhas = sorted(cluster_df['linha'].unique())
        print(f"\n🚌 Linhas ({len(linhas)}): {', '.join(linhas)}")
        
        # Características médias
        stats = {
            'atraso_medio': cluster_df['atraso_medio'].mean(),
            'prop_atrasadas': cluster_df['prop_atrasadas'].mean(),
            'velocidade_media': cluster_df['velocidade_media'].mean(),
            'conformidade': cluster_df['conformidade_shape'].mean(),
            'variabilidade': cluster_df['atraso_std'].mean()
        }
        
        print(f"\n📈 Características médias:")
        print(f"   Atraso médio:       {stats['atraso_medio']:+6.1f} min")
        print(f"   % atrasadas:        {stats['prop_atrasadas']*100:6.1f}%")
        print(f"   Velocidade:         {stats['velocidade_media']:6.1f} km/h")
        print(f"   Conformidade:       {stats['conformidade']*100:6.1f}%")
        print(f"   Variabilidade:      {stats['variabilidade']:6.1f} min")
        
        # Identificar perfil
        if stats['atraso_medio'] > 10 and stats['prop_atrasadas'] > 0.5:
            perfil = "🔴 CRONICAMENTE ATRASADO - Alta irregularidade"
        elif stats['atraso_medio'] > 5 and stats['prop_atrasadas'] > 0.3:
            perfil = "🟡 MODERADAMENTE IRREGULAR - Atrasos frequentes"
        elif stats['velocidade_media'] < 15:
            perfil = "🐌 LENTO MAS REGULAR - Baixa velocidade operacional"
        elif stats['atraso_medio'] < 0:
            perfil = "🟢 PONTUAL E RÁPIDO - Acima da expectativa"
        else:
            perfil = "🟢 REGULAR E PREVISÍVEL - Operação normal"
        
        print(f"\n🏷️  Perfil: {perfil}")
        
        resultados_clusters.append({
            'cluster': cluster_id,
            'perfil': perfil,
            'linhas': linhas,
            **stats
        })
    
    return pd.DataFrame(resultados_clusters)

# ============================================================================
# 6. VISUALIZAÇÕES
# ============================================================================

def visualizar_pca_clusters(X_pca, clusters, df_agregado):
    """
    Visualiza clusters no espaço PCA
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PC1 vs PC2
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=clusters, cmap='viridis', 
                               alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel('PC1 (Componente Principal 1)', fontsize=11)
    axes[0].set_ylabel('PC2 (Componente Principal 2)', fontsize=11)
    axes[0].set_title('Clusters no Espaço PCA (PC1 vs PC2)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Cluster', fontsize=10)
    
    # PC1 vs PC3
    if X_pca.shape[1] > 2:
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 2], 
                                   c=clusters, cmap='viridis', 
                                   alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1].set_xlabel('PC1 (Componente Principal 1)', fontsize=11)
        axes[1].set_ylabel('PC3 (Componente Principal 3)', fontsize=11)
        axes[1].set_title('Clusters no Espaço PCA (PC1 vs PC3)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label('Cluster', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualizar_rankings(ranking_atraso, ranking_prop, ranking_velocidade):
    """
    Visualiza rankings das piores linhas
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Atraso médio
    top_atraso = ranking_atraso.head(12)
    axes[0].barh(range(len(top_atraso)), top_atraso.values, 
                 color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(len(top_atraso)))
    axes[0].set_yticklabels(top_atraso.index)
    axes[0].set_xlabel('Atraso Médio (minutos)', fontsize=10)
    axes[0].set_ylabel('Linha', fontsize=10)
    axes[0].set_title('Top 12 - Maior Atraso Médio', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # % atrasadas
    top_prop = ranking_prop.head(12)
    axes[1].barh(range(len(top_prop)), top_prop.values * 100, 
                 color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(range(len(top_prop)))
    axes[1].set_yticklabels(top_prop.index)
    axes[1].set_xlabel('% Viagens Atrasadas', fontsize=10)
    axes[1].set_ylabel('Linha', fontsize=10)
    axes[1].set_title('Top 12 - Maior % de Atrasos', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    # Velocidade
    top_vel = ranking_velocidade.head(12)
    axes[2].barh(range(len(top_vel)), top_vel.values, 
                 color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[2].set_yticks(range(len(top_vel)))
    axes[2].set_yticklabels(top_vel.index)
    axes[2].set_xlabel('Velocidade Média (km/h)', fontsize=10)
    axes[2].set_ylabel('Linha', fontsize=10)
    axes[2].set_title('Top 12 - Menor Velocidade', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('rankings.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 7. PIPELINE COMPLETO
# ============================================================================

def pipeline_completo(billing_project_id, teste=False):
    """
    Pipeline completo de análise
    
    Args:
        billing_project_id: ID do projeto GCP
        teste: Se True, usa amostra pequena
    """
    print("\n" + "=" * 80)
    print("")
    print(" PROJETO COCADA 2025-2 ".center(80, "="))
    print(" Análise de Atrasos - Linhas do Fundão ".center(80))
    print(" João Victor Borges Nascimento - 121064604 ".center(80))
    print("")
    print("=" * 80)
    
    # 1. Extração
    df = extrair_dados_fundao(billing_project_id, teste=teste)
    if df is None:
        return None
    
    # 2. Pré-processamento
    df = preprocessar_dados(df)
    
    # 3. Agregação
    df_agregado = agregar_por_linha_dia(df)
    
    # 4. Análise Exploratória
    ranking_atraso, ranking_prop, ranking_vel = analise_exploratoria(df_agregado)
    
    # 5. PCA
    X_pca, pca, scaler, X_scaled, features = aplicar_pca(df_agregado, n_components=3)
    
    # 6. Determinar k ótimo
    inertias, silhouettes = determinar_k_otimo(X_scaled, k_range=range(2, 7))
    
    # 7. K-Means (usar k=4 ou ajustar baseado no silhouette)
    clusters, kmeans = aplicar_kmeans(X_scaled, n_clusters=4)
    
    # 8. Análise de Clusters
    df_clusters = analisar_clusters(df_agregado, clusters)
    
    # 9. Visualizações
    print("\n📊 Gerando visualizações...")
    visualizar_pca_clusters(X_pca, clusters, df_agregado)
    visualizar_rankings(ranking_atraso, ranking_prop, ranking_vel)
    
    # 10. Resultados finais
    print("\n" + "=" * 80)
    print(" ANÁLISE CONCLUÍDA ".center(80, "="))
    print("=" * 80)
    print("\n✅ Arquivos gerados:")
    print("   - pca_clusters.png")
    print("   - k_otimo.png")
    print("   - pca_clusters.png")
    print("   - rankings.png")
    
    # Salvar resultados em CSV
    df_agregado['cluster'] = clusters
    df_agregado.to_csv('resultados_agregados.csv', index=False)
    df_clusters.to_csv('caracterizacao_clusters.csv', index=False)
    
    print("   - resultados_agregados.csv")
    print("   - caracterizacao_clusters.csv")
    
    print("\n📋 Resumo dos Resultados:")
    print(f"   Total de viagens analisadas: {len(df):,}")
    print(f"   Linhas com dados completos: {df_agregado['linha'].nunique()}")
    print(f"   Clusters identificados: {len(df_clusters)}")
    print(f"   Período analisado: {DATA_INICIO} a {DATA_FIM}")
    
    return {
        'df_original': df,
        'df_agregado': df_agregado,
        'X_pca': X_pca,
        'clusters': clusters,
        'pca': pca,
        'kmeans': kmeans,
        'rankings': (ranking_atraso, ranking_prop, ranking_vel),
        'df_clusters': df_clusters
    }

# Teste rápido (5000 viagens)
resultados = pipeline_completo("universal-helix-468201-g9", teste=True)

# Análise completa (todos os dados)
# resultados = pipeline_completo("universal-helix-468201-g9", teste=False)