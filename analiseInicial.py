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

BILLING_PROJECT_ID = "universal-helix-468201-g9"

# Linhas que atendem o Fundão
LINHAS_FUNDAO = [
    '321', '323', '325', '327', '485', '635', 'SP635', '486', '616', '913', '945'
    ]

# Coordenadas da Cidade Universitária (bounding box aproximado)
FUNDAO_LAT_MIN = -22.870
FUNDAO_LAT_MAX = -22.838
FUNDAO_LON_MIN = -43.250
FUNDAO_LON_MAX = -43.198

# Período de análise
DATA_INICIO = '2025-03-01'
DATA_FIM = '2025-11-30'

# Threshold de atraso
ATRASO_THRESHOLD = 15  # minutos

# 0. FUNÇÃO DE TESTE DE CONEXÃO

def testar_conexao(billing_project_id):
    """
    Testa a conexão com o BigQuery e verifica dados disponíveis
    """
    print("=" * 80)
    print(" TESTE DE CONEXÃO ".center(80, "="))
    print("=" * 80)
    
    try:
        print("\n[TESTE 1] Testando conexão com BigQuery...")
        df_test = bd.read_sql("""
            SELECT COUNT(*) as total
            FROM `datario.transporte_rodoviario_municipal.viagem_onibus`
            LIMIT 1
        """, billing_project_id=billing_project_id)
        print(f"✅ Conexão OK! Total de registros na tabela: {df_test['total'].iloc[0]:,}")
        
        print("\n[TESTE 2] Verificando linhas disponíveis...")
        linhas_str = "', '".join(LINHAS_FUNDAO)
        df_linhas = bd.read_sql(f"""
            SELECT servico, COUNT(*) as total
            FROM `datario.transporte_rodoviario_municipal.viagem_onibus`
            WHERE servico IN ('{linhas_str}')
            AND data BETWEEN '{DATA_INICIO}' AND '{DATA_FIM}'
            GROUP BY servico
            ORDER BY total DESC
            LIMIT 10000
        """, billing_project_id=billing_project_id)
        
        if len(df_linhas) == 0:
            print("⚠️  Nenhuma das linhas especificadas tem dados no período!")
            print(f"   Linhas buscadas: {LINHAS_FUNDAO}")
            return False
        
        print(f"✅ {len(df_linhas)} linhas encontradas:")
        for _, row in df_linhas.iterrows():
            print(f"   Linha {row['servico']:>6s}: {row['total']:>6,} viagens")
        
        print("\n[TESTE 3] Verificando dados GPS...")
        df_gps_test = bd.read_sql(f"""
            SELECT servico, COUNT(*) as total
            FROM `datario.transporte_rodoviario_municipal.gps_onibus`
            WHERE servico IN ('{linhas_str}')
            AND data BETWEEN '{DATA_INICIO}' AND '{DATA_FIM}'
            GROUP BY servico
            LIMIT 10
        """, billing_project_id=billing_project_id)
        
        if len(df_gps_test) == 0:
            print("⚠️  Nenhum dado GPS encontrado para essas linhas!")
            return False
        
        print(f"✅ Dados GPS encontrados para {len(df_gps_test)} linhas")
        
        print("\n[TESTE 4] Verificando GPS dentro do Fundão...")
        df_fundao_test = bd.read_sql(f"""
            SELECT COUNT(*) as total
            FROM `datario.transporte_rodoviario_municipal.gps_onibus`
            WHERE servico IN ('{linhas_str}')
            AND data BETWEEN '{DATA_INICIO}' AND '{DATA_FIM}'
            AND SAFE_CAST(latitude AS FLOAT64) BETWEEN {FUNDAO_LAT_MIN} AND {FUNDAO_LAT_MAX}
            AND SAFE_CAST(longitude AS FLOAT64) BETWEEN {FUNDAO_LON_MIN} AND {FUNDAO_LON_MAX}
            LIMIT 1
        """, billing_project_id=billing_project_id)
        
        total_fundao = df_fundao_test['total'].iloc[0]
        if total_fundao == 0:
            print("⚠️  Nenhum registro GPS dentro das coordenadas do Fundão!")
            print(f"   Coordenadas: Lat [{FUNDAO_LAT_MIN}, {FUNDAO_LAT_MAX}]")
            print(f"                Lon [{FUNDAO_LON_MIN}, {FUNDAO_LON_MAX}]")
            return False
        
        print(f"✅ {total_fundao:,} registros GPS dentro do Fundão!")
        
        print("\n" + "=" * 80)
        print(" TODOS OS TESTES PASSARAM! ".center(80, "="))
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Erro nos testes: {e}")
        print("\n💡 Verifique:")
        print("   1. BILLING_PROJECT_ID está correto")
        print("   2. Credenciais do GCP estão configuradas")
        print("   3. gcloud auth application-default login")
        return False

# 1. EXTRAÇÃO DE DADOS - GPS NOS PONTOS DO FUNDÃO

def extrair_gps_fundao(billing_project_id, teste=False):
    """
    Extrai dados de GPS quando os ônibus estão DENTRO da Cidade Universitária
    
    IMPORTANTE: Usa tabela de GPS para pegar localizações exatas
    """
    print("=" * 80)
    print(" EXTRAÇÃO DE DADOS GPS - CIDADE UNIVERSITÁRIA ".center(80, "="))
    print("=" * 80)
    
    linhas_str = "', '".join(LINHAS_FUNDAO)
    limite = "LIMIT 10000" if teste else ""
    
    # Query para GPS dentro do Fundão
    query = f"""
    SELECT 
        data,
        servico AS linha,
        timestamp_gps,
        SAFE_CAST(latitude AS FLOAT64) AS latitude,
        SAFE_CAST(longitude AS FLOAT64) AS longitude,
        velocidade_instantanea AS velocidade,
        id_veiculo
    FROM 
        `datario.transporte_rodoviario_municipal.gps_onibus`

    WHERE 
        servico IN ('{linhas_str}')
        AND data BETWEEN '{DATA_INICIO}' AND '{DATA_FIM}'
        AND EXTRACT(DAYOFWEEK FROM data) NOT IN (1, 7)
        AND timestamp_gps IS NOT NULL
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
        -- Filtrar apenas coordenadas dentro do Fundão
        AND SAFE_CAST(latitude AS FLOAT64) BETWEEN {FUNDAO_LAT_MIN} AND {FUNDAO_LAT_MAX}
        AND SAFE_CAST(longitude AS FLOAT64) BETWEEN {FUNDAO_LON_MIN} AND {FUNDAO_LON_MAX}
    ORDER BY data, servico, timestamp_gps
    {limite}
    """
    
    print(f"\n📊 Consultando dados GPS dentro do Fundão...")
    print(f"   Área: Lat [{FUNDAO_LAT_MIN}, {FUNDAO_LAT_MAX}]")
    print(f"         Lon [{FUNDAO_LON_MIN}, {FUNDAO_LON_MAX}]")
    print(f"   Período: {DATA_INICIO} a {DATA_FIM}")
    print(f"   Modo: {'TESTE' if teste else 'COMPLETO'}")
    
    try:
        df_gps = bd.read_sql(query, billing_project_id=billing_project_id)
        
        # Validar dados retornados
        if len(df_gps) == 0:
            print(f"\n⚠️  Nenhum dado encontrado!")
            print(f"   Possíveis causas:")
            print(f"   1. Nenhuma linha da lista passa pelo Fundão")
            print(f"   2. Coordenadas incorretas")
            print(f"   3. Período sem dados")
            print(f"\n   Linhas buscadas: {LINHAS_FUNDAO}")
            return None
        
        # Converter tipos se necessário
        if 'latitude' in df_gps.columns:
            df_gps['latitude'] = pd.to_numeric(df_gps['latitude'], errors='coerce')
        if 'longitude' in df_gps.columns:
            df_gps['longitude'] = pd.to_numeric(df_gps['longitude'], errors='coerce')
        
        # Remover linhas com coordenadas inválidas
        df_gps = df_gps.dropna(subset=['latitude', 'longitude'])
        
        print(f"\n✅ Dados GPS carregados!")
        print(f"   Registros GPS no Fundão: {len(df_gps):,}")
        print(f"   Linhas únicas: {df_gps['linha'].nunique()}")
        print(f"   Veículos únicos: {df_gps['id_veiculo'].nunique()}")
        print(f"   Período: {df_gps['data'].min()} a {df_gps['data'].max()}")
        
        return df_gps
        
    except Exception as e:
        print(f"\n❌ Erro ao executar query: {e}")
        print(f"\n💡 Dicas:")
        print(f"   1. Verifique se as linhas existem na tabela gps_onibus")
        print(f"   2. Teste com uma query mais simples primeiro")
        print(f"   3. Verifique o período de dados disponíveis")
        return None

def extrair_viagens_completas(billing_project_id, teste=False):
    """
    Extrai dados completos de viagens para ter contexto
    """
    print("\n" + "=" * 80)
    print(" EXTRAÇÃO DE VIAGENS COMPLETAS (CONTEXTO) ".center(80, "="))
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
        id_viagem,
        id_veiculo,
        sentido
    FROM 
        `datario.transporte_rodoviario_municipal.viagem_onibus`
    WHERE 
        servico IN ('{linhas_str}')
        AND data BETWEEN '{DATA_INICIO}' AND '{DATA_FIM}'
        AND EXTRACT(DAYOFWEEK FROM data) NOT IN (1, 7)
        AND datetime_partida IS NOT NULL
        AND datetime_chegada IS NOT NULL
    ORDER BY data, servico, datetime_partida
    {limite}
    """
    
    print(f"\n📊 Carregando viagens completas...")
    
    try:
        df_viagens = bd.read_sql(query, billing_project_id=billing_project_id)
        print(f"✅ {len(df_viagens):,} viagens carregadas")
        return df_viagens
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

# 2. PRÉ-PROCESSAMENTO - FOCO NO FUNDÃO

def calcular_tempo_no_fundao(df_gps):
    print("\n" + "=" * 80)
    print(" ANÁLISE DE TEMPO NO FUNDÃO ".center(80, "="))
    print("=" * 80)

    df = df_gps.copy()

    df['timestamp_gps'] = pd.to_datetime(df['timestamp_gps'])

    # Ordenação
    df = df.sort_values(['linha', 'id_veiculo', 'data', 'timestamp_gps'])

    # diff() em minutos
    df['tempo_desde_anterior'] = (
        df.groupby(['linha','id_veiculo','data'])['timestamp_gps']
        .diff()
        .dt.total_seconds() / 60
    )

    # limpar NaN e valores inválidos
    df['tempo_desde_anterior'] = df['tempo_desde_anterior'].fillna(0)
    df['tempo_desde_anterior'] = df['tempo_desde_anterior'].clip(lower=0, upper=10)

    # hora correta → início do intervalo
    df['hora'] = (df['timestamp_gps'] - pd.to_timedelta(df['tempo_desde_anterior'], unit='m')).dt.hour

    tempo_fundao = df.groupby(['linha','data','hora']).agg({
        'tempo_desde_anterior': 'sum',
        'timestamp_gps': 'count',
        'velocidade': 'mean',
        'id_veiculo': 'nunique'
    }).reset_index()

    tempo_fundao.columns = [
        'linha','data','hora',
        'tempo_total_min','num_pings',
        'velocidade_media','num_veiculos'
    ]

    # períodos
    tempo_fundao['periodo'] = tempo_fundao['hora'].apply(
        lambda h: 'pico_manha' if 6 <= h < 9 else ('pico_tarde' if 17 <= h < 20 else 'fora_pico')
    )

    print(f"\n✅ Tempo no Fundão calculado")
    print(f"   Registros: {len(tempo_fundao):,}")
    print(f"   Tempo médio no Fundão por passagem: {tempo_fundao['tempo_total_min'].mean():.1f} min")

    return tempo_fundao

def calcular_atrasos_com_contexto(df_viagens, tempo_fundao):
    """
    Calcula atrasos usando Timestamps para garantir dados válidos
    """
    print("\n" + "=" * 80)
    print(" CÁLCULO DE ATRASOS (METODOLOGIA: TIMESTAMP REAL) ".center(80, "="))
    print("=" * 80)
    
    df = df_viagens.copy()
    
    # 1. Converter timestamps para datetime garantido
    df['datetime_partida'] = pd.to_datetime(df['datetime_partida'])
    df['datetime_chegada'] = pd.to_datetime(df['datetime_chegada'])
    
    # 2. CALCULAR TEMPO MANUALMENTE (Ignorando a coluna 'tempo_viagem' que pode estar vazia)
    # Resultado em minutos
    df['tempo_viagem_min'] = (df['datetime_chegada'] - df['datetime_partida']).dt.total_seconds() / 60.0
    
    # Debug: Mostrar quantos dados temos antes de filtrar
    print(f"   Registros antes do filtro: {len(df)}")
    print(f"   Tempo médio calculado: {df['tempo_viagem_min'].mean():.1f} min")
    
    # 3. Limpeza (remover erros de GPS: viagens < 5 min ou > 4 horas)
    df = df[(df['tempo_viagem_min'] > 5) & (df['tempo_viagem_min'] < 240)]
    print(f"   Registros após filtro de consistência (>5min): {len(df)}")

    # 4. Calcular o "Tempo Ideal" (Baseline) - Fluxo Livre (10% mais rápidas)
    cols_agrupamento = ['linha']
    if df['sentido'].nunique() > 1:
        cols_agrupamento.append('sentido')
    
    baseline = df.groupby(cols_agrupamento)['tempo_viagem_min'].quantile(0.10).reset_index()
    baseline.rename(columns={'tempo_viagem_min': 'tempo_ideal_sem_transito'}, inplace=True)
    
    # Junta o baseline na tabela principal
    df = df.merge(baseline, on=cols_agrupamento, how='left')
    
    # 5. Cálculo do Atraso Real
    df['atraso_min'] = df['tempo_viagem_min'] - df['tempo_ideal_sem_transito']
    df['atrasada'] = df['atraso_min'] > ATRASO_THRESHOLD

    # Métricas auxiliares para o PCA
    # Usa distância planejada se existir, senão assume velocidade média baseada no tempo
    if 'distancia_planejada' in df.columns:
        df['velocidade_kmh'] = np.where(
            df['tempo_viagem_min'] > 0,
            (df['distancia_planejada'] / 1000) / (df['tempo_viagem_min'] / 60),
            0
        )
    else:
        df['velocidade_kmh'] = 0 # Fallback
    
    # Variabilidade
    df['tempo_iqr'] = df.groupby(cols_agrupamento)['tempo_viagem_min'].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))

    print(f"\n✅ Atrasos recalculados com sucesso")
    print(f"   Atraso médio geral: {df['atraso_min'].mean():+.1f} min")
    
    return df

def agregar_por_linha_dia(df_viagens, tempo_fundao):
    """
    Agrega incluindo métricas do Fundão (FILTROS RELAXADOS PARA TESTE)
    """
    print("\n" + "=" * 80)
    print(" AGREGAÇÃO LINHA × DIA (COM MÉTRICAS DO FUNDÃO) ".center(80, "="))
    print("=" * 80)
    
    # Agregar tempo no Fundão por linha×dia
    fundao_agg = tempo_fundao.groupby(['linha', 'data']).agg({
        'tempo_total_min': 'mean',
        'velocidade_media': 'mean',
        'num_pings': 'sum'
    }).reset_index()
    
    fundao_agg.columns = ['linha', 'data', 'tempo_fundao_total', 
                          'velocidade_fundao', 'pings_fundao']
    
    # Agregar viagens
    viagens_agg = df_viagens.groupby(['linha', 'data']).agg({
        'atraso_min': ['mean', 'std', 'max'],
        'atrasada': 'mean',
        'velocidade_kmh': ['mean', 'std'],
        'tempo_viagem_min': ['mean', 'std'],
        'tempo_iqr': 'mean',
        'perc_conformidade_shape': 'mean',
        'id_viagem': 'count'
    }).reset_index()
    
    viagens_agg.columns = [
        'linha', 'data',
        'atraso_medio', 'atraso_std', 'atraso_max',
        'prop_atrasadas',
        'velocidade_media', 'velocidade_std',
        'tempo_viagem_medio', 'tempo_viagem_std',
        'variabilidade_iqr',
        'conformidade_shape',
        'num_viagens'
    ]
    
    # Merge com dados do Fundão
    agregado = viagens_agg.merge(fundao_agg, on=['linha', 'data'], how='left')
    
    # Preencher NaN
    agregado['tempo_fundao_total'] = agregado['tempo_fundao_total'].fillna(0)
    agregado['velocidade_fundao'] = agregado['velocidade_fundao'].fillna(0)
    agregado['pings_fundao'] = agregado['pings_fundao'].fillna(0)
    
    # Features temporais
    agregado['dia_semana'] = pd.to_datetime(agregado['data']).dt.dayofweek
    agregado['mes'] = pd.to_datetime(agregado['data']).dt.month
    
    # --- ALTERAÇÃO AQUI: Baixei de 3 para 1 para garantir que o teste funcione ---
    agregado = agregado[agregado['num_viagens'] >= 1] 
    # -----------------------------------------------------------------------------
    
    print(f"\n✅ Agregação concluída")
    print(f"   Registros linha×dia: {len(agregado):,}")
    print(f"   Tempo médio no Fundão por dia: {agregado['tempo_fundao_total'].mean():.1f} min")
    
    return agregado

# 3. ANÁLISE EXPLORATÓRIA

def analise_exploratoria(df_agregado):
    print("\n" + "=" * 80)
    print(" ANÁLISE EXPLORATÓRIA ".center(80, "="))
    print("=" * 80)

    df = df_agregado.copy()
    df['tempo_fundao_total'] = df['tempo_fundao_total'].replace(0, np.nan)
    df['velocidade_fundao'] = df['velocidade_fundao'].replace(0, np.nan)
    linhas_min = df['linha'].value_counts()
    linhas_validas = linhas_min[linhas_min >= 1].index
    df = df[df['linha'].isin(linhas_validas)]
    if len(df) == 0:
        print("⚠️  Dados insuficientes para gerar rankings no modo de teste.")
        return df, df, df, df

    # Rankings
    ranking_atraso = df.groupby('linha')['atraso_medio'].mean().sort_values(ascending=False)
    ranking_prop = df.groupby('linha')['prop_atrasadas'].mean().sort_values(ascending=False)
    ranking_tempo_fundao = df.groupby('linha')['tempo_fundao_total'].mean().sort_values(ascending=False)
    ranking_vel_fundao = df[df['velocidade_fundao'].notna()].groupby('linha')['velocidade_fundao'].mean().sort_values()

    print("\n🔴 TOP 10 LINHAS - MAIOR ATRASO MÉDIO")
    print("-" * 50)
    for i, (linha, atraso) in enumerate(ranking_atraso.head(10).items(), 1):
        print(f"  {i:2d}. Linha {linha:>4s}: {atraso:+6.1f} min")

    print("\n🟡 TOP 10 LINHAS - MAIOR % VIAGENS ATRASADAS")
    print("-" * 50)
    for i, (linha, prop) in enumerate(ranking_prop.head(10).items(), 1):
        print(f"  {i:2d}. Linha {linha:>4s}: {prop*100:5.1f}%")

    print("\n⏱️  TOP 10 LINHAS - MAIS TEMPO NO FUNDÃO")
    print("-" * 50)
    for i, (linha, tempo) in enumerate(ranking_tempo_fundao.head(10).items(), 1):
        print(f"  {i:2d}. Linha {linha:>4s}: {tempo:6.1f} min/dia")

    print("\n🐌 TOP 10 LINHAS - MENOR VELOCIDADE NO FUNDÃO")
    print("-" * 50)
    if ranking_vel_fundao.empty:
        print("⚠ Nenhuma linha com velocidade válida no Fundão!")
    else:
        for i, (linha, vel) in enumerate(ranking_vel_fundao.head(10).items(), 1):
            print(f"  {i:2d}. Linha {linha:>4s}: {vel:5.1f} km/h")

    print("\n📊 ESTATÍSTICAS GERAIS")
    print("-" * 50)
    print(f"  Atraso médio geral: {df['atraso_medio'].mean():+.1f} min")
    print(f"  Tempo médio no Fundão: {df['tempo_fundao_total'].mean():.1f} min/dia")
    print(f"  Velocidade média no Fundão: {df['velocidade_fundao'].mean():.1f} km/h")

    return ranking_atraso, ranking_prop, ranking_tempo_fundao, ranking_vel_fundao

# 4. PCA

def aplicar_pca(df_filtrado, n_components=3):
    print("\n" + "=" * 80)
    print(" PCA - ANÁLISE DE COMPONENTES PRINCIPAIS ".center(80, "="))
    print("=" * 80)
    
    features = [
        'atraso_medio', 'atraso_std', 'atraso_max',
        'prop_atrasadas',
        'velocidade_media', 'velocidade_std',
        'tempo_viagem_medio', 'tempo_viagem_std',
        'variabilidade_iqr',
        'conformidade_shape',
        'tempo_fundao_total', 
        'velocidade_fundao' 
    ]
    
    # Preenchimento de nulos
    X = df_filtrado[features].fillna(0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    n_samples, n_features = X.shape    
    n_components_possivel = min(n_components, n_samples, n_features)
    
    if n_components_possivel < 2:
        print(f"\n⚠️  Dados insuficientes para PCA! (Amostras: {n_samples}, Features: {n_features})")
        print("   Retornando dados originais sem redução.")
        # Retorna estruturas vazias ou dummy para não quebrar o resto do pipeline
        return X.values, None, StandardScaler(), X.values, features

    if n_components_possivel < n_components:
        print(f"\n⚠️  Ajustando n_components de {n_components} para {n_components_possivel} devido à baixa quantidade de dados.")
        n_components = n_components_possivel

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print("\n📈 VARIÂNCIA EXPLICADA")
    print("-" * 50)
    cumsum = 0
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        cumsum += var
        print(f"   PC{i}: {var*100:5.1f}% (acumulado: {cumsum*100:5.1f}%)")
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i}' for i in range(1, n_components+1)],
        index=features
    )
    
    print("\n🔍 PRINCIPAIS FATORES POR COMPONENTE")
    for i in range(n_components):
        print(f"\n   PC{i+1}:")
        top_features = loadings[f'PC{i+1}'].abs().nlargest(3)
        for feat, _ in top_features.items():
            val = loadings.loc[feat, f'PC{i+1}']
            sinal = "+" if val > 0 else ""
            print(f"    {feat:30s}: {sinal}{val:.3f}")
    
    return X_pca, pca, scaler, X_scaled, features

# 5-6. K-MEANS E VISUALIZAÇÕES (MANTÉM O CÓDIGO ORIGINAL)

def determinar_k_otimo(X, k_range=range(2, 8)):
    """Método do cotovelo + silhouette"""
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
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inércia', fontsize=11)
    axes[0].set_title('Método do Cotovelo', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Coeficiente de Silhueta', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_otimo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    k_sugerido = k_range[np.argmax(silhouettes)]
    print(f"\n💡 K sugerido: {k_sugerido}")
    
    return inertias, silhouettes

def aplicar_kmeans(X, n_clusters=4):
    """Clusterização K-Means"""
    print(f"\n⚙️  Aplicando K-Means com k={n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    sil_score = silhouette_score(X, clusters)
    
    print(f"✅ Clustering concluído")
    print(f"   Silhouette Score: {sil_score:.3f}")
    
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"\n   Distribuição:")
    for cluster_id, count in zip(unique, counts):
        print(f"     Cluster {cluster_id}: {count:5d} registros ({count/len(clusters)*100:4.1f}%)")
    
    return clusters, kmeans

def analisar_clusters(df_filtrado, clusters):
    """Caracteriza clusters"""
    print("\n" + "=" * 80)
    print(" CARACTERIZAÇÃO DOS CLUSTERS ".center(80, "="))
    print("=" * 80)
    
    df_filtrado = df_filtrado.copy()
    df_filtrado['cluster'] = clusters
    
    resultados = []
    
    for cid in sorted(df_filtrado['cluster'].unique()):
        print(f"\n{'='*80}")
        print(f" CLUSTER {cid} ".center(80, "="))
        print('='*80)
        
        cdf = df_filtrado[df_filtrado['cluster'] == cid]
        
        print(f"\n📊 Tamanho: {len(cdf):,} registros ({len(cdf)/len(df_filtrado)*100:.1f}%)")
        
        linhas = sorted(cdf['linha'].unique())
        print(f"\n🚌 Linhas ({len(linhas)}): {', '.join(linhas)}")
        
        stats = {
            'atraso_medio': cdf['atraso_medio'].mean(),
            'prop_atrasadas': cdf['prop_atrasadas'].mean(),
            'velocidade_media': cdf['velocidade_media'].mean(),
            'tempo_fundao': cdf['tempo_fundao_total'].mean(),
            'velocidade_fundao': cdf[cdf['velocidade_fundao'] > 0]['velocidade_fundao'].mean()
        }
        
        print(f"\n📈 Características:")
        print(f"   Atraso médio:            {stats['atraso_medio']:+6.1f} min")
        print(f"   % atrasadas:             {stats['prop_atrasadas']*100:6.1f}%")
        print(f"   Velocidade geral:        {stats['velocidade_media']:6.1f} km/h")
        print(f"   Tempo no Fundão/dia:     {stats['tempo_fundao']:6.1f} min")
        print(f"   Velocidade no Fundão:    {stats['velocidade_fundao']:6.1f} km/h")
        
        # Perfil
        if stats['atraso_medio'] > 10 and stats['prop_atrasadas'] > 0.5:
            perfil = "🔴 CRONICAMENTE ATRASADO"
        elif pd.notna(stats['velocidade_fundao']) and stats['velocidade_fundao'] < 10:
            perfil = "🐌 LENTO NO FUNDÃO - Congestionamento frequente"
        elif stats['atraso_medio'] > 5:
            perfil = "🟡 MODERADAMENTE IRREGULAR"
        elif stats['atraso_medio'] < 0:
            perfil = "🟢 PONTUAL E RÁPIDO"
        else:
            perfil = "🟢 REGULAR"
        
        print(f"\n🏷️  Perfil: {perfil}")
        
        resultados.append({
            'cluster': cid,
            'perfil': perfil,
            'linhas': linhas,
            **stats
        })
    
    return pd.DataFrame(resultados)

def visualizar_pca_clusters(X_pca, clusters, df_filtrado):
    """Visualiza clusters no PCA"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=clusters, cmap='viridis', 
                               alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel('PC1', fontsize=11)
    axes[0].set_ylabel('PC2', fontsize=11)
    axes[0].set_title('Clusters (PC1 vs PC2)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    if X_pca.shape[1] > 2:
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 2], 
                                   c=clusters, cmap='viridis', 
                                   alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1].set_xlabel('PC1', fontsize=11)
        axes[1].set_ylabel('PC3', fontsize=11)
        axes[1].set_title('Clusters (PC1 vs PC3)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualizar_rankings(ranking_atraso, ranking_prop, ranking_tempo_fundao):
    """Visualiza rankings incluindo métricas do Fundão"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Atraso médio
    top_atraso = ranking_atraso.head(12)
    axes[0, 0].barh(range(len(top_atraso)), top_atraso.values, 
                    color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_yticks(range(len(top_atraso)))
    axes[0, 0].set_yticklabels(top_atraso.index)
    axes[0, 0].set_xlabel('Atraso Médio (minutos)', fontsize=10)
    axes[0, 0].set_ylabel('Linha', fontsize=10)
    axes[0, 0].set_title('Top 12 - Maior Atraso Médio', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].invert_yaxis()
    
    # % atrasadas
    top_prop = ranking_prop.head(12)
    axes[0, 1].barh(range(len(top_prop)), top_prop.values * 100, 
                    color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_yticks(range(len(top_prop)))
    axes[0, 1].set_yticklabels(top_prop.index)
    axes[0, 1].set_xlabel('% Viagens Atrasadas', fontsize=10)
    axes[0, 1].set_ylabel('Linha', fontsize=10)
    axes[0, 1].set_title('Top 12 - Maior % de Atrasos', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    axes[0, 1].invert_yaxis()
    
    # Tempo no Fundão
    top_tempo = ranking_tempo_fundao.head(12)
    axes[1, 0].barh(range(len(top_tempo)), top_tempo.values, 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_yticks(range(len(top_tempo)))
    axes[1, 0].set_yticklabels(top_tempo.index)
    axes[1, 0].set_xlabel('Tempo no Fundão (min/dia)', fontsize=10)
    axes[1, 0].set_ylabel('Linha', fontsize=10)
    axes[1, 0].set_title('Top 12 - Mais Tempo no Fundão', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # Legenda resumo
    axes[1, 1].axis('off')
    resumo_texto = f"""
    RESUMO DA ANÁLISE
    
    Foco: Atrasos nas linhas que passam pela
          Cidade Universitária (Fundão)
    
    Período: {DATA_INICIO} a {DATA_FIM}
    
    Métricas principais:
    • Atraso médio por viagem
    • % de viagens atrasadas (>15min)
    • Tempo gasto no Fundão
    • Velocidade dentro do Fundão
    
    Técnicas aplicadas:
    • PCA (redução de dimensionalidade)
    • K-Means (clusterização)
    
    Arquivo: rankings.png
    """
    axes[1, 1].text(0.1, 0.5, resumo_texto, fontsize=11, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('rankings.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. PIPELINE COMPLETO

def pipeline_completo(billing_project_id, teste=False, skip_test=False):
    """
    Pipeline completo focado em atrasos na Cidade Universitária
    
    Args:
        billing_project_id: ID do projeto GCP
        teste: Se True, usa amostra pequena
        skip_test: Se True, pula teste de conexão (não recomendado)
    """
    print("\n" + "=" * 80)
    print("")
    print(" PROJETO COCADA 2025-2 ".center(80, "="))
    print(" Análise de Atrasos - Cidade Universitária ".center(80))
    print(" João Victor Borges Nascimento - 121064604 ".center(80))
    print("")
    print("=" * 80)
    
    # Teste de conexão
    if not skip_test:
        print("\n[PASSO 0/10] Testando conexão e validando dados...")
        if not testar_conexao(billing_project_id):
            print("\n❌ Testes falharam. Corrija os problemas antes de continuar.")
            return None
        input("\nPressione ENTER para continuar com a análise completa...")
    
    # 1. Extração GPS no Fundão
    print("\n[PASSO 1/10] Extraindo dados GPS do Fundão...")
    df_gps = extrair_gps_fundao(billing_project_id, teste=teste)
    if df_gps is None or len(df_gps) == 0:
        print("❌ Não há dados GPS suficientes. Verifique as coordenadas ou período.")
        return None
    
    # 2. Extração de viagens completas
    print("\n[PASSO 2/10] Extraindo dados completos de viagens...")
    df_viagens = extrair_viagens_completas(billing_project_id, teste=teste)
    if df_viagens is None:
        return None
    
    # 3. Calcular tempo no Fundão
    print("\n[PASSO 3/10] Calculando tempo gasto no Fundão...")
    tempo_fundao = calcular_tempo_no_fundao(df_gps)
    
    # 4. Calcular atrasos
    print("\n[PASSO 4/10] Calculando atrasos das viagens...")
    df_viagens = calcular_atrasos_com_contexto(df_viagens, tempo_fundao)
    
    # 5. Agregar por linha × dia
    print("\n[PASSO 5/10] Agregando dados por linha × dia...")
    df_filtrado = agregar_por_linha_dia(df_viagens, tempo_fundao)
    
    # 6. Análise Exploratória
    print("\n[PASSO 6/10] Análise exploratória...")
    ranking_atraso, ranking_prop, ranking_tempo, ranking_vel = analise_exploratoria(df_filtrado)
    
    # 7. PCA
    print("\n[PASSO 7/10] Aplicando PCA...")
    X_pca, pca, scaler, X_scaled, features = aplicar_pca(df_filtrado, n_components=3)
    
    # 8. Determinar k ótimo
    print("\n[PASSO 8/10] Determinando k ótimo...")
    inertias, silhouettes = determinar_k_otimo(X_scaled, k_range=range(2, 7))
    
    # 9. K-Means
    print("\n[PASSO 9/10] Aplicando K-Means...")
    clusters, kmeans = aplicar_kmeans(X_scaled, n_clusters=4)
    df_clusters = analisar_clusters(df_filtrado, clusters)
    
    # 10. Visualizações
    print("\n[PASSO 10/10] Gerando visualizações...")
    visualizar_pca_clusters(X_pca, clusters, df_filtrado)
    visualizar_rankings(ranking_atraso, ranking_prop, ranking_tempo)
    
    # Salvar resultados
    print("\n" + "=" * 80)
    print(" SALVANDO RESULTADOS ".center(80, "="))
    print("=" * 80)
    
    df_filtrado['cluster'] = clusters
    df_filtrado.to_csv('resultados_agregados.csv', index=False)
    df_clusters.to_csv('caracterizacao_clusters.csv', index=False)
    tempo_fundao.to_csv('tempo_fundao_detalhado.csv', index=False)
    
    print("\n✅ Arquivos gerados:")
    print("   - k_otimo.png")
    print("   - pca_clusters.png")
    print("   - rankings.png")
    print("   - resultados_agregados.csv")
    print("   - caracterizacao_clusters.csv")
    print("   - tempo_fundao_detalhado.csv")
    
    print("\n📋 Resumo Final:")
    print(f"   Total de registros GPS no Fundão: {len(df_gps):,}")
    print(f"   Total de viagens analisadas: {len(df_viagens):,}")
    print(f"   Linhas com dados completos: {df_filtrado['linha'].nunique()}")
    print(f"   Clusters identificados: {len(df_clusters)}")
    print(f"   Período analisado: {DATA_INICIO} a {DATA_FIM}")
    print(f"   Tempo médio no Fundão/dia: {df_filtrado['tempo_fundao_total'].mean():.1f} min")
    
    print("\n" + "=" * 80)
    print(" ANÁLISE CONCLUÍDA COM SUCESSO! ".center(80, "="))
    print("=" * 80)
    
    return {
        'df_gps': df_gps,
        'df_viagens': df_viagens,
        'tempo_fundao': tempo_fundao,
        'df_filtrado': df_filtrado,
        'X_pca': X_pca,
        'clusters': clusters,
        'pca': pca,
        'kmeans': kmeans,
        'rankings': (ranking_atraso, ranking_prop, ranking_tempo, ranking_vel),
        'df_clusters': df_clusters
    }

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" INSTRUÇÕES DE USO ".center(80))
    print("=" * 80)
    print("\n1. Instale as dependências:")
    print("   pip install basedosdados pandas numpy scikit-learn matplotlib seaborn")
    print("\n2. Configure suas credenciais GCP:")
    print("   gcloud auth application-default login")
    print("\n3. Execute:")
    print("   # Teste com amostra (10k GPS + 5k viagens)")
    print("   resultados = pipeline_completo(BILLING_PROJECT_ID, teste=True)")
    print("\n   # Análise completa (pode demorar vários minutos)")
    print("   resultados = pipeline_completo(BILLING_PROJECT_ID, teste=False)")
    print("\n" + "=" * 80)
    
    resultados = pipeline_completo(BILLING_PROJECT_ID, teste=False)