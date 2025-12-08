"""
Teste de Coordenadas - Cidade Universitária (Fundão)
Visualiza a área de cobertura do bounding box
"""

import basedosdados as bd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

BILLING_PROJECT_ID = "universal-helix-468201-g9"

# Coordenadas do Fundão (do código principal)
FUNDAO_LAT_MIN = -22.875
FUNDAO_LAT_MAX = -22.837
FUNDAO_LON_MIN = -43.245
FUNDAO_LON_MAX = -43.200

# Linhas que atendem o Fundão
LINHAS_FUNDAO = [
    '321', '323', '325', '327', '485', '635', 'SP635', '486', '616', '913', '945'
]

# ============================================================================
# FUNÇÕES DE TESTE
# ============================================================================

def mostrar_coordenadas():
    """
    Mostra as coordenadas configuradas
    """
    print("=" * 80)
    print(" COORDENADAS CONFIGURADAS ".center(80, "="))
    print("=" * 80)
    
    print("\n📍 Bounding Box da Cidade Universitária (Fundão):")
    print(f"   Latitude  MIN: {FUNDAO_LAT_MIN}")
    print(f"   Latitude  MAX: {FUNDAO_LAT_MAX}")
    print(f"   Longitude MIN: {FUNDAO_LON_MIN}")
    print(f"   Longitude MAX: {FUNDAO_LON_MAX}")
    
    # Calcular dimensões aproximadas
    lat_diff = FUNDAO_LAT_MAX - FUNDAO_LAT_MIN
    lon_diff = FUNDAO_LON_MAX - FUNDAO_LON_MIN
    
    # 1 grau de latitude ≈ 111 km
    # 1 grau de longitude ≈ 111 km * cos(latitude)
    lat_km = lat_diff * 111
    lon_km = lon_diff * 111 * np.cos(np.radians((FUNDAO_LAT_MIN + FUNDAO_LAT_MAX) / 2))
    
    print(f"\n📏 Dimensões aproximadas da área:")
    print(f"   Largura  (Lat): {lat_diff:.6f}° ≈ {lat_km:.2f} km")
    print(f"   Altura   (Lon): {lon_diff:.6f}° ≈ {lon_km:.2f} km")
    print(f"   Área aproximada: {lat_km * lon_km:.2f} km²")
    
    # Coordenadas dos cantos
    print(f"\n📌 Cantos do bounding box:")
    print(f"   Canto Superior Esquerdo: ({FUNDAO_LAT_MAX}, {FUNDAO_LON_MIN})")
    print(f"   Canto Superior Direito:  ({FUNDAO_LAT_MAX}, {FUNDAO_LON_MAX})")
    print(f"   Canto Inferior Esquerdo: ({FUNDAO_LAT_MIN}, {FUNDAO_LON_MIN})")
    print(f"   Canto Inferior Direito:  ({FUNDAO_LAT_MIN}, {FUNDAO_LON_MAX})")
    
    # Ponto central
    centro_lat = (FUNDAO_LAT_MIN + FUNDAO_LAT_MAX) / 2
    centro_lon = (FUNDAO_LON_MIN + FUNDAO_LON_MAX) / 2
    print(f"\n🎯 Ponto central: ({centro_lat:.6f}, {centro_lon:.6f})")
    
    # Link do Google Maps
    print(f"\n🗺️  Visualizar no Google Maps:")
    print(f"   https://www.google.com/maps/@{centro_lat},{centro_lon},14z")

def buscar_amostra_gps():
    """
    Busca amostra de dados GPS para visualização
    """
    print("\n" + "=" * 80)
    print(" BUSCANDO AMOSTRA DE DADOS GPS ".center(80, "="))
    print("=" * 80)
    
    linhas_str = "', '".join(LINHAS_FUNDAO)
    
    # Query para buscar amostra DENTRO do Fundão
    query_dentro = f"""
    SELECT 
        CAST(latitude AS FLOAT64) AS latitude,
        CAST(longitude AS FLOAT64) AS longitude,
        servico AS linha
    FROM 
        `datario.transporte_rodoviario_municipal.gps_onibus`
    WHERE 
        servico IN ('{linhas_str}')
        AND data BETWEEN '2025-03-01' AND '2025-03-10'
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
        AND CAST(latitude AS FLOAT64) BETWEEN {FUNDAO_LAT_MIN} AND {FUNDAO_LAT_MAX}
        AND CAST(longitude AS FLOAT64) BETWEEN {FUNDAO_LON_MIN} AND {FUNDAO_LON_MAX}
    LIMIT 1000
    """
    
    # Query para buscar amostra GERAL (região mais ampla)
    lat_buffer = 0.05
    lon_buffer = 0.05
    query_geral = f"""
    SELECT 
        CAST(latitude AS FLOAT64) AS latitude,
        CAST(longitude AS FLOAT64) AS longitude,
        servico AS linha
    FROM 
        `datario.transporte_rodoviario_municipal.gps_onibus`
    WHERE 
        servico IN ('{linhas_str}')
        AND data BETWEEN '2025-03-01' AND '2025-03-10'
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
        AND CAST(latitude AS FLOAT64) BETWEEN {FUNDAO_LAT_MIN - lat_buffer} AND {FUNDAO_LAT_MAX + lat_buffer}
        AND CAST(longitude AS FLOAT64) BETWEEN {FUNDAO_LON_MIN - lon_buffer} AND {FUNDAO_LON_MAX + lon_buffer}
    LIMIT 5000
    """
    
    try:
        print("\n📡 Buscando dados GPS DENTRO do Fundão...")
        df_dentro = bd.read_sql(query_dentro, billing_project_id=BILLING_PROJECT_ID)
        print(f"✅ {len(df_dentro):,} registros encontrados DENTRO do bounding box")
        
        print("\n📡 Buscando dados GPS na REGIÃO AMPLA (contexto)...")
        df_geral = bd.read_sql(query_geral, billing_project_id=BILLING_PROJECT_ID)
        print(f"✅ {len(df_geral):,} registros encontrados na região ampla")
        
        return df_dentro, df_geral
        
    except Exception as e:
        print(f"\n❌ Erro ao buscar dados: {e}")
        return None, None

def visualizar_mapa(df_dentro, df_geral):
    """
    Visualiza os pontos GPS e o bounding box
    """
    print("\n" + "=" * 80)
    print(" GERANDO VISUALIZAÇÃO ".center(80, "="))
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ===== MAPA 1: Região ampla com contexto =====
    ax1 = axes[0]
    
    if df_geral is not None and len(df_geral) > 0:
        # Plotar todos os pontos GPS (contexto)
        ax1.scatter(df_geral['longitude'], df_geral['latitude'], 
                   c='lightblue', s=1, alpha=0.3, label='GPS geral')
    
    if df_dentro is not None and len(df_dentro) > 0:
        # Plotar pontos DENTRO do Fundão
        ax1.scatter(df_dentro['longitude'], df_dentro['latitude'], 
                   c='red', s=5, alpha=0.6, label='GPS no Fundão', zorder=3)
    
    # Desenhar bounding box
    rect = patches.Rectangle(
        (FUNDAO_LON_MIN, FUNDAO_LAT_MIN),
        FUNDAO_LON_MAX - FUNDAO_LON_MIN,
        FUNDAO_LAT_MAX - FUNDAO_LAT_MIN,
        linewidth=3, edgecolor='green', facecolor='none',
        label='Bounding Box Fundão', zorder=4
    )
    ax1.add_patch(rect)
    
    # Marcar ponto central
    centro_lat = (FUNDAO_LAT_MIN + FUNDAO_LAT_MAX) / 2
    centro_lon = (FUNDAO_LON_MIN + FUNDAO_LON_MAX) / 2
    ax1.plot(centro_lon, centro_lat, 'g*', markersize=20, 
             label='Centro', zorder=5)
    
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title('Contexto Regional - Ilha do Fundão', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # ===== MAPA 2: Zoom no Fundão =====
    ax2 = axes[1]
    
    if df_dentro is not None and len(df_dentro) > 0:
        # Plotar apenas pontos dentro do Fundão
        scatter = ax2.scatter(df_dentro['longitude'], df_dentro['latitude'], 
                             c=df_dentro['linha'].astype('category').cat.codes, 
                             cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Adicionar legenda de linhas
        linhas_unicas = df_dentro['linha'].unique()
        if len(linhas_unicas) <= 10:
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                 markersize=8, label=linha)
                      for i, linha in enumerate(linhas_unicas)]
            ax2.legend(handles=handles, title='Linhas', loc='upper right', fontsize=9)
    
    # Desenhar bounding box
    rect2 = patches.Rectangle(
        (FUNDAO_LON_MIN, FUNDAO_LAT_MIN),
        FUNDAO_LON_MAX - FUNDAO_LON_MIN,
        FUNDAO_LAT_MAX - FUNDAO_LAT_MIN,
        linewidth=3, edgecolor='green', facecolor='none', linestyle='--'
    )
    ax2.add_patch(rect2)
    
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title('Zoom - Cidade Universitária (Fundão)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(FUNDAO_LON_MIN - 0.005, FUNDAO_LON_MAX + 0.005)
    ax2.set_ylim(FUNDAO_LAT_MIN - 0.005, FUNDAO_LAT_MAX + 0.005)
    ax2.set_aspect('equal', adjustable='box')
    
    # Adicionar anotações dos limites
    ax2.axhline(y=FUNDAO_LAT_MIN, color='green', linestyle=':', alpha=0.5)
    ax2.axhline(y=FUNDAO_LAT_MAX, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=FUNDAO_LON_MIN, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=FUNDAO_LON_MAX, color='green', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('coordenadas_fundao.png', dpi=300, bbox_inches='tight')
    print("\n✅ Mapa salvo: coordenadas_fundao.png")
    plt.show()

def gerar_relatorio_detalhado(df_dentro):
    """
    Gera relatório detalhado sobre os dados GPS
    """
    print("\n" + "=" * 80)
    print(" RELATÓRIO DETALHADO ".center(80, "="))
    print("=" * 80)
    
    if df_dentro is None or len(df_dentro) == 0:
        print("\n⚠️  Nenhum dado disponível para análise")
        return
    
    print(f"\n📊 Estatísticas dos dados GPS no Fundão:")
    print(f"   Total de registros: {len(df_dentro):,}")
    print(f"   Linhas únicas: {df_dentro['linha'].nunique()}")
    
    print(f"\n📍 Coordenadas encontradas:")
    print(f"   Latitude  - Min: {df_dentro['latitude'].min():.6f}")
    print(f"   Latitude  - Max: {df_dentro['latitude'].max():.6f}")
    print(f"   Longitude - Min: {df_dentro['longitude'].min():.6f}")
    print(f"   Longitude - Max: {df_dentro['longitude'].max():.6f}")
    
    print(f"\n🚌 Distribuição por linha:")
    contagem_linhas = df_dentro['linha'].value_counts()
    for linha, count in contagem_linhas.items():
        pct = (count / len(df_dentro)) * 100
        print(f"   Linha {linha:>6s}: {count:5,} registros ({pct:5.1f}%)")
    
    # Verificar cobertura
    lat_cobertura = ((df_dentro['latitude'].max() - df_dentro['latitude'].min()) / 
                     (FUNDAO_LAT_MAX - FUNDAO_LAT_MIN)) * 100
    lon_cobertura = ((df_dentro['longitude'].max() - df_dentro['longitude'].min()) / 
                     (FUNDAO_LON_MAX - FUNDAO_LON_MIN)) * 100
    
    print(f"\n📐 Cobertura da área:")
    print(f"   Cobertura Latitude:  {lat_cobertura:.1f}%")
    print(f"   Cobertura Longitude: {lon_cobertura:.1f}%")

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def executar_teste_completo():
    """
    Executa todos os testes de coordenadas
    """
    print("\n" + "=" * 80)
    print("")
    print(" TESTE DE COORDENADAS - CIDADE UNIVERSITÁRIA ".center(80, "="))
    print(" Validação do Bounding Box ".center(80))
    print("")
    print("=" * 80)
    
    # 1. Mostrar coordenadas configuradas
    mostrar_coordenadas()
    
    # 2. Buscar amostra de dados GPS
    df_dentro, df_geral = buscar_amostra_gps()
    
    if df_dentro is None:
        print("\n❌ Não foi possível buscar dados. Verifique a conexão.")
        return
    
    # 3. Gerar relatório detalhado
    gerar_relatorio_detalhado(df_dentro)
    
    # 4. Visualizar mapa
    visualizar_mapa(df_dentro, df_geral)
    
    print("\n" + "=" * 80)
    print(" TESTE CONCLUÍDO ".center(80, "="))
    print("=" * 80)
    print("\n✅ Arquivo gerado: coordenadas_fundao.png")
    print("\n💡 Dica: Abra o arquivo PNG para visualizar a área de cobertura")

# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    executar_teste_completo()