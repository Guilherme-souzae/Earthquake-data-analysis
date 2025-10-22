import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('earthquakes.csv', sep=',', encoding='utf-8')

# Pergunta 1: Qual é a porcentagem de terremotos ocorridos em zonas costeiras que efetivamente geraram um alerta de tsunami?
def ask1(df):
    # Definir palavras-chave costeiras
    coastal_keywords = ['coast', 'off ', 'sea', 'ocean', 'costa', 'mar', 'ridge']

    # Filtrar eventos costeiros
    df['is_coastal'] = df['place'].astype(str).str.contains('|'.join(coastal_keywords), case=False, na=False)
    df_coastal = df[df['is_coastal'] == True].copy()

    # Contar eventos com e sem tsunami
    count_tsunami = df_coastal[df_coastal['tsunami'] == 1].shape[0]
    count_no_tsunami = df_coastal[df_coastal['tsunami'] == 0].shape[0]

    # Preparar dados para o gráfico
    sizes = [count_tsunami, count_no_tsunami]
    labels = ['Gerou Alerta', 'Não Gerou Alerta']
    colors = ['#d62728', '#1f77b4']  # Vermelho e Azul

    # Criar o gráfico de pizza
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'white'}
            )

    plt.title('Proporção de Alertas de Tsunami em Eventos Costeiros')

    # Mostrar o gráfico
    plt.show()

# Pergunta 2: Quais são os eventos com o maior número de relatos de percepção humana? Esses eventos correlacionam-se mais com a magnitude ou com a proximidade a áreas densamente povoadas?
def ask2(df):
    # Parte A: Top 10 Eventos por 'felt'

    # 1. Obter Top 10
    df_top_felt = df.dropna(subset=['felt']).nlargest(10, 'felt')

    # 2. Plotar Gráfico de Barras
    plt.figure(figsize=(12, 7))
    bar_labels = df_top_felt['title']  # .str[:10] # Encurtados
    plt.bar(bar_labels, df_top_felt['felt'], color='skyblue')

    plt.xlabel('Título do Evento (iniciais)')
    plt.ylabel('Número de Relatos (Felt)')
    plt.title('Top 10 Eventos por Percepção Humana (felt)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

    # --- Parte B: Gráficos - Análise de Correlação ---

    df_corr = df.dropna(subset=['felt', 'magnitude', 'distanceKM'])
    df_corr = df_corr[df_corr['felt'] > 0]

    if len(df_corr) > 2:
        # --- Gráfico de Dispersão: Felt vs. Magnitude ---
        x_mag = df_corr['magnitude']
        y_felt = df_corr['felt']
        m_mag, b_mag = np.polyfit(x_mag, y_felt, 1)  # Calcular linha de tendência

        plt.figure(figsize=(10, 6))
        plt.scatter(x_mag, y_felt, alpha=0.5, label='Eventos')
        plt.plot(x_mag, m_mag * x_mag + b_mag, color='red', linestyle='--', label='Linha de Tendência')

        plt.title('Correlação: Percepção (felt) vs. Magnitude')
        plt.xlabel('Magnitude')
        plt.ylabel('Número de Relatos (Felt)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plt.show()

        # --- Gráfico de Dispersão: Felt vs. Distância ---
        x_dist = df_corr['distanceKM']
        m_dist, b_dist = np.polyfit(x_dist, y_felt, 1)  # Calcular linha de tendência

        plt.figure(figsize=(10, 6))
        plt.scatter(x_dist, y_felt, alpha=0.5, label='Eventos', color='green')
        plt.plot(x_dist, m_dist * x_dist + b_dist, color='red', linestyle='--', label='Linha de Tendência')

        plt.title('Correlação: Percepção (felt) vs. Distância (km)')
        plt.xlabel('Distância (km)')
        plt.ylabel('Número de Relatos (Felt)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plt.show()

# Pergunta 3: Como os níveis de alerta (alert - 'green', 'yellow', 'red') se distribuem geograficamente? Onde se concentram os alertas de maior impacto (vermelho), indicando áreas prioritárias para infraestrutura de resposta rápida?
def ask3(df):
    # --- Parte A: Mapa com distribuição geográfica dos alertas ---
    alert_levels = ['green', 'yellow', 'red']
    df_alerts = df[df['alert'].isin(alert_levels)].copy()

    # 1. Mapeamento de cores
    color_map = {'green': '#2ca02c', 'yellow': '#ffdd00', 'red': '#d62728'}

    # 2. Criar figura com projeção geográfica
    plt.figure(figsize=(15, 8))

    # Adicionar mapa base
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()

    # 3. Plotar pontos por nível de alerta
    for alert_level in alert_levels:
        df_subset = df_alerts[df_alerts['alert'] == alert_level]
        ax.scatter(
            df_subset['longitude'],
            df_subset['latitude'],
            color=color_map[alert_level],
            label=f'Alerta {alert_level.capitalize()}',
            alpha=0.7,
            s=df_subset['magnitude'] * 12,  # tamanho proporcional à magnitude
            transform=ccrs.PlateCarree()
        )

    # 4. Títulos e legenda
    plt.title('Distribuição Geográfica dos Níveis de Alerta de Terremotos', fontsize=14)
    ax.legend(title='Nível de Alerta', loc='lower left')

    plt.show()

    # --- Parte B: Gráfico de barras dos alertas vermelhos ---
    df_red_counts = df[df['alert'] == 'red']['country'].value_counts()

    if not df_red_counts.empty:
        plt.figure(figsize=(12, 7))
        df_red_counts.plot(kind='bar', color='crimson')

        plt.title('Concentração de Alertas Vermelhos por País')
        plt.xlabel('País')
        plt.ylabel('Contagem de Alertas Vermelhos')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum alerta vermelho encontrado nos dados.")

# Pergunta 4: Qual é a correlação estatística entre a magnitude, a profundidade e a intensidade de dano reportada? Terremotos mais rasos tendem a gerar MMI mais alto para a mesma magnitude?
def ask4(df):
    df = df[['magnitude', 'depth', 'mmi']].dropna()

    correlation = df.corr()
    print(correlation)

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlação entre Magnitude, Profundidade e MMI")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="magnitude",
        y="mmi",
        hue="depth",
        palette="viridis",
        alpha=0.7
    )
    plt.title("Relação entre Magnitude, Profundidade e Intensidade (MMI)")
    plt.xlabel("Magnitude")
    plt.ylabel("Intensidade (MMI)")
    plt.legend(title="Profundidade (km)")
    plt.show()

# Pergunta 5: Terremotos mais rasos estão associados a maiores índices de dano?​
def ask5(df):
    df = df[['depth', 'mmi']].dropna()

    corr = df['depth'].corr(df['mmi'])
    print(f"Correlação entre profundidade e MMI: {corr:.2f}")

    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x='depth',
        y='mmi',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'}
    )
    plt.title("Relação entre Profundidade e Intensidade de Dano (MMI)")
    plt.xlabel("Profundidade (km)")
    plt.ylabel("Intensidade (MMI)")
    plt.show()

# Pergunta 6: Qual é a proporção de terremotos por continente?
def ask6(df):
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['continent'])

    continentProportion = df['continent'].value_counts(normalize=True) * 100
    continentProportion = continentProportion.reset_index()
    continentProportion.columns = ['continent', 'percentage']

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=continentProportion,
        x='continent',
        y='percentage',
    )
    plt.title("Proporção de Terremotos por Continente")
    plt.xlabel("Continente")
    plt.ylabel("Proporção (%)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Pergunta 7: Qual é a profundidade média dos sismos em diferentes zonas geográficas de interesse?
def ask7(df):
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    continent_df = df.groupby('continent')
    mean_depth_df = continent_df['depth'].mean()
    mean_depth_sorted_df = mean_depth_df.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    mean_depth_sorted_df.plot(kind='bar')
    plt.title(f'Média da profundidade dos terremodos por continente')
    plt.xlabel('Continente')
    plt.ylabel('Profundidade média (km)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Pergunta 8: Terremotos mais rasos estão associados a maiores índices de dano, exigindo códigos de construção mais rigorosos?
def ask8(df):
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df['mmi'] = pd.to_numeric(df['mmi'], errors='coerce')
    df = df.dropna(subset=['depth', 'mmi'])

    plt.figure(figsize=(10, 6))
    plt.scatter(df['depth'], df['mmi'], alpha=0.25)
    plt.title('Relação entre profundidade e intensidade (MMI)')
    plt.xlabel('Profundidade (km)')
    plt.ylabel('Intensidade MMI')
    plt.grid(True)
    plt.show()

# Pergunta 9: Quais terremotos de maior magnitude foram mais sentidos pela população (felt)?
def ask9(df):
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df['felt'] = pd.to_numeric(df['felt'], errors='coerce')
    df = df.dropna(subset=['magnitude', 'felt'])
    df = df.drop_duplicates(subset=['magnitude', 'felt'])
    df_top = df.sort_values(by=['felt', 'magnitude'], ascending=False).head(5)

    plt.scatter(df_top['magnitude'], df_top['felt'], s=df_top['magnitude'] * df_top['felt'] / 1000)
    plt.title('Relação entre magnitude e valor sentido')
    plt.xlabel('Magnitude')
    plt.ylabel('Valor sentido (felt)')
    plt.grid(True)
    plt.show()

# Pergunta 10: Existe correlação entre a magnitude e a profundidade do terremoto?
def ask10(df):
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df = df.dropna(subset=['magnitude', 'depth'])

    correlacao = df['magnitude'].corr(df['depth'])

    plt.figure(figsize=(10, 6))
    sns.regplot(x='depth', y='magnitude', data=df, scatter_kws={'alpha': 0.25})
    plt.title('Correlação entre profundidade e magnitude dos terremotos')
    plt.xlabel('Profundidade (km)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


# --- Dicionário para associar número → função ---
menu_options = {
    1: ask1,
    2: ask2,
    3: ask3,
    4: ask4,
    5: ask5,
    6: ask6,
    7: ask7,
    8: ask8,
    9: ask9,
    10: ask10
}

# --- Loop do menu ---
while True:
    print("\n=== MENU DE ANÁLISE DE TERREMOTOS ===")
    for i in range(1, 11):
        print(f"{i} - Pergunta {i}")
    print("0 - Sair")

    try:
        opcao = int(input("Escolha uma opção (0-10): "))

        if opcao == 0:
            print("Encerrando o programa. Até mais!")
            break

        elif opcao in menu_options:
            menu_options[opcao](df)  # chama a função correspondente
        else:
            print("❌ Opção inválida. Digite um número de 0 a 10.")
    except ValueError:
        print("⚠️ Entrada inválida. Digite apenas números.")