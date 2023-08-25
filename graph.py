import pandas as pd
import networkx as nx
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np 
import threading
import os



# Função para carregar os dados de um ano específico dos arquios .txt
def load_data(year):
    # Criar um grafo vazio
    graph = nx.Graph()
    graph.name = f"Graph {year}"
    parties_in_file = set()

    # Carregar os dados dos arquivos .txt
    politicians_file = f"politicians{year}.txt"
    graph_file = f"graph{year}.txt"

    try:
        politicians_data = pd.read_csv(politicians_file, delimiter=';', header=None, names=['nome', 'partido', 'votos'])
        parties_in_file = set(politicians_data['partido'].str.upper())

        graph_data = pd.read_csv(graph_file, delimiter=';', header=None, names=['deputado1', 'deputado2', 'votos_iguais'])
    
    except FileNotFoundError:
        messagebox.showerror("Erro", f"Arquivo '{politicians_file}' não encontrado. Verifique o ano informado.")
        return None
    
    # Adicionar os nós e as arestas ao grafo
    for index, row in politicians_data.iterrows():
        politician_name = row['nome']
        party = row['partido']
        num_votes = row['votos']
        graph.add_node(politician_name, partido=party, votos=num_votes)

    # Adicionar as arestas ao grafo
    for index, row in graph_data.iterrows():
        deputy_1 = row['deputado1']
        deputy_2 = row['deputado2']
        weight = row['votos_iguais']
        
        if deputy_1 in graph.nodes and deputy_2 in graph.nodes and weight > 0:
            graph.add_edge(deputy_1, deputy_2, weight=weight)

    return graph, parties_in_file



# Função para filtrar os dados e normalizar os pesos das arestas
def apply_filters_and_normalize(graph, parties):
    # Criar uma cópia do grafo original
    filtered_normalized_graph = graph.copy()

    # Remover nós que não pertencem aos partidos informados
    if parties:
        nodes_to_remove = [node for node, data in filtered_normalized_graph.nodes(data=True) if data['partido'] not in parties]
        filtered_normalized_graph.remove_nodes_from(nodes_to_remove)

    # Remover arestas com peso 0
    for u, v, data in filtered_normalized_graph.edges(data=True):
        weight = data['weight']
        votes_u = filtered_normalized_graph.nodes[u]['votos']
        votes_v = filtered_normalized_graph.nodes[v]['votos']

        normalized_weight = weight / min(votes_u, votes_v)
        filtered_normalized_graph[u][v]['weight'] = normalized_weight

    return filtered_normalized_graph



# Função para aplicar o threshold e inverter os pesos das arestas
def apply_threshold_and_invert_weights(filtered_normalized_graph, threshold):
    # Criar uma cópia do grafo filtrado e normalizado
    thresholded_inverted_graph = filtered_normalized_graph.copy()

    # Remover arestas com peso menor que o threshold
    edges_to_remove = [(u, v) for u, v, data in thresholded_inverted_graph.edges(data=True) if data['weight'] < threshold]
    thresholded_inverted_graph.remove_edges_from(edges_to_remove)

    # Inverter os pesos das arestas
    for u, v, data in thresholded_inverted_graph.edges(data=True):
        old_weight = data['weight']
        new_weight = 1 - old_weight
        thresholded_inverted_graph[u][v]['weight'] = new_weight

    return thresholded_inverted_graph


# Função para calcular a centralidade de betweenness
def calculate_betweenness_centrality(thresholded_inverted_graph):
    betweenness_graph = nx.betweenness_centrality(thresholded_inverted_graph)
    return betweenness_graph



# Função para plotar o gráfico de centralidade de betweenness
def plot_centrality_graph(betweenness_graph, graph, parties, filename):
    # Ordenar os deputados por centralidade de betweenness
    sorted_data = sorted(betweenness_graph.items(), key=lambda x: x[1], reverse=False)
    deputados = [item[0] for item in sorted_data]
    centralidades = [item[1] for item in sorted_data]
    
    # Plotar o gráfico de barras
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(deputados)), centralidades, color='blue')
    
    # Definir os nomes dos deputados e dos partidos como rótulos dos eixos
    plt.subplots_adjust(bottom=0.3)
    deputados_com_partido = [f"({graph.nodes[deputado]['partido']}) {deputado}" for deputado in deputados]
    plt.xticks(range(len(deputados)), deputados_com_partido, rotation=52, ha="right",fontsize=5)
    
    # Definir título e rótulos dos eixos
    plt.title(f"Medida de Centralidade")
    plt.xlabel("Deputados")
    plt.ylabel("Betweenness")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()



# Função para plotar o heatmap de correlação
def plot_heatmap(filtered_normalized_graph, parties, filename):  
    deputados_por_partido = {}
    
    # Filtrar os nós que pertencem aos partidos informados
    for node, data in filtered_normalized_graph.nodes(data=True):
        partido = data['partido']
        if partido not in deputados_por_partido:
            deputados_por_partido[partido] = []
        deputados_por_partido[partido].append(node)
     
    # Ordenar os partidos e os deputados dentro de cada partido 
    partidos_ordenados = sorted(deputados_por_partido.keys())
    deputados_ordenados = [deputado for partido in partidos_ordenados for deputado in deputados_por_partido[partido]]
    
    # Montar a matriz de correlação  
    correlation_matrix = []

    # Calcular a matriz de correlação entre deputados com base nas arestas
    for deputado_1 in deputados_ordenados:
        row = []
        for deputado_2 in deputados_ordenados:
            if deputado_1 == deputado_2:
                row.append(1.0)  # Correlação máxima consigo mesmo
            else:
                if filtered_normalized_graph.has_edge(deputado_1, deputado_2):
                    weight = filtered_normalized_graph[deputado_1][deputado_2]['weight']
                    row.append(weight)
                else:
                    row.append(0.0) # Correlação mínima entre deputados não conectados
        correlation_matrix.append(row)

    # Converter a matriz em um array numpy
    correlation_array = np.array(correlation_matrix)

    # Plotar o heatmap usando Matplotlib
    plt.figure(figsize=(11, 9))
    plt.imshow(correlation_array, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Correlação')
    
    # Definir os nomes dos deputados e dos partidos como rótulos dos eixos
    deputados_partidos_ordenados = [f"{deputado} ({filtered_normalized_graph.nodes[deputado]['partido']})" for deputado in deputados_ordenados]
    plt.xticks(range(len(deputados_partidos_ordenados)), deputados_partidos_ordenados, rotation=45, fontsize=5, ha="right")
    plt.yticks(range(len(deputados_partidos_ordenados)), deputados_partidos_ordenados, fontsize=5)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.2)
    
    # Definir título e rótulos dos eixos
    plt.title("Heatmap de Correlação entre Deputados")
    
    # Mostrar o heatmap
    plt.tight_layout()
    plt.savefig(filename)  
    plt.show()



# Função para plotar o grafo 
def plot_graph(thresholded_inverted_graph, filename, graph_saved_label=None, graph_canvas=None):
    plt.figure(figsize=(10, 15))
    
    # Calculando medidas de centralidade (exemplo: centralidade de grau)
    node_degrees = dict(thresholded_inverted_graph.degree())
    max_degree = max(node_degrees.values())
    
    # Calculando tamanho dos nós com base na centralidade de grau
    node_sizes = [60 + 200 * (node_degrees[node] / max_degree) for node in thresholded_inverted_graph.nodes()]
    
    # Calculando posição dos nós com base no algoritmo de layout
    k_value = 0.9 / np.sqrt(len(thresholded_inverted_graph.nodes()))
    pos = nx.spring_layout(thresholded_inverted_graph, seed=42, k=k_value) 
    rotation_agle= 130
    rotation_agle_rad = np.radians(rotation_agle)
    pos = {node: (x*np.cos(rotation_agle_rad) - y*np.sin(rotation_agle_rad), x*np.sin(rotation_agle_rad) + y*np.cos(rotation_agle_rad)) 
                   for node, (x, y) in pos.items()}
    
    # Rotacionando os nós para facilitar a visualização
    party_colors = {}
    unique_parties = set(nx.get_node_attributes(thresholded_inverted_graph, 'partido').values())
    
    dark_color_palette = plt.get_cmap('tab10')
    
    party_colors = {}
    for idx, party in enumerate(unique_parties):
        party_colors[party] = dark_color_palette(idx)
    
    # Definindo cores dos nós com base no partido
    node_colors = [party_colors[thresholded_inverted_graph.nodes[node]['partido']] for node in thresholded_inverted_graph.nodes()]
    
    # Desenhando o grafo
    nx.draw(
        thresholded_inverted_graph,
        #rotated_pos,
        pos,
        with_labels=False,
        node_size=node_sizes,
        #node_size=60,
        font_size=5,
        alpha=0.7,
        edge_color='black',
        node_color=node_colors,
        verticalalignment="center",
    )
    
    # Definindo rótulos dos nós
    labels = {node: f"{node} ({thresholded_inverted_graph.nodes[node]['partido']})" for node in thresholded_inverted_graph.nodes()}  # Removendo os colchetes aqui
    for node, label in labels.items():
        x, y = pos[node]
        plt.text(x, y, label, fontsize=8, ha="center", va="center", path_effects=[], alpha=1.0)
    
    # Definindo legenda dos partidos    
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{party}', markerfacecolor=party_colors[party])
                      for party in unique_parties]
    
    # Definindo legenda dos partidos e título do gráfico
    plt.legend(handles=legend_handles, title="Partidos", loc='upper right')
    plt.title("Grafo de Redes Políticas")
    plt.axis('off')    
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    # Atualizar label para mostrar que o gráfico foi salvo
    if graph_saved_label:
        graph_saved_label.config(text=f"Gráfico salvo em: {filename}", fg="green")
        graph_saved_label.update_idletasks()




# Função para executar a análise 
def run_analysis(year, parties, threshold, save_choice, save_directory, processing_label):
    processing_label.config(text="Processando...", fg="blue")
    processing_label.update_idletasks()
    
    try:
        year = int(year)
        if year < 2001 or year > 2023:
            raise ValueError()
    except ValueError:
        messagebox.showerror("Erro", "Ano inválido. Informe um ano entre 2001 e 2023.")
        return
    
    parties = parties.upper().split()
    
    if not threshold:
        messagebox.showerror("Erro", "Threshold não informado.")
        return
    threshold = float(threshold.replace(',', '.'))
    if not 0 <= threshold <= 1:
        messagebox.showerror("Erro", "Threshold inválido. Informe um valor entre 0 e 1.")
        return
    
    graph, parties_in_file = load_data(year)
    if graph is None:
        return
    
    partidos_nao_encontrados = [partido for partido in parties if partido not in parties_in_file]
    if partidos_nao_encontrados:
        messagebox.showerror("Erro", f"Partido(s) não encontrado(s): {', '.join(partidos_nao_encontrados)}")
        return
    
    # Aplicar filtros e normalizar os pesos das arestas
    filtered_normalized_graph = apply_filters_and_normalize(graph, parties)
    thresholded_inverted_graph = apply_threshold_and_invert_weights(filtered_normalized_graph, threshold)
    betweenness_graph = calculate_betweenness_centrality(thresholded_inverted_graph)
    
    
    # Salvar gráficos e dados 
    if save_choice == 's':
        
        # Criar diretório para salvar os gráficos
        centrality_graph_filename = os.path.join(save_directory, "centralidade_deputados.png")
        plot_centrality_graph(betweenness_graph, graph, parties, centrality_graph_filename)
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        heatmap_filename = os.path.join(save_directory, "heatmap_correlacao.png")
        plot_heatmap(filtered_normalized_graph, parties, heatmap_filename)
        
       
        graph_filename = os.path.join(save_directory, "grafo.png")
        graph_saved_label = tk.Label(processing_label.master, text="", fg="green")
        graph_saved_label.pack()
        plot_graph(thresholded_inverted_graph, graph_filename, graph_saved_label)
        
        betweenness_centrality_filename = os.path.join(save_directory, "betweenness_centrality.csv")
       
    
    # Mostrar mensagem de sucesso
    messagebox.showinfo("Sucesso", "Análise concluída e gráficos salvos com sucesso!")
    processing_label.config(text="Processamento concluído", fg="green")



# Função para escolher o diretório para salvar os gráficos
def choose_directory(entry):
    chosen_directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, chosen_directory)


# Função principal para executar a interface gráfica
def main():
    root = tk.Tk()
    root.title("Análise de Redes Políticas")

    year_label = tk.Label(root, text="Informe o ano a considerar (de 2001 a 2023):")
    year_label.pack()

    year_entry = tk.Entry(root)
    year_entry.pack()

    parties_label = tk.Label(root, text="Informe os partidos a analisar, separados por espaço (ex. PT MDB PL):")
    parties_label.pack()

    parties_entry = tk.Entry(root)
    parties_entry.pack()

    threshold_label = tk.Label(root, text="Informe o percentual mínimo de conconrdância (threshold) (de 0.0 a 1.0):")
    threshold_label.pack()

    threshold_entry = tk.Entry(root)
    threshold_entry.pack()

    save_choice_label = tk.Label(root, text="Salvar gráficos?")
    save_choice_label.pack()

    save_choice_var = tk.StringVar(value="s")
    save_yes_radio = tk.Radiobutton(root, text="Sim", variable=save_choice_var, value="s")
    save_yes_radio.pack()

    save_no_radio = tk.Radiobutton(root, text="Não", variable=save_choice_var, value="n")
    save_no_radio.pack()

    directory_label = tk.Label(root, text="Diretório para salvar gráficos:")
    directory_label.pack()

    directory_entry = tk.Entry(root)
    directory_entry.pack()

    choose_directory_button = tk.Button(root, text="Escolher Diretório", command=lambda: choose_directory(directory_entry))
    choose_directory_button.pack()

    graph_saved_label = tk.Label(root, text="", fg="green")
    graph_saved_label.pack()

    run_button = tk.Button(root, text="Executar Análise", command=lambda: run_analysis(year_entry.get(), parties_entry.get(), threshold_entry.get(), save_choice_var.get(), directory_entry.get(), processing_label))
    run_button.pack()

    processing_label = tk.Label(root, text="", fg="black")
    processing_label.pack()
    
    root.mainloop()

# Execução do programa
if __name__ == "__main__":
    main()
