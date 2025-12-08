import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def draw_graph(G, pos=None, highlight_nodes=None, highlight_edges=None, 
               title="", directed=False):
    """
    Vẽ đồ thị và trả về hình ảnh dạng base64
    """
    plt.figure(figsize=(8, 6))
    
    if pos is None:
        if directed:
            pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
    
    # Vẽ tất cả các node và edge
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    nx.draw_networkx_labels(G, pos)
    
    if directed:
        nx.draw_networkx_edges(G, pos, arrowstyle='->', 
                              arrowsize=20, alpha=0.5)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Highlight nodes nếu có
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, 
                              node_color='red', node_size=700)
    
    # Highlight edges nếu có
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, 
                              edge_color='red', width=3)
    
    # Vẽ trọng số nếu có
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(title)
    plt.axis('off')
    
    # Chuyển thành base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_str}"

def adjacency_matrix_to_graph(matrix, directed=False):
    """
    Chuyển ma trận kề thành đồ thị NetworkX
    """
    G = nx.DiGraph() if directed else nx.Graph()
    n = len(matrix)
    
    for i in range(n):
        G.add_node(i)
    
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=matrix[i][j])
    
    return G

def edge_list_to_graph(edges, directed=False):
    """
    Chuyển danh sách cạnh thành đồ thị NetworkX
    """
    G = nx.DiGraph() if directed else nx.Graph()
    
    for edge in edges:
        if len(edge) == 2:
            u, v = edge
            weight = 1
        else:
            u, v, weight = edge
        G.add_edge(u, v, weight=weight)
    
    return G