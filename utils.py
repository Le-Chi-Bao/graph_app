import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

import io
from PIL import Image

# THÊM HÀM create_smart_layout
def create_smart_layout(G):
    """Tạo layout thông minh cho đồ thị"""
    try:
        if len(G.nodes()) <= 10:
            return nx.circular_layout(G)
        else:
            return nx.spring_layout(G, seed=42, k=2, iterations=100)
    except:
        return nx.spring_layout(G, seed=42)
    
def draw_graph(G, highlight_nodes=None, highlight_edges=None, title="", directed=False):
    plt.figure(figsize=(10, 8))
    
    # TẠO LAYOUT THÔNG MINH
    pos = create_smart_layout(G)
    
    # Đảm bảo tất cả node có vị trí
    if not pos:
        pos = nx.spring_layout(G, seed=42, k=2, iterations=100)
    
    # Vẽ nodes với kích thước theo bậc
    node_sizes = []
    for node in G.nodes():
        degree = G.degree(node)
        node_sizes.append(300 + degree * 50)  # Node lớn hơn nếu có nhiều kết nối
    
    node_color = ['lightblue'] * len(G.nodes())
    if highlight_nodes:
        node_color = ['red' if node in highlight_nodes else 'lightblue' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Vẽ edges với độ dày theo trọng số
    edge_list = list(G.edges())
    edge_colors = ['gray'] * len(edge_list)
    edge_widths = []
    edge_alphas = []
    
    for i, (u, v) in enumerate(edge_list):
        # Độ dày theo trọng số
        weight = G[u][v].get('weight', 1)
        edge_widths.append(0.5 + weight * 0.3)
        edge_alphas.append(0.7)
        
        if highlight_edges:
            if (u, v) in highlight_edges or (v, u) in highlight_edges:
                edge_colors[i] = 'red'
                edge_widths[-1] = 4
                edge_alphas[-1] = 1.0
    
    if directed:
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_colors, 
                              width=edge_widths, alpha=edge_alphas,
                              arrows=True, arrowstyle='-|>', arrowsize=15,
                              connectionstyle='arc3,rad=0.0')
    else:
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_colors, 
                              width=edge_widths, alpha=edge_alphas,
                              connectionstyle='arc3,rad=0.0')
    
    # Thêm trọng số với màu sắc
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                    font_size=9, font_weight='bold',
                                    label_pos=0.5)  # 0.5 = giữa cạnh
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Tạo padding xung quanh
    plt.tight_layout(pad=2.0)
    
    # Convert to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
    plt.close()
    buf.seek(0)
    
    img = Image.open(buf)
    return np.array(img)

def adjacency_matrix_to_graph(matrix, directed=False):
    """
    Chuyen ma tran ke thanh do thi NetworkX
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
    Chuyen danh sach canh thanh do thi NetworkX
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