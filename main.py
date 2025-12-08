# main.py - FIX LỖI PATH TOO LONG
import gradio as gr
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import io
import tempfile
import os
from pathlib import Path

# ==================== GLOBAL STATE ====================
current_graph = nx.Graph()
is_directed = False

# ==================== UTILITY FUNCTIONS ====================
def safe_int_convert(val):
    """Chuyển đổi an toàn sang int"""
    try:
        return int(float(val))
    except:
        return 0

def draw_and_save_graph(G, directed, highlight_path=None, title=""):
    """Vẽ đồ thị và lưu ra file TEMP - FIX PATH TOO LONG"""
    if not G.nodes():
        return None
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Cấu hình cơ bản
    node_color = ['lightblue'] * len(G.nodes())
    edge_color = ['gray'] * len(G.edges())
    
    # Highlight path
    if highlight_path:
        path_edges = [(highlight_path[i], highlight_path[i+1]) 
                     for i in range(len(highlight_path)-1)]
        node_color = ['red' if node in highlight_path else 'lightblue' 
                     for node in G.nodes()]
        edge_color = ['red' if edge in path_edges or (edge[1], edge[0]) in path_edges 
                     else 'gray' for edge in G.edges()]
    
    # Vẽ
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    if directed:
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, 
                              arrows=True, arrowstyle='->', arrowsize=20)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=edge_color)
    
    # Thêm trọng số
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(title or f"Đồ thị ({len(G.nodes())} nodes, {len(G.edges())} edges)")
    plt.axis('off')
    
    # Tạo file tạm thời ngắn - FIX LỖI
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "graph_temp.png")
    
    # Đảm bảo đường dẫn ngắn
    if len(temp_file) > 100:
        temp_file = "C:/temp/graph.png"  # Đường dẫn cực ngắn
    
    plt.savefig(temp_file, bbox_inches='tight', dpi=100)
    plt.close()
    
    return temp_file

# ==================== MAIN HANDLERS ====================
def create_graph_handler(text, directed):
    """Xử lý tạo đồ thị"""
    global current_graph, is_directed
    
    is_directed = directed
    edges = []
    
    # Xử lý từng dòng
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                u = safe_int_convert(parts[0])
                v = safe_int_convert(parts[1])
                w = float(parts[2]) if len(parts) > 2 else 1.0
                edges.append((u, v, w))
            except:
                continue
    
    if not edges:
        return "❌ Không có dữ liệu hợp lệ", None
    
    # Tạo đồ thị
    current_graph = nx.DiGraph() if directed else nx.Graph()
    for u, v, w in edges:
        current_graph.add_edge(u, v, weight=w)
    
    img_path = draw_and_save_graph(current_graph, directed, 
                                   title=f"Đã tạo {len(edges)} cạnh")
    return f"✅ Tạo thành công {len(edges)} cạnh", img_path

def shortest_path_handler(start, end):
    """Tìm đường đi ngắn nhất"""
    if not current_graph.nodes():
        return "❌ Chưa có đồ thị", None
    
    try:
        start = int(start)
        end = int(end)
        
        try:
            path = nx.dijkstra_path(current_graph, start, end)
            length = nx.dijkstra_path_length(current_graph, start, end)
            img_path = draw_and_save_graph(current_graph, is_directed, 
                                          highlight_path=path,
                                          title=f"Đường đi: {path} (dài: {length})")
            return f"📏 Đường đi: {path}\n📊 Độ dài: {length}", img_path
        except nx.NetworkXNoPath:
            img_path = draw_and_save_graph(current_graph, is_directed)
            return "⚠ Không tìm thấy đường đi", img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "❌ Node không hợp lệ", img_path

def bfs_handler(start):
    """Xử lý BFS"""
    if not current_graph.nodes():
        return "❌ Chưa có đồ thị", None
    
    try:
        start = int(start)
        bfs_nodes = list(nx.bfs_tree(current_graph, start).nodes())
        img_path = draw_and_save_graph(current_graph, is_directed,
                                      highlight_path=bfs_nodes,
                                      title=f"BFS: {bfs_nodes}")
        return f"🔄 BFS: {bfs_nodes}", img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "❌ Node không hợp lệ", img_path

def dfs_handler(start):
    """Xử lý DFS"""
    if not current_graph.nodes():
        return "❌ Chưa có đồ thị", None
    
    try:
        start = int(start)
        dfs_nodes = list(nx.dfs_tree(current_graph, start).nodes())
        img_path = draw_and_save_graph(current_graph, is_directed,
                                      highlight_path=dfs_nodes,
                                      title=f"DFS: {dfs_nodes}")
        return f"🔍 DFS: {dfs_nodes}", img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "❌ Node không hợp lệ", img_path

def bipartite_handler():
    """Kiểm tra đồ thị 2 phía"""
    if not current_graph.nodes():
        return "❌ Chưa có đồ thị", None
    
    try:
        is_bip = nx.is_bipartite(current_graph)
        result = "✅ Là đồ thị 2 phía" if is_bip else "❌ Không phải đồ thị 2 phía"
        img_path = draw_and_save_graph(current_graph, is_directed, title=result)
        return result, img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "⚠ Không thể kiểm tra", img_path

# ==================== GRADIO UI ====================
with gr.Blocks(title="Graph Visualizer", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("# 📊 **TRÌNH XỬ LÝ ĐỒ THỊ**")
    gr.Markdown("Nhập đồ thị và thực hiện các thuật toán cơ bản")
    
    with gr.Tabs():
        # TAB 1: NHẬP ĐỒ THỊ
        with gr.Tab("📝 Nhập đồ thị"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### **Nhập danh sách cạnh**")
                    input_text = gr.Textbox(
                        label="Mỗi dòng: u v [weight]",
                        placeholder="Ví dụ:\n0 1 5\n0 2 3\n1 2 2",
                        lines=10,
                        value="0 1 5\n0 2 3\n1 2 2"
                    )
                    
                    with gr.Row():
                        directed_cb = gr.Checkbox(label="Đồ thị có hướng", value=False)
                        create_btn = gr.Button("Tạo đồ thị", variant="primary", size="lg")
                    
                    status = gr.Textbox(label="Trạng thái", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### **Hiển thị**")
                    output_img = gr.Image(label="Đồ thị")
            
            # Kết nối
            create_btn.click(
                fn=create_graph_handler,
                inputs=[input_text, directed_cb],
                outputs=[status, output_img]
            )
        
        # TAB 2: THUẬT TOÁN CƠ BẢN
        with gr.Tab("🔍 Thuật toán"):
            with gr.Row():
                with gr.Column():
                    # Dijkstra
                    gr.Markdown("### **Đường đi ngắn nhất**")
                    with gr.Row():
                        start_node = gr.Number(label="Node bắt đầu", value=0, precision=0)
                        end_node = gr.Number(label="Node kết thúc", value=1, precision=0)
                    
                    dijkstra_btn = gr.Button("Tìm đường đi", variant="primary")
                    dijkstra_result = gr.Textbox(label="Kết quả")
                    
                    # BFS/DFS
                    gr.Markdown("### **Duyệt đồ thị**")
                    traversal_start = gr.Number(label="Node bắt đầu", value=0, precision=0)
                    
                    with gr.Row():
                        bfs_btn = gr.Button("BFS")
                        dfs_btn = gr.Button("DFS")
                    
                    traversal_result = gr.Textbox(label="Kết quả duyệt")
                    
                    # Bipartite
                    gr.Markdown("### **Kiểm tra tính chất**")
                    bipartite_btn = gr.Button("Kiểm tra 2 phía")
                    bipartite_result = gr.Textbox(label="Kết quả")
                
                with gr.Column():
                    algo_img = gr.Image(label="Kết quả trực quan")
            
            # Kết nối sự kiện
            dijkstra_btn.click(
                fn=shortest_path_handler,
                inputs=[start_node, end_node],
                outputs=[dijkstra_result, algo_img]
            )
            
            bfs_btn.click(
                fn=bfs_handler,
                inputs=[traversal_start],
                outputs=[traversal_result, algo_img]
            )
            
            dfs_btn.click(
                fn=dfs_handler,
                inputs=[traversal_start],
                outputs=[traversal_result, algo_img]
            )
            
            bipartite_btn.click(
                fn=bipartite_handler,
                outputs=[bipartite_result, algo_img]
            )
        
        # TAB 3: CHUYỂN ĐỔI
        with gr.Tab("🔄 Chuyển đổi"):
            gr.Markdown("### **Chuyển đổi biểu diễn**")
            
            format_type = gr.Radio(
                choices=["Ma trận kề", "Danh sách kề", "Danh sách cạnh"],
                label="Chọn định dạng",
                value="Danh sách cạnh"
            )
            
            convert_btn = gr.Button("Chuyển đổi", variant="primary")
            conversion_output = gr.Textbox(label="Kết quả", lines=10)
            
            def convert_handler(format_type):
                if not current_graph.nodes():
                    return "❌ Chưa có đồ thị"
                
                try:
                    if format_type == "Ma trận kề":
                        nodes = sorted(current_graph.nodes())
                        matrix = nx.to_numpy_array(current_graph, nodelist=nodes)
                        result = "Ma trận kề:\n"
                        result += str(matrix)
                        
                    elif format_type == "Danh sách kề":
                        result = "Danh sách kề:\n"
                        for node in sorted(current_graph.nodes()):
                            neighbors = list(current_graph.neighbors(node))
                            result += f"{node}: {neighbors}\n"
                    
                    else:  # Danh sách cạnh
                        result = "Danh sách cạnh:\n"
                        for u, v, data in current_graph.edges(data=True):
                            w = data.get('weight', 1)
                            result += f"({u}, {v}, {w})\n"
                    
                    return result
                except Exception as e:
                    return f"❌ Lỗi: {str(e)}"
            
            convert_btn.click(
                fn=convert_handler,
                inputs=[format_type],
                outputs=[conversion_output]
            )
        
        # TAB 4: LƯU/TẢI
        with gr.Tab("💾 Lưu/Tải"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### **Lưu đồ thị**")
                    save_btn = gr.Button("Xuất JSON", variant="primary")
                    json_output = gr.Textbox(label="Dữ liệu JSON", lines=8)
                    
                    def save_handler():
                        if not current_graph.nodes():
                            return "❌ Chưa có đồ thị"
                        
                        edges = [(u, v, current_graph[u][v].get('weight', 1)) 
                                for u, v in current_graph.edges()]
                        
                        data = {
                            "directed": is_directed,
                            "nodes": list(current_graph.nodes()),
                            "edges": edges
                        }
                        return json.dumps(data, indent=2)
                    
                    save_btn.click(fn=save_handler, outputs=[json_output])
                
                with gr.Column():
                    gr.Markdown("### **Tải đồ thị**")
                    json_input = gr.Textbox(
                        label="Dán JSON ở đây",
                        placeholder='{"directed": false, "edges": [[0,1,5], [0,2,3]]}',
                        lines=8
                    )
                    
                    load_btn = gr.Button("Tải từ JSON")
                    load_status = gr.Textbox(label="Trạng thái")
                    
                    def load_handler(json_str):
                        try:
                            data = json.loads(json_str)
                            global current_graph, is_directed
                            
                            is_directed = data.get("directed", False)
                            current_graph = nx.DiGraph() if is_directed else nx.Graph()
                            
                            for u, v, w in data.get("edges", []):
                                current_graph.add_edge(u, v, weight=w)
                            
                            img_path = draw_and_save_graph(current_graph, is_directed,
                                                         title="Đồ thị đã tải")
                            return "✅ Đã tải thành công", img_path
                        except:
                            return "❌ JSON không hợp lệ", None
                    
                    load_btn.click(
                        fn=load_handler,
                        inputs=[json_input],
                        outputs=[load_status, output_img]
                    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    ### 📌 **Hướng dẫn nhanh:**
    1. **Tab 1**: Nhập đồ thị (mỗi dòng: `u v weight`)
    2. **Tab 2**: Chạy các thuật toán
    3. **Tab 3**: Chuyển đổi định dạng
    4. **Tab 4**: Lưu/tải đồ thị
    """)

# ==================== CHẠY ỨNG DỤNG ====================
if __name__ == "__main__":
    print("🚀 Ứng dụng đang chạy tại: http://localhost:7860")
    
    # Tạo thư mục temp nếu chưa có
    temp_dir = "C:/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7858,
        share=False,
        show_error=True
    )