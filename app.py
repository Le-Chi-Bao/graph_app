# main.py - HOÀN CHỈNH VỚI IMPORT
import gradio as gr
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import tempfile
import os
import warnings

# ==================== IMPORT MODULES ====================
from graph_operations import GraphOperations
from utils import draw_graph, adjacency_matrix_to_graph, edge_list_to_graph

# ==================== GLOBAL STATE ====================
current_graph = nx.Graph()
is_directed = False
graph_ops = GraphOperations()  # Instance của GraphOperations

# ==================== UTILITY FUNCTIONS ====================
def safe_int_convert(val):
    """Chuyển đổi an toàn sang int"""
    try:
        return int(float(val))
    except:
        return 0

def draw_and_save_graph(G, directed, highlight_path=None, highlight_edges=None, title=""):
    """Vẽ đồ thị và lưu ra file TEMP"""
    if not G.nodes():
        return None
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Cấu hình cơ bản
    node_color = ['lightblue'] * len(G.nodes())
    edge_color = ['gray'] * len(G.edges())
    edge_width = [1] * len(G.edges())
    
    # Highlight PATH (cho Dijkstra, BFS, DFS)
    path_edges = []
    if highlight_path:
        # Tạo danh sách cạnh từ path
        path_edges = [(highlight_path[i], highlight_path[i+1]) 
                     for i in range(len(highlight_path)-1)]
        
        # Highlight nodes trong path
        node_color = ['red' if node in highlight_path else 'lightblue' 
                     for node in G.nodes()]
        
        # Highlight edges trong path
        for i, edge in enumerate(G.edges()):
            if edge in path_edges or (edge[1], edge[0]) in path_edges:
                edge_color[i] = 'red'
                edge_width[i] = 3
    
    # Highlight EDGES cụ thể (cho Prim, Kruskal)
    if highlight_edges:
        for i, edge in enumerate(G.edges()):
            if edge in highlight_edges or (edge[1], edge[0]) in highlight_edges:
                edge_color[i] = 'red'
                edge_width[i] = 3
    
    # Vẽ nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    # Vẽ edges với màu sắc và độ dày khác nhau
    if directed:
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width,
                              arrows=True, arrowstyle='-|>', arrowsize=20)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width)
    
    # Thêm trọng số
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(title or f"Đồ thị ({len(G.nodes())} nodes, {len(G.edges())} edges)")
    plt.axis('off')
    
    # Tạo file tạm thời
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "graph_temp.png")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    
    plt.savefig(temp_file, bbox_inches='tight', dpi=100)
    plt.close()
    
    return temp_file

# ==================== HANDLERS CƠ BẢN ====================
def create_graph_handler(text, directed):
    """Xử lý tạo đồ thị"""
    global current_graph, is_directed, graph_ops
    
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
        return "Không có dữ liệu hợp lệ", None
    
    # Tạo đồ thị
    current_graph = nx.DiGraph() if directed else nx.Graph()
    for u, v, w in edges:
        current_graph.add_edge(u, v, weight=w)
    
    # Cập nhật graph_ops
    graph_ops.set_graph(current_graph, directed)
    
    img_path = draw_and_save_graph(current_graph, directed, 
                                   title=f"Đã tạo {len(edges)} cạnh")
    return f"Tạo thành công {len(edges)} cạnh", img_path

def shortest_path_handler(start, end):
    """Tìm đường đi ngắn nhất"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        start = int(start)
        end = int(end)
        
        try:
            path = nx.dijkstra_path(current_graph, start, end)
            length = nx.dijkstra_path_length(current_graph, start, end)
            
            # Tạo danh sách cạnh cần highlight
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            img_path = draw_and_save_graph(
                current_graph, is_directed, 
                highlight_path=path,
                highlight_edges=path_edges,
                title=f"Đường đi ngắn nhất: {path} (dài: {length})"
            )
            return f"Đường đi: {' -> '.join(map(str, path))}\nĐộ dài: {length}", img_path
        except nx.NetworkXNoPath:
            img_path = draw_and_save_graph(current_graph, is_directed)
            return "Không tìm thấy đường đi", img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "Node không hợp lệ", img_path

def bfs_handler(start):
    """Xử lý BFS"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        start = int(start)
        # Lấy cây BFS
        bfs_tree = nx.bfs_tree(current_graph, start)
        bfs_nodes = list(bfs_tree.nodes())
        
        # Lấy các cạnh trong cây BFS
        bfs_edges = list(bfs_tree.edges())
        
        img_path = draw_and_save_graph(
            current_graph, is_directed,
            highlight_path=bfs_nodes,
            highlight_edges=bfs_edges,
            title=f"BFS Tree từ node {start}"
        )
        return f"BFS: {bfs_nodes}", img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "Node không hợp lệ", img_path

def dfs_handler(start):
    """Xử lý DFS"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        start = int(start)
        # Lấy cây DFS
        dfs_tree = nx.dfs_tree(current_graph, start)
        dfs_nodes = list(dfs_tree.nodes())
        
        # Lấy các cạnh trong cây DFS
        dfs_edges = list(dfs_tree.edges())
        
        img_path = draw_and_save_graph(
            current_graph, is_directed,
            highlight_path=dfs_nodes,
            highlight_edges=dfs_edges,
            title=f"DFS Tree từ node {start}"
        )
        return f"DFS: {dfs_nodes}", img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "Node không hợp lệ", img_path

def bipartite_handler():
    """Kiểm tra đồ thị 2 phía - Dùng GraphOperations"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        # Dùng graph_ops
        is_bip = graph_ops.is_bipartite()
        result = "Là đồ thị 2 phía" if is_bip else "Không phải đồ thị 2 phía"
        img_path = draw_and_save_graph(current_graph, is_directed, title=result)
        return result, img_path
    except:
        img_path = draw_and_save_graph(current_graph, is_directed)
        return "Không thể kiểm tra", img_path

# ==================== HANDLERS NÂNG CAO ====================
def prim_handler():
    """Thuật toán Prim"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        mst_edges = graph_ops.prim_mst()
        if mst_edges:
            total_weight = sum(weight for _, _, weight in mst_edges)
            
            # Lấy danh sách cạnh MST để highlight
            mst_edge_list = [(u, v) for u, v, _ in mst_edges]
            
            img_path = draw_and_save_graph(
                current_graph, is_directed,
                highlight_edges=mst_edge_list,
                title=f"Prim MST - Tổng trọng số: {total_weight}"
            )
            result = f"Cây khung nhỏ nhất (Prim):\n"
            for u, v, w in mst_edges:
                result += f"  {u} -> {v} (w={w})\n"
            result += f"Tổng trọng số: {total_weight}"
            return result, img_path
        else:
            return "Đồ thị không liên thông", None
    except Exception as e:
        return f"Lỗi: {str(e)}", None

def kruskal_handler():
    """Thuật toán Kruskal"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        mst_edges = graph_ops.kruskal_mst()
        if mst_edges:
            total_weight = sum(weight for _, _, weight in mst_edges)
            
            # Lấy danh sách cạnh MST để highlight
            mst_edge_list = [(u, v) for u, v, _ in mst_edges]
            
            img_path = draw_and_save_graph(
                current_graph, is_directed,
                highlight_edges=mst_edge_list,
                title=f"Kruskal MST - Tổng trọng số: {total_weight}"
            )
            result = f"Cây khung nhỏ nhất (Kruskal):\n"
            for u, v, w in mst_edges:
                result += f"  ({u}, {v}) - {w}\n"
            result += f"Tổng trọng số: {total_weight}"
            return result, img_path
        else:
            return "Đồ thị không liên thông", None
    except Exception as e:
        return f"Lỗi: {str(e)}", None

def ford_fulkerson_handler(source, sink):
    """Thuật toán Ford-Fulkerson"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        source = int(source)
        sink = int(sink)
        
        max_flow = graph_ops.ford_fulkerson(source, sink)
        
        img_path = draw_and_save_graph(
            current_graph, is_directed,
            title=f"Ford-Fulkerson - Luồng cực đại: {max_flow}"
        )
        
        return f"Luồng cực đại từ {source} -> {sink}: {max_flow}", img_path
    except Exception as e:
        return f"Lỗi: {str(e)}", None

def fleury_handler(start_node):
    """Thuật toán Fleury (tìm chu trình Euler)"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        start = int(start_node)
        circuit = graph_ops.fleury_eulerian_path(start)
        
        if circuit:
            img_path = draw_and_save_graph(
                current_graph, is_directed,
                highlight_edges=circuit,
                title=f"Fleury - Chu trình Euler"
            )
            
            result = f"Chu trình Euler (Fleury):\n"
            for u, v in circuit:
                result += f"  {u} -> {v}\n"
            return result, img_path
        
        return "Đồ thị không có chu trình Euler", None
    except Exception as e:
        return f"Lỗi: {str(e)}", None

def hierholzer_handler(start_node):
    """Thuật toán Hierholzer (tìm chu trình Euler)"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    try:
        start = int(start_node)
        circuit = graph_ops.hierholzer_eulerian_circuit(start)
        
        if circuit:
            img_path = draw_and_save_graph(
                current_graph, is_directed,
                highlight_edges=circuit,
                title=f"Hierholzer - Chu trình Euler"
            )
            
            result = f"Chu trình Euler (Hierholzer):\n"
            for u, v in circuit:
                result += f"  {u} -> {v}\n"
            return result, img_path
        
        return "Đồ thị không có chu trình Euler", None
    except Exception as e:
        return f"Lỗi: {str(e)}", None

def advanced_algo_handler(algo_choice, param1=None, param2=None):
    """Xử lý thuật toán nâng cao"""
    if not current_graph.nodes():
        return "Chưa có đồ thị", None
    
    if algo_choice == "Prim":
        return prim_handler()
    elif algo_choice == "Kruskal":
        return kruskal_handler()
    elif algo_choice == "Ford-Fulkerson":
        if param1 is None or param2 is None:
            return "Vui lòng nhập source và sink", None
        return ford_fulkerson_handler(param1, param2)
    elif algo_choice == "Fleury":
        if param1 is None:
            return "Vui lòng nhập node bắt đầu", None
        return fleury_handler(param1)
    elif algo_choice == "Hierholzer":
        if param1 is None:
            return "Vui lòng nhập node bắt đầu", None
        return hierholzer_handler(param1)
    
    return "Vui lòng chọn thuật toán", None

# ==================== GRADIO UI ====================
with gr.Blocks(title="Trình Xử Lý Đồ Thị", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("# TRÌNH XỬ LÝ ĐỒ THỊ")
    gr.Markdown("Nhập đồ thị và thực hiện các thuật toán cơ bản & nâng cao")
    
    with gr.Tabs():
        # TAB 1: NHẬP ĐỒ THỊ
        with gr.Tab("Nhập đồ thị"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Nhập danh sách cạnh")
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
                    gr.Markdown("### Hiển thị")
                    output_img = gr.Image(label="Đồ thị")
            
            create_btn.click(
                fn=create_graph_handler,
                inputs=[input_text, directed_cb],
                outputs=[status, output_img]
            )
        
        # TAB 2: THUẬT TOÁN CƠ BẢN
        with gr.Tab("Thuật toán cơ bản"):
            with gr.Row():
                with gr.Column():
                    # Dijkstra
                    gr.Markdown("### Đường đi ngắn nhất")
                    with gr.Row():
                        start_node = gr.Number(label="Node bắt đầu", value=0, precision=0)
                        end_node = gr.Number(label="Node kết thúc", value=1, precision=0)
                    
                    dijkstra_btn = gr.Button("Tìm đường đi", variant="primary")
                    dijkstra_result = gr.Textbox(label="Kết quả")
                    
                    # BFS/DFS
                    gr.Markdown("### Duyệt đồ thị")
                    traversal_start = gr.Number(label="Node bắt đầu", value=0, precision=0)
                    
                    with gr.Row():
                        bfs_btn = gr.Button("BFS")
                        dfs_btn = gr.Button("DFS")
                    
                    traversal_result = gr.Textbox(label="Kết quả duyệt")
                    
                    # Bipartite
                    gr.Markdown("### Kiểm tra tính chất")
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
        
        # TAB 3: THUẬT TOÁN NÂNG CAO
        with gr.Tab("Thuật toán nâng cao"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Lựa chọn thuật toán")
                    
                    algo_choice = gr.Radio(
                        choices=["Prim", "Kruskal", "Ford-Fulkerson", "Fleury", "Hierholzer"],
                        label="Chọn thuật toán",
                        value="Prim"
                    )
                    
                    # Dynamic inputs based on algorithm
                    with gr.Group() as param_group:
                        source_input = gr.Number(
                            label="Source node (cho Ford-Fulkerson)",
                            value=0,
                            precision=0,
                            visible=False
                        )
                        sink_input = gr.Number(
                            label="Sink node (cho Ford-Fulkerson)",
                            value=1,
                            precision=0,
                            visible=False
                        )
                        start_input = gr.Number(
                            label="Node bắt đầu (cho Fleury/Hierholzer)",
                            value=0,
                            precision=0,
                            visible=False
                        )
                    
                    # Update input visibility based on algorithm choice
                    def update_inputs(algo):
                        vis_source = (algo == "Ford-Fulkerson")
                        vis_sink = (algo == "Ford-Fulkerson")
                        vis_start = (algo in ["Fleury", "Hierholzer"])
                        
                        return [
                            gr.update(visible=vis_source),
                            gr.update(visible=vis_sink),
                            gr.update(visible=vis_start)
                        ]
                    
                    algo_choice.change(
                        fn=update_inputs,
                        inputs=[algo_choice],
                        outputs=[source_input, sink_input, start_input]
                    )
                    
                    run_algo_btn = gr.Button("Chạy thuật toán", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Kết quả")
                    advanced_result = gr.Textbox(label="Kết quả thuật toán", lines=6)
                    advanced_img = gr.Image(label="Trực quan hóa")
            
            # Handler cho nút chạy thuật toán
            def run_advanced_algo(algo, source, sink, start):
                if algo == "Ford-Fulkerson":
                    return advanced_algo_handler(algo, source, sink)
                elif algo in ["Fleury", "Hierholzer"]:
                    return advanced_algo_handler(algo, start, None)
                else:
                    return advanced_algo_handler(algo, None, None)
            
            run_algo_btn.click(
                fn=run_advanced_algo,
                inputs=[algo_choice, source_input, sink_input, start_input],
                outputs=[advanced_result, advanced_img]
            )
        
        # TAB 4: CHUYỂN ĐỔI
        with gr.Tab("Chuyển đổi"):
            gr.Markdown("### Chuyển đổi biểu diễn")
            
            format_type = gr.Radio(
                choices=["Ma trận kề", "Danh sách kề", "Danh sách cạnh"],
                label="Chọn định dạng",
                value="Danh sách cạnh"
            )
            
            convert_btn = gr.Button("Chuyển đổi", variant="primary")
            conversion_output = gr.Textbox(label="Kết quả", lines=10)
            
            def convert_handler(format_type):
                if not current_graph.nodes():
                    return "Chưa có đồ thị"
                
                try:
                    if format_type == "Ma trận kề":
                        # Dùng graph_ops
                        matrix, nodes = graph_ops.to_adjacency_matrix()
                        result = "Ma trận kề:\n"
                        result += "   " + " ".join(str(n) for n in nodes) + "\n"
                        for i, row in enumerate(matrix):
                            result += f"{nodes[i]}: " + " ".join(str(x) for x in row) + "\n"
                        
                    elif format_type == "Danh sách kề":
                        # Dùng graph_ops
                        adj_list = graph_ops.to_adjacency_list()
                        result = "Danh sách kề:\n"
                        for node in sorted(adj_list.keys()):
                            neighbors = adj_list[node]
                            neighbor_str = ", ".join(f"{n}({w})" for n, w in neighbors)
                            result += f"{node}: {neighbor_str}\n"
                    
                    else:  # Danh sách cạnh
                        # Dùng graph_ops
                        edges = graph_ops.to_edge_list()
                        result = "Danh sách cạnh:\n"
                        for u, v, w in edges:
                            result += f"({u}, {v}, {w})\n"
                    
                    return result
                except Exception as e:
                    return f"Lỗi: {str(e)}"
            
            convert_btn.click(
                fn=convert_handler,
                inputs=[format_type],
                outputs=[conversion_output]
            )
        
        # ==================== CHỈ SỬA TAB 5 ====================

        # ==================== TAB 5: LƯU/TẢI - ĐƠN GIẢN ====================

        with gr.Tab(" Lưu/Tải"):
            gr.Markdown("###  Lưu đồ thị bằng danh sách kề")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # PHẦN LƯU
                    gr.Markdown("####  Xuất đồ thị")
                    
                    save_btn = gr.Button(" Copy danh sách kề", variant="primary", size="lg")
                    save_status = gr.Textbox(label="Trạng thái", interactive=False)
                    
                    # Textbox hiển thị danh sách kề
                    json_output = gr.Code(
                        label="Danh sách kề (JSON)",
                        language="json",
                        lines=10,
                        interactive=False,
                        value="{}"
                    )
                    
                    # Nút copy tự động
                    copy_btn = gr.Button(" Copy vào clipboard", variant="secondary")
                    
                    def save_adjacency_list():
                        """Xuất danh sách kề dạng JSON đẹp"""
                        if not current_graph.nodes():
                            return "Chưa có đồ thị", "{}"
                        
                        try:
                            # Lấy danh sách kề từ graph_ops
                            adj_list = graph_ops.to_adjacency_list()
                            
                            # Tạo JSON có cấu trúc đẹp
                            data = {
                                "directed": is_directed,
                                "weighted": any('weight' in data for _, _, data in current_graph.edges(data=True)),
                                "adjacency_list": {}
                            }
                            
                            # Format đẹp hơn
                            for node in sorted(adj_list.keys()):
                                neighbors = []
                                for neighbor, weight in adj_list[node]:
                                    neighbors.append([neighbor, weight])
                                data["adjacency_list"][str(node)] = neighbors
                            
                            json_str = json.dumps(data, indent=2, ensure_ascii=False)
                            return " Đã tạo danh sách kề", json_str
                            
                        except Exception as e:
                            return f" Lỗi: {str(e)}", "{}"
                    
                    def copy_to_clipboard(json_str):
                        """Copy JSON vào clipboard (giả lập)"""
                        if json_str != "{}":
                            # Trong Gradio, có thể dùng js để copy thật
                            return " Đã copy vào clipboard"
                        return " Không có dữ liệu để copy"
                    
                    save_btn.click(
                        fn=save_adjacency_list,
                        outputs=[save_status, json_output]
                    )
                    
                    copy_btn.click(
                        fn=copy_to_clipboard,
                        inputs=[json_output],
                        outputs=[save_status]
                    )
                    
                with gr.Column(scale=1):
                    # PHẦN TẢI
                    gr.Markdown("####  Tải đồ thị")
                    
                    # Hướng dẫn
                    gr.Markdown("""
                    **Định dạng JSON:**
                    ```json
                    {
                    "directed": false,
                    "adjacency_list": {
                        "0": [[1, 5], [2, 3]],
                        "1": [[0, 5], [2, 2]]
                    }
                    }
                    ```
                    """)
                    
                    json_input = gr.Textbox(
                        label="Dán JSON danh sách kề",
                        placeholder='{"directed": false, "adjacency_list": {"0": [[1,5]], "1": [[0,5]]}}',
                        lines=6
                    )
                    
                    load_btn = gr.Button(" Tạo đồ thị từ JSON", variant="primary", size="lg")
                    load_status = gr.Textbox(label="Trạng thái tải", interactive=False)
                    
                    def load_from_json(json_str):
                        """Tải đồ thị từ JSON danh sách kề"""
                        if not json_str.strip():
                            return " Vui lòng nhập JSON", None
                        
                        try:
                            global current_graph, is_directed, graph_ops
                            
                            data = json.loads(json_str)
                            
                            # Lấy thông tin
                            is_directed = data.get("directed", False)
                            adj_list = data.get("adjacency_list", {})
                            
                            if not adj_list:
                                return " JSON không có adjacency_list", None
                            
                            # Tạo đồ thị mới
                            current_graph = nx.DiGraph() if is_directed else nx.Graph()
                            
                            # Thêm các cạnh
                            edge_count = 0
                            for u_str, neighbors in adj_list.items():
                                u = int(u_str)
                                for neighbor_info in neighbors:
                                    if isinstance(neighbor_info, list) and len(neighbor_info) >= 2:
                                        v = int(neighbor_info[0])
                                        w = float(neighbor_info[1]) if len(neighbor_info) > 1 else 1.0
                                        current_graph.add_edge(u, v, weight=w)
                                        edge_count += 1
                            
                            # Cập nhật graph_ops
                            graph_ops.set_graph(current_graph, is_directed)
                            
                            # Vẽ đồ thị
                            img_path = draw_and_save_graph(
                                current_graph, 
                                is_directed,
                                title=f"Đồ thị ({len(current_graph.nodes())} nodes, {edge_count} edges)"
                            )
                            
                            return f" Đã tải: {len(current_graph.nodes())} nodes, {edge_count} edges", img_path
                            
                        except json.JSONDecodeError:
                            return " JSON không hợp lệ", None
                        except Exception as e:
                            return f" Lỗi: {str(e)}", None
                    
                    load_btn.click(
                        fn=load_from_json,
                        inputs=[json_input],
                        outputs=[load_status, output_img]
                    )
            
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    ### Hướng dẫn nhanh:
    1. Tab 1: Nhập đồ thị (mỗi dòng: u v weight)
    2. Tab 2: Thuật toán cơ bản (Dijkstra, BFS, DFS, 2 phía)
    3. Tab 3: Thuật toán nâng cao (Prim, Kruskal, Ford-Fulkerson, Fleury, Hierholzer)
    4. Tab 4: Chuyển đổi định dạng
    5. Tab 5: Lưu/tải đồ thị
    """)

# ==================== CHẠY ỨNG DỤNG ====================
if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("Ứng dụng đang chạy tại: http://localhost:7872")
    print("Tab 1: Nhập đồ thị")
    print("Tab 2: Thuật toán cơ bản (Dijkstra, BFS, DFS)")
    print("Tab 3: Thuật toán nâng cao (Prim, Kruskal, Ford-Fulkerson, Fleury, Hierholzer)")
    print("Tab 4: Chuyển đổi định dạng")
    print("Tab 5: Lưu/tải đồ thị")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7880,
        share=False,
        show_error=True,
        quiet=True
    )