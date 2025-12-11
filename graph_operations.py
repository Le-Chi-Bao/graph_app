import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq
import warnings

warnings.filterwarnings("ignore")

class GraphOperations:
    def __init__(self):
        self.graph = None
        self.directed = False
    
    def set_graph(self, G, directed=False):
        """Thiết lập đồ thị"""
        self.graph = G
        self.directed = directed
    
    # 1. Tìm đường đi ngắn nhất (Dijkstra)
    def shortest_path(self, start, end):
        """Tìm đường đi ngắn nhất bằng Dijkstra"""
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return None, float('inf')
        
        try:
            path = nx.dijkstra_path(self.graph, start, end)
            length = nx.dijkstra_path_length(self.graph, start, end)
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    # 2. Duyệt BFS
    def bfs_traversal(self, start):
        """Duyệt đồ thị theo BFS"""
        visited = []
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.append(node)
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
        
        return visited
    
    # 3. Duyệt DFS
    def dfs_traversal(self, start):
        """Duyệt đồ thị theo DFS"""
        visited = []
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                neighbors = list(self.graph.neighbors(node))
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return visited
    
    # 4. Kiểm tra đồ thị 2 phía
    def is_bipartite(self):
        """Kiểm tra đồ thị có phải là 2 phía không"""
        try:
            return nx.is_bipartite(self.graph)
        except:
            return False
    
    # 5. Chuyển đổi biểu diễn
    def to_adjacency_matrix(self):
        """Chuyển sang ma trận kề"""
        nodes = sorted(self.graph.nodes())
        n = len(nodes)
        matrix = [[0] * n for _ in range(n)]
        node_index = {node: i for i, node in enumerate(nodes)}
        
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1)
            i, j = node_index[u], node_index[v]
            matrix[i][j] = weight
            if not self.directed:
                matrix[j][i] = weight
        
        return matrix, nodes
    
    def to_adjacency_list(self):
        """Chuyển sang danh sách kề"""
        adj_list = {}
        for node in self.graph.nodes():
            neighbors = []
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor].get('weight', 1)
                neighbors.append((neighbor, weight))
            adj_list[node] = neighbors
        return adj_list
    
    def to_edge_list(self):
        """Chuyển sang danh sách cạnh"""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1)
            edges.append((u, v, weight))
        return edges

    # 6. Thuật toán Prim
    def prim_mst(self):
        """Tìm cây khung nhỏ nhất bằng Prim"""
        if not nx.is_connected(self.graph.to_undirected()):
            return None

        mst_edges = []
        visited = set()
        start_node = list(self.graph.nodes())[0]
        visited.add(start_node)
        
        edges = []
        for neighbor in self.graph.neighbors(start_node):
            weight = self.graph[start_node][neighbor].get('weight', 1)
            heapq.heappush(edges, (weight, start_node, neighbor))
        
        while edges and len(visited) < len(self.graph.nodes()):
            weight, u, v = heapq.heappop(edges)
            if v not in visited:
                visited.add(v)
                mst_edges.append((u, v, weight))
                for neighbor in self.graph.neighbors(v):
                    if neighbor not in visited:
                        w = self.graph[v][neighbor].get('weight', 1)
                        heapq.heappush(edges, (w, v, neighbor))
        
        return mst_edges
    
    # 7. Thuật toán Kruskal
    def kruskal_mst(self):
        """Tìm cây khung nhỏ nhất bằng Kruskal"""
        parent = {}
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u
                return True
            return False
        
        # Khởi tạo
        for node in self.graph.nodes():
            parent[node] = node
        
        # Sắp xếp các cạnh theo trọng số
        edges = []
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1)
            edges.append((weight, u, v))
        edges.sort()
        
        mst_edges = []
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst_edges.append((u, v, weight))
        
        return mst_edges
    
    # 8. Thuật toán Ford-Fulkerson
    def ford_fulkerson(self, source, sink):
        """Tìm luồng cực đại bằng Ford-Fulkerson"""
        # Tạo đồ thị residual
        R = nx.DiGraph() if self.directed else nx.Graph()
        
        # Thêm các cạnh với capacity
        for u, v, data in self.graph.edges(data=True):
            capacity = data.get('weight', 1)
            R.add_edge(u, v, capacity=capacity, flow=0)
            if not self.directed:
                R.add_edge(v, u, capacity=capacity, flow=0)
        
        max_flow = 0
        
        while True:
            # Tìm đường tăng luồng bằng BFS
            visited = {source: None}
            queue = deque([source])
            found = False
            
            while queue and not found:
                u = queue.popleft()
                for v in R.neighbors(u):
                    if v not in visited and R[u][v]['capacity'] - R[u][v]['flow'] > 0:
                        visited[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
            
            if not found:
                break
            
            # Tìm giá trị luồng tăng
            path_flow = float('inf')
            v = sink
            while v != source:
                u = visited[v]
                path_flow = min(path_flow, R[u][v]['capacity'] - R[u][v]['flow'])
                v = u
            
            # Cập nhật luồng
            v = sink
            while v != source:
                u = visited[v]
                R[u][v]['flow'] += path_flow
                R[v][u]['flow'] -= path_flow
                v = u
            
            max_flow += path_flow
        
        return max_flow
    
    # 9. Thuật toán Fleury (CẢI THIỆN)
    def fleury_eulerian_path(self, start=None):
        """Tìm chu trình Euler bằng Fleury"""
        if self.graph is None or len(self.graph.nodes()) == 0:
            return None
        
        # Tạo bản sao đồ thị
        G = self.graph.copy()
        
        # Kiểm tra điều kiện Euler
        if not nx.is_eulerian(G):
            return None
        
        # Chọn node bắt đầu
        if start is None:
            # Chọn node có bậc lẻ nếu có, không thì node đầu tiên
            odd_nodes = [node for node in G.nodes() if G.degree(node) % 2 == 1]
            start = odd_nodes[0] if odd_nodes else list(G.nodes())[0]
        
        circuit = []
        current = start
        
        # Hàm kiểm tra cạnh có phải là cầu không
        def is_bridge(u, v):
            # Đếm số thành phần liên thông trước khi xóa cạnh
            G_temp = G.copy()
            G_temp.remove_edge(u, v)
            return nx.number_connected_components(G_temp.to_undirected()) > nx.number_connected_components(G.to_undirected())
        
        while G.number_of_edges() > 0:
            neighbors = list(G.neighbors(current))
            
            # Chọn cạnh
            chosen_edge = None
            for neighbor in neighbors:
                if not is_bridge(current, neighbor):
                    chosen_edge = (current, neighbor)
                    break
            
            # Nếu tất cả đều là cầu, chọn cạnh đầu tiên
            if chosen_edge is None and neighbors:
                chosen_edge = (current, neighbors[0])
            
            if chosen_edge:
                circuit.append(chosen_edge)
                G.remove_edge(chosen_edge[0], chosen_edge[1])
                current = chosen_edge[1]
            else:
                break
        
        return circuit
    
    # 10. Thuật toán Hierholzer (CẢI THIỆN)
    def hierholzer_eulerian_circuit(self, start=None):
        """Tìm chu trình Euler bằng Hierholzer"""
        if self.graph is None or len(self.graph.nodes()) == 0:
            return None
        
        # Tạo bản sao để sửa đổi
        G = self.graph.copy()
        
        # Kiểm tra điều kiện Euler
        if not nx.is_eulerian(G):
            return None
        
        # Chọn node bắt đầu
        if start is None:
            start = list(G.nodes())[0]
        
        # Thuật toán Hierholzer
        circuit = []
        stack = [start]
        
        while stack:
            current = stack[-1]
            
            # Nếu node còn cạnh
            if G.degree(current) > 0:
                # Lấy một cạnh bất kỳ
                next_node = list(G.neighbors(current))[0]
                
                # Thêm cạnh vào stack
                stack.append(next_node)
                
                # Xóa cạnh
                G.remove_edge(current, next_node)
            else:
                # Nếu node không còn cạnh, thêm vào circuit
                circuit.append(stack.pop())
        
        # Chuyển circuit thành các cạnh
        edges = []
        for i in range(len(circuit) - 1):
            edges.append((circuit[i], circuit[i+1]))
        
        return edges
    
    # 11. Phương thức bổ trợ: Kiểm tra đồ thị Euler
    def is_eulerian(self):
        """Kiểm tra xem đồ thị có chu trình Euler không"""
        if self.graph is None:
            return False
        return nx.is_eulerian(self.graph)
    
    # 12. Phương thức bổ trợ: Lấy thông tin đồ thị
    def get_graph_info(self):
        """Lấy thông tin chi tiết về đồ thị"""
        if self.graph is None:
            return "Chưa có đồ thị"
        
        info = []
        info.append(f" Số node: {len(self.graph.nodes())}")
        info.append(f" Số cạnh: {len(self.graph.edges())}")
        info.append(f" Loại: {'Có hướng' if self.directed else 'Vô hướng'}")
        
        # Kiểm tra tính liên thông
        if not self.directed:
            connected = nx.is_connected(self.graph)
            info.append(f" Liên thông: {'Có' if connected else 'Không'}")
        
        # Kiểm tra Euler
        eulerian = self.is_eulerian()
        info.append(f" Có chu trình Euler: {'Có' if eulerian else 'Không'}")
        
        # Kiểm tra 2 phía
        bipartite = self.is_bipartite()
        info.append(f" Là đồ thị 2 phía: {'Có' if bipartite else 'Không'}")
        
        return "\n".join(info)