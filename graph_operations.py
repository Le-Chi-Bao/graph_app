import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

class GraphOperations:
    def __init__(self):
        self.graph = None
        self.directed = False
    
    def set_graph(self, G, directed=False):
        """Thiet lap do thi"""
        self.graph = G
        self.directed = directed
    
    # 1. Tim duong di ngan nhat (Dijkstra)
    def shortest_path(self, start, end):
        """Tim duong di ngan nhat bang Dijkstra"""
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return None, float('inf')
        
        try:
            path = nx.dijkstra_path(self.graph, start, end)
            length = nx.dijkstra_path_length(self.graph, start, end)
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    # 2. Duyet BFS
    def bfs_traversal(self, start):
        """Duyet do thi theo BFS"""
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
    
    # 3. Duyet DFS
    def dfs_traversal(self, start):
        """Duyet do thi theo DFS"""
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
    
    # 4. Kiem tra do thi 2 phia
    def is_bipartite(self):
        """Kiem tra do thi co phai la 2 phia khong"""
        try:
            return nx.is_bipartite(self.graph)
        except:
            return False
    
    # 5. Chuyen doi bieu dien
    def to_adjacency_matrix(self):
        """Chuyen sang ma tran ke"""
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
        """Chuyen sang danh sach ke"""
        adj_list = {}
        for node in self.graph.nodes():
            neighbors = []
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor].get('weight', 1)
                neighbors.append((neighbor, weight))
            adj_list[node] = neighbors
        return adj_list
    
    def to_edge_list(self):
        """Chuyen sang danh sach canh"""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1)
            edges.append((u, v, weight))
        return edges
    
    # 6. Thuat toan Prim
    def prim_mst(self):
        """Tim cay khung nho nhat bang Prim"""
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
    
    # 7. Thuat toan Kruskal
    def kruskal_mst(self):
        """Tim cay khung nho nhat bang Kruskal"""
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
        
        # Khoi tao
        for node in self.graph.nodes():
            parent[node] = node
        
        # Sap xep cac canh theo trong so
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
    
    # 8. Thuat toan Ford-Fulkerson
    def ford_fulkerson(self, source, sink):
        """Tim luong cuc dai bang Ford-Fulkerson"""
        # Tao do thi residual
        R = nx.DiGraph() if self.directed else nx.Graph()
        
        # Them cac canh voi capacity
        for u, v, data in self.graph.edges(data=True):
            capacity = data.get('weight', 1)
            R.add_edge(u, v, capacity=capacity, flow=0)
            if not self.directed:
                R.add_edge(v, u, capacity=capacity, flow=0)
        
        max_flow = 0
        
        while True:
            # Tim duong tang luong bang BFS
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
            
            # Tim gia tri luong tang
            path_flow = float('inf')
            v = sink
            while v != source:
                u = visited[v]
                path_flow = min(path_flow, R[u][v]['capacity'] - R[u][v]['flow'])
                v = u
            
            # Cap nhat luong
            v = sink
            while v != source:
                u = visited[v]
                R[u][v]['flow'] += path_flow
                R[v][u]['flow'] -= path_flow
                v = u
            
            max_flow += path_flow
        
        return max_flow