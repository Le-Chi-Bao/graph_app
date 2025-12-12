## Graph Visualization & Algorithms Tool
## Một ứng dụng web tương tác để trực quan hóa, phân tích và thực thi các thuật toán đồ thị, được phát triển cho môn học Cấu trúc rời rạc.

##  Tính năng nổi bật
##  Thuật toán cơ bản
Tìm đường đi ngắn nhất (Dijkstra)

Duyệt đồ thị (BFS & DFS)

Kiểm tra đồ thị hai phía (Bipartite)

Chuyển đổi biểu diễn đồ thị (Ma trận kề ↔ Danh sách kề ↔ Danh sách cạnh)

## Thuật toán nâng cao
Tìm cây khung nhỏ nhất (Prim & Kruskal)

Tìm luồng cực đại (Ford-Fulkerson)

Tìm chu trình Euler (Fleury & Hierholzer)

## Giao diện thân thiện
Trực quan hóa đồ thị với NetworkX & Matplotlib

Giao diện web tương tác với Gradio

Hiển thị động theo từng thuật toán

Hỗ trợ đồ thị có hướng/vô hướng, có trọng số

## Cài đặt và chạy ứng dụng
### 1. Tạo môi trường ảo với Conda

conda create --name graph_app python=3.9 -y

conda activate graph_app
### 2. Clone repository

git clone https://github.com/Le-Chi-Bao/graph_app.git

cd graph_app
### 3. Cài đặt thư viện

pip install -r requirements.txt
### 4. Khởi chạy ứng dụng

python app.py

Ứng dụng sẽ khởi động và có thể truy cập tại:
http://localhost:7880

## Yêu cầu hệ thống
Python 3.9+

Conda (khuyến nghị) hoặc pip

RAM tối thiểu: 4GB

Kết nối internet để tải các thư viện

## Hướng dẫn sử dụng nhanh
Tab 1 - Nhập đồ thị: Nhập cạnh theo định dạng u v [weight]

Tab 2 - Thuật toán cơ bản: Chạy Dijkstra, BFS, DFS, kiểm tra 2 phía

Tab 3 - Thuật toán nâng cao: Chọn và chạy Prim, Kruskal, Ford-Fulkerson,...

Tab 4 - Chuyển đổi: Xem đồ thị dưới dạng ma trận kề, danh sách kề, danh sách cạnh

Tab 5 - Lưu/Tải: Xuất/nhập đồ thị dạng JSON
