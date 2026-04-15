import sys
import os
import math
import numpy as np

# Thiết lập đường dẫn để import Matrix từ part1
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.part1_skeleton import Matrix, Vector
import part2.diagonalization


def svd(self):
    """
    Phân rã SVD: A = U * Sigma * V^T
    Sử dụng Numpy chỉ để tìm trị riêng/vector riêng của A^T * A.
    Các bước còn lại dùng code tự viết.
    """
    m_dim, n_dim = self.num_row, self.num_col
    a_t = self.transpose()
    
    # 1. TÌM V: Chỉ dùng numpy để giải bài toán trị riêng cho A^T * A
    ata = a_t.matmul(self)
    ata_np = np.array(ata.data, dtype=float)
    
    # Sử dụng eigh (chuyên cho ma trận đối xứng) để lấy trị riêng và vector riêng
    eig_vals, eig_vecs = np.linalg.eigh(ata_np)
    
    # 2. SẮP XẾP & CHUẨN HÓA V
    v_pairs = []
    for i in range(n_dim):
        lam = max(0.0, eig_vals[i]) # Tránh số âm rất nhỏ do sai số
        v_vec = eig_vecs[:, i].tolist()
        # Chuẩn hóa vector v_i
        norm_v = math.sqrt(sum(x*x for x in v_vec))
        v_vec = [x/norm_v for x in v_vec] if norm_v > 1e-12 else v_vec
        v_pairs.append((lam, v_vec))
    
    # Sắp xếp trị riêng giảm dần
    v_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # 3. TẠO SIGMA VÀ V^T
    sigma_data = [[0.0 for _ in range(n_dim)] for _ in range(m_dim)]
    v_t_data = []
    for i in range(n_dim):
        lam, v_vec = v_pairs[i]
        sig = math.sqrt(lam)
        if i < min(m_dim, n_dim):
            sigma_data[i][i] = sig
        v_t_data.append(v_vec)
    
    sigma = Matrix(sigma_data)
    v_t = Matrix(v_t_data)
    
    # 4. TÌM U: u_i = A * v_i / sigma_i
    u_cols = []
    for i in range(min(m_dim, n_dim)):
        sig = sigma_data[i][i]
        vi = v_t_data[i]
        
        if sig > 1e-9:
            # u_i = A * v_i / sig
            avi = [sum(self.data[r][c] * vi[c] for c in range(n_dim)) for r in range(m_dim)]
            u_cols.append([val/sig for val in avi])
            
    # 5. BỔ SUNG CỘT CHO U
    # Nếu m > n, U cần đủ m cột. Các cột còn lại thuộc Nullspace của A^T.
    if len(u_cols) < m_dim:
        _, (_, _, ns) = a_t.rank_and_basis()
        
        for vec in ns:
            if len(u_cols) >= m_dim: break
            w = vec.data[:]
            
            # Trực giao hóa Gram-Schmidt với các cột u đã có để đảm bảo U trực giao
            for u_existing in u_cols:
                dot_wu = sum(w[k] * u_existing[k] for k in range(m_dim))
                w = [w[k] - dot_wu * u_existing[k] for k in range(m_dim)]
            
            norm_w = math.sqrt(sum(x*x for x in w))
            if norm_w > 1e-9:
                u_cols.append([x/norm_w for x in w])

    # 6. LẮP RÁP MA TRẬN U
    # Chuyển từ danh sách cột (u_cols) sang dữ liệu hàng cho Matrix class
    u_final_data = [[u_cols[j][i] for j in range(len(u_cols))] for i in range(m_dim)]
    u = Matrix(u_final_data)
    
    return u, sigma, v_t

# Gán hàm vào class Matrix để sử dụng
Matrix.svd = svd


    

    # Kiểm chứng
if __name__ == "__main__":
    A = Matrix([
        [1, 5, -2, 4],
        [-3, 2, 7, 1],
        [8, -1, 3, -5]
    ])
    
    u, sigma, v_t = A.svd()
    print("Kích thước U:", len(u.data), "x", len(u.data[0]))
    print("Kích thước Sigma:", len(sigma.data), "x", len(sigma.data[0]))
    print("Kích thước V_T:", len(v_t.data), "x", len(v_t.data[0]))
    
    A_reconstructed = np.matmul(u.data, np.matmul(sigma.data, v_t.data))
    print(A_reconstructed)
    if np.allclose(A_reconstructed, A.data):
        print("SVD decomposition is correct!")