import sys
import os
import math

# Thiết lập đường dẫn để import Matrix từ part1
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.part1_skeleton import Matrix
import diagonalization

def svd(self):
    """
    Phân rã SVD đầy đủ (Full SVD): A = U * Sigma * V^T
    U: m x m, Sigma: m x n, V^T: n x n
    """
    a_t = self.transpose()
    m_dim, n_dim = self.num_row, self.num_col
    
    # 1. Tìm V và trị riêng từ A^T * A (n x n)
    ata = a_t.matmul(self)
    p_v, d_v = ata.diagonalize()
    
    eigen_pairs_v = []
    for i in range(n_dim):
        lam = d_v.data[i][i]
        v = [p_v.data[j][i] for j in range(n_dim)]
        eigen_pairs_v.append((max(0, lam), v)) # max(0, lam) để tránh số âm cực nhỏ do sai số
    
    # Sắp xếp trị riêng giảm dần
    eigen_pairs_v.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Tìm U từ A * A^T (m x m)
    # Tính trực tiếp từ AA^T để lấy đủ m cột trực giao của U
    aat = self.matmul(a_t)
    p_u, d_u = aat.diagonalize()
    
    eigen_pairs_u = []
    for i in range(m_dim):
        lam = d_u.data[i][i]
        u = [p_u.data[j][i] for j in range(m_dim)]
        eigen_pairs_u.append((max(0, lam), u))
    
    eigen_pairs_u.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Lắp ráp ma trận Sigma (m x n)
    sigma_data = [[0.0 for _ in range(n_dim)] for _ in range(m_dim)]
    for i in range(min(m_dim, n_dim)):
        lam = eigen_pairs_v[i][0]
        sigma_data[i][i] = math.sqrt(lam)
    sigma = Matrix(sigma_data)
    
    # 4. Lắp ráp V^T (n x n) và chuẩn hóa vector v
    v_rows = []
    for _, v in eigen_pairs_v:
        norm_v = math.sqrt(sum(x*x for x in v))
        v_rows.append([x/norm_v for x in v] if norm_v > 1e-12 else v)
    v_t = Matrix(v_rows)
    
    # 5. Lắp ráp U (m x m) và đồng bộ hóa dấu (Sign Correction)
    # Vì vector riêng có thể ngược dấu (+/-), cần đảm bảo A*vi = sigma_i * ui
    u_cols_temp = []
    for _, u in eigen_pairs_u:
        norm_u = math.sqrt(sum(x*x for x in u))
        u_cols_temp.append([x/norm_u for x in u] if norm_u > 1e-12 else u)
        
    for i in range(min(m_dim, n_dim)):
        sig = sigma_data[i][i]
        if sig > 1e-9:
            vi = v_rows[i]
            ui = u_cols_temp[i]
            
            # Tính A * vi
            avi = [sum(self.data[r][c] * vi[c] for c in range(n_dim)) for r in range(m_dim)]
            
            # Nếu A*vi ngược hướng với ui, ta đổi dấu cột ui
            dot = sum(avi[k] * ui[k] for k in range(m_dim))
            if dot < 0:
                u_cols_temp[i] = [-x for x in ui]
                
    # Chuyển u_cols_temp (danh sách cột) thành u_data (danh sách dòng)
    u_data = [[u_cols_temp[j][i] for j in range(m_dim)] for i in range(m_dim)]
    u = Matrix(u_data)
    
    return u, sigma, v_t

Matrix.svd = svd

# Kiểm chứng
if __name__ == "__main__":
    import numpy as np
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
    if np.allclose(A_reconstructed, A.data):
        print("SVD decomposition is correct!")