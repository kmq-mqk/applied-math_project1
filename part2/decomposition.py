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
    # 5 Test Cases cho SVD
    test_cases_svd = [
        [[1, 5, -2, 4], [-3, 2, 7, 1], [8, -1, 3, -5]], # Case 1: Chữ nhật ngang (m < n)
        [[1, 2], [3, 4], [5, 6]],                       # Case 2: Chữ nhật đứng (m > n)
        [[1, 1], [0, 1]],                               # Case 3: Ma trận vuông
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],             # Case 4: Hạng thiếu (Rank-deficient)
        [[1, 0], [0, 1]]                                # Case 5: Ma trận đơn vị (Edge case)
    ]

    for idx, A_data in enumerate(test_cases_svd):
        print(f"\n{'='*25} SVD TEST CASE {idx+1} {'='*25}")
        try:
            A_obj = Matrix(A_data)
            u, sigma, v_t = A_obj.svd()
            
            # Chuyển sang numpy để kiểm tra
            U_np = np.array(u.data, dtype=float)
            S_np = np.array(sigma.data, dtype=float)
            VT_np = np.array(v_t.data, dtype=float)
            A_np = np.array(A_data, dtype=float)
            
            # Tái tạo ma trận: A_new = U * Sigma * V^T
            A_reconstructed = np.matmul(U_np, np.matmul(S_np, VT_np))
            
            print(f"Ma trận gốc {len(A_data)}x{len(A_data[0])}:\n{A_np}")
            print(f"Ma trận Sigma (làm tròn):\n{np.round(S_np, 4)}")
            
            # Kiểm tra A == U * Sigma * V^T
            is_correct = np.allclose(A_np, A_reconstructed, atol=1e-7)
            
            # Kiểm tra tính trực giao của U (U^T * U = I)
            # Vì U là m x m, ta kiểm tra U.T @ U có ra ma trận đơn vị không
            I_check = np.matmul(U_np.T, U_np)
            is_u_orthogonal = np.allclose(I_check, np.eye(U_np.shape[1]), atol=1e-7)
            
            if is_correct:
                print(">>> Kết quả: TÁI TẠO CHÍNH XÁC!")
                if is_u_orthogonal:
                    print(">>> Tính chất: Ma trận U trực giao hoàn hảo.")
                else:
                    # Nếu báo dòng này, nghĩa là bước Gram-Schmidt bổ sung cột đang có vấn đề
                    print(">>> Lưu ý: Tái tạo đúng nhưng U chưa chuẩn trực giao.")
            else:
                print(">>> Kết quả: THẤT BẠI (Sai số quá lớn)!")
                
        except Exception as e:
            print(f"Có lỗi xảy ra tại Test Case {idx+1}: {e}")