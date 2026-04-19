import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.part1_skeleton import Matrix


# --- ĐỊNH NGHĨA CÁC HÀM ---
# dùng numpy tìm trị riêng 
def find_eigenvalues(self):
    # if self.num_row != self.num_col:
    #     raise ValueError("Chỉ ma trận vuông mới có giá trị riêng.")
    
    A_np = np.array(self.data, dtype=float)
    eigenvalues = np.linalg.eigvals(A_np)
    
    # Bỏ phần ảo nếu quá nhỏ
    eigenvalues = np.where(np.abs(eigenvalues.imag) < 1e-10, 
                           eigenvalues.real, 
                           eigenvalues)
    
    # Làm tròn để khử sai số dấu phẩy động
    # 5 chữ số thập phân là đủ an toàn cho các bài toán đại số tuyến tính cơ bản
    eigenvalues_rounded = np.round(eigenvalues.real, 8)
    
    # Lọc lấy các giá trị riêng duy nhất
    unique_eigenvalues = np.unique(eigenvalues_rounded).tolist()
    unique_eigenvalues.sort(reverse=True)
    return unique_eigenvalues

Matrix.find_eigenvalues = find_eigenvalues
    
def diagonalize(self):
    # if self.num_row != self.num_col:
    #     #raise ValueError("Ma trận phải là ma trận vuông để chéo hóa.")
        
    eigenvalues = self.find_eigenvalues() 
    
    eigenvectors_matrix = [] 
    diagonal_elements = []   
    
    for lam in eigenvalues:
        M_data = []
        for i in range(self.num_row):
            row = []
            for j in range(self.num_col):
                val = self.data[i][j] - (lam if i == j else 0)
                # Làm tròn nhẹ một lần nữa khi tạo ma trận để đảm bảo các số 0 tròn trĩnh
                row.append(round(val, 5))
            M_data.append(row)
            
        M = Matrix(M_data)
        
        result = M.rank_and_basis()
        nullspace_vectors = result[1][2]
        
        for v in nullspace_vectors:
            eigenvectors_matrix.append(v)
            diagonal_elements.append(lam)

    # if len(eigenvectors_matrix) < self.num_row:
    #    # raise ValueError("Ma trận bị khiếm khuyết, không đủ vector riêng độc lập để chéo hóa.")

    P_data = [[eigenvectors_matrix[j].data[i] for j in range(self.num_col)] for i in range(self.num_row)]
    P = Matrix(P_data)

    D_data = [[diagonal_elements[i] if i == j else 0 for j in range(self.num_col)] for i in range(self.num_row)]
    D = Matrix(D_data)

    return P, D

Matrix.diagonalize = diagonalize


# --- KIỂM TRA CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Danh sách các ma trận kiểm thử
    # Case 1: Ma trận cơ bản (3x3)
    # Case 2: Ma trận đơn vị (Trường hợp biên: P và D là chính nó)
    # Case 3: Ma trận có trị riêng lặp lại (Kiểm tra nghiệm bội)
    # Case 4: Ma trận tam giác (Trị riêng nằm trên đường chéo)
    # Case 5: Ma trận số thực (Kiểm tra độ chính xác số thập phân)
    
    test_matrices = [
        [[2, 0, -2], [0, 3, 0], [0, 0, 3]],         # Case 1
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],          # Case 2
        [[4, 2, 2], [2, 4, 2], [2, 2, 4]],          # Case 3
        [[1, 2, 3], [0, 4, 5], [0, 0, 6]],          # Case 4
        [[1.5, 0.5], [0.5, 1.5]]                    # Case 5
    ]

    for idx, A_data in enumerate(test_matrices):
        print(f"\n{'='*20} TEST CASE {idx+1} {'='*20}")
        try:
            A_obj = Matrix(A_data)
            P, D = A_obj.diagonalize()
            
            print(f"Ma trận đầu vào A:\n{np.array(A_data)}")
            print(f"Ma trận P (Vecto riêng):\n{P}")
            print(f"Ma trận D (Trị riêng):\n{D}")

            # Kiểm tra logic: A * P = P * D
            A_np = np.array(A_data, dtype=float)
            P_np = np.array(P.data, dtype=float)
            D_np = np.array(D.data, dtype=float)
            
            AP = A_np.dot(P_np)
            PD = P_np.dot(D_np)

            if np.allclose(AP, PD, atol=1e-5):
                print(">>> Kết quả: CHÍNH XÁC! (A*P = P*D)")
            else:
                print(">>> Kết quả: SAI LỆCH!")
                
        except Exception as e:
            print(f"Có lỗi xảy ra tại Test Case {idx+1}: {e}")