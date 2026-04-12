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
    if self.num_row != self.num_col:
        raise ValueError("Chỉ ma trận vuông mới có giá trị riêng.")
    
    A_np = np.array(self.data, dtype=float)
    eigenvalues = np.linalg.eigvals(A_np)
    
    # Bỏ phần ảo nếu quá nhỏ
    eigenvalues = np.where(np.abs(eigenvalues.imag) < 1e-10, 
                           eigenvalues.real, 
                           eigenvalues)
    
    # Làm tròn để khử sai số dấu phẩy động
    # 5 chữ số thập phân là đủ an toàn cho các bài toán đại số tuyến tính cơ bản
    eigenvalues_rounded = np.round(eigenvalues.real, 5)
    
    # Lọc lấy các giá trị riêng duy nhất
    unique_eigenvalues = np.unique(eigenvalues_rounded).tolist()
    
    return unique_eigenvalues

Matrix.find_eigenvalues = find_eigenvalues
    
def diagonalize(self):
    if self.num_row != self.num_col:
        raise ValueError("Ma trận phải là ma trận vuông để chéo hóa.")
        
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

    if len(eigenvectors_matrix) < self.num_row:
        raise ValueError("Ma trận bị khiếm khuyết, không đủ vector riêng độc lập để chéo hóa.")

    P_data = [[eigenvectors_matrix[j].data[i] for j in range(self.num_col)] for i in range(self.num_row)]
    P = Matrix(P_data)

    D_data = [[diagonal_elements[i] if i == j else 0 for j in range(self.num_col)] for i in range(self.num_row)]
    D = Matrix(D_data)

    return P, D

Matrix.diagonalize = diagonalize


# --- KIỂM TRA CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    A = [[4, -2], 
         [1, 1]]
         
    try:
        P, D = Matrix(A).diagonalize()
        print("Ma trận P:")
        print(P) 
        print("\nMa trận D:")
        print(D)  

        A_test = np.array(A)
        P_test = np.array(P.data)
        D_test = np.array(D.data)
        
        AP = A_test.dot(P_test)
        PD = P_test.dot(D_test)

        print("\n--- Kiểm tra kết quả ---")
        if np.allclose(AP, PD):
            print("Kết quả chính xác! Thỏa mãn A * P = P * D")     
        else:
            print("Kết quả chưa chính xác, A * P khác P * D")
            
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")