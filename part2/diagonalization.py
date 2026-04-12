import sys
import os
import numpy as np

# Lấy đường dẫn của thư mục hiện tại (part2) và lùi lại 1 cấp ra thư mục cha (Lab1)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Thêm thư mục Lab1 vào danh sách tìm kiếm của Python để import được part1
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.part1_skeleton import Matrix

# --- ĐỊNH NGHĨA CÁC HÀM ---

def find_eigenvalues(self):
    if self.num_row != self.num_col:
        raise ValueError("Chỉ ma trận vuông mới có giá trị riêng.")
    
    A_np = np.array(self.data, dtype=float)
    eigenvalues = np.linalg.eigvals(A_np)
    
    # Làm tròn số phức nếu phần ảo rất nhỏ
    eigenvalues = np.where(np.abs(eigenvalues.imag) < 1e-10, 
                           eigenvalues.real, 
                           eigenvalues)
    
    return eigenvalues.tolist()

# Gắn hàm find_eigenvalues vào lớp Matrix
Matrix.find_eigenvalues = find_eigenvalues
    
def diagonalize(self):
    if self.num_row != self.num_col:
        raise ValueError("Ma trận phải là ma trận vuông để chéo hóa.")
        
    # Bước 1: Tìm danh sách các giá trị riêng (Eigenvalues)
    eigenvalues = self.find_eigenvalues() 
    
    eigenvectors_matrix = [] # Chứa các cột vector riêng
    diagonal_elements = []   # Chứa các lambda tương ứng
    
    # Bước 2: Tìm vector riêng cho từng giá trị riêng
    for lam in eigenvalues:
        # Tạo ma trận M = A - lam * I
        M_data = []
        for i in range(self.num_row):
            row = []
            for j in range(self.num_col):
                val = self.data[i][j] - (lam if i == j else 0)
                row.append(val)
            M_data.append(row)
            
        M = Matrix(M_data)
        
        # Gọi hàm lấy cơ sở không gian nghiệm (Null space)
        # result có dạng: (rank, (col_space, row_space, null_space))
        result = M.rank_and_basis()
        nullspace_vectors = result[1][2]
        
        # Lặp qua các vector trong Null space (Đã sửa lỗi ở đây)
        for v in nullspace_vectors:
            eigenvectors_matrix.append(v)
            diagonal_elements.append(lam)

    # Bước 3: Kiểm tra điều kiện chéo hóa
    if len(eigenvectors_matrix) < self.num_row:
        raise ValueError("Ma trận bị khiếm khuyết, không đủ vector riêng độc lập để chéo hóa.")

    # Bước 4: Lắp ráp P và D
    # Xoay eigenvectors_matrix thành các cột để tạo P
    P_data = [[eigenvectors_matrix[j].data[i] for j in range(self.num_col)] for i in range(self.num_row)]
    P = Matrix(P_data)

    # Đặt diagonal_elements lên đường chéo tạo D
    D_data = [[diagonal_elements[i] if i == j else 0 for j in range(self.num_col)] for i in range(self.num_row)]
    D = Matrix(D_data)

    return P, D

# Gắn hàm diagonalize vào lớp Matrix
Matrix.diagonalize = diagonalize


# --- KIỂM TRA CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    A = [[4, -2], 
         [1, 1]]
         
    try:
        P, D = Matrix(A).diagonalize()
        print("P:")
        P.print()
        print("D:")
        D.print()

        # Kiểm tra bằng cách nhân ma trận (A * P = P * D)
        A_test = np.array(A)
        P_test = np.array(P.data)
        D_test = np.array(D.data)
        
        AP = A_test.dot(P_test)
        PD = P_test.dot(D_test)

        # Đối chiếu kết quả
        print("\n--- Kiểm tra kết quả ---")
        if np.allclose(AP, PD):
            print("Kết quả chính xác! Thỏa mãn A * P = P * D")     
        else:
            print("Kết quả chưa chính xác, A * P khác P * D")
            
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")