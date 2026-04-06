from part1.part1_skeleton import Matrix
import numpy as np

# Định nghĩa hàm find_eigenvalues
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
Matrix.find_eigenvalues = find_eigenvalues()
    
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
        # Hàm này bạn sẽ phải code trong yêu cầu rank_and_basis() của phần 1
        basis_vectors = M.get_null_space_basis() 
        
        for v in basis_vectors:
            eigenvectors_matrix.append(v)
            diagonal_elements.append(lam)

    # Bước 3: Kiểm tra điều kiện chéo hóa
    if len(eigenvectors_matrix) < self.num_row:
        raise ValueError("Ma trận bị khiếm khuyết, không đủ vector riêng độc lập để chéo hóa.")

    # Bước 4: Lắp ráp P và D
    # Xoay eigenvectors_matrix thành các cột để tạo P
    P_data = [[eigenvectors_matrix[j][i] for j in range(self.num_col)] for i in range(self.num_row)]
    P = Matrix(P_data)

    # Đặt diagonal_elements lên đường chéo tạo D
    D_data = [[diagonal_elements[i] if i == j else 0 for j in range(self.num_col)] for i in range(self.num_row)]
    D = Matrix(D_data)

    return P, D

# Gắn hàm diagonalize vào lớp Matrix
Matrix.diagonalize = diagonalize()

#Kiểm tra bằng numPy
import numpy as np
A = [[4, -2], 
     [1, 1]]
P, D = Matrix(A).diagonalize()
print("P:")
P.print()
print("D:")
D.print()
# Kiểm tra bằng NumPy
# Chuyển A sang dạng numpy array để tính toán
A_np = np.array(A)
# Tính giá trị riêng và vector riêng bằng NumPy
eigenvalues, eigenvectors = np.linalg.eig(A_np)
print("Eigenvalues (NumPy):", eigenvalues)
print("Eigenvectors (NumPy):\n", eigenvectors)
# So sánh với kết quả từ hàm diagonalize
if np.allclose(np.diag(D.data), eigenvalues) and np.allclose(P.data, eigenvectors):
    print("Kết quả từ diagonalize khớp với NumPy.")     
else:
        print("Kết quả từ diagonalize không khớp với NumPy.")
        