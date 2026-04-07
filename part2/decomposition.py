import math
from part1.part1_skeleton import Matrix

def svd(self):
	"""
	Phân rã ma trận SVD: a = u * sigma * v_t
	Trả về (u, sigma, v_t) dưới dạng các đối tượng Matrix.
	"""
	# Bước 1: Tính ma trận m = a_t * a
	a_t = self.transpose() 
	m = a_t.matmul(self)
	
	# Bước 2: Chéo hóa m để tìm trị riêng và vector riêng
	p, d = m.diagonalize()
	
	# Bước 3: Trích xuất và sắp xếp các cặp (trị riêng, vector riêng)
	eigen_pairs = []
	for i in range(m.num_col):
		lam = d.data[i][i]
		v = [p.data[j][i] for j in range(p.num_row)]
		eigen_pairs.append((lam, v))
		
	# Sắp xếp giảm dần dựa vào giá trị riêng lam
	eigen_pairs.sort(key=lambda x: x[0], reverse=True)
	
	# Bước 4: Xây dựng các ma trận u, sigma và v
	u_cols = []
	v_cols = []
	sigma_vals = []
	
	for lam, v in eigen_pairs:
		if lam > 1e-10:
			sigma = math.sqrt(lam)
			sigma_vals.append(sigma)
			
			# Chuẩn hóa vector v (tạo thành một cột của ma trận trực giao v_cols)
			norm_v = math.sqrt(sum(x*x for x in v))
			v_norm = [x/norm_v for x in v]
			v_cols.append(v_norm)
			
			# Tính vector u = (a * v_norm)/sigma
			u = []
			for i in range(self.num_row):
				val = sum(self.data[i][k]*v_norm[k] for k in range(self.num_col))
				u.append(val/sigma)
			u_cols.append(u)
			
	# Bước 5: Chuyển đổi dữ liệu danh sách thành đối tượng Matrix
	v_t = Matrix(v_cols)
	
	sigma_data = [[0.0 for _ in range(self.num_col)] for _ in range(self.num_row)]
	for i in range(len(sigma_vals)):
		sigma_data[i][i] = sigma_vals[i]
	sigma = Matrix(sigma_data)
	
	u_data = [[u_cols[j][i] for j in range(len(u_cols))] for i in range(self.num_row)]
	u = Matrix(u_data)
	
	return u, sigma, v_t

# Thêm phương thức svd vào lớp Matrix
Matrix.svd = svd

#Kiểm chứng bằng numPy
import numpy as np
#test ma trân 3x4
A = Matrix([
	[1, 5, -2, 4],
	[-3, 2, 7, 1],
	[8, -1, 3, -5]
])
u, sigma, v_t = A.svd()
if np.allclose(np.matmul(u.data, np.matmul(sigma.data, v_t.data)), A.data):
        print("SVD decomposition is correct!")