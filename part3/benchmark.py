import random

# Phần 3.2

# Phần 3.3
def create_hilbert_matrix(n):
	# Tạo ma trận Hilbert cấp n: H_ij = 1 / (i + j + 1)
	data = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
	return data # Trả về list of lists để tương thích với class Matrix

def create_random_spd_matrix(n):
	# Tạo ma trận đối xứng xác định dương (SPD) ngẫu nhiên [cite: 245]
	# Bước 1: Tạo ma trận A ngẫu nhiên
	a_matrix = [[random.uniform(0, 1) for _ in range(n)] for _ in range(n)]
	
	# Bước 2: S = A^T * A (Đảm bảo đối xứng và xác định dương)
	# Bạn có thể viết một hàm nhân ma trận đơn giản ở đây
	spd_data = [[0.0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(n):
			for k in range(n):
				spd_data[i][j] += a_matrix[k][i] * a_matrix[k][j]
				
	# Bước 3: Cộng thêm n vào đường chéo chính để tăng tính chéo trội (tùy chọn)
	for i in range(n):
		spd_data[i][i] += n
		
	return spd_data