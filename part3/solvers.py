class Iterative_Solver:
	def __init__(self, matrix, b):
		self.matrix = matrix
		self.b = b

	def is_strictly_diagonally_dominant(self):
		# Kiểm tra điều kiện hội tụ: chéo trội chặt
		n = self.matrix.num_row
		for i in range(n):
			sum_row = sum(abs(self.matrix.data[i][j]) for j in range(n) if i != j)
			if abs(self.matrix.data[i][i]) <= sum_row:
				return False
		return True

	def gauss_seidel(self, eps = 1e-6, max_iterations = 1000):
		if not self.is_strictly_diagonally_dominant():
			print("Warning: Matrix is not strictly diagonally dominant.")
		
		n = self.matrix.num_row
		x = [0.0] * n # Khởi tạo nghiệm ban đầu bằng 0
		
		for k in range(max_iterations):
			x_old = list(x)
			for i in range(n):
				sum_val = self.b[i]
				for j in range(n):
					if i != j:
						sum_val -= self.matrix.data[i][j] * x[j]
				x[i] = sum_val / self.matrix.data[i][i]
			
			# Kiểm tra điều kiện dừng (chuẩn vô cùng)
			diff = max(abs(x[i] - x_old[i]) for i in range(n))
			if diff < eps:
				return x, k + 1
		return x, max_iterations