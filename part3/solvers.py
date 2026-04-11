class Iterative_Solver:
	def __init__(self, matrix, b):
		self.matrix = matrix
		self.b = b

	def is_strictly_diagonally_dominant(self):
		# Kiểm tra điều kiện hội tụ: chéo trội chặt
		n = self.matrix.num_row
		for i in range(n):
			sum_row = sum(abs(self.matrix[i][j]) for j in range(n) if i != j)
			if abs(self.matrix[i][i]) <= sum_row:
				return False
		return True

	def gauss_seidel(self, eps=1e-6, max_iterations=1000):
		"""
		Giải Ax = b bằng phương pháp lặp Gauss-Seidel.

		Công thức lặp:
			x_i^{k+1} = (1/a_ii) * (b_i
			             - Σ_{j<i} a_ij * x_j^{k+1}   ← đã cập nhật
			             - Σ_{j>i} a_ij * x_j^k  )      ← chưa cập nhật

		Điều kiện dừng: max|x_i^{k+1} - x_i^k| < eps  (chuẩn vô cùng)

		Returns:
			x            : vector nghiệm
			n_iterations : số vòng lặp đã thực hiện
		"""
		if not self.is_strictly_diagonally_dominant():
			print("Warning: Matrix is not strictly diagonally dominant.")

		n = self.matrix.num_row
		x = [0.0] * n		# khởi tạo nghiệm ban đầu bằng 0

		for k in range(max_iterations):
			x_old = list(x)

			for i in range(n):
				# Σ_{j<i} a_ij * x_j^{k+1}  ← dùng x[j] đã cập nhật
				sigma_lower = sum(self.matrix[i][j]*x[j]     for j in range(i))
				# Σ_{j>i} a_ij * x_j^k       ← dùng x_old[j] chưa cập nhật
				sigma_upper = sum(self.matrix[i][j]*x_old[j] for j in range(i + 1, n))

				x[i] = (self.b[i] - sigma_lower - sigma_upper)/self.matrix[i][i]

			# kiểm tra điều kiện dừng (chuẩn vô cùng)
			diff = max(abs(x[i] - x_old[i]) for i in range(n))
			if diff < eps:
				return x, k + 1

		return x, max_iterations