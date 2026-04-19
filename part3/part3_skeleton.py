import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
	sys.path.append(parent_dir)

from part1.part1_skeleton import Matrix
import part1.gauss as gauss


# 3.1 
def gaussian_solve(a: Matrix, b: list[float]) -> list[float]:
	"""
	Giải Ax = b bằng Gauss elimination
	Gọi gauss.gaussian_eliminate() rồi lấy nghiệm x.

	Params:
		a : Matrix hệ số n×n
		b : vector vế phải

	Returns:
		x : vector nghiệm (list[float])
	"""
	_, x, _ = gauss.gaussian_eliminate(a.data, b)
	return x


def lu_solve(a: Matrix, b: list[float]) -> list[float]:
	"""
	Giải Ax = b thông qua phân rã LU 

	Params:
		a : Matrix hệ số n×n
		b : vector vế phải

	Returns:
		x : vector nghiệm (list[float])
	"""
	n = a.num_row
	upper = [a[i][:] for i in range(n)]
	lower = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
	perm = list(range(n))

	for k in range(n):
		max_row = max(range(k, n), key=lambda i: abs(upper[i][k]))
		if abs(upper[max_row][k]) < 1e-12:
			raise ValueError("Ma trận suy biến")
		upper[k], upper[max_row] = upper[max_row], upper[k]
		perm[k], perm[max_row] = perm[max_row], perm[k]
		for j in range(k):
			lower[k][j], lower[max_row][j] = lower[max_row][j], lower[k][j]
		for i in range(k + 1, n):
			lower[i][k] = upper[i][k]/upper[k][k]
			for j in range(k, n):
				upper[i][j] -= lower[i][k]*upper[k][j]

	pb = [b[perm[i]] for i in range(n)]
	y = [0.0] * n
	for i in range(n):
		y[i] = (pb[i] - sum(lower[i][j]*y[j] for j in range(i)))/lower[i][i]

	x = [0.0] * n
	for i in range(n - 1, -1, -1):
		x[i] = (y[i] - sum(upper[i][j]*x[j] for j in range(i + 1, n)))/upper[i][i]
	return x


# Phương pháp lặp 
class Iterative_Solver:
	def __init__(self, matrix: Matrix, b: list[float]):
		self.matrix = matrix
		self.b = b

	def is_strictly_diagonally_dominant(self):
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
	

# Test cases
if __name__ == "__main__":
	def allclose(x, expected, tol=1e-4):
		return all(abs(x[i] - expected[i]) < tol for i in range(len(x)))
 
	def run_test(name, passed):
		print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
 
	print("=" * 45)
	print("TEST — Iterative_Solver.gauss_seidel()")
	print("=" * 45)
 
	# Test 1 — Hệ 3x3 chéo trội, nghiệm đúng = [1, 2, 3]
	a = Matrix([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]])
	b = [14.0, 18.0, 30.0]
	x, _ = Iterative_Solver(a, b).gauss_seidel(eps=1e-10)
	run_test("3x3 cơ bản — nghiệm [1, 2, 3]", allclose(x, [1.0, 2.0, 3.0]))
 
	# Test 2 — Edge case: hệ 1x1
	x, _ = Iterative_Solver(Matrix([[5.0]]), [10.0]).gauss_seidel()
	run_test("1x1 — nghiệm [2.0]", allclose(x, [2.0]))
 
	# Test 3 — Đường chéo rất lớn, hội tụ <= 5 vòng
	a = Matrix([[1000.0, 1.0, 1.0], [1.0, 1000.0, 1.0], [1.0, 1.0, 1000.0]])
	x_true = [1.0, -1.0, 2.0]
	b = [sum(a[i][j]*x_true[j] for j in range(3)) for i in range(3)]
	x, n_iter = Iterative_Solver(a, b).gauss_seidel(eps=1e-10)
	run_test("Duong cheo rat lon — hoi tu nhanh", allclose(x, x_true) and n_iter <= 5)
 
	# Test 4 — Edge case: b = [0, 0, 0], nghiem = [0, 0, 0]
	a = Matrix([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
	x, _ = Iterative_Solver(a, [0.0, 0.0, 0.0]).gauss_seidel(eps=1e-10)
	run_test("b = [0,0,0] — nghiem [0, 0, 0]", allclose(x, [0.0, 0.0, 0.0]))
 
	# Test 5 — Edge case: ma tran KHONG cheo troi → is_dd = False, khong crash
	a = Matrix([[1.0, 2.0, 3.0], [4.0, 1.0, 6.0], [7.0, 8.0, 1.0]])
	solver = Iterative_Solver(a, [1.0, 1.0, 1.0])
	run_test("Khong cheo troi — is_dd = False", not solver.is_strictly_diagonally_dominant())
 
	print("=" * 45)
 