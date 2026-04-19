from part1 import utils
from part1.part1_skeleton import Matrix

def gauss_jordan_eliminate(A) -> Matrix:
	"""
	Khu Gauss-Jordan co chon phan tu chot (Partial Pivoting)
	"""
	ret = Matrix(A.data)
	mat = ret.data
	m = ret.num_row
	n = ret.num_col

	target_row = 0
	# buoc khu $k \in [0, n)$
	for k in range(n):
		if target_row >= m:
			break

		# Find desired `p` (max row)
		p = target_row
		# iterrate `i`: `target_row <= i < m` to find max row
		for i in range(target_row, m):
			if abs(mat[p][k]) < abs(mat[i][k]):
				p = i	# update max row
		# swap `r_k` <-> `r_p`
		mat[target_row], mat[p] = mat[p], mat[target_row]

		# `mat[k][k] == 0` ->> skip column `k`
		if utils.is_zero(mat[target_row][k]):	# 100% will bug later
			continue
		
		# if `pivot != 1` (`mat[target_row][k]` is `pivot`) then divide the current row by `pivot`, the `pivot`'s value after will be 1
		pivot = mat[target_row][k]
		if not utils.is_zero(pivot - 1):	
			utils.row_multiply(mat[target_row], 1/pivot)

		# eliminate values in `pivot`'s column for all rows
		for l in range(m):
			if l == target_row:	# don't have to eliminate row of pivot
				continue
			factor = mat[l][k]/mat[target_row][k]
			utils.row_add(mat[l], -factor, mat[target_row])
		target_row += 1
	return ret
		
def inverse(mat):
	"""
	Tra ve ma tran nghich dao cua `mat`
	Su dung phuong phap khu Gauss-Jordan
	"""
	if mat.det() == 0:
		return None
	
	aug_mat = mat.augment()
	aug_mat = aug_mat.gauss_jordan_eliminate()
	return aug_mat.take_cols(mat.num_col, mat.num_col*2 - 1)