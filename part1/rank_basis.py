from utils import utils
from .part1_skeleton import Matrix, Vector

def rank_and_basis(mat) -> tuple[int, tuple[list[Vector], list[Vector], list[Vector]]]:
	# return (rank, ([basis of C(self)], [basis of R(self)], [basis of N(self)]))
	col_space = []
	row_space = []
	null_space = []

	""" calculate rank and find basis for row_space """
	rref = mat.gauss_jordan_eliminate()
	data = rref.data

	rank = 0
	for row in data:
		# find rows that is not 0 -> them rows will have the pivot value 1
		if 1 in row:
			row_space.append(Vector(row[:], is_column=False))
			rank += 1
	
	""" find basis for col_space """
	pivot_row = 0
	for j in range(mat.num_col):
		# find pivot columns
		if pivot_row in range(mat.num_row) and data[pivot_row][j] == 1:
			col_space.append(Vector([row[j] for row in mat.data]))
			pivot_row += 1

	""" find basis of null_space """
	free_vars = []
	pivot_row = 0
	# find free variables, x_i is a free variable if i-th column is not a pivot column
	for j in range(mat.num_col):
		if pivot_row in range(mat.num_row) and data[pivot_row][j] == 1:
			pivot_row += 1
			continue
		free_vars.append(j)
	free_vars.sort(reverse=True)
	# mimic back substitution;
	# for each basis vector, choose one free variable and let its value = 1, others' value = 0
	for free_var in free_vars:
		basis_vec_components = [0] * mat.num_col
		basis_vec_components[free_var] = 1
		for i in range(mat.num_row - 1, -1, -1):
			current_row = data[i]
			for j in range(mat.num_col):
				if data[i][j] == 1:
					for k in range(j + 1, mat.num_col):
						basis_vec_components[j] -= data[i][k]*basis_vec_components[k]
					break

		# add each basis vector corresponding to each free var into null space
		null_space.append(Vector(basis_vec_components))

	return (rank, (col_space, row_space, null_space))