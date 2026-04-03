def gauss_jordan_eliminate(mat: Matrix):
	"""
	Khu Gauss-Jordan co chon phan tu chot (Partial Pivoting)
	"""
	m = mat.num_row
	n = mat.num_col

	# buoc khu $k \in [0, m)$
	for k in range(m):
		# Find desired `p` (max row)
		p = 0
		# iterrate `i`: `k <= i < n` to find max row
		for i in range(k, m):
			if abs(mat[p][k]) < abs(mat[i][k]):
				p = i	# update max row
		# swap `r_k` <-> `r_p`
		mat[k], mat[p] = mat[p], mat[k]

		# `mat[k][k] == 0` ->> skip column `k`
		if is_zero(mat[k][k]):
			continue
		
		# if `pivot != 1` (`mat[k][k]` is `pivot`) then divide the current row by `pivot`, the `pivot`'s value after will be 1
		pivot = mat[k][k]
		if pivot != 1:	
			row_multiply(mat[k], 1/pivot)

		# eliminate values in `pivot`'s column for rows below
		