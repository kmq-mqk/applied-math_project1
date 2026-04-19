def is_zero(x: float) -> bool:
	return abs(x) < 1e-10

""" START : ELEMENTARY ROW OPERATIONS """
def row_multiply(row: list, k):
	"""
	row <- row * k
	"""
	n = len(row)
	for i in range(n):
		row[i] *= k
	return

def row_add(row: list, k, another: list):
	"""
	row <- row + (k * another)
	"""
	if len(row) != len(another):
		raise ValueError("Both rows must have the same length")

	n = len(row)
	for i in range(n):
		row[i] += k*another[i]
""" END   : ELEMENTARY ROW OPERATIONS """