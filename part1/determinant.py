from part1.gauss import gaussian_eliminate

def determinant(A):
	# check 
	if (len(A) != len(A[0])):
		return None
	
	A, _ , s = gaussian_eliminate(A) 

	n = len(A)
	res = 1.0

	for i in range(n):
		res *= A[i][i]
	res *= (-1)**s
	return round(float(res), 8) + 0.0

