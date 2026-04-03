from gauss import gaussian_eliminate

def determinant(A):
	A_bar, _ , s = gaussian_eliminate(A) # Take augmented A_bar

	n = len(A_bar)
	res = 1

	for i in range(n):
		res *= A_bar[i][i]
	res *= (-1)**s
	return round(res, 2)

# Test
	A = [
		[3, 2, -4], 
		[2, 3, 3], 
		[5, -3, 1]
	]
print(determinant(A))

