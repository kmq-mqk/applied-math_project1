from .gauss import gaussian_eliminate # type: ignore

def determinant(A):
	A, _ , s = gaussian_eliminate(A) 

	n = len(A)
	res = 1

	for i in range(n):
		res *= A[i][i]
	res *= (-1)**s
	return round(res,2 )

# Test
'''A = [[3, 2, -4], 
     [2, 3, 3], 
     [5, -3, 1]]
print(determinant(A))'''

