import numpy as np

def gaussian_eliminate(A, b = None):
    if b is None:
        b = [0] * len(A)

    n = len(A) # Take the number of A's rows
    m = len(A[0])
    count = 0

    A_bar = [row[:] + [val] for row, val in zip(A, b)] # Aumented A_bar (A|b)

    # Forward elimination
    for i in range(n):

        # Find pivot (max row) among rows
        max_row = i
        for k in range(i+ 1, n):
            if abs(A_bar[k][i]) > abs(A_bar[max_row][i]):
                max_row = k

        A_bar[i], A_bar[max_row] = A_bar[max_row], A_bar[i] # swap i_th row and pivot 
        if (max_row != i): # count swap
            count += 1

        if abs(A_bar[i][i]) < 1e-10: # max_row = 0 <=> rank A_bar < n
            return "Vo so nghiem"
        
        # elimination below current max row
        for k in range(i+ 1, n): 
            factor = A_bar[k][i] / A_bar[i][i]
            for j in range(i, n + 1):
                A_bar[k][j] -= factor * A_bar[i][j]

    x = back_subtitution(A_bar, n)
    A = [row[:m] for row in A_bar]
    return A, x, count


# back substition = calculate from last row to first row 
# because the last row has the least variable (1 variable) 
# then use the available variables to calculate the rest
def back_subtitution(A_bar, n):
    x = [0] * n
    for i in range(n - 1, -1, -1):
        total = sum(A_bar[i][j] * x[j] for j in range(i + 1, n)) 
        x[i] = round((A_bar[i][n] - total) / A_bar[i][i], 3) # round result

    return x

def verify_solution(A, x, b):
   return np.allclose(np.matmul(A, x) , b)



                     
# Test
A = [[3, 2, -4], 
     [2, 3, 3], 
     [5, -3, 1]]
b = [3, 15, 14]

result = gaussian_eliminate(A, b)
A_bar, x, count = result
print("NGHIỆM CỦA HỆ PHƯƠNG TRÌNH:", x)
if (verify_solution(A, x, b)):
    print("PHƯƠNG PHÁP TÍNH ĐÚNG")





    


