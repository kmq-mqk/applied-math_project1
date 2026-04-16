import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
	sys.path.append(parent_dir)


from part1.determinant import *
from part1.gauss import *
from part1.inverse import *

if __name__ == "__main__":
    # Test gauss
    '''A = [
        [1, 0, 0], 
        [0, 0, 3], 
        [0, 0, 1],
    ]
    b = [3, 15, 14]

    result = gaussian_eliminate(A, b)
    A_bar, x, count = result
    print("NGHIỆM CỦA HỆ PHƯƠNG TRÌNH:", x)
    if (verify_solution(A, x, b)):
        print("PHƯƠNG PHÁP TÍNH ĐÚNG")'''

    # Test det
    '''A = [
        [1, 0, 0], 
        [0, 0, 3], 
        [0, 0, 1],
    ]

    if determinant is not None:
        print(determinant(A))
    '''

    # Test inverse
    '''  m1 = Matrix([[4, 7], [2, 6]])
    r1 = inverse(m1)
    if r1:
        print([[round(v, 4) for v in row] for row in r1.data])
    else:
        print(None)

    m2 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    r2 = inverse(m2)
    print(r2)'''


