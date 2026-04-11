from utils import utils
from .part1_skeleton import Matrix

def rank_and_basis(mat):
	rref = mat.gauss_jordan_eliminate()
	rank = 0
	for row in rref.data:
		if 1 in row:
			rank += 1
	return rank