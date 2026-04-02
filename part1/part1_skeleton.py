import determinant
import gauss

class Matrix:
	def __str__(self):
		return "\n".join(map(str, self.data))

	def shape(self):
		return (self.rows, self.cols)

	""" START CONSTRUCTORS """
	def __init__(self, data: list[list[float]]):
		self.data = data
		self.num_row = len(data)
		self.num_col = len(data[0])

	@classmethod
	def diag(cls, data: list[float]):
		pass

	@classmethod
	def identity(cls, n: int):
		pass
	""" END CONSTRUCTORS """

	""" START ATTRIBUTE CHECKING """
	def is_ref(self):
		# return True `self` is REF
		pass
	def is_diagonal(self):
		pass
	def is_identity(self):
		pass
	""" END ATTRIBUTE CHECKING """

	def det(self):
		return determinant.determinant(self)

	def gaussian_eliminate(self):
		# return (REF matrix of self, solution, number of steps)
		return gauss.gaussian_eliminate(self.data)
	
	def gauss_jordan_eliminate(self):
		# return RREF matrix of self
		pass

	def back_subtitution(self):
		# return solution of [REF matrix | b']
		if self.is_ref():
			return gauss.back_subtitution(self.data, self.num_row)
	
	def inverse(self):
		# return self^{-1}
		pass

	def rank_and_basis(sefl):
		# return (rank, ([basis of C(self)], [basis of R(self)], [basis of N(self)]))
		pass