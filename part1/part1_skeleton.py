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
		n = len(data)
		return cls([
			[data[i] if i == j else 0 for j in range(n)]
			for i in range(n)
		])

	@classmethod
	def identity(cls, n: int):
		return cls([
			[1 if i == j else 0 for j in range(n)]
			for i in range(n)
		])
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

	""" START : GENERATE NEW MATRIX """
	def augment(self, other = None):
		"""
		return augmented matrix [self | other]
		if other == None then let other = Identity matrix
		"""
		if self.num_row != other.num_row:
			raise ValueError("Matrices must have the same number of rows")

		n = self.num_row
		if other == None:
			other = Matrix.identity(n)

		new_data = [self.data[i] + other.data[i] for i in range(n)]
		return Matrix(new_data)

	def inverse(self):
		# return self^{-1}
		pass

	def gaussian_eliminate(self):
		# return (REF matrix of self, solution, number of steps)
		return gauss.gaussian_eliminate(self.data)
	
	def gauss_jordan_eliminate(self):
		# return RREF matrix of self
		pass
	""" END : GENERATE NEW MATRIX """

	""" START : CALCULATE ON MATRIX """
	def det(self):
		return determinant.determinant(self)

	def back_subtitution(self):
		# return solution of [REF matrix | b']
		if self.is_ref():
			return gauss.back_subtitution(self.data, self.num_row)

	def rank_and_basis(sefl):
		# return (rank, ([basis of C(self)], [basis of R(self)], [basis of N(self)]))
		pass
	""" END : CALCULATE ON MATRIX """