from . import determinant
from . import gauss

class Matrix:
	""" START CONSTRUCTORS """
	def __init__(self, data: list[list[float]]):
		self.data = [row[:] for row in data]
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

	def __str__(self):
		return "\n".join(
			"[ " + " ".join(f"{x:.2f}" for x in row) + " ]"
			for row in self.data
		)

	def shape(self):
		return (self.rows, self.cols)

	""" START ATTRIBUTE CHECKING """
	def is_ref(self):
		# return True if `self` is REF
		pivot_col = -1
		mat = self.data
		for row in mat:
			for j in range(self.num_col):
				if row[j]:
					if j <= pivot_col:
						return False;
					pivot_col = j
					continue
		return True

	def is_diagonal(self):
		mat = self.data
		for i in range(self.num_row):
			for j in range(self.num_col):
				if i != j and mat[i][j]:
					return False
		return True
			
	def is_identity(self):
		mat = self.data
		for i in range(self.num_row):
			for j in range(self.num_col):
				if (i != j and mat[i][j]) or (i == j and mat[i][j] != 1):
					return False
		return True
	""" END ATTRIBUTE CHECKING """

	""" START : GENERATE NEW MATRIX """
	def augment(self, other = None):
		"""
		return augmented matrix [self | other]
		if other == None then let other = Identity matrix
		"""
		if other != None and self.num_row != other.num_row:
			raise ValueError("Matrices must have the same number of rows")

		n = self.num_row
		if other == None:
			other = Matrix.identity(n)

		new_data = [self.data[i] + other.data[i] for i in range(n)]
		return Matrix(new_data)

	def take_cols(self, *args):
		# case 1: list of targeted cols
		if len(args) == 1 and isistance(args[0], list):
			selected_cols = args[0]
		# case 2: <<first_col_id>>, <<last_col_id>> -> identify range [<<first_col_id>>, <<last_col_id>>]
		elif len(args) == 2:
			start_col, end_col = args
			selected_cols = list(range(start_col, end_col + 1))
		
		selected_cols.sort()
		return Matrix([[row[j] for j in selected_cols] for row in self.data])

	def inverse(self):
		# return self^{-1}
		from .inverse import inverse
		return inverse(self)

	def gaussian_eliminate(self):
		# return (REF matrix of self, solution, number of steps)
		return gauss.gaussian_eliminate(self.data)
	
	def gauss_jordan_eliminate(self):
		# return RREF matrix of self
		from .inverse import gauss_jordan_eliminate as gj
		return gj(self)
	""" END : GENERATE NEW MATRIX """

	""" START : CALCULATE ON MATRIX """
	def det(self):
		if self.num_col != self.num_row:
			raise ValueError(
				f"Determinant undefined for non-square matrix: "
				f"Shape of matrix: {self.shape}"
				)
		return determinant.determinant(self.data)

	def back_subtitution(self):
		# return solution of [REF matrix | b']
		if self.is_ref():
			return gauss.back_subtitution(self.data, self.num_row)

	def rank_and_basis(self):
		# return (rank, ([basis of C(self)], [basis of R(self)], [basis of N(self)]))
		from .rank_basis import rank_and_basis as rb
		return rb(self)
	""" END : CALCULATE ON MATRIX """

""" ASSIGN CLASS METHODS TO FUNCTIONS FROM `inverse.py` """
""" ASSIGN CLASS METHODS TO FUNCTIONS FROM `rank_basis.py` """

class Vector:
	def __init__(self, components, is_column=True):
		self.data = list(components)
		self.num_row = self.num_col = 1
		if is_column == True:
			self.num_col = len(self.data)
		else:
			self.num_row = len(self.data)

	def transpose(self):
		return Vector(self.components, is_column=not self.is_column)

	def __repr__(self):
		direction = "Column" if self.is_column else "Row"
		return f"{direction} Vector: {self.components}"