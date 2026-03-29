class Matrix:
	def __init__(self, data: list[list[float]]):
		self.data = data
		self.num_row = len(data)
		self.num_col = len(data[0])
	
	def __str__(self):
		return "\n".join(map(str, self.data))

	def shape(self):
		return (self.rows, self.cols)