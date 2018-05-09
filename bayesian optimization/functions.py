import numpy as np


class F1():
	def pred(self, x):
		return 40.0 * np.sin(x / 1.0) - np.power(0.3 * (x + 6.0), 2)\
			- np.power(0.2 * (x - 4.0), 2) - 1.0 * np.abs(x + 2.0) + np.random.normal(0, 1, 1)


class F2():
	def __init__(self):
		pass

	def pred(self, x):
		return np.log(1 + np.abs(x) ** (2 + np.sin(x)))
