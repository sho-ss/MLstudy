import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm


class GPRegressor():

	def __init__(self, kernel):
		self.kernel = kernel

	def calc_kernel_matrix(self, X):
		N = len(X)
		K = np.zeros((N, N))
		for i in range(N):
			for j in range(N):
				K[i][j] = self.kernel(X[i], X[j])
		# symmetric matrix
		return np.maximum(K, K.transpose())

	def calc_kernel_sequence(self, X, x):
		N = len(X)
		seq = np.zeros(N)
		for i in range(N):
			seq[i] = self.kernel(X[i], x)
		return seq.reshape(-1, 1)

	def predict(self, X_train, Y_train, X, sigma2):
		N_train = len(X_train)
		N = len(X)
		K = self.calc_kernel_matrix(X_train)
		invmat = np.linalg.inv(sigma2 * np.eye(N_train) + K)
		mu = np.zeros(N)
		sigma2_new = np.zeros(N)
		for i in range(N):
			seq = self.calc_kernel_sequence(X_train, X[i])
			mu[i] = np.dot(seq.T, np.dot(invmat, Y_train))
			sigma2_new[i] = sigma2 + self.kernel(X[i], X[i]) - np.dot(seq.T, np.dot(invmat, seq))
		return mu, sigma2_new


def plot_gp(X, mu, sigma2, ax):
	ax.fill_between(X, mu + np.sqrt(sigma2), mu - np.sqrt(sigma2), color='skyblue')
	ax.plot(X, mu, ls="-", color="darkblue")


def gif_plot(i, ax, X_samples, Y_samples, X, sigma2, kernel, model):
	if i != 0:
		ax.cla()

	# plot true func
	ax.plot(X, true_func(X), ls='-', c='r')

	mu_new, sigma2_new = model.predict(X_samples[0: i + 1], Y_samples[0: i + 1], X, sigma2)
	plot_gp(X, mu_new, sigma2_new, ax)
	ax.scatter(X_samples[0: i + 1], Y_samples[0: i + 1])
	ax.set_ylim([-10, 10])


#############
# kernels

def kernel_r(x1, x2):
	alpha = 200
	beta = 1.0
	return alpha * np.exp(-0.5 * np.sum((x1 - x2) ** 2) / (beta ** 2))


r = lambda x1, x2: np.abs(x1 - x2)


def matern52(x1, x2):
	sigma_f = 1.0
	sigma_l = 1.0
	r_val = r(x1, x2)
	return sigma_f**2 * (1 + np.sqrt(5) * r_val / sigma_l + (5 + r_val**2) / (3 * sigma_l**2))\
		* np.exp(-np.sqrt(5) * r_val / sigma_l)


#############
# true function
true_func = lambda x: x * np.sin(x)


def test():
	x_min = 0.0
	x_max = 10.0
	X = np.linspace(x_min, x_max)
	N = len(X)
	kernel = matern52
	gp = GPRegressor(kernel=kernel)
	K = gp.calc_kernel_matrix(X)

	n_samples = 3
	tiny = 1e-5

	Y_r = [np.random.multivariate_normal(np.zeros(N), (np.eye(N) * tiny + K)) for _ in range(n_samples)]

	# prepare plots
	fig, (ax1, ax2) = plt.subplots(2, 1)
	for i in range(n_samples):
		ax1.plot(X, Y_r[i], "-")

	# sampling data for update prior
	N_samples = 40
	X_samples = np.linspace(x_min + 0.5, x_max - 0.5, N_samples)
	Y_samples = true_func(X_samples)

	# plot prior
	sigma2 = 0.1
	ani = anm.FuncAnimation(fig, gif_plot, frames=N_samples, interval=300,
		fargs=(ax2, X_samples, Y_samples, X, sigma2, kernel, gp))

	ani.save("test.gif", writer='imagemagick')


if __name__ == '__main__':
	test()
