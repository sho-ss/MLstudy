import gaussian_process as gp
import sklearn.gaussian_process as skgp
import functions
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


true_func = functions.F2()


def plot_gp(gaussian_process, xp, yp, bounds, i, ax):
	X = np.linspace(bounds[0], bounds[1], 100)

	ax.cla()
	ax.plot(X, true_func.pred(X), ls="-", c="orange")
	mu, sigma2 = gaussian_process.predict(xp, yp, X, 0.0)
	ax.fill_between(X, mu + np.sqrt(sigma2), mu - np.sqrt(sigma2), color="aqua")
	ax.plot(X, mu, ls="-", c='k')
	ax.scatter(xp, yp, c='red')


def plot_af(model, xp, yp, bounds, greater, ax):
	X = np.linspace(bounds[0], bounds[1])
	af_vals = -1 * acquitision_func(X, model, xp, yp, yp, greater=greater)

	ax.cla()
	ax.plot(X, af_vals, ls="-", c="darkorange")


def acquitision_func(x, gaussian_process, X_train, Y_train, evaluated_point,
	greater, n_params=1):
	x = x.reshape(-1, n_params)
	mu, sigma2 = gaussian_process.predict(x, return_std=True)

	if greater:
		opt_num = np.max(evaluated_point)
	else:
		opt_num = np.min(evaluated_point)

	scaling_factor = (-1) ** (not greater)

	with np.errstate(divide='ignore'):
		Z = scaling_factor * (mu - opt_num) / sigma2
		expected_improvement = scaling_factor * (mu - opt_num) * norm.cdf(Z) + sigma2 * norm.pdf(Z)
		expected_improvement[sigma2 == 0.0] = 0.0

	return (-1) * expected_improvement


def sample_next_point(gaussian_process, X, Y, bounds, evaluated_point, greater=True, n_restart=10):
	"""

	Arguments:
	------------
		n_restart, int.
			num of restarting point that is init value in opt.
	"""
	best_af_val = 1
	x_best = None

	n_params = len(bounds)
	for init_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restart, n_params)):
		res = minimize(acquitision_func, init_point,
			bounds=bounds,
			args=(gaussian_process, X, Y, evaluated_point, greater, n_params),
			method='L-BFGS-B')

		if res.fun < best_af_val:
			best_af_val = res.fun
			x_best = res.x

	return x_best


def kernel_r(x1, x2):
	alpha = 20.0
	beta = 1.0
	return alpha * np.exp(-1 * np.sum((x1 - x2) ** 2) * 0.5 / (beta ** 2))

r = lambda x1, x2: np.abs(x1 - x2)


def matern52(x1, x2):
	sigma_f = 2.0
	sigma_l = 2.0
	r_val = r(x1, x2)
	return sigma_f**2 * (1 + np.sqrt(5) * r_val / sigma_l + (5 + r_val**2) / (3 * sigma_l**2))\
		* np.exp(-np.sqrt(5) * r_val / sigma_l)


def main(X, Y, sample_loss, alpha=1e-05, n_iter=30, bounds=(-5, 5)):
	"""

	Arguments:
	------------
		n_iter: int
			number of optimization loop.
		bounds: ndarray, float.
			element is touple, (min_value, max_value)
	"""
	x_samples = []
	y_samples = []

	kernel = skgp.kernels.Matern()
	gaussian_process = skgp.GaussianProcessRegressor(kernel=kernel,
		alpha=alpha, normalize_y=True)

	# initialize x
	x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
	x_samples.append(x0)
	y_samples.append(sample_loss(x0, X, Y))

	xp = np.array(x_samples)
	yp = np.array(y_samples)

	# minimize or maximize
	greater = True

	# prepare plot
	#fid, (ax1, ax2) = plt.subplots(2, 1)
	for i in range(n_iter):
		"""
		plot_gp(model, xp, yp, bounds, i, ax1)
		plot_af(model, xp, yp, bounds, greater=greater, ax=ax2)
		if i < 10:
			plt.savefig('./pngs/gps/test_0{}.png'.format(i))
		else:
			plt.savefig('./pngs/gps/test_{}.png'.format(i))
		"""
		gaussian_process.fit(xp, yp)
		xn = sample_next_point(gaussian_process, xp, yp, bounds, yp, greater=greater)
		if xn is None:
			print('end epoch: ', i)
			return xp, yp
		x_samples.append(xn)
		y_samples.append(sample_loss(xn, X, Y))

		xp = np.array(x_samples)
		yp = np.array(y_samples)
	return xp, yp


if __name__ == '__main__':
	main()
