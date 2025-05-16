import numpy as np
import matplotlib.pyplot as plt

# mean = [10.0,9.5]
# variance = [27.0, 18.0]
# gauss = lambda x, mean_x, variance_x : 2.645/(variance_x * np.sqrt(2.0*np.pi)) * np.exp(-0.5*np.square(((x-mean_x)/variance_x)))

# x_axis = np.linspace(0, 26, 27)
# y_axis = np.linspace(0, 17, 18)
# x_axis = gauss(x_axis, mean[0], variance[0])
# y_axis = gauss(y_axis, mean[1], variance[1])
# print(np.sum(x_axis))
# print(np.sum(y_axis))

# gauss_matrix = x_axis * y_axis[:, None]
# print(np.sum(gauss_matrix.reshape(-1)))
# # fig = plt.figure(figsize=(8, 6))
# # plt.imshow(gauss_matrix)
# # plt.show()

x_axis = np.linspace(0, 26, 27)
y_axis = np.linspace(0, 17, 18)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)

mean = np.array([10.0, 9.5])
variance = np.array([[18.0, 0], [0, 27.0]])

pos = np.empty(x_axis.shape + (2,))
pos[:, :, 0] = x_axis
pos[:, :, 1] = y_axis

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

gauss = multivariate_gaussian(pos, mean, variance)
# gauss.reshape(-1)
# print(np.sum(gauss))
fig = plt.figure()
plt.imshow(gauss)
plt.show()