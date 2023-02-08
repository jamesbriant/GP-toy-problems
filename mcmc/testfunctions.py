# The following functions were taken from the wikipedia page below.
# Most are 2D functions but some have no dimension requirements.
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np
from typing import Tuple

def arrays_to_arraymesh(*X):
    return np.vstack(list(map(np.ravel, np.meshgrid(*X)))).T # list() here avoides a depreciation warning

def arraymesh_to_arrays(arraymesh, shape: Tuple):
    return arraymesh.reshape(shape)

class base_function:
    """
    Base class outlines the common functions across the test functions
    """
    def __init__(self):
        pass

    def eval(self, x):
        """
        Args:
            x (ndarray): a numpy 2d-array where each row is an input data point and 
                each column is an input dimension. 
        """
        pass

    def get_global_minimum_location(self, dimensions: int = None):
        return self.global_minimum_location

    def get_global_minimum_value(self):
        return self.global_minimum_value

class Rastrigin(base_function):

    def __init__(self, A: float):
        self.A = A
        self.global_minimum_value = 0

    def eval(self, X):
        """
        Args:
            x (ndarray): a numpy 2d-array where each row is an input data point and 
                each column is an input dimension. 
        """
        n_data_points, n_dimensions = X.shape

        output = self.A * n_dimensions * np.ones((n_data_points,))
        for i_dimension in range(n_dimensions):
            output += X[:, i_dimension]**2 - self.A*np.cos(2*np.pi*X[:, i_dimension])

        return output

    def get_global_minimum_location(self, dimensions: int):
        assert isinstance(dimensions, int) and dimensions > 0

        return tuple([0 for i in range(dimensions)])
    

class Ackley(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (0, 0)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        output = np.e + 20*np.ones((n_data_points,))
        output += -20 * np.exp(-0.2*np.sqrt(0.5*(X[:, 0]**2 + X[:, 1]**2)))
        output += -np.exp(0.5*(np.cos(2*np.pi*X[:, 0]) + np.cos(2*np.pi*X[:, 1])))

        return output


class Sphere(base_function):
    def __init__(self):
        self.global_minimum_value = 0

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        output = np.zeros((n_data_points,))
        for i_dimension in range(n_dimensions):
            output += X[:, i_dimension]**2

        return output

    def get_global_minimum_location(self, dimensions: int):
        assert isinstance(dimensions, int) and dimensions > 0

        return tuple([0 for i in range(dimensions)])


class Rosenbrock(base_function):
    def __init__(self):
        self.global_minimum_value = 0

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions >= 2, "input dimension must be at least 2, {} provided".format(n_dimensions)

        output = np.zeros((n_data_points,))
        for i_dimension in range(n_dimensions - 1):
            X0 = X[:, i_dimension]
            X1 = X[:, i_dimension+1]
            output += 100*(X1 - X0**2)**2 
            output += (1 - X0)**2

        return output

    def get_global_minimum_location(self, dimensions: int):
        assert isinstance(dimensions, int) and dimensions > 1

        return tuple([1 for i in range(dimensions)])


class Beale(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (3, 0.5)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        output = (1.5 - X0 + X0*X1)**2
        output += (2.25 - X0 + X0*X1**2)**2
        output += (2.625 - X0 + X0*X1**3)**2

        return output


class GoldsteinPrice(base_function):
    def __init__(self):
        self.global_minimum_value = 3
        self.global_minimum_location = (0, -1)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = (X0 + X1 + 1)**2
        B = 19 - 14*X0 + 3*X0**2 - 14*X1 + 6*X0*X1 + 3*X1**2
        C = (2*X0 - 3*X1)**2
        D = 18 - 32*X0 + 12*X0**2 + 48*X1 - 36*X0*X1 + 27*X1**2

        return (1 + A*B)*(30 + C*D)


class Booth(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (1, 3)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = (X0 + 2*X1 - 7)**2
        B = (2*X0 + X1 - 5)**2

        return A + B


class Bukin6(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (-10, 1)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = 100*np.sqrt(np.abs(X1 - 0.01*X0**2))
        B = 0.01*np.abs(X0 + 10)

        return A + B


class Matyas(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (0, )

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        return 0.26*(X0**2 + X1**2) - 0.48*X0*X1


class Levi13(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (1, 1)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = np.sin(3*np.pi*X0)**2
        B = ((X0 - 1)**2)*(1 + np.sin(3*np.pi*X1)**2)
        C = ((X1 - 1)**2)*(1 + np.sin(2*np.pi*X1)**2)

        return A + B + C


class Himmelblau(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = ((3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126))

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        return (X0**2 + X1 - 11)**2 + (X0 + X1**2 - 7)**2


class ThreeHumpCamel(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (0, 0)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        return 2*X0**2 - 1.05*X0**4 + (X0**6)/6 + X0*X1 + X1**2


class Easom(base_function):
    def __init__(self):
        self.global_minimum_value = -1
        self.global_minimum_location = (np.pi, np.pi)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        return -np.cos(X0)*np.cos(X1)*np.exp(-((X0 - np.pi)**2 + (X1 - np.pi)**2))


class CrossInTray(base_function):
    def __init__(self):
        self.global_minimum_value = -2.06261
        self.global_minimum_location = ((1.34941, -1.34941), (1.34941, 1.34941), (-1.34941, 1.34941), (-1.34941, -1.34941))

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = np.sin(X0)*np.sin(X1)*np.exp(np.abs(100 - np.sqrt(X0**2 + X1**2)/np.pi))

        return -0.0001*(np.abs(A) + 1)**0.1


class Eggholder(base_function):
    def __init__(self):
        self.global_minimum_value = -959.6407
        self.global_minimum_location = (512, 404.2319)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = -(X1 + 47)*np.sin(np.sqrt(np.abs(X0/2 + X1 + 47)))
        B = -X0*np.sin(np.sqrt(np.abs(X0 - X1 - 47)))

        return A + B


class HoelderTable(base_function):
    def __init__(self):
        self.global_minimum_value = -19.2085
        self.global_minimum_location = ((8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, -9.66459))

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        return -np.abs(np.sin(X0)*np.cos(X1)(np.exp(np.abs(1 - np.sqrt(X0**2 + X1**2)/np.pi))))


class McCormick(base_function):
    def __init__(self):
        self.global_minimum_value = -1.9133
        self.global_minimum_location = (-0.54719, -1.54719)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        return np.sin(X0 + X1) + (X0 - X1)**2 - 1.5*X0 + 2.5*X1 + 1


class Schaffer2(base_function):
    def __init__(self):
        self.global_minimum_value = 0
        self.global_minimum_location = (0, 0)

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = np.sin(X0**2 - X1**2)**2 - 0.5
        B = (1 + 0.001*(X0**2 + X1**2))**2

        return 0.5 + A/B


class Schaffer4(base_function):
    def __init__(self):
        self.global_minimum_value = 0.292579
        self.global_minimum_location = ((0, 1.25313), (0, -1.25313), (1.25313, 0), (-1.25313, 0))

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        assert n_dimensions == 2, "input dimension must be 2, {} provided".format(n_dimensions)

        X0 = X[:, 0]
        X1 = X[:, 1]

        A = np.cos(np.sin(np.abs(X0**2 - X1**2)))**2 - 0.5
        B = (1 + 0.001*(X0**2 + X1**2))**2

        return 0.5 + A/B


class StyblinskiTang(base_function):
    def __init__(self):
        pass

    def eval(self, X):
        n_data_points, n_dimensions = X.shape

        output = np.zeros((n_data_points,))
        for i_dimension in range(n_dimensions):
            X0 = X[:, i_dimension]
            output += X0**4 - 16*X0**2 + 5*X0

        return output/2

    def get_global_minimum_location(self, dimensions: int):
        assert isinstance(dimensions, int) and dimensions > 1

        return tuple([-2.903534 for i in range(dimensions)])

    def get_global_minimum_value(self, dimensions: int):
        """
        This global minimum is within the interval (-39.16617n, -39.16616n).
        """
        assert isinstance(dimensions, int) and dimensions > 1

        return (-39.16617*dimensions, -39.16616*dimensions)