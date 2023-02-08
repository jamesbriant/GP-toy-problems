import numpy as np
from scipy.stats import norm, beta, gamma


# For now, choose h1() = h2() = (1)
# implying beta1 and beta2 are unknown constants
# representing the mean of eta and delta.

class Model:
    """Class representing a model based on the Kennedy & O'Hagan (2001) framework.
    
    Model
    -----
    y_f(x) = rho*eta(x,theta) + delta(x) + e(x)
    """

    def __init__(self, xc, t, y, xf, z):
        """Initiate model with n regression basis vectors

        """

        # delete this line once the cholesky decomposition is fixed
        self.rank_errors = 0

        self.params = {
            'beta1': np.array([0]),
            'beta2': np.array([0]),
            'theta': np.array([1]), #1,
            'rho': np.array([1])#1
        }
        self.hyperparams = {
            'l_c1_x': np.array([0.25, 0.25]),
            'l_c1_t': np.array([0.25]),
            'sigma_c1': np.array([1]), #1,
            'l_c2_x': np.array([0.1, 0.1]),
            'sigma_c2': np.array([1]), #1,
            'lambda': np.array([1]) #1
        }

        ###### Model inputs
        ### Simulator variables
        # Variable inputs
        self.xc = xc
        # Calibration inputs
        self.t = t
        # Simulator data
        self.y = y
        ### Field data
        # Measurement locations
        self.xf = xf
        # observations
        self.z = z

        # data vector
        self.d = np.hstack([self.y, self.z])

        # dimensions
        self.y_n = self.y.shape[0]
        self.z_n = self.z.shape[0]

        # DO NOT DELETE THESE LINES
        self.m_d = None
        self.V_d = None
        self.V_d_chol = None
        self.logpost = np.array([None]*11)

        self.H1D1_eval()
        self.H1D2_eval()
        self.H2D2_eval()
        self.H_eval()
        self.m_d_eval()

        self.V1D1_eval()
        self.V1D2_eval()
        self.C1D1D2_eval()
        self.V2D2_eval()
        self.V_d_eval()

        self.calculate_logpost()


    def h1_eval(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, c: np.ndarray):
        """Hermite polynonial basis function for regression over emulator."""
        
        return np.polynomial.hermite_e.hermeval3d(x, y, theta, c)


    def h2_eval(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, c: np.ndarray):
        """Hermite polynonial basis function for regression over discrepancy."""
        
        return np.polynomial.hermite_e.hermeval3d(x, y, theta, c)


    def H1D1_eval(self):
        self.H1D1 = np.ones((self.y_n, 1))


    def H1D2_eval(self):
        self.H1D2 = np.ones((self.z_n, 1))


    def H2D2_eval(self):
        self.H2D2 = np.ones((self.z_n, 1))


    def H_eval(self):
        self.H = np.zeros((self.y_n + self.z_n, 2))

        self.H[0:self.y_n, :1] = self.H1D1
        self.H[self.y_n:, :1] = self.params['rho']*self.H1D2
        self.H[self.y_n:, :2] = self.H2D2

    
    def m_d_eval(self):
        self.m_d = np.dot(self.H, np.hstack([self.params['beta1'], self.params['beta2']]))


    def r1_eval(self, x1: np.ndarray, x2: np.ndarray, t1: np.ndarray = None, t2: np.ndarray = None):
        """Squared exponential kernel."""
        
        lx = self.hyperparams['l_c1_x']
        lt = self.hyperparams['l_c1_t']

        assert lx.shape[0] == x2.shape[1], "length scale parameters not equal to number of variable parameter dimensions."

        dx = np.abs(x1 - x2)**2
        if t1 is None and t2 is None:
            dt = np.zeros((x2.shape[0], 1))
        else:
            dt = np.abs(t1 - t2)**2

        return np.exp(np.sum(-dx/(2*lx**2), axis=1) - np.ravel(dt/(2*lt**2)))


    def r1_periodic_eval(self, x1: np.ndarray, x2: np.ndarray, t1: np.ndarray = None, t2: np.ndarray = None):
        """Periodic kernel."""
        
        lx = self.hyperparams['l_c1_x']
        lt = self.hyperparams['l_c1_t']

        assert lx.shape[0] == x2.shape[1], "length scale parameters not equal to number of variable parameter dimensions."

        dx = np.abs(x1 - x2)**2
        if t1 is None and t2 is None:
            dt = np.zeros((x2.shape[0], 1))
        else:
            dt = np.abs(t1 - t2)**2

        return np.exp(np.sum(-dx/(2*lx**2), axis=1) - np.ravel(dt/(2*lt**2)))


    def r2_eval(self, x1: np.ndarray, x2: np.ndarray):
        """Squared exponential kernel."""

        lx = self.hyperparams['l_c2_x']

        assert lx.shape[0] == x2.shape[1], "length scale parameters not equal to number of variable parameter dimensions."

        dx = np.abs(x1 - x2)**2

        return np.exp(np.sum(-dx/(2*lx**2), axis=1))


    def V1D1_eval(self):
        V1D1 = np.zeros((self.y_n, self.y_n))
        for i in range(self.y_n - 1):
            temp = self.r1_eval(
                self.xc[i, :].reshape(1, -1), 
                self.xc[(i+1):, :], 
                self.t[i, :].reshape(1, -1), 
                self.t[(i+1):, :]
            )
            with open('temp.txt', 'a') as f:
                f.write(str(temp))
            V1D1[i, (i+1):] = temp
        V1D1 += V1D1.T
        self.V1D1_unscaled = V1D1 + np.identity(self.y_n)
        self.V1D1_scale()

    
    def V1D1_scale(self):
        self.V1D1 = self.hyperparams['sigma_c1'] * self.V1D1_unscaled

    
    def V1D2_eval(self):
        V1D2 = np.zeros((self.z_n, self.z_n))
        for i in range(self.z_n - 1):
            V1D2[i, (i+1):] = self.r1_eval(
                self.xf[i, :].reshape(1, -1), 
                self.xf[(i+1):, :]
            )
        V1D2 += V1D2.T
        self.V1D2_unscaled = V1D2 + np.identity(self.z_n)
        self.V1D2_scale()


    def V1D2_scale(self):
        self.V1D2 = self.hyperparams['sigma_c1'] * self.V1D2_unscaled


    def C1D1D2_eval(self):
        C1D1D2 = np.zeros((self.z_n, self.y_n))
        theta = self.params['theta']
        for i in range(self.z_n):
            C1D1D2[i, :] = self.r1_eval(
                self.xf[i, :].reshape(1, -1),
                self.xc[:, :],
                np.array([theta]).reshape(1, -1),
                self.t[:, :]
            )
        self.C1D1D2_unscaled = C1D1D2
        self.C1D1D2_scale()

    
    def C1D1D2_scale(self):
        self.C1D1D2 = self.hyperparams['sigma_c1'] * self.C1D1D2_unscaled


    def V2D2_eval(self):
        V2D2 = np.zeros((self.z_n, self.z_n))
        for i in range(self.z_n - 1):
            V2D2[i, (i+1):] = self.r2_eval(
                self.xf[i, :].reshape(1, -1), 
                self.xf[(i+1):, :]
            )
        V2D2 += V2D2.T
        self.V2D2_unscaled = V2D2 + np.identity(self.z_n)
        self.V2D2_scale()

    
    def V2D2_scale(self):
        self.V2D2 = self.hyperparams['sigma_c2'] * self.V2D2_unscaled

    
    def V_d_eval(self):
        a = self.y_n
        b = self.z_n

        self.V_d = np.zeros((a+b, a+b))

        self.V_d[0:a, 0:a] = self.V1D1
        self.V_d[a:, 0:a] = self.params['rho']*self.C1D1D2
        self.V_d[0:a, a:] = np.transpose(self.V_d[a:, 0:a])
        self.V_d[a:, a:] = self.V2D2 + self.V1D2*self.params['rho']**2 + self.hyperparams['lambda']*np.identity(b)

        # rank = np.linalg.matrix_rank(self.V_d)
        # if rank < self.y_n + self.z_n:
        #     # print(rank)
        #     self.rank_errors += 1
        self.V_d_chol = np.linalg.cholesky(self.V_d)


    def loglike(self) -> float:
        if self.m_d is None:
            self.m_d_eval()
        if self.V_d_chol is None:
            self.V_d_eval()
        u = np.linalg.solve(self.V_d_chol, self.d - self.m_d)
        Q = np.dot(u, u.T)
        logdet = np.sum(np.log(np.diag(self.V_d_chol)))
        return -logdet - 0.5*Q
        

    def prior_beta_1(self) -> float:
        # Gaussian(0.5, 10)
        mu = [0.5]
        var = np.diag([10])
        return norm.logpdf(self.params['beta1'], loc=mu, scale=var)


    def prior_beta_2(self) -> float:
        # Gaussian(0, 100)
        mu = [0]
        var = np.diag([100])
        return norm.logpdf(self.params['beta2'], loc=mu, scale=var)

    
    def prior_theta(self) -> float:
        # Gaussian(0.5, 3)
        mu = [0.5]
        var = np.diag([3])
        return norm.logpdf(self.params['theta'], loc=mu, scale=var)

    
    def prior_rho(self) -> float:
        # Gaussian(1, 1)
        mu = [1]
        var = np.diag([1])
        return norm.logpdf(self.params['rho'], loc=mu, scale=var)


    def prior_l_c1_x(self) -> float:
        # Beta(2,3)
        return np.sum(beta.logpdf(self.hyperparams['l_c1_x'], 2, 3, loc=0, scale=1))


    def prior_l_c1_t(self) -> float:
        # Beta(2,3)
        return beta.logpdf(self.hyperparams['l_c1_t'], 2, 3, loc=0, scale=1)


    def prior_l_c2_x(self) -> float:
        # Beta(2,3)
        return np.sum(beta.logpdf(self.hyperparams['l_c2_x'], 2, 3, loc=0, scale=1))

    
    def prior_sigma_c1(self) -> float:
        # Gamma(2, 2)
        return gamma.logpdf(self.hyperparams['sigma_c1'], a=2, scale=2)


    def prior_sigma_c2(self) -> float:
        # Gamma(2, 2)
        return gamma.logpdf(self.hyperparams['sigma_c2'], a=2, scale=2)


    def prior_lambda(self) -> float:
        # Gamma(0.5, 1)
        return gamma.logpdf(self.hyperparams['lambda'], a=0.5, scale=1)


    def calculate_logpost(self):
        self.logpost[0] = self.loglike()
        self.logpost[1] = self.prior_beta_1()
        self.logpost[2] = self.prior_beta_2()
        self.logpost[3] = self.prior_theta()
        self.logpost[4] = self.prior_rho()
        self.logpost[5] = self.prior_l_c1_x()
        self.logpost[6] = self.prior_l_c1_t()
        self.logpost[7] = self.prior_l_c2_x()
        self.logpost[8] = self.prior_sigma_c1()
        self.logpost[9] = self.prior_sigma_c2()
        self.logpost[10] = self.prior_lambda()
    

    def get_logpost(self) -> float:
        # print(self.logpost)
        return np.sum(self.logpost)[0][0]

    
    def get_model_params(self):
        keys = []
        items = []
        for key, item in self.params.items():
            if isinstance(item, np.ndarray):
                for element in item.tolist():
                    items.append(element)
                    keys.append(key)
            else:
                keys.append(key)
                items.append(item[0])
        for key, item in self.hyperparams.items():
            if isinstance(item, np.ndarray):
                for element in item.tolist():
                    items.append(element)
                    keys.append(key)
            else:
                keys.append(key)
                items.append(item[0])

        return {'parameters': keys, 'values': items}
