import scipy as sp
import time
import pdb
from scipy.stats import rv_discrete
from scipy.optimize import fmin_l_bfgs_b as bfgs
from scipy.linalg import norm
from scipy.cluster.vq import kmeans
import scipy.cluster.vq as scvq
from scipy.linalg import lstsq


class CvxFit:
    """Class representing the convex fit to measurements.


    Attributes
    ----------
    pts: N x (n+1) scipy.ndarray,
    This contains the scaled training data.

    mean_training_error:
    This contains the training error.

    Functions
    -------------
    fit ()
    evaluate ()
    test_error (points)
    get_coefficients ()

    """

    def __init__(
        self,
        X=[],
        Y=[],
        type="pwl",
        k=None,
        penalty="square",
        penalty_param=None,
        extra_param=[1, 0.001, 20],
        tol=1e-4,
        *args,
        **kwargs
    ):
        """This is the constructor for the CvxFit class.

        Args:
            X: scipy.ndarray, each row of which represents a domain point.
            Y: array_like, function values evaluated at corresponding domain points.
            type: string,
                'pwl' for piecewise linear fits
                'pwq' for piecewise quadratic fits.
           k: integer, opt., overrides extra_param[0] if provided
               Number of clusters.
           penalty: string, one of
               'square'
               'asymmetric_l1'
               'l1'
               'huber'
               'custom'.
           penalty_param: python object, according to penalty argument,
               'square': None,
               'asymmetric_l1': (wp/wn), the weights add up to 1,
               'l1': None,
               'huber': (delta) as defined in http://en.wikipedia.org/wiki/Huber_loss_function,
               'custom': python function taking array and performing elementwise
                   operations on it.
           extra_param: list, depends on type
               'pwl': [number of affine functions, regularization parameter, maxiter]
               'pwq': [number of quadratic functions, regularization parameter, maxiter].
           tol: Tolerance, opt., lower numbers give more accurate fit, but may take more time.
        """

        _ = sp.array(X)
        __ = sp.array(Y).reshape(len(Y), 1)
        self.pts = sp.hstack((_, __))
        self._orig_pts = self.pts.copy()
        self._scale()

        self._dim = self.pts.shape[1] - 1
        self._N = self.pts.shape[0]
        self._extra_param = extra_param
        if k is not None:
            self._extra_param[0] = k
        self._tol = tol
        self._penalty_param = penalty_param
        self._coeffs = None

        # assign loss function
        dict_of_loss_functions = {
            "square": self._square,
            "asymmetric_l1": self._asymmetric_l1,
            "huber": self._huber,
            "l1": self._l1,
            "custom": None,
        }
        if penalty is not "custom":
            self._loss = dict_of_loss_functions[penalty]
        else:
            self._loss = penalty_param

        # assign fitting function
        dict_of_fitting_functions = {
            "pwl": self._fit_pwl,
            "pwq": self._fit_pwq,
            "soc": self._fit_soc,
        }
        self.fit = dict_of_fitting_functions[type]

        # assign evaluate function
        dict_of_evaluate_functions = {
            "pwl": self._eval_pwl,
            "pwq": self._eval_pwq,
            "soc": self._eval_soc,
        }
        self.evaluate = dict_of_evaluate_functions[type]

    def _fit_pwl(self):
        """Fit a PWL function.

        This performs a piecewise linear fit on the
        parameters and data with which the class
        was initialized.
        """

        num_clsts = self._extra_param[0]
        lambd = self._extra_param[1]
        pts = self.pts
        _ = self._initialize_kmeans_plus_plus()
        # centroids = kmeans(pts[:, :-1], _)[0]
        # cluster_id = self._assign_initial_clusters(centroids, pts)
        # clst_id = scvq.kmeans2(pts[:, :-1], k=_, minit='matrix')[1]
        centroids = kmeans(pts[:, :-1], _)[0]
        clst_id = self._assign_initial_clsts(centroids, pts)

        # print cluster_id
        p_coeffs = sp.randn(num_clsts, self._dim + 1)
        n_coeffs = p_coeffs.copy() + 1
        aval = self._mean_sq(n_coeffs - p_coeffs)
        max_iter = self._extra_param[2]
        iter = 0
        while aval > self._tol and iter < max_iter:
            iter = iter + 1
            n_coeffs = self._pwl_coeff_update(clst_id, p_coeffs, lambd)
            clst_id = self._assign_clsts(n_coeffs)
            aval = self._mean_sq(n_coeffs - p_coeffs)
            p_coeffs = n_coeffs

        self._coeffs = n_coeffs
        self.mean_training_error = self.test_error(self._orig_pts) / (self._N + 0.0)
        return n_coeffs

    def _eval_pwl(self, pt_unscaled):
        """Evaluates the PWL fit.

        This evaluates the PWL approximation at
        given points.

        Args:
            pt: scipy.ndarray each row of which contains points.

        Returns:
            scipy.ndarray containing function values evaluated at the given
            data points.
        """

        pt = self._scale_input(pt_unscaled)
        coeffs = self._coeffs
        num_clsts = self._extra_param[0]
        _ = sp.dot(coeffs[:, :-1], pt.transpose()) + sp.tile(
            coeffs[:, -1].reshape(num_clsts, 1), pt.shape[0]
        )
        return self._std[-1] * sp.amax(_, axis=0) + self._mean[-1]

    def _fit_pwq(self):
        """Fit a PWQ function.

        This performs a piecewise quadratic fit on the
        parameters and data with which the class
        was initialized.
        """

        k = self._extra_param[0]
        rho = self._extra_param[1]
        iter_convex = self._extra_param[2]
        niter = 100
        change_limit = 1e-3
        grid_pt = self.pts[:, :-1]
        Y = self.pts[:, -1].reshape(self._N, 1)

        N = self._N
        dim = self._dim

        # Running k-means
        clst_identity = scvq.kmeans2(grid_pt, k)[1]
        Pqr = sp.zeros([k, dim * (dim + 1) / 2 + dim + 1])

        # Generating the grid for easy least squares estimation
        X = sp.ones([N, Pqr.shape[1]])
        X[:, -dim - 1 : -1] = grid_pt
        Upp = sp.triu(sp.ones([dim, dim]))
        for ind in range(N):
            temp = sp.dot(
                sp.reshape(grid_pt[ind, :], [dim, 1]),
                sp.reshape(grid_pt[ind, :], [1, dim]),
            )
            X[ind, 0 : dim * (dim + 1) / 2] = temp[Upp == 1]

        # Regularization parameter
        mat = rho * sp.diagflat(
            sp.concatenate(
                (sp.ones([dim * (dim + 1) / 2, 1]), sp.zeros([dim + 1, 1])), 0
            )
        )

        # Running the iterative algorithm
        ind = 0
        change = 1
        while ind < niter and change > change_limit:
            ind = ind + 1
            change = 0
            # Going through clusters
            for cl_ind in range(k):
                pos_ind = clst_identity == cl_ind
                if sp.sum(pos_ind) > 2:
                    _ = sp.dot(X[pos_ind, :].T, X[pos_ind, :])
                    __ = sp.dot(X[pos_ind, :].T, Y[pos_ind])
                    temp = lstsq(_ + mat, __)[0]

                    def f_obj(alpha):
                        alpha = alpha.reshape(dim * (dim + 1) / 2 + dim + 1, 1)
                        _ = sp.dot(alpha.T, sp.dot(mat, alpha))
                        _ = (
                            sp.sum(
                                self._loss(sp.dot(X[pos_ind, :], alpha) - Y[pos_ind])
                            )
                            + _
                        )
                        return _

                    id_cvx = 0
                    while id_cvx < iter_convex:
                        id_cvx = id_cvx + 1
                        res = bfgs(f_obj, temp, approx_grad=1, factr=1e12)
                        temp = res[0].reshape(len(temp), 1)

                        # Eliminating non-convexity
                        Z = sp.zeros([dim, dim])
                        Z[Upp == 1] = temp[: dim * (dim + 1) / 2, 0]
                        Z = 0.5 * (Z + Z.T)
                        [_eval, evec] = sp.linalg.eigh(Z)
                        _ = sp.diagflat(_eval * (_eval >= 0))
                        Z = sp.dot(evec, sp.dot(_, evec.T))
                        Z = 2 * Z - sp.diagflat(sp.diag(Z))
                        temp[0 : dim * (dim + 1) / 2, 0] = Z[Upp == 1]

                    # Measuring change
                    change = max(change, sp.linalg.norm(temp.T - Pqr[cl_ind, :]))
                    Pqr[cl_ind, :] = temp[:, 0]

            clst_identity = sp.argmax(sp.dot(Pqr, X.T), 0)
            self._coeffs = Pqr

        self.mean_training_error = self.test_error(self._orig_pts) / (self._N + 0.0)
        return Pqr

    def _eval_pwq(self, pt_unscaled):
        """Evaluates the PWQ fit.

        This evaluates the PWQ approximation at
        given points.

        Args:
            pt: scipy.ndarray each row of which contains points.

        Returns:
            scipy.ndarray containing function values evaluated at the given
            data points.
        """

        pt = _scale_input(pt_unscaled)
        Pqr = self._coeffs
        N = pt.shape[0]
        dim = pt.shape[1]
        X = sp.ones([N, Pqr.shape[1]])
        X[:, -dim - 1 : -1] = pt
        Upp = sp.triu(sp.ones([dim, dim]))
        for ind in range(N):
            _ = sp.reshape(pt[ind, :], [dim, 1])
            __ = sp.reshape(pt[ind, :], [1, dim])
            temp = sp.dot(_, __)
            X[ind, 0 : dim * (dim + 1) / 2] = temp[Upp == 1]
        val = sp.max(sp.dot(Pqr, X.T), axis=0)
        return self._std[-1] * val + self._mean[-1]

    def _fit_soc(self):
        """Fit a SOC (second order cone).

        This performs a second order cone fit on the
        parameters and data with which the class
        was initialized.
        """

        num_rows = self._extra_param[0]
        lambd = self._extra_param[1]
        pts = self.pts
        N = pts.shape[0]
        dim = self._dim

        def f_obj(coeffs):
            # no regularization since problem is non-convex and
            # there are no theoretical guarantees
            self._coeffs = coeffs
            _ = self._eval_soc(pts[:, :-1]) - pts[:, -1]
            return sp.sum(self._loss(_))

        _ = sp.randn(num_rows * dim + num_rows + dim + 1)
        res = bfgs(f_obj, _, approx_grad=1, factr=1e10, maxfun=100)
        self._coeffs = res[0]
        return res[0]

    def _eval_soc(self, pt):
        """Evaluates the SOC approximation.

        Args:
            pt: scipy.ndarray each row of which contains data points.

        Returns:
            scipy.ndarray containing function values evaluated at the given
            data points.
        """

        num_rows = self._extra_param[0]
        coeffs = self._coeffs
        dim = self._dim
        N = self._N
        A = coeffs[0 : num_rows * dim].reshape(num_rows, dim)
        b = coeffs[num_rows * dim : num_rows * dim + num_rows]
        e = coeffs[num_rows * dim + num_rows : num_rows * dim + dim + num_rows]
        f = coeffs[-1]
        pts_domain = pt.transpose()
        _ = sp.dot(A, pts_domain)
        __ = sp.tile(b.reshape(num_rows, 1), N)
        _ = sp.sqrt(sp.sum((_ + __) ** 2, axis=0))
        __ = sp.dot(e.reshape(1, dim), pts_domain)
        _ = _ - __ - f
        return _

    ##These are utility functions for retrieving coefficients and/or test error

    def get_coefficients(self):
        """Returns internal coefficients for the model."""
        return self._coeffs

    def test_error(self, test_data, loss_function=None):
        """Evaluates test error on the given data.

        Args:
            test_data: scipy.ndarray each row of which contains data point.
            loss_function: python function, opt., defining penalty.

        Returns:
            Total test error.
        """
        if loss_function is None:
            loss_function = self._loss
        _ = self.evaluate(test_data[:, :-1])
        return sp.sum(loss_function(_ - test_data[:, -1]))

    ##These are utility functions for pwl/pwq/soc modules

    def _rnd_wd_pwl(self, pmf):
        """This returns a sample according to pmf."""

        pmf = pmf / sp.sum(pmf)
        xk = sp.arange(self._N)
        _ = rv_discrete(name="custm", values=(xk, pmf))
        return _.rvs()

    def _update_random_weights(self, list_centrs, num_centroids, pmf):
        """This computes the distance of each point from centroid list."""

        pts = self.pts
        (N, __) = pts.shape
        (__, dim) = list_centrs.shape

        pts_domain = pts[:, :-1]
        _ = sp.tile(pts_domain, 1)
        _lc = list_centrs[num_centroids - 1, :].flatten()
        __ = sp.tile(_lc, [N, 1])
        _ = (_ - __) ** 2
        _ = sp.reshape(_, (N, dim))
        _ = sp.sum(_, axis=1)
        _ = sp.reshape(_, (N, 1))
        sq_distances = sp.minimum(pmf.reshape(N, 1), _)
        return sq_distances

    def _initialize_kmeans_plus_plus(self):
        """This returns list of centroids drawn via kmeans++."""

        num_clsts = self._extra_param[0]
        list_centrs = sp.ones((num_clsts, self._dim))
        pmf = (1e9) * sp.ones(self._N)
        for ctr in sp.arange(num_clsts):
            list_centrs[ctr, :] = self.pts[self._rnd_wd_pwl(pmf), :-1]
            pmf = self._update_random_weights(list_centrs, ctr + 1, pmf)
        return list_centrs

    def _assign_initial_clsts(self, list_centrs, pts):
        """This assigns points to clusters."""

        (N, __) = pts.shape
        num_centroids = list_centrs.shape[0]
        dim = self._dim

        # Use array operations to compute distances efficiently
        _ = pts[:, :-1]
        pts_repeated = sp.tile(_, num_centroids)
        _ = list_centrs[0:num_centroids, :].flatten()
        flat_centroid_repeated = sp.tile(_, [N, 1])
        _ = pts_repeated - flat_centroid_repeated
        _ = _**2
        _ = sp.reshape(_, (num_centroids * N, dim))
        _ = sp.sum(_, axis=1)
        _ = sp.reshape(_, (N, num_centroids))
        indices = sp.argmin(_, axis=1)
        return indices

    def _assign_clsts(self, lst_coeffs):
        """This assigns points to clusters based on current estimates.

        Args:
            lst_coeffs: k x (n+1) scipy.ndarray containing current estimates.
        Returns:
            indices: cluster identities for each point in self.pts.
        """

        pts = self.pts
        (N, __) = pts.shape
        (num_clst, _) = lst_coeffs.shape
        dim = self._dim

        # Use array operations to compute dot products efficiently
        _ = pts[:, :-1]
        _ = sp.hstack((_, sp.ones((N, 1)), -pts[:, -1].reshape(N, 1)))
        _pt = sp.tile(_, num_clst)
        _ = sp.hstack((lst_coeffs, sp.ones((num_clst, 1))))
        _flt = _[0:num_clst, :].flatten()
        _ = _pt * sp.tile(_flt, [N, 1])
        _ = sp.reshape(_, (num_clst * N, dim + 2))
        _ = sp.sum(_, axis=1)
        _ = sp.reshape(_, (N, num_clst))
        indices = sp.argmax(_, axis=1)
        return indices

    def _pwl_coeff_update(self, clst_identities, p_coeffs, lambd):
        """This updates the coefficients for pwl."""

        pts = self.pts
        (N, __) = pts.shape
        (num_clst, _) = p_coeffs.shape

        def clst_updt(ctr):
            _ = sp.where(clst_identities == ctr)[0]
            clst_pts = pts[_, :]

            def f(coeffs):
                _ = sp.dot(clst_pts[:, :-1], coeffs[:-1]) + coeffs[-1] - clst_pts[:, -1]
                _loss = 0.5 * sp.sum(self._loss(_))
                _reg = lambd / 2.0 * sp.sum(coeffs * coeffs)
                return _loss + _reg

            _ = p_coeffs[ctr, :]
            res = bfgs(f, _, fprime=None, approx_grad=1, factr=1e12, maxfun=20)
            return res[0]

        n_coeffs = sp.array([clst_updt(ctr) for ctr in range(num_clst)])
        return n_coeffs

    def _mean_sq(self, a):
        """This returns the mean square norm of a 2D scipy.ndarray."""
        return sp.sum(a * a) / (0.0 + a.shape[0] * a.shape[1])

    def _square(self, x):
        """Computes the elementwise square."""

        return x * x

    def _asymmetric_l1(self, x):
        """Computes elementwise asymmetric l1 norm."""

        _plus = sp.maximum(x, 0)
        _minus = sp.maximum(-x, 0)

        if self._penalty_param is None:
            self._penalty_param = 1.0
        wp = (self._penalty_param + 0.0) / (self._penalty_param + 1)
        wn = 1 - wp
        return wp * _plus + wn * _minus

    def _huber(self, x):
        """Computes the elementwise huber norm."""

        if self._penalty_param is None:
            self._penalty_param = 1
        delt = self._penalty_param
        _quad = 0.5 * x * x * ((sp.absolute(x) < delt) + 0.0)
        _lin = delt * (sp.absolute(x) - delt / 2.0) * ((sp.absolute(x) > delt) + 0.0)
        return _lin + _quad

    def _l1(self, x):
        """Computes the elementwise l1 norm."""
        return sp.absolute(x)

    def _scale(self):
        """Normalize training data."""

        self._mean = sp.mean(self.pts, axis=0)
        self._std = sp.std(self.pts, axis=0)
        self.pts = self.pts - sp.tile(self._mean, (self.pts.shape[0], 1))
        _ = sp.tile(self._std, (self.pts.shape[0], 1))
        self.pts = self.pts / _

    def _scale_input(self, x):
        """Normalize test data."""

        dim = x.shape[1]
        x = x - sp.tile(self._mean[:dim], (x.shape[0], 1))
        _ = sp.tile(self._std[:dim], (x.shape[0], 1))
        x = x / _
        return x


def main():
    N = 100
    dim = 4
    coeffs_actual = sp.randn(17, dim + 1)

    def f_actual(x):
        return sp.amax(
            sp.dot(coeffs_actual[:, :-1], x.reshape(dim, 1)) + coeffs_actual[:, -1]
        )

    X = sp.randn(N, dim)
    Y = sp.array([f_actual(pt) for pt in list(X)])
    # pts = sp.hstack((pts_tmp, sp.reshape(f_val, (N, 1))))

    fit_object = CvxFit(X=X, Y=Y, type="pwl", extra_param=(2, 1e-2, 10))
    fit_object.fit()


if __name__ == "__main__":
    main()
