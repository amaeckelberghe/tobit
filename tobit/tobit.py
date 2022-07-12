import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
import scipy.stats
from sklearn.linear_model import LinearRegression


def tobit_neg_log_likelihood(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]

    to_cat = []

    cens = False
    if y_left is not None:
        cens = True
        left = (y_left - np.dot(x_left, b))
        to_cat.append(left)
    if y_right is not None:
        cens = True
        right = (np.dot(x_right, b) - y_right)
        to_cat.append(right)
    if cens:
        concat_stats = np.concatenate(to_cat, axis=0) / s
        log_cum_norm = scipy.stats.norm.logcdf(concat_stats)
        cens_sum = log_cum_norm.sum()
    else:
        cens_sum = 0

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        mid = scipy.stats.norm.logpdf(mid_stats) - math.log(max(np.finfo('float').resolution, s))
        mid_sum = mid.sum()
    else:
        mid_sum = 0

    loglik = cens_sum + mid_sum

    return - loglik


def tobit_neg_log_likelihood_der(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]

    beta_jac = np.zeros(len(b))
    sigma_jac = 0

    if y_left is not None:
        left_stats = (y_left - np.dot(x_left, b)) / s
        l_pdf = scipy.stats.norm.logpdf(left_stats)
        l_cdf = log_ndtr(left_stats)
        left_frac = np.exp(l_pdf - l_cdf)
        beta_left = np.dot(left_frac, x_left / s)
        beta_jac -= beta_left

        left_sigma = np.dot(left_frac, left_stats)
        sigma_jac -= left_sigma

    if y_right is not None:
        right_stats = (np.dot(x_right, b) - y_right) / s
        r_pdf = scipy.stats.norm.logpdf(right_stats)
        r_cdf = log_ndtr(right_stats)
        right_frac = np.exp(r_pdf - r_cdf)
        beta_right = np.dot(right_frac, x_right / s)
        beta_jac += beta_right

        right_sigma = np.dot(right_frac, right_stats)
        sigma_jac -= right_sigma

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        beta_mid = np.dot(mid_stats, x_mid / s)
        beta_jac += beta_mid

        mid_sigma = (np.square(mid_stats) - 1).sum()
        sigma_jac += mid_sigma

    combo_jac = np.append(beta_jac, sigma_jac / s)  # by chain rule, since the expression above is dloglik/dlogsigma

    return -combo_jac


class TobitRegression:
    def __init__(self, lower_censoring: float = None, upper_censoring: float = None, fit_intercept: bool = True):
        if lower_censoring is None and upper_censoring is None:
            raise ValueError("Both lower_censoring and upper_censoring are None. "
                             "Either form of censoring is expected in tobit regressions.")
        self.lower_censoring = lower_censoring
        self.upper_censoring = upper_censoring
        self.fit_intercept = fit_intercept
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None
        self.params = None

    def fit(self, X, y, verbose=False):
        """
        Fit a maximum-likelihood Tobit regression
        :param X: Pandas DataFrame (n_samples, n_features): Data
        :param y: Pandas Series (n_samples,): Target
        :param cens: Pandas Series (n_samples,): -1 indicates left-censored samples, 0 for uncensored, 1 for right-censored
        :param verbose: boolean, show info from minimization
        :return:
        """
        x_copy = X.copy()
        if self.fit_intercept:
            x_copy.insert(0, 'intercept', 1.0)
        else:
            x_copy = x_copy - x_copy.mean()  # Rescale by demeaning in case no intercept is fitted.
        init_reg = LinearRegression(fit_intercept=False).fit(x_copy, y)
        b0 = init_reg.coef_
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)
        xs, ys = self._split_left_right_censored(x_copy, y)

        result = minimize(lambda params: tobit_neg_log_likelihood(xs, ys, params), params0, method='BFGS',
                          jac=lambda params: tobit_neg_log_likelihood_der(xs, ys, params), options={'disp': verbose})
        if verbose:
            print(result)
        self.ols_coef_ = b0[1:]
        self.ols_intercept = b0[0]
        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:-1]
        else:
            self.coef_ = result.x[:-1]
            self.intercept_ = 0
        self.sigma_ = result.x[-1]
        self.params = {'intercept': self.intercept_} if self.fit_intercept else {}
        self.params.update({var: coef for var, coef in zip(X.columns, self.coef_)})
        return self

    def _define_censoring(self, y):
        censored = pd.Series(np.zeros((len(y),)))
        if self.lower_censoring is not None:
            censored[y == self.lower_censoring] = -1
        if self.upper_censoring is not None:
            censored[y == self.upper_censoring] = 1
        return censored

    def _split_left_right_censored(self, x, y):
        cens = self._define_censoring(y)
        counts = cens.value_counts()
        if -1 not in counts and 1 not in counts:
            warnings.warn("No censored observations; use regression methods for uncensored data")
        xs = []
        ys = []

        for value in [-1, 0, 1]:
            if value in counts:
                split = cens == value
                y_split = np.squeeze(y[split].values)
                x_split = x[split].values

            else:
                y_split, x_split = None, None
            xs.append(x_split)
            ys.append(y_split)
        return xs, ys

    def predict(self, X):
        return self.intercept_ + np.dot(X, self.coef_)
