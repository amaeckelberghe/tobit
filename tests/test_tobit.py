import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append('../tobit')
from tobit import TobitRegression


@pytest.fixture(scope="module")
def setup():
    data = pd.read_csv(os.path.join(os.getcwd(), 'tests', 'tobit_data.txt'), sep=" ")
    data.loc[data.gender == 'male', 'gender'] = 1
    data.loc[data.gender == 'female', 'gender'] = 0
    data.loc[data.children == 'yes', 'children'] = 1
    data.loc[data.children == 'no', 'children'] = 0
    data = data.astype(float)
    y = data.affairs
    x = data.drop(['affairs', 'gender', 'education', 'children'], axis=1)
    return y, x


def test_tobit(setup):
    y, x = setup
    tr = TobitRegression(lower_censoring=0, fit_intercept=True)
    tr = tr.fit(X=x, y=y, verbose=False)

    actual = tr.coef_
    desired = [-0.17933256, 0.55414179, -1.68622027, 0.32605329, -2.2849727]
    np.testing.assert_almost_equal(actual, desired)


def test_tobit_no_intercept(setup):
    y, x = setup
    tr = TobitRegression(lower_censoring=0, fit_intercept=False)
    tr = tr.fit(X=x, y=y, verbose=False)

    actual = tr.coef_
    desired = [-0.1117397, 0.3457717, -1.0394346, 0.2146717, -1.4844]
    np.testing.assert_almost_equal(actual, desired)


if __name__ == '__main__':
    pytest.main()
