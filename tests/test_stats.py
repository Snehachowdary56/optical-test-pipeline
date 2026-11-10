import pandas as pd
import pytest

from otp.stats import describe, linear_regression, yield_rate


def test_describe_basic_values():
    stats = describe(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert stats["count"] == 5
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["std"] == pytest.approx(1.5811, abs=1e-4)  # sample standard deviation


def test_regression_recovers_a_perfect_line():
    reg = linear_regression([0, 10, 20, 30], [1.0, 1.5, 2.0, 2.5])
    assert reg.slope == pytest.approx(0.05)
    assert reg.intercept == pytest.approx(1.0)
    assert reg.r_squared == pytest.approx(1.0)


def test_regression_needs_distinct_x_values():
    with pytest.raises(ValueError):
        linear_regression([25, 25, 25], [1.0, 2.0, 3.0])


def test_yield_rate():
    summary = pd.DataFrame({"passed": [True, True, True, False]})
    assert yield_rate(summary) == 0.75


def test_yield_rate_of_empty_summary_is_zero():
    assert yield_rate(pd.DataFrame({"passed": []})) == 0.0
