"""
models package — shared classes for serializable ML wrappers.

ForecastWrapper must live here (not in train_models.py) so that joblib
pickle references resolve to "models.ForecastWrapper" regardless of which
script calls joblib.dump or joblib.load.
"""


class ForecastWrapper:
    """
    Uniform interface over pmdarima and statsmodels ARIMA fitted models.
    Provides forecast(n) so the finance agent doesn't need to know which
    backend was used.
    """
    def __init__(self, backend, order: tuple):
        self._backend = backend
        self.order    = order

    def forecast(self, n_periods: int = 1):
        """Predict the next n_periods values as a numpy array."""
        try:
            return self._backend.predict(n_periods=n_periods)   # pmdarima API
        except TypeError:
            return self._backend.forecast(n_periods)            # statsmodels API

    def predict(self, n_periods: int = 1):
        return self.forecast(n_periods)
