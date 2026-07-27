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
        """
        Predict the next n_periods values as a numpy array.

        Real bug this guards against: statsmodels' ARIMAResults.predict()
        does NOT raise TypeError for the pmdarima-style n_periods= kwarg --
        it silently accepts it via **kwargs and falls back to its own
        default (start=None, end=None), which returns IN-SAMPLE predictions
        for the entire training range instead of a forward forecast. That
        previously went unnoticed because pmdarima (the primary backend)
        was always available; the statsmodels fallback path only actually
        ran once pmdarima failed to import, which is what surfaced this.
        Checking the returned length -- not just catching TypeError --is
        what actually catches it, regardless of why predict() misbehaved.
        """
        try:
            result = self._backend.predict(n_periods=n_periods)   # pmdarima API
            if len(result) == n_periods:
                return result
        except TypeError:
            pass
        return self._backend.forecast(n_periods)                 # statsmodels API

    def predict(self, n_periods: int = 1):
        return self.forecast(n_periods)
