"""
model_utils.py -- houses BoundedLinearModel so it lives in a stable,
importable module path (domain.model_utils) rather than inside whichever
script happens to be run as __main__.

WHY THIS FILE EXISTS: pickle records a class's location as (module,
qualname) at pickle time. If a class is defined directly inside a script
that gets executed directly (e.g. `python train_ltwr.py`), Python treats
that script as the module "__main__", so the pickle stores the class as
living in "__main__". Any OTHER script that later unpickles that file is
itself a different "__main__" -- so pickle looks for the class in the
wrong place and raises AttributeError, even though the code is otherwise
completely correct. Keeping the class here instead means its __module__ is
always "domain.model_utils", regardless of whether train_ltwr.py or
run_experiment.py is run directly, as `-m`, or imported from elsewhere.
"""
import numpy as np


class BoundedLinearModel:
    """Minimal sklearn-compatible wrapper (.predict()) around a bounded
    least-squares fit -- a drop-in replacement for Ridge in
    pipeline/academic_retrieval.py's ltwr_model.predict(X) call."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef)
        self.intercept_ = intercept

    def predict(self, X):
        return X @ self.coef_ + self.intercept_