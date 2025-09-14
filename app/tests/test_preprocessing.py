import numpy as np
import pandas as pd
from app.preprocessing import Preprocessor

def test_preprocessor_load_and_transform():
    prep = Preprocessor()
    cols = prep.features_order
    X = pd.DataFrame([[1.0]*len(cols)], columns=cols)
    Xt = prep.transform(X)
    assert Xt.shape == (1, len(cols))
    assert np.isfinite(Xt).all()
