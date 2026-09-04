import numpy as np
import pandas as pd

import data_collection


def test_fill_numeric_missing_preserves_string_missing_values():
    frame = pd.DataFrame({
        "numeric": [1.5, np.nan],
        "string": ["present", None],
    })

    out = data_collection.fill_numeric_missing(frame)

    assert out.loc[1, "numeric"] == 0
    assert pd.isna(out.loc[1, "string"])
    assert pd.isna(frame.loc[1, "numeric"])
