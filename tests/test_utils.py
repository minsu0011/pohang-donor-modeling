from __future__ import annotations

import pandas as pd

from pohang_donor_models.utils import chronological_split, quarter_to_index, safe_divide


def test_quarter_to_index_is_consecutive() -> None:
    assert quarter_to_index(20234) + 1 == quarter_to_index(20241)
    assert quarter_to_index(20241) + 1 == quarter_to_index(20242)


def test_chronological_split_uses_last_periods() -> None:
    frame = pd.DataFrame({"period": [1, 1, 2, 2, 3, 3], "value": range(6)})
    train, test, periods = chronological_split(frame, "period", 1)
    assert periods == [3]
    assert set(train["period"]) == {1, 2}
    assert set(test["period"]) == {3}


def test_safe_divide_zero_is_nan() -> None:
    result = safe_divide(pd.Series([1, 2]), pd.Series([1, 0]))
    assert result[0] == 1
    assert pd.isna(result[1])
