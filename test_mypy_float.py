import pandas as pd
from typing import Any, List

def test():
    result = pd.Series([1.0, 2.0])
    constants: List[Any] = [1.0, 2.0]
    for c in constants:
        result = result - float(c)
        result = result * float(c)
