from typing import NamedTuple, Any


class MVNStandard(NamedTuple):
    """
    https://github.com/EEA-sensors/sqrt-parallel-smoothers
    """
    mean: Any
    cov: Any
