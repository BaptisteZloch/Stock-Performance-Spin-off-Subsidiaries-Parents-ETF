import pandas as pd
from utility.types import SpinOff


def get_spin_off_parent_behavior(
    universe_returns: pd.DataFrame, spin_off: SpinOff, days_before: int, day_after: int
):
    assert (
        200 >= days_before > 1 and 200 >= day_after > 1
    ), "Error provide valide day_after and days_before parameter [1,200]"
    spin_off_indice = universe_returns.index.get_loc(spin_off.spin_off_ex_date)
    return (
        (
            universe_returns[spin_off.parent_company].iloc[
                spin_off_indice - days_before : spin_off_indice + days_before
            ]
            + 1
        )
        .cumprod()
        .to_numpy()
    )
