from datetime import date, datetime
from typing import Dict, List, Set, Union
import numpy as np
import numpy.typing as npt

import pandas as pd

from utility.types import RebalanceFrequencyEnum


def get_rebalance_dates(
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    frequency: RebalanceFrequencyEnum = RebalanceFrequencyEnum.MONTH_START,
) -> Set[Union[pd.Timestamp, datetime]]:
    """Generate a Series of the rebalance date during the backtest period based on the frequency.

    Args:
        start_date (Union[datetime, str]): The start date.
        end_date (Union[datetime, str]):  The start date.
        frequency (RebalanceFrequencyEnum, optional): The chosen frequency, daily, monthly, quarterly... Defaults to "monthly".

    Returns:
        Set[Union[pd.Timestamp, datetime]]: The rebalance dates.
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date, infer_datetime_format=True)
    return set(
        [start_date] + pd.date_range(start_date, end_date, freq=frequency).to_list()
    )


def compute_weights_drift(
    securities: List[str],
    old_weights: npt.NDArray[np.float32],
    current_returns: npt.NDArray[np.float32],
) -> Dict[str, float]:
    """Take the old weights and the current returns and return the new weights (impacted with the drift)

    Args:
        securities (List[str]): The list of securities.
        old_weights (npt.NDArray[np.float32]): The old weights associated with the securities.
        current_returns (npt.NDArray[np.float32]): The current returns associated with the securities.

    Returns:
        Dict[str, float]: The new weights in a dict format: key=security, value=new weight.
    """
    return {
        security: unit_weight
        for security, unit_weight in zip(
            securities,
            (old_weights * (current_returns + 1))
            / ((current_returns + 1) @ old_weights),
        )
    }


def is_business_day(date_to_check: Union[datetime, date, str, pd.Timestamp]):
    return bool(len(pd.bdate_range(date_to_check, date_to_check)))
