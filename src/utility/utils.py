from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import numpy.typing as npt
from numbers import Number
import pandas as pd
import pytz
import blpapi
from utility.types import RebalanceFrequencyEnum, SpinOff


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


def wrangle_spin_off_dataframe(
    spin_off_dataframe: pd.DataFrame,
) -> Dict[date, List[SpinOff]]:
    """Clean the spin off dataframe and convert it into a dict with special format that will be used in the backtester

    Args:
        spin_off_dataframe (pd.DataFrame): The spin off dataframe bloomberg.

    Returns:
        Dict[date, List[SpinOff]]: The desired formatted dict.
    """
    spin_off_announcements = {}
    for index, row in spin_off_dataframe.iterrows():
        if row["ANNOUNCED_DATE"].date() in spin_off_announcements.keys():
            spin_off_announcements[row["ANNOUNCED_DATE"].date()].append(
                SpinOff(
                    parent_company=row["SPINOFF_TICKER_PARENT"],
                    subsidiary_company=row["SPINOFF_TICKER"],
                    spin_off_ex_date=row["EFFECTIVE_DATE"].date(),
                )
            )
        else:
            spin_off_announcements[row["ANNOUNCED_DATE"].date()] = [
                SpinOff(
                    parent_company=row["SPINOFF_TICKER_PARENT"],
                    subsidiary_company=row["SPINOFF_TICKER"],
                    spin_off_ex_date=row["EFFECTIVE_DATE"].date(),
                )
            ]
    return spin_off_announcements


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


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


####################################### BELOW FOR BLOOMBERG API ONLY #######################################


def datetime_converter(value: Union[str, date, datetime]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz:
        ts = ts.tz_convert(pytz.FixedOffset(ts.tz.getOffsetInMinutes()))
    return ts


def element_to_value(elem: blpapi.Element) -> Union[pd.Timestamp, str, Number, None]:
    """Convert a blpapi.Element to its value defined in its datatype with some possible coercisions.

    datetime -> pd.Timestamp
    date -> pd.Timestamp
    blp.name.Name -> str
    null value -> None
    ValueError Exception -> None

    Args:
        elem: Element to convert

    Returns: A value

    """
    if elem.isNull():
        return None
    else:
        try:
            value = elem.getValue()
            if isinstance(value, blpapi.name.Name):
                return str(value)
            if isinstance(value, datetime) or isinstance(value, date):
                return datetime_converter(value)

            return value
        except ValueError:
            return None


def _element_to_dict(elem: Union[str, blpapi.Element]) -> Any:
    if isinstance(elem, str):
        return elem
    dtype = elem.datatype()
    if dtype == blpapi.DataType.CHOICE:
        return {f"{elem.name()}": _element_to_dict(elem.getChoice())}
    elif elem.isArray():
        return [_element_to_dict(v) for v in elem.values()]
    elif dtype == blpapi.DataType.SEQUENCE:
        return {
            f"{elem.name()}": {
                f"{e.name()}": _element_to_dict(e) for e in elem.elements()
            }
        }
    else:
        return element_to_value(elem)


def element_to_dict(elem: blpapi.Element) -> Dict:
    """Convert a blpapi.Element to an equivalent dictionary representation.

    Args:
        elem: A blpapi.Element

    Returns: A dictionary representation of blpapi.Element

    """
    return _element_to_dict(elem)


def message_to_dict(msg: blpapi.Message) -> Dict:
    """Convert a blpapi.Message to a dictionary representation.

    Args:
        msg: A blpapi.Message

    Returns: A dictionary with relevant message metadata and data

    """
    return {
        "fragmentType": msg.fragmentType() if hasattr(msg, "fragmentType") else None,
        "correlationIds": [cid.value() for cid in msg.correlationIds()],
        "messageType": f"{msg.messageType()}",
        "timeReceived": _get_time_received(msg),
        "element": element_to_dict(msg.asElement()),
    }


def _get_time_received(msg: blpapi.Message) -> Optional[pd.Timestamp]:
    try:
        return datetime_converter(msg.timeReceived())
    except ValueError:
        return None


def dict_to_req(request: blpapi.Request, request_data: Dict) -> blpapi.Request:
    """Populate request with data from request_data.

    Args:
        request: Request to populate
        request_data: Data used for populating the request

    Returns: A blpapi.Request

    Notes: An example request data dictionary is

      rdata = {'fields': ['SETTLE_DT'], 'securities': ['AUD Curncy'],
               'overrides': [{'overrides': {'fieldId': 'REFERENCE_DATE', 'value': '20180101'}}]}

    """
    for key, value in request_data.items():
        elem = request.getElement(key)
        if elem.datatype() == blpapi.DataType.SEQUENCE:
            for elem_dict in value:
                if elem.isArray():
                    el = elem.appendElement()
                    for k, vv in elem_dict[key].items():
                        el.setElement(k, vv)
                else:
                    elem.setElement(elem_dict, value[elem_dict])
        elif elem.isArray():
            for v in value:
                elem.appendValue(v)
        elif elem.datatype() == blpapi.DataType.CHOICE:
            for k, v in value.items():
                c = elem.getElement(k)
                if c.isArray():
                    for v_i in v:
                        c.appendValue(v_i)
                else:
                    c.setValue(v)
        else:
            elem.setValue(value)
    return request
