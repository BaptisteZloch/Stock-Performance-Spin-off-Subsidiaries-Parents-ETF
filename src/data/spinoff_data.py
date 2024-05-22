from typing import Literal, Union
from datetime import date, datetime
import pandas as pd
from data.bloomberg_api import BlpQuery


def get_spin_off_history(
    index_universe: Literal["RTY Index", "SPX Index", "SX5E Index", "SXXP Index"],
    start_date: Union[date, datetime, str],
    end_date: Union[date, datetime, str],
) -> pd.DataFrame:
    """_summary_ahaha

    Args:
        index_universe (Literal[&quot;RTY Index&quot;, &quot;SPX Index&quot;, &quot;SX5E Index&quot;, &quot;SXXP Index&quot;]): _description_
        start_date (Union[date, datetime, str]): _description_
        end_date (Union[date, datetime, str]): _description_

    Returns:
        pd.DataFrame: Returns a pd.DataFrame with columns : SPINOFF_TICKER_PARENT	ANNOUNCED_DATE	EFFECTIVE_DATE	SPINOFF_TICKER
    """
    assert index_universe in {
        "RTY Index",
        "SPX Index",
        "SX5E Index",
        "SXXP Index",
    }, "Error, provide a valid index universe."
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    bquery = BlpQuery().start()
    spin_off_raw_dataframe = bquery.bql(
        f"""let(#Data = Spinoffs(Effective_Date=range({start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}));
                #Filtered_Data = dropna(matches(#Data,#Data().DISTRIBUTION_RATIO >= 0.0),true);)
                get(#Filtered_Data)
                for(members('{index_universe}'))
                with(currency=USD)
                preferences(addcols=all)"""
    )
    bquery.stop()
    spin_off_raw_dataframe = (
        (
            spin_off_raw_dataframe.pivot_table(
                values=["secondary_value"],
                columns="secondary_name",
                index="security",
                aggfunc="first",
            )["secondary_value"]
        )
        .reset_index()[
            ["security", "ANNOUNCED_DATE", "EFFECTIVE_DATE", "SPINOFF_TICKER"]
        ]
        .rename(columns={"security": "SPINOFF_TICKER_PARENT"})
    )
    spin_off_raw_dataframe["ANNOUNCED_DATE"] = spin_off_raw_dataframe[
        "ANNOUNCED_DATE"
    ].apply(lambda x: pd.to_datetime(x).replace(tzinfo=None))
    spin_off_raw_dataframe["EFFECTIVE_DATE"] = spin_off_raw_dataframe[
        "EFFECTIVE_DATE"
    ].apply(lambda x: pd.to_datetime(x).replace(tzinfo=None))
    return spin_off_raw_dataframe
