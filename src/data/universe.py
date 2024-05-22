from datetime import date, datetime
from typing import Iterable, Literal, Optional, Union
import yfinance as yf
import pandas as pd

from data.bloomberg_api import BlpQuery
from utility.constants import INDEXES
from utility.types import Indexes


class Universe:
    spin_off_raw_dataframe: Optional[pd.DataFrame] = None
    spinoff_price_returns_dataframe: Optional[pd.DataFrame] = None

    def __init__(
        self,
        index_universe: Indexes,
        start_date: Union[date, datetime, str],
        end_date: Union[date, datetime, str],
    ) -> None:
        """_summary_

        Args:
            index_universe (Literal[&quot;RTY Index&quot;, &quot;SPX Index&quot;, &quot;SX5E Index&quot;, &quot;SXXP Index&quot;]): _description_
            start_date (Union[date, datetime, str]): _description_
            end_date (Union[date, datetime, str]): _description_
        """
        assert index_universe in INDEXES, "Error, provide a valid index universe."
        self.__INDEX_UNIVERSE = index_universe
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.__START_DATE = start_date
        self.__END_DATE = end_date

    def get_price_history_from_spinoff(self) -> pd.DataFrame:
        if self.spinoff_price_returns_dataframe is None:
            self.spin_off_raw_dataframe = self.get_spin_off_history()
            bquery = BlpQuery().start()
            # BDH to get the price history of the spinoff parents and children
            DATA_ALL = bquery.bdh(
                list(
                    set(
                        self.spin_off_raw_dataframe["SPINOFF_TICKER_PARENT"].tolist()
                    ).union(set(self.spin_off_raw_dataframe["SPINOFF_TICKER"].tolist()))
                ),
                ["PX_LAST"],
                start_date=self.__START_DATE.strftime("%Y%m%d"),
                end_date=self.__END_DATE.strftime("%Y%m%d"),
                options={"adjustmentSplit": True},
            )
            bquery.stop()
            # Process the dataframe to have a security by column
            self.spinoff_price_returns_dataframe = (
                DATA_ALL.pivot(index="date", columns="security", values="PX_LAST")
                .ffill()
                .pct_change()
                .fillna(0)
            )
            self.spinoff_price_returns_dataframe.index = (
                self.spinoff_price_returns_dataframe.index.date
            )
            self.spinoff_price_returns_dataframe = (
                self.spinoff_price_returns_dataframe.asfreq("1B", method="ffill")
            )

        return self.spinoff_price_returns_dataframe

    def get_spin_off_history(
        self,
    ) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: Returns a pd.DataFrame with columns : SPINOFF_TICKER_PARENT	ANNOUNCED_DATE	EFFECTIVE_DATE	SPINOFF_TICKER
        """
        if self.spin_off_raw_dataframe is None:
            bquery = BlpQuery().start()
            self.spin_off_raw_dataframe = bquery.bql(
                f"""let(#Data = Spinoffs(Effective_Date=range({self.__START_DATE.strftime('%Y-%m-%d')},{self.__END_DATE.strftime('%Y-%m-%d')}));
                        #Filtered_Data = dropna(matches(#Data,#Data().DISTRIBUTION_RATIO >= 0.0),true);)
                        get(#Filtered_Data)
                        for(members('{self.__INDEX_UNIVERSE}'))
                        with(currency=USD)
                        preferences(addcols=all)"""
            )
            bquery.stop()
            self.spin_off_raw_dataframe = (
                (
                    self.spin_off_raw_dataframe.pivot_table(
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
            self.spin_off_raw_dataframe["ANNOUNCED_DATE"] = self.spin_off_raw_dataframe[
                "ANNOUNCED_DATE"
            ].apply(lambda x: pd.to_datetime(x).replace(tzinfo=None))
            self.spin_off_raw_dataframe["EFFECTIVE_DATE"] = self.spin_off_raw_dataframe[
                "EFFECTIVE_DATE"
            ].apply(lambda x: pd.to_datetime(x).replace(tzinfo=None))
        return self.spin_off_raw_dataframe


# class Universe:
#     __companies = [
#         "MMM",
#         "AAPL",
#         "AMZN",
#         "AFL",
#         "MSFT",
#         "BLK",
#         "BSX",
#         "IP",
#         "JPM",
#         "MC.PA",
#     ]

#     def __init__(self) -> None:
#         df: pd.DataFrame = yf.download(" ".join(self.__companies))["Close"]
#         self.__universe_data = df.dropna().asfreq("B", method="ffill")

#     def get_universe_securities(self) -> Iterable[str]:
#         return self.__companies

#     def get_universe_returns(self) -> pd.DataFrame:
#         return self.__universe_data.pct_change().fillna(0)

#     def get_universe_history(self) -> pd.DataFrame:
#         return self.__universe_data

#     def get_universe_perf(self) -> pd.DataFrame:
#         return (self.__universe_data.pct_change().fillna(0) + 1).cumprod()
