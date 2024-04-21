from datetime import datetime
from typing import Literal
import pandas as pd

from data.bloomberg_api import BlpQuery


class Benchmark:
    INDICE_DATA_PATH = "../data/Equity_index.xlsx"

    def __init__(
        self,
        force_refresh: bool = False,
    ) -> None:
        self.__bench_history: pd.DataFrame = Benchmark.__load_indexes(
            force_refresh=force_refresh
        )

    @staticmethod
    def __load_indexes(force_refresh: bool = False) -> pd.DataFrame:
        """Load historical price for the indices in the database.

        Args:
            force_refresh (bool, optional): Whether to refresh or not the database, this uses Bloomberg requets. Defaults to False.

        Returns:
            pd.DataFrame: The dataframe of indices
        """
        DF_INDEXES = (
            pd.read_excel(Benchmark.INDICE_DATA_PATH, index_col="date")
            .fillna(method="ffill")  # forward fill the NaN
            .asfreq(
                "1B", method="ffill"
            )  # Convert the data to business day and forward fill the NaN
        )
        # Check if there are new business dates available
        new_dates_to_add = pd.bdate_range(
            end=datetime.now().date(),
            start=DF_INDEXES.index[-1].date(),
            inclusive="neither",
        )
        if new_dates_to_add.shape[0] != 0 and force_refresh is True:
            try:
                bquery = BlpQuery().start()
                LATEST_DATA_POINTS = bquery.bdh(
                    ["RTY Index", "SPX Index", "SX5E Index", "SXXP Index"],
                    ["PX_LAST"],  # Get closing price
                    start_date=new_dates_to_add[0].strftime("%Y%m%d"),
                    end_date=new_dates_to_add[-1].strftime("%Y%m%d"),
                    options={"adjustmentSplit": True},
                ).pivot(index="date", columns="security", values="PX_LAST")
                bquery.stop()  # Stop the BBG session
                DF_INDEXES = pd.concat(
                    [DF_INDEXES, LATEST_DATA_POINTS]
                )  # Merge the 2 dataframes
                DF_INDEXES.to_excel(Benchmark.INDICE_DATA_PATH)  # Update the database
            except:
                print("Something went wrong retry tomorrow")
        return DF_INDEXES.fillna(method="ffill").asfreq("1B", method="ffill")

    def get_benchmark_returns(
        self,
        benchmark_ticker: Literal[
            "RTY Index", "SPX Index", "SX5E Index", "SXXP Index"
        ] = "RTY Index",
    ) -> pd.Series:
        return self.__bench_history[benchmark_ticker].pct_change().fillna(0)

    def get_benchmark_history(
        self,
        benchmark_ticker: Literal[
            "RTY Index", "SPX Index", "SX5E Index", "SXXP Index"
        ] = "RTY Index",
    ) -> pd.Series:
        return self.__bench_history[benchmark_ticker]

    def get_benchmark_perf(
        self,
        benchmark_ticker: Literal[
            "RTY Index", "SPX Index", "SX5E Index", "SXXP Index"
        ] = "RTY Index",
    ) -> pd.Series:
        return (
            self.__bench_history[benchmark_ticker].pct_change().fillna(0) + 1
        ).cumprod()
