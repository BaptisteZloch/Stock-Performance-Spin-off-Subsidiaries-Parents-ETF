from typing import Iterable
import yfinance as yf
import pandas as pd


class Universe:
    __companies = [
        "MMM",
        "AAPL",
        "AMZN",
        "AFL",
        "MSFT",
        "BLK",
        "BSX",
        "IP",
        "JPM",
        "MC.PA",
    ]

    def __init__(self) -> None:
        df: pd.DataFrame = yf.download(" ".join(self.__companies))["Close"]
        self.__universe_data = df.dropna().asfreq("B", method="ffill")

    def get_universe_securities(self) -> Iterable[str]:
        return self.__companies

    def get_universe_returns(self) -> pd.DataFrame:
        return self.__universe_data.pct_change().fillna(0)

    def get_universe_history(self) -> pd.DataFrame:
        return self.__universe_data

    def get_universe_perf(self) -> pd.DataFrame:
        return (self.__universe_data.pct_change().fillna(0) + 1).cumprod()
