from typing import Iterable
import yfinance as yf
import pandas as pd


class Benchmark:
    __benchmark_ticker = "^RUT"

    def __init__(self) -> None:
        self.__bench_history: pd.DataFrame = (
            (yf.download(self.__benchmark_ticker)[["Close"]])
            .dropna()
            .asfreq("B", method="ffill")
            .rename(columns={"Close": self.__benchmark_ticker})[self.__benchmark_ticker]
        )

    def get_benchmark_returns(self) -> pd.Series:
        return self.__bench_history.pct_change().fillna(0)

    def get_benchmark_history(self) -> pd.Series:
        return self.__bench_history

    def get_benchmark_perf(self) -> pd.Series:
        return (self.__bench_history.pct_change().fillna(0) + 1).cumprod()
