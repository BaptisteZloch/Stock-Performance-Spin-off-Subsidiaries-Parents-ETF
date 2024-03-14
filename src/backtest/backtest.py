from datetime import datetime
from itertools import groupby
from typing import Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from backtest.metrics import drawdown
from backtest.reports import plot_from_trade_df, print_portfolio_strategy_report
from data.universe import Universe
from strategy.allocation import ALLOCATION_DICT
from utility.types import (
    AllocationMethodsEnum,
    SpinOff,
    RebalanceFrequencyEnum,
)

from utility.utils import (
    compute_weights_drift,
    get_rebalance_dates,
)


class Backtester:
    TRADING_DAYS_IN_A_MONTH = 21

    def __init__(
        self,
        universe_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> None:
        """Constructor for the backtester class

        Args:
        ----
            universe_returns (pd.DataFrame): The returns of the universe each columns is an asset, each row represent a date and returns for the assets. The DataFrame must have a `DatetimeIndex` with freq=B.
            benchmark_returns (pd.Series): The returns of the benchmark in order to measure the performance of the backtest. The Series must have a `DatetimeIndex` with freq=B.
        """
        lower_bound = max(
            [
                universe_returns.index[0],
                benchmark_returns.index[0],
            ]
        )
        upper_bound = min(
            [
                universe_returns.index[-1],
                benchmark_returns.index[-1],
            ]
        )
        self.__universe_returns = universe_returns.loc[lower_bound:upper_bound]
        self.__benchmark_returns = benchmark_returns.loc[lower_bound:upper_bound]

    def run_backtest(
        self,
        allocation_type: AllocationMethodsEnum,
        spin_off_events_annoucement: Dict[datetime, List[SpinOff]],
        holding_period_in_months: Optional[int] = None,
        backtest_type: Literal["subsidiaries", "parents"] = "parents",
        transaction_cost: float = 0.010,
        days_holding_securities_before_spin_off: Optional[int] = None,
        verbose: bool = True,
        print_metrics: bool = True,
        plot_performance: bool = True,
        starting_offset: int = 20,
    ) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
        """_summary_

         Args:
             allocation_type (AllocationMethodsEnum): The allocation method to use. Only Equally weighted here.
             spin_off_events (Dict[datetime, List[SpinOff]]): a dictionary with keys as spin off announcement date and values as list of SpinOff object containing the parent company and the subsidiary created the spin off ex_date.
             holding_period_in_months (Optional[int]): The holding period in month. None means owning stocks until the end of the backtest. Usually the holding period is between 12 and 36 months. Defaults to None.
             backtest_type (Literal[&quot;subsidiaries&quot;, &quot;parents&quot;], optional): _description_. Defaults to "parents".
             transaction_cost (float, optional): The transaction costs. Defaults to 0.010.
             verbose (bool, optional): Print rebalance dates... Defaults to True.
             print_metrics (bool, optional): Print the metrics of the strategy. Defaults to True.
             plot_performance (bool, optional): Plot several Chart showing the performances of the strategy. Defaults to True.
             starting_offset (int, optional): _description_. Defaults to 20.

        Returns:
         ----
             Tuple[Union[pd.Series, pd.DataFrame], ...]: Return a tuple of DataFrame/Series respectively : The returns of the strategy and the bench ptf_and_bench (Series), the historical daily weights of the portfolio ptf_weights_df (DataFrame), The regime a the beta at each detection date ptf_regime_beta_df (DataFrame), all risk/perf metrics of the strategy ptf_metrics_df (DataFrame)
        """
        assert starting_offset >= 0, "Error, provide a positive starting offset."
        assert isinstance(
            self.__universe_returns.index, pd.DatetimeIndex
        ), "Error provide a dataframe with datetime index"
        returns_histo = pd.Series(name="strategy_returns", dtype=float)
        weights_histo: List[Dict[str, float]] = []
        # Create in the same format as the spin_off_events_annoucement but instead on announcement date as key it is the ex-date
        spin_off_events_list = [
            sp for _, v in spin_off_events_annoucement.items() for sp in v
        ]
        spin_off_events_list.sort(key=lambda spinoff: spinoff.spin_off_ex_date)
        spin_off_events = {
            key: list(group)
            for key, group in groupby(
                spin_off_events_list, key=lambda x: x.spin_off_ex_date
            )
        }
        # Will store the asset name and the corresponding duration in days
        securities_and_holding_periods: Dict[str, int] = {}

        for index, _ in tqdm(
            self.__universe_returns.iloc[starting_offset:].iterrows(),
            desc="Backtesting portfolio...",
            total=self.__universe_returns.shape[0],
            leave=False,
        ):
            # if (
            #     index in spin_off_events_annoucement.keys() or first_rebalance is False
            # ) and backtest_type == "parents":
            #     assets = Universe.get_universe_securities()
            #     first_rebalance = True
            #     if verbose:
            #         print(f"Rebalancing the portfolio on {index}...")
            #     weights = ALLOCATION_DICT[allocation_type](
            #         assets,
            #         self.__universe_returns[assets].loc[:index]
            #         # self.__universe_returns.columns, self.__universe_returns.loc[:index]
            #     )
            #     # Row returns with applied fees
            #     returns = (
            #         self.__universe_returns[list(weights.keys())].loc[index].to_numpy()
            #         - transaction_cost
            #     )
            # el
            if index in spin_off_events.keys():
                # We get the securities in the portfolio and filter in order to remove the
                # assets where the holding period is > than the threshold
                securities_and_holding_periods = {
                    k: v
                    for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                    if v / self.TRADING_DAYS_IN_A_MONTH
                    < holding_period_in_months  # Filter based on theur current holding period
                }

                # Get the subsidiary or parents to add to the portfolio
                for assets_to_add in map(
                    lambda x: x.parent_company
                    if backtest_type == "parents"
                    else x.subsidiary_company,
                    spin_off_events.get(index),  # This is a list of SpinOff object,
                ):
                    securities_and_holding_periods[
                        assets_to_add
                    ] = 0  # Add the asset to the portfolio with zero day holding period

                # get their names only
                assets_in_portfolio = list(securities_and_holding_periods.keys())
                if verbose:
                    print(f"Rebalancing the portfolio on due to spin off on {index}...")
                # Create weights dictionary for allocation given an allocation method.
                # The allocation method needs assets and their corresponding weights
                weights = ALLOCATION_DICT[allocation_type](
                    assets_in_portfolio,
                    self.__universe_returns[assets_in_portfolio].loc[:index],
                )
                returns = (
                    self.__universe_returns[list(weights.keys())]
                    .loc[index]
                    .to_numpy()  # Get the latest returns
                    - transaction_cost  # Substract transaction costs
                )
            elif (
                len(
                    [
                        k
                        for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                        if v / self.TRADING_DAYS_IN_A_MONTH >= holding_period_in_months
                    ]
                )
                > 0
            ):
                # We get the securities in the portfolio and filter in order to remove the
                # assets where the holding period is > than the threshold
                securities_and_holding_periods = {
                    k: v
                    for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                    if v / self.TRADING_DAYS_IN_A_MONTH
                    < holding_period_in_months  # Filter based on theur current holding period
                }

                # get their names only
                assets_in_portfolio = list(securities_and_holding_periods.keys())
                if len(assets_in_portfolio) == 0:
                    returns = (
                        np.array([self.__benchmark_returns.loc[index]])
                        - transaction_cost
                    )
                    weights = {self.__benchmark_returns.name: 1.0}
                    if verbose:
                        print(
                            f"Rebalancing the portfolio on due to nothing in the portfolio at {index}... (No more securities in the portfolio)"
                        )
                else:
                    if verbose:
                        print(
                            f"Rebalancing the portfolio on due to max holding period reached {index}..."
                        )
                    # Create weights dictionary for allocation given an allocation method.
                    # The allocation method needs assets and their corresponding weights
                    weights = ALLOCATION_DICT[allocation_type](
                        assets_in_portfolio,
                        self.__universe_returns[assets_in_portfolio].loc[:index],
                    )
                    returns = (
                        self.__universe_returns[list(weights.keys())]
                        .loc[index]
                        .to_numpy()  # Get the latest returns
                        - transaction_cost  # Substract transaction costs
                    )
            else:  # Nothing to do
                if len(securities_and_holding_periods.keys()) == 0:
                    returns = np.array([self.__benchmark_returns.loc[index]])
                    weights = {self.__benchmark_returns.name: 1.0}
                    # if verbose:
                    #     print(
                    #         f"Rebalancing the portfolio on due to nothing in the portfolio at {index}... (in the else part)"
                    #     )

                # self.__benchmark_returns
                else:
                    returns = (
                        self.__universe_returns[list(weights.keys())]
                        .loc[index]
                        .to_numpy()
                        - transaction_cost
                    )
            # Append the current weights to the list
            weights_histo.append(weights)
            # Create numpy weights for matrix operations
            weights_np = np.array(list(weights.values()))
            # Append the current return to the list
            returns_histo.loc[index] = returns @ weights_np

            # Compute the weight drift due to assets price fluctuations
            weights = compute_weights_drift(
                list(weights.keys()),
                weights_np,
                returns,
            )
            # Add new day to holding period
            securities_and_holding_periods = {
                k: v + 1 for k, v in securities_and_holding_periods.items()
            }

        # Construct dataframe with the returns, the perf, and the drawdown for the plots.
        self.__benchmark_returns.name = "returns"
        ptf_and_bench = pd.merge(
            returns_histo, self.__benchmark_returns, left_index=True, right_index=True
        )
        ptf_and_bench["cum_returns"] = (ptf_and_bench["returns"] + 1).cumprod()
        ptf_and_bench["strategy_cum_returns"] = (
            ptf_and_bench["strategy_returns"] + 1
        ).cumprod()
        ptf_and_bench["drawdown"] = drawdown(ptf_and_bench["returns"])
        ptf_and_bench["strategy_drawdown"] = drawdown(ptf_and_bench["strategy_returns"])

        # The weights of the ptf
        ptf_weights_df = pd.DataFrame(
            weights_histo,
            index=self.__universe_returns.iloc[starting_offset:].index,
            dtype=float,
        ).fillna(0)

        if print_metrics is True:
            ptf_metrics_df = print_portfolio_strategy_report(
                ptf_and_bench["strategy_returns"],
                ptf_and_bench["returns"],
            )
        if plot_performance is True:
            plot_from_trade_df(
                ptf_and_bench,
                ptf_weights_df,
            )
        if print_metrics is True:
            return ptf_and_bench, ptf_weights_df, ptf_metrics_df
        return ptf_and_bench, ptf_weights_df
