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
from utility.constants import LONG_SHORT_DICT, TRADING_DAYS_IN_A_MONTH
from utility.types import (
    AllocationMethodsEnum,
)

from utility.utils import (
    compute_weights_drift,
    wrangle_spin_off_dataframe,
)


class Backtester:
    def __init__(self, universe_obj: Universe) -> None:
        """Constructor for the backtester class

        Args:
        ----
            universe_returns (pd.DataFrame): The returns of the universe each columns is an asset, each row represent a date and returns for the assets. The DataFrame must have a `DatetimeIndex` with freq=B.
            benchmark_returns (pd.Series): The returns of the benchmark in order to measure the performance of the backtest. The Series must have a `DatetimeIndex` with freq=B.
            spin_off_dataframe (pd.DataFrame): dataframe with spin off using the `get_spin_off_history` function.
        """
        universe_returns = universe_obj.get_returns_history_from_spinoff()
        benchmark_returns = universe_obj.get_benchmark_returns()
        spin_off_dataframe = universe_obj.get_spin_off_history()
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
        self.__spin_off_dataframe = spin_off_dataframe
        # This for loop removes the securities that are not tradable after the spin off (i.e not retrieved with bloomberg so we remove the spin off)
        for col in list(
            set(self.__spin_off_dataframe["SPINOFF_TICKER_PARENT"].tolist()).union(
                set(self.__spin_off_dataframe["SPINOFF_TICKER"].tolist())
            )
        ):
            if (
                col not in self.__universe_returns.columns
            ):  # Check if spin off exists in the price dataframe
                # Filter it out
                self.__spin_off_dataframe = self.__spin_off_dataframe[
                    (self.__spin_off_dataframe["SPINOFF_TICKER_PARENT"] != col)
                    & (self.__spin_off_dataframe["SPINOFF_TICKER"] != col)
                ]

    def run_backtest(
        self,
        allocation_type: AllocationMethodsEnum,
        holding_period_in_months: Optional[int] = None,
        backtest_type: Literal["subsidiaries", "parents"] = "parents",
        transaction_cost: float = 0.010,
        position_type: Literal["long", "short"] = "long",
        hold_parent_before_spin_off: bool = True,
        verbose: bool = True,
        print_metrics: bool = True,
        plot_performance: bool = True,
        starting_offset: int = 20,
    ) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
        """_summary_

         Args:
             allocation_type (AllocationMethodsEnum): The allocation method to use. Only Equally weighted here.
             holding_period_in_months (Optional[int]): The holding period in month. None means owning stocks until the end of the backtest. Usually the holding period is between 12 and 36 months. Defaults to None.
             backtest_type (Literal[&quot;subsidiaries&quot;, &quot;parents&quot;], optional): _description_. Defaults to "parents".
             transaction_cost (float, optional): The transaction costs. Defaults to 0.010.
             position_type (Literal[&quot;long&quot;,&quot;short&quot;], optional): _description_. Defaults to "long".
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
        returns_histo = pd.Series(
            name="strategy_returns", dtype=float
        )  # Will store the returns of the portfolio
        weights_histo: List[Dict[str, float]] = []  # Will store the weights

        spin_off_events_announcement = wrangle_spin_off_dataframe(
            self.__spin_off_dataframe
        )
        # Create in the same format as the spin_off_events_announcement but instead on announcement date as key it is the ex-date
        spin_off_events_list = [
            sp for _, v in spin_off_events_announcement.items() for sp in v
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
            if (
                index.date() in spin_off_events_announcement.keys()
                and backtest_type == "parents"
                and hold_parent_before_spin_off is True
            ):
                # We get the securities in the portfolio and filter in order to remove the
                # assets where the holding period is > than the threshold
                securities_and_holding_periods = {
                    k: v
                    for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                    if v / TRADING_DAYS_IN_A_MONTH
                    < holding_period_in_months  # Filter based on their current holding period
                }

                # Get the subsidiary or parents to add to the portfolio
                for assets_to_add in map(
                    lambda x: x.parent_company,
                    spin_off_events_announcement.get(
                        index.date()
                    ),  # This is a list of SpinOff object,
                ):
                    if assets_to_add not in securities_and_holding_periods.keys():
                        securities_and_holding_periods[
                            assets_to_add
                        ] = 0  # Add the asset to the portfolio with zero day holding period

                # get their names only
                assets_in_portfolio = list(securities_and_holding_periods.keys())
                if verbose:
                    print(
                        f"Rebalancing the portfolio on due to spin off announcement on {index.date()}..."
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
            elif index.date() in spin_off_events.keys():
                # We get the securities in the portfolio and filter in order to remove the
                # assets where the holding period is > than the threshold
                securities_and_holding_periods = {
                    k: v
                    for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                    if v / TRADING_DAYS_IN_A_MONTH
                    < holding_period_in_months  # Filter based on their current holding period
                }

                # Get the subsidiary or parents to add to the portfolio
                for assets_to_add in map(
                    lambda x: x.parent_company
                    if backtest_type == "parents"
                    else x.subsidiary_company,
                    spin_off_events.get(
                        index.date()
                    ),  # This is a list of SpinOff object,
                ):
                    if assets_to_add not in securities_and_holding_periods.keys():
                        securities_and_holding_periods[
                            assets_to_add
                        ] = 0  # Add the asset to the portfolio with zero day holding period

                # get their names only
                assets_in_portfolio = list(securities_and_holding_periods.keys())
                if verbose:
                    print(
                        f"Rebalancing the portfolio on due to spin off event on {index.date()}..."
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
            elif (
                len(
                    [
                        k
                        for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                        if v / TRADING_DAYS_IN_A_MONTH >= holding_period_in_months
                    ]
                )
                > 0
            ):
                # We get the securities in the portfolio and filter in order to remove the
                # assets where the holding period is > than the threshold
                securities_and_holding_periods = {
                    k: v
                    for k, v in securities_and_holding_periods.items()  # Iteration over the securities and their current holding period
                    if v / TRADING_DAYS_IN_A_MONTH
                    < holding_period_in_months  # Filter based on their current holding period
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
                            f"Rebalancing the portfolio on due to nothing in the portfolio at {index.date()}... (No more securities in the portfolio)"
                        )
                else:
                    if verbose:
                        print(
                            f"Rebalancing the portfolio on due to max holding period reached {index.date()}..."
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
            returns_histo.loc[index] = LONG_SHORT_DICT.get(position_type, 1) * (
                returns @ weights_np
            )

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
