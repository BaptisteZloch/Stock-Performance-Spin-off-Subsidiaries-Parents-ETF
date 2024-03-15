# ALL THE CODE BELOW COMES FROM THE LIBRARY CREATED BY BAPTISTE ZLOCH :
# https://pypi.org/project/quant-invest-lab/
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from typing import Optional
import matplotlib.pyplot as plt


from backtest.metrics import (
    omega_ratio,
    construct_report_dataframe,
)


def print_portfolio_strategy_report(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    report_df = construct_report_dataframe(portfolio_returns, benchmark_returns)
    if benchmark_returns is not None:
        print(f"\n{'  Returns statistical information  ':-^50}")

        print(
            f"Expected return annualized: {100*report_df.loc['Expected return', 'Portfolio']:.2f} % vs {100*report_df.loc['Expected return', 'Benchmark']:.2f} % (benchmark)"
        )
        print(
            f"CAGR: {100*report_df.loc['CAGR', 'Portfolio']:.2f} % vs {100*report_df.loc['CAGR', 'Benchmark']:.2f} % (benchmark)"
        )
        print(
            f"Expected volatility annualized: {100*report_df.loc['Expected volatility', 'Portfolio']:.2f} % vs {100*report_df.loc['Expected volatility', 'Benchmark']:.2f} % (benchmark)"
        )
        print(
            f"Specific volatility (diversifiable) annualized: {100*report_df.loc['Specific risk', 'Portfolio'] :.2f} %"
        )
        print(
            f"Systematic volatility annualized: {100*report_df.loc['Systematic risk', 'Portfolio'] :.2f} %"
        )
        print(
            f"Skewness: {report_df.loc['Skewness', 'Portfolio']:.2f} vs {report_df.loc['Skewness', 'Benchmark']:.2f} (benchmark), <0 = left tail, >0 = right tail"
        )
        print(
            f"Kurtosis: {report_df.loc['Kurtosis', 'Portfolio']:.2f} vs {report_df.loc['Skewness', 'Benchmark']:.2f} (benchmark)",
            ", >3 = fat tails, <3 = thin tails",
        )
        print(
            f"95%-VaR: {100*report_df.loc['VaR', 'Portfolio']:.2f} % vs {100*report_df.loc['VaR', 'Benchmark']:.2f} % (benchmark) -> the lower the better"
        )
        print(
            f"95%-CVaR: {100*report_df.loc['CVaR', 'Portfolio']:.2f} % vs {100*report_df.loc['CVaR', 'Benchmark']:.2f} % (benchmark) -> the lower the better"
        )

        print(f"\n{'  Strategy statistical information  ':-^50}")
        print(
            f"Max drawdown: {100*report_df.loc['Max drawdown', 'Portfolio']:.2f} % vs {100*report_df.loc['Max drawdown', 'Benchmark']:.2f} % (benchmark)"
        )
        print(
            f"Kelly criterion: {100*report_df.loc['Kelly criterion', 'Portfolio']:.2f} % vs {100*report_df.loc['Kelly criterion', 'Benchmark']:.2f} % (benchmark)"
        )
        print(
            f"Benchmark sensitivity (beta): {report_df.loc['Portfolio beta', 'Portfolio']:.2f} vs 1 (benchmark)"
        )
        print(f"Determination coefficient R²: {report_df.loc['R2', 'Portfolio']:.2f}")
        print(
            f"Tracking error annualized: {100*report_df.loc['Tracking error', 'Portfolio']:.2f} %"
        )
        print(f"\n{'  Strategy ratios  ':-^50}")
        print("No risk free rate considered for the following ratios.\n")
        print(
            f"Sharpe ratio annualized: {report_df.loc['Sharpe ratio', 'Portfolio']:.2f} vs {report_df.loc['Sharpe ratio', 'Benchmark']:.2f} (benchmark)"
        )
        print(
            f"Sortino ratio annualized: {report_df.loc['Sortino ratio', 'Portfolio']:.2f} vs {report_df.loc['Sortino ratio', 'Benchmark']:.2f} (benchmark)"
        )
        print(
            f"Burke ratio annualized: {report_df.loc['Burke ratio', 'Portfolio']:.2f} vs {report_df.loc['Burke ratio', 'Benchmark']:.2f} (benchmark)"
        )
        print(
            f"Calmar ratio annualized: {report_df.loc['Calmar ratio', 'Portfolio']:.2f} vs {report_df.loc['Calmar ratio', 'Benchmark']:.2f} (benchmark)"
        )
        print(
            f"Tail ratio annualized: {report_df.loc['Tail ratio', 'Portfolio']:.2f} vs {report_df.loc['Tail ratio', 'Benchmark']:.2f} (benchmark)"
        )
        print(
            f"Treynor ratio annualized: {report_df.loc['Treynor ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Information ratio annualized: {report_df.loc['Information ratio', 'Portfolio']:.2f}"
        )
    else:
        print(f"\n{'  Returns statistical information  ':-^50}")

        print(
            f"Expected return annualized: {100*report_df.loc['Expected return', 'Portfolio']:.2f} %"
        )
        print(
            f"Expected volatility annualized: {100*report_df.loc['Expected volatility', 'Portfolio']:.2f} %"
        )
        print(
            f"Skewness: {report_df.loc['Skewness', 'Portfolio'] :.2f}, <0 = left tail, >0 = right tail"
        )
        print(
            f"Kurtosis: {report_df.loc['Kurtosis', 'Portfolio']:.2f}",
            ", >3 = fat tails, <3 = thin tails",
        )
        print(
            f"95%-VaR: {100*report_df.loc['VaR', 'Portfolio']:.2f} % -> the lower the better"
        )
        print(
            f"95%-CVaR: {100*report_df.loc['CVaR', 'Portfolio']:.2f} % -> the lower the better"
        )

        print(f"\n{'  Strategy statistical information  ':-^50}")
        print(f"Max drawdown: {100*report_df.loc['Max drawdown', 'Portfolio']:.2f} %")
        print(
            f"Kelly criterion: {100*report_df.loc['Kelly criterion', 'Portfolio']:.2f} %"
        )
        print(f"\n{'  Strategy ratios  ':-^50}")
        print("No risk free rate considered for the following ratios.\n")
        print(
            f"Sharpe ratio annualized: {report_df.loc['Sharpe ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Sortino ratio annualized: {report_df.loc['Sortino ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Burke ratio annualized: {report_df.loc['Burke ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Calmar ratio annualized: {report_df.loc['Calmar ratio', 'Portfolio']:.2f}"
        )
        print(f"Tail ratio annualized: {report_df.loc['Tail ratio', 'Portfolio']:.2f}")
    return report_df


def plot_from_trade_df(
    ptf_and_bench: pd.DataFrame,
    ptf_weights_evolution: pd.DataFrame,
) -> None:
    """Plot historical price, equity progression, drawdown evolution and return distribution.

    Args:
        ptf_and_bench (pd.DataFrame): Dataframe containing the returns, cum returns, drawdown for both the strategy portflio and the benchmark
        ptf_weights_evolution (pd.DataFrame): A dataframe with the weights of the portfolio at each date
    """
    fig, ax = plt.subplots(4, 2, figsize=(25, 30))

    ############################################## Perf and Regimes
    ax[0, 0].plot(
        ptf_and_bench["strategy_cum_returns"], color="orange", label="Portfolio"
    )
    ax[0, 0].plot(ptf_and_bench["cum_returns"], color="blue", label="Benchmark")
    ax[0, 0].set_xlabel("Datetime", fontsize=15)
    ax[0, 0].set_ylabel("Returns", fontsize=15)
    ax[0, 0].set_title(
        f"Performance benchmark vs portfolio with regime detected", fontsize=20
    )
    ax[0, 0].grid()
    ax[0, 0].legend(loc="lower right", fontsize=15)
    ############################################## DRAWDOWN
    ax[1, 0].fill_between(
        ptf_and_bench.index,
        ptf_and_bench["strategy_drawdown"],
        color="orange",
        alpha=0.3,
        label="Strategy drawdown",
    )
    ax[1, 0].fill_between(
        ptf_and_bench.index,
        ptf_and_bench["drawdown"],
        color="blue",
        alpha=0.3,
        label="Benchmark drawdown",
    )

    ax[1, 0].set_xlabel("Datetime", fontsize=18)
    ax[1, 0].set_ylabel("Drawdown", fontsize=18)
    ax[1, 0].set_title(
        "Underwater (drawdown) plot",
        fontsize=20,
    )
    ax[1, 0].grid()
    ax[1, 0].legend(loc="lower right", fontsize=15)
    ########################################### HISTORIGRAM
    samples_bench = sorted(ptf_and_bench["returns"].to_numpy())
    samples_strat = sorted(ptf_and_bench["strategy_returns"].to_numpy())
    ax[0, 1].fill_between(
        samples_strat,
        gaussian_kde(samples_strat, bw_method="scott").pdf(samples_strat),
        color="orange",
        alpha=0.3,
        label="KDE Strategy returns",
    )
    ax[0, 1].fill_between(
        samples_bench,
        gaussian_kde(samples_bench, bw_method="scott").pdf(samples_bench),
        color="blue",
        alpha=0.3,
        label="KDE Benchmark returns",
    )

    ax[0, 1].set_xlabel("Returns", fontsize=18)
    ax[0, 1].set_ylabel("Density", fontsize=18)
    ax[0, 1].set_title(
        "Returns distribution",
        fontsize=20,
    )
    ax[0, 1].grid()
    ax[0, 1].legend(fontsize=15)
    ########################################### WEIGHTS DRIFT
    ax[3, 0].stackplot(
        ptf_weights_evolution.index,
        ptf_weights_evolution.to_numpy().T,
    )
    ax[3, 0].set_xlabel("Datetime", fontsize=18)
    ax[3, 0].set_ylabel("Weights", fontsize=18)
    ax[3, 0].set_title("Portfolio weights evolution over time", fontsize=20)
    ax[3, 0].grid()
    ax[3, 0].legend(ptf_weights_evolution.columns.to_list(), fontsize=15)

    ########################################### WEIGHTS DRIFT
    df_rets = pd.DataFrame(
        {
            "returns": ptf_and_bench["returns"],
            "strategy_returns": ptf_and_bench["strategy_returns"],
        }
    )

    deciles = np.array(
        [
            (chunks["returns"].mean(), chunks["strategy_returns"].mean())
            for chunks in np.array_split(
                df_rets.sort_values(by="returns", ascending=True), 10
            )
        ]
    )

    ax[1, 1].bar(
        np.arange(1, deciles[:, 0].shape[0] + 1) + 0.2,
        deciles[:, -1],
        0.4,
        color="orange",
        label="Portfolio",
    )
    ax[1, 1].bar(
        np.arange(1, deciles[:, 0].shape[0] + 1) - 0.2,
        deciles[:, 0],
        0.4,
        color="blue",
        label="Benchmark",
    )
    ax[1, 1].set_xlabel("Return deciles", fontsize=15)
    ax[1, 1].set_ylabel("Average return", fontsize=15)
    ax[1, 1].set_title(f"Performance by decile", fontsize=20)
    ax[1, 1].grid()
    ax[1, 1].legend(fontsize=7)

    ############################################## Perf horizon
    ptf_and_bench["strategy_cum_returns"]
    windows_bh = [
        day for day in range(5, ptf_and_bench["strategy_cum_returns"].shape[0] // 3, 30)
    ]
    bench_expected_return_profile = [
        ptf_and_bench["returns"]
        .rolling(window)
        .apply(lambda prices: (prices.iloc[-1] / prices.iloc[0]) - 1)
        .mean()
        for window in windows_bh
    ]
    ptf_expected_return_profile = [
        ptf_and_bench["strategy_returns"]
        .rolling(window)
        .apply(lambda prices: (prices.iloc[-1] / prices.iloc[0]) - 1)
        .mean()
        for window in windows_bh
    ]
    ax[2, 1].plot(
        windows_bh, ptf_expected_return_profile, color="orange", label="Portfolio"
    )
    ax[2, 1].scatter(
        windows_bh,
        ptf_expected_return_profile,
        color="orange",
    )
    ax[2, 1].plot(
        windows_bh, bench_expected_return_profile, color="blue", label="Benchmark"
    )
    ax[2, 1].scatter(windows_bh, bench_expected_return_profile, color="blue")
    ax[2, 1].set_xlabel("Investment horizon in days", fontsize=15)
    ax[2, 1].set_ylabel("Returns", fontsize=15)
    ax[2, 1].set_title(
        f"Expected return with respect to investment horizon", fontsize=20
    )
    ax[2, 1].grid()
    ax[2, 1].legend(fontsize=15)
    ############################################## Rolling sharpe ratio
    n_rolling = ptf_and_bench["strategy_cum_returns"].shape[0] // 10

    ax[2, 0].plot(
        ptf_and_bench.index,
        ptf_and_bench["strategy_cum_returns"]
        .rolling(n_rolling)
        .apply(
            lambda rets: (252 * rets.mean()) / (rets.std() * (252**0.5)),
        )
        .fillna(0),
        color="orange",
        label="Portfolio",
    )
    ax[2, 0].plot(
        ptf_and_bench.index,
        ptf_and_bench["cum_returns"]
        .rolling(n_rolling)
        .apply(
            lambda rets: (252 * rets.mean()) / (rets.std() * (252**0.5)),
        )
        .fillna(0),
        color="blue",
        label="Benchmark",
    )
    ax[2, 0].set_xlabel("Datetime", fontsize=15)
    ax[2, 0].set_ylabel("Returns", fontsize=15)
    ax[2, 0].set_title(f"{n_rolling}-days rolling Sharpe ratio", fontsize=20)
    ax[2, 0].grid()
    ax[2, 0].legend(fontsize=15)
    ############################################## Omega curve
    thresholds = np.linspace(0.01, 0.75, 100)
    omega_bench = []
    omega_ptf = []
    for threshold in thresholds:
        omega_ptf.append(omega_ratio(ptf_and_bench["strategy_returns"], threshold))
        omega_bench.append(omega_ratio(ptf_and_bench["returns"], threshold))
    ax[3, 1].plot(thresholds, omega_ptf, color="orange", label="Portfolio")
    ax[3, 1].plot(thresholds, omega_bench, color="blue", label="Benchmark")
    ax[3, 1].set_xlabel("Thresholds", fontsize=15)
    ax[3, 1].set_ylabel("Omega ratio", fontsize=15)
    ax[3, 1].set_title(f"Omega curve", fontsize=20)
    ax[3, 1].grid()
    ax[3, 1].legend(fontsize=15)
