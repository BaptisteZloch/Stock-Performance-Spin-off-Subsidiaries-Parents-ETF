import argparse
from utility.constants import INDEXES
from utility.types import AllocationMethodsEnum
from data.universe import Universe
from backtest.backtest import Backtester


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments using argparse."""

    parser = argparse.ArgumentParser(
        description="Run a backtest with specified parameters."
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        choices=INDEXES,
        help='The index universe to use (e.g., "RTY Index").',
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        help="The start date for the backtest (YYYY-MM-DD format).",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        help="The end date for the backtest (YYYY-MM-DD format).",
    )

    # Optional arguments
    parser.add_argument(
        "-p",
        "--position_type",
        type=str,
        default="long",
        choices=["long", "short"],
        help="The position type (long or short). Default: long.",
    )
    parser.add_argument(
        "-b",
        "--backtest_type",
        type=str,
        default="parents",
        choices=["parents", "subsidiaries"],
        help="The type of backtest to run (e.g., parents, subsidiaries). Default: parents.",
    )
    parser.add_argument(
        "-t",
        "--transaction_cost",
        type=float,
        default=0.001,
        help="The transaction cost per trade in percent (default: 0.001).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    universe_obj = Universe(
        index_universe=args.index_universe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    bk_test = Backtester(universe_obj=universe_obj)

    ptf_and_bench, ptf_weights_df, ptf_metrics_df = bk_test.run_backtest(
        allocation_type=AllocationMethodsEnum.EQUALLY_WEIGHTED,
        backtest_type=args.backtest_type,
        holding_period_in_months=15,
        transaction_cost=args.transaction_cost,
        position_type=args.position_type,
        plot_performance=True,
        verbose=False,
    )

    print("Weights to implement in PORT:")
    print(ptf_weights_df.iloc[-1])
