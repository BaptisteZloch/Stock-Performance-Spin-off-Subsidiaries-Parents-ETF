from copy import copy
from typing import Any, Callable, Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd

from scipy.optimize import minimize
from sklearn.covariance import ShrunkCovariance

from utility.types import AllocationMethodsEnum


class Allocation:
    @staticmethod
    def equal_weighted_allocation(
        selected_assets: List[str], *arg, **kwargs
    ) -> Dict[str, float]:
        return {asset: 1 / len(selected_assets) for asset in selected_assets}

    @staticmethod
    def max_sharpe_allocation(
        selected_assets: List[str],
        selected_assets_returns: pd.DataFrame,
        covariance_matrix_shrinkage: bool = True,
        *arg,
        **kwargs
    ) -> Dict[str, float]:
        def mean_variance_objective(weights: npt.NDArray[np.float32], arg: List[Any]):
            cov_matrix = arg[0]
            expected_return_vector = arg[1]
            # rf = float(arg[2])
            # Risk of the portfolio
            sigma_p = np.sqrt(weights @ cov_matrix @ weights.T)
            expected_return = expected_return_vector @ weights.T
            return -expected_return / sigma_p

        w0 = np.array([1 / len(selected_assets) for _ in selected_assets])

        cons = (
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1,
            },  # return 0 if sum of the weights is 1
            {"type": "ineq", "fun": lambda x: x},  # Long only
        )

        bounds = tuple([(0.0, 1.0) for _ in selected_assets])
        if covariance_matrix_shrinkage:
            covariance_matrix = selected_assets_returns.cov().to_numpy()
        else:
            cov = ShrunkCovariance().fit(selected_assets_returns.cov().to_numpy())
            covariance_matrix = cov.covariance_
        weights = minimize(
            mean_variance_objective,
            x0=w0,
            args=[
                covariance_matrix,
                selected_assets_returns.mean().to_numpy(),
            ],
            method="SLSQP",
            constraints=cons,
            bounds=bounds,
            options={"disp": False},
            tol=1e-10,
        ).x
        return {
            security: unit_weight
            for security, unit_weight in zip(
                selected_assets,
                map(lambda w: w if w >= 0.0001 else 0, weights),
            )
        }

    @staticmethod
    def risk_parity_allocation(
        selected_assets: List[str],
        selected_assets_returns: pd.DataFrame,
        *arg,
        **kwargs
    ) -> Dict[str, float]:
        def risk_budget_objective(weights: npt.NDArray[np.float32], arg: List[Any]):
            v = arg[0]  # Variance
            risk_budget = arg[1]
            # Risk of the portfolio
            sigma_p = np.sqrt(weights @ v @ weights.T)
            # Marginal contribution of each asset to the risk of the portfolio
            marginal_risk_contribution = v * weights.T
            # Contribution of each asset to the risk of the portfolio
            assets_risk_contribution = (
                marginal_risk_contribution @ weights.T
            ) / sigma_p

            # We calculate the desired contribution of each asset to the risk of the weights distribution
            assets_risk_target = sigma_p * risk_budget

            # Error between the desired contribution and the calculated contribution of each asset
            residual = assets_risk_contribution - assets_risk_target
            return residual @ residual.T

        w0 = np.array([1 / len(selected_assets) for _ in selected_assets])
        budget = copy(w0)

        cons = (
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1,
            },  # return 0 if sum of the weights is 1
            {"type": "ineq", "fun": lambda x: x},  # Long only
        )

        bounds = tuple([(0.0, 1.0) for _ in selected_assets])
        weights = minimize(
            risk_budget_objective,
            x0=w0,
            args=[selected_assets_returns.cov().to_numpy(), budget],
            method="SLSQP",
            constraints=cons,
            bounds=bounds,
            options={"disp": False},
            tol=1e-10,
        ).x
        return {
            security: unit_weight
            for security, unit_weight in zip(
                selected_assets,
                map(lambda w: w if w >= 0.001 else 0, weights),
            )
        }


ALLOCATION_DICT: Dict[
    AllocationMethodsEnum, Callable[[List[str], pd.DataFrame], Dict[str, float]]
] = {
    AllocationMethodsEnum.EQUALLY_WEIGHTED: Allocation.equal_weighted_allocation,
    AllocationMethodsEnum.MAX_SHARPE: Allocation.max_sharpe_allocation,
    AllocationMethodsEnum.RISK_PARITY: Allocation.risk_parity_allocation,
}
