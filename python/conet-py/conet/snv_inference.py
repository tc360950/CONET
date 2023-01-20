import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Parameters:
    m: float  # per allele coverage
    e: float  # sequencing error
    q: float  # read success probablity

    def __str__(self) -> str:
        return (
            f"Per allele coverage: {self.m}, "
            f"sequencing error: {self.e}, "
            f"read success probability: {self.q}."
        )


class MMEstimator:
    # ROW = cells
    DEFAULT_SEQUENCING_ERROR: float = 0.0001

    @staticmethod
    def estimate(d: np.ndarray, CN_profiles: np.ndarray, cluster_sizes) -> Parameters:
        zero_copy_bins = MMEstimator.__count_bin_cell_pairs_with_zero_copy(
            cluster_sizes, CN_profiles
        )
        cn_sum = MMEstimator.__sum_copy_numbers(cluster_sizes, CN_profiles)

        if zero_copy_bins == 0 or np.sum(d[CN_profiles == 0]):
            warnings.warn(
                f"Can't estimate sequencing error - there are no "
                f"reads for full deletions in the dataset!"
                f"Setting sequencing error to {MMEstimator.DEFAULT_SEQUENCING_ERROR}."
            )
            sequencing_error = MMEstimator.DEFAULT_SEQUENCING_ERROR
        else:
            sequencing_error = np.sum(d[CN_profiles == 0]) / zero_copy_bins
        per_allele_coverage = np.sum(d[CN_profiles != 0]) / cn_sum
        return Parameters(m=per_allele_coverage, e=sequencing_error, q=0.5)

    @staticmethod
    def __sum_copy_numbers(cluster_sizes, CN_profiles: np.ndarray) -> int:
        return sum(
            [
                cluster_sizes[c]
                * np.sum(
                    CN_profiles[
                    c, :
                    ]
                )
                for c in range(0, len(cluster_sizes))
            ]
        )

    @staticmethod
    def __count_bin_cell_pairs_with_zero_copy(
            cluster_sizes, CN_profiles: np.ndarray
    ) -> int:
        return sum(
            [
                cluster_sizes[c]
                * np.sum(
                    CN_profiles[
                        c,
                    ]
                    == 0
                )
                for c in range(0, len(cluster_sizes))
            ]
        )


class NewtonRhapsonEstimator:
    MAX_Q: float = 0.999

    def __init__(self, d: np.ndarray, CN: np.ndarray, cluster_sizes):
        self.d = d
        self.cluster_sizes = cluster_sizes
        self.CN = CN
        self.S_0 = self.__create_S_matrix()
        self.SCN = self.CN * self.S_0

    def solve(self, params: Parameters) -> Parameters:
        p_vec = np.array([params.e, params.m, params.q])

        i2 = 0
        target = self.calculate_target(params)
        while np.max(np.abs(target)) >= 0.1 and i2 < 10000:
            target = self.calculate_target(params)
            print(f"Target after {i2} iterations {target} with {params}")
            jacobian = self.calculate_jacobian(params)
            if np.sum(self.S_0) == 0.0:
                inverse_sub_jacobian = np.linalg.inv(jacobian[1:3, 1:3])
                inverse_jacobian = np.zeros((3, 3))
                inverse_jacobian[1:3, 1:3] = inverse_sub_jacobian
            else:
                inverse_jacobian = np.linalg.inv(self.calculate_jacobian(params))
            diff_vec = np.matmul(inverse_jacobian, target)

            for i in range(0, diff_vec.shape[0]):
                if p_vec[i] - diff_vec[i] < 0:
                    diff_vec[i] = 0.7 * p_vec[i]
            p_vec = p_vec - diff_vec
            params.e = p_vec[0]
            params.m = p_vec[1]
            params.q = p_vec[2]
            params.q = min(params.q, NewtonRhapsonEstimator.MAX_Q)
            i2 += 1
        if i2 == 10000:
            warnings.warn("NewtonRhapson estimator did not converge.")
        return params

    def __create_S_matrix(self) -> np.ndarray:
        result = np.zeros(self.d.shape)
        for cell in range(0, self.d.shape[0]):
            result[
                cell,
            ] = self.cluster_sizes[cell]
        return result
    def __create_alpha_matrix(self, params: Parameters) -> np.ndarray:
        b = self.CN.copy().astype(float)
        for cell in range(0, self.d.shape[0]):
            b[cell,] *= (
                    self.cluster_sizes[cell] * params.m
            )
        b = b + self.S_0* params.e
        return b

    def __create_b_special_matrix(
            self,
            alpha: np.ndarray,
            params: Parameters,
            func: Callable[[float, float], float],
    ) -> np.ndarray:
        result = np.zeros(self.d.shape)
        coef = math.exp(math.log(1 - params.q) - math.log(params.q))
        for cell in range(0, result.shape[0]):
            for bin in range(0, result.shape[1]):
                result[cell, bin] = func(
                    coef * alpha[cell, bin], self.d[cell, bin]
                )
        return result

    def calculate_target(self, p: Parameters) -> np.ndarray:
        A = self.__create_alpha_matrix(p)
        B_inv = self.__create_b_special_matrix(
            A,
            p,
            lambda a, d: 0.0
            if d == 0.0 or a == 0.0
            else sum([1 / (a + i) for i in range(0, int(d))]),
        )
        S_0 = self.S_0
        SCN = self.SCN

        log_coef = math.log(1 - p.q) - math.log(p.q)
        f_1 = math.log(1 - p.q) * math.exp(log_coef + math.log(np.sum(S_0))) + (
            0.0
            if np.sum(S_0 * B_inv) == 0.0
            else math.exp(log_coef + math.log(np.sum(S_0 * B_inv)))
        )
        f_2 = math.log(1 - p.q) * math.exp(log_coef + math.log(np.sum(SCN))) + math.exp(
            log_coef + math.log(np.sum(SCN * B_inv))
        )
        f_3 = (
                -math.log(1 - p.q) * math.exp(math.log(np.sum(A)) - 2.0 * math.log(p.q))
                - math.exp(math.log(np.sum(A)) - math.log(p.q))
                + math.exp(math.log(np.sum(self.d)) - math.log(p.q))
                - math.exp(math.log(np.sum(A * B_inv)) - 2.0 * math.log(p.q))
        )

        return np.array([f_1, f_2, f_3])

    # def calculate_likelihood(self, p: Parameters) -> np.ndarray:
    #     log_coef = math.log(1 - p.q) - math.log(p.q)
    #     coef = math.exp(log_coef)
    #     A = self.__create_alpha_matrix(p)
    #     B = self.__create_b_special_matrix(A, p, func=lambda a, d: 0.0 if d == 0.0 or a == 0.0 else sum(
    #         [math.log(a * coef + i) for i in range(0, int(d))]), )
    #     return coef * math.log(1 - p.q) * np.sum(A) + math.log(p.q) * np.sum(self.d) + np.sum(B)

    def calculate_jacobian(self, p: Parameters) -> np.ndarray:
        A = self.__create_alpha_matrix(p)
        S_0 = self.S_0
        SCN = self.SCN
        B_inv = self.__create_b_special_matrix(
            A,
            p,
            lambda a, d: 0.0
            if d == 0.0 or a == 0.0
            else sum([1 / (a + i) for i in range(0, int(d))]),
        )
        B_inv_2 = self.__create_b_special_matrix(
            A,
            p,
            lambda a, d: 0.0
            if d == 0.0 or a == 0.0
            else sum([1 / (a + i) ** 2 for i in range(0, int(d))]),
        )

        log_coef = math.log(1 - p.q) - math.log(p.q)

        f_1_e = -math.exp(math.log(np.sum(S_0 * S_0 * B_inv_2)) + 2.0 * log_coef)
        f_2_m = -math.exp(math.log(np.sum(SCN * SCN * B_inv_2)) + 2.0 * log_coef)
        f_1_m = -math.exp(math.log(np.sum(SCN * S_0 * B_inv_2)) + 2.0 * log_coef)
        f_2_e = f_1_m
        f_1_q = (
            -math.log(1 - p.q)
            * math.exp(math.log(np.sum(S_0)) - 2.0 * math.log(p.q))
            - math.exp(math.log(np.sum(S_0)) - math.log(p.q))
            +  math.exp(
                math.log(np.sum(S_0 * A * B_inv_2)) - 2.0 * math.log(p.q) + log_coef
            )
            - math.exp(math.log(np.sum(S_0 * B_inv)) - 2.0 * math.log(p.q))
        )
        f_2_q = (
            -math.log(1 - p.q)
            * math.exp(math.log(np.sum(SCN)) - 2.0 * math.log(p.q))
            - math.exp(math.log(np.sum(SCN)) - math.log(p.q))
            +  math.exp(
                math.log(np.sum(SCN * A * B_inv_2)) - 2.0 * math.log(p.q) + log_coef
            )
            - math.exp(math.log(np.sum(SCN * B_inv)) - 2.0 * math.log(p.q))
        )

        f_3_q = (
                2.0
                * math.log(1 - p.q)
                * math.exp(math.log(np.sum(A)) - 3.0 * math.log(p.q))
                + math.exp(math.log(np.sum(A)) - 2.0 * math.log(p.q) - math.log(1 - p.q))
                + math.exp(math.log(np.sum(A)) - 2.0 * math.log(p.q))
                - math.exp(math.log(np.sum(self.d)) - 2.0 * math.log(p.q))
                - math.exp(math.log(np.sum(A * A * B_inv_2)) - 4.0 * math.log(p.q))
                + 2.0 * math.exp(math.log(np.sum(A * B_inv)) - 3.0 * math.log(p.q))
        )

        f_3_m = f_2_q
        f_3_e = f_1_q

        return np.array(
            [[f_1_e, f_1_m, f_1_q], [f_2_e, f_2_m, f_2_q], [f_3_e, f_3_m, f_3_q]]
        )
