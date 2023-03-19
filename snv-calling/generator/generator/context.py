import random
from dataclasses import dataclass
from typing import List

import numpy as np
from generator.core.model_data import Parameters
from generator.core.types import CNEvent, SNVEvent
from generator.generator.gen_utils import sample_conditionally_with_replacement

class ContextException(Exception):
     pass
@dataclass
class SNVGeneratorContext:
    p: Parameters

    def sequencing_error(self) -> float:
        return self.p.e

    def read_success_prob(self) -> float:
        return self.p.q

    def per_allele_coverage(self) -> float:
        return self.p.m

    def snv_in_cn_proportion(self) -> float:
        return 0.1

    def max_cn(self) -> int:
        return 9

    def neutral_cn(self) -> int:
        return 2

    def number_of_bins(self) -> int:
        return 1000

    def sample_cn_change(
            self, event: CNEvent, parent_cn_profile: np.ndarray, overlap_bit_map: np.ndarray
    ) -> int:
        """
        Should return CN change caused by event @event.
        parent_cn_profile - CN profile of @event's parent in the tree
        overlap_bit_map - boolean bitmap, @overlap_bit_map[bin] == True
        means that @bin is subject to some
        event in the subtree of @event (and hence no full CN deleteion is possible in that bin)
        """
        max_amplification = min(
            2, self.max_cn() - np.max(parent_cn_profile[range(event[0], event[1])])
        )
        if np.all(
                overlap_bit_map == False
        ):  # No bin is present in child nodes - we can do full deletion
            max_possible_deletion = -min(
                2, np.min(parent_cn_profile[[i for i in range(event[0], event[1])]])
            )
        else:
            min_cn_in_bins_with_overlap = np.min(parent_cn_profile[overlap_bit_map])
            max_possible_deletion = -min(2, int(min_cn_in_bins_with_overlap) - 1, int(np.min(parent_cn_profile[event[0]: event[1]])))

        if max_possible_deletion == 0 and max_amplification == 0:
            raise ContextException("Cant generate cn change")
        return sample_conditionally_with_replacement(
            1,
            lambda: random.randint(max_possible_deletion, max_amplification),
            lambda x: x != 0,
        )[0]

    def get_cn_event_candidates(self) -> List[CNEvent]:
        return [
            (a, b)
            for a in range(0, self.number_of_bins())
            for b in range(0, self.number_of_bins())
            if a < b and b - a < 100
        ]

    def get_snv_event_candidates(self) -> List[SNVEvent]:
        return [i for i in range(0, self.number_of_bins())]

    def sample_number_of_snvs_for_edge(self, num_available_snvs: int) -> int:
        return min(num_available_snvs, np.random.poisson(lam=1.0, size=None))

    def get_number_of_alterations(
            self, cn_before: int, cn_after: int, parent_altered_counts: int
    ) -> int:
        """
        How many altered copies shall be present in a bin, which had @cn_before CN
        and @parent_altered_counts altered copies
        and then went through mutation changing its CN to @cn_after.
        """
        if cn_after == 0:
            return 0
        if cn_after == cn_before:
            return parent_altered_counts
        if cn_after > cn_before:
            return (
                parent_altered_counts + (cn_after - cn_before)
                if random.randint(1, cn_before) <= parent_altered_counts
                else parent_altered_counts
            )
        else:
            copies_for_deletion = random.sample(
                range(0, cn_before), cn_before - cn_after
            )
            return parent_altered_counts - len(
                [x for x in copies_for_deletion if x < parent_altered_counts]
            )
