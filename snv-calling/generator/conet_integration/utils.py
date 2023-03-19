import numpy as np
import pandas as pd
from generator.core.event_tree import EventTree


def raw_cc_matrix_to_CONET_format(
    cc: np.ndarray, event_tree: EventTree
) -> pd.DataFrame:
    BIN_START_COLUMN = 1
    BIN_END_COLUMN = 2
    BREAKPOINT_CANDIDATE_COLUMN = 4
    ADDITIONAL_COLUMNS = 5
    no_cells, no_loci = cc.shape
    cc = np.transpose(cc)

    # Add columns required by CONET
    add = np.full([no_loci, ADDITIONAL_COLUMNS], 1.0, dtype=np.float64)
    add[:, BIN_START_COLUMN] = range(0, no_loci)
    add[:, BIN_END_COLUMN] = range(1, no_loci + 1)
    add[:, BREAKPOINT_CANDIDATE_COLUMN] = 0
    add[event_tree.get_breakpoint_loci(), BREAKPOINT_CANDIDATE_COLUMN] = 1
    full_counts = np.hstack([add, cc])
    return pd.DataFrame(
        full_counts,
        columns=["chr", "start", "end", "width", "candidate_brkp"]
        + [f"cell{i}" for i in range(0, no_cells)],
    )
