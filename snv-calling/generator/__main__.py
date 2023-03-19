import logging
import pickle

import numpy
import numpy as np

from generator.conet_integration.utils import raw_cc_matrix_to_CONET_format
from generator.generator.context import SNVGeneratorContext
from generator.generator.gen import SNVModelGenerator, sample_cells_data

import argparse

from generator.core.model_data import create_cn_matrix, EventTreeWithCounts
from generator.statistics.statistics_calculator import StatisticsCalculator

parser = argparse.ArgumentParser(description='Generate CONET SNV tree.')
parser.add_argument('--sequencing_error', type=float,default=0.001)
parser.add_argument('--read_success_prob', type=float, default=0.0001)
parser.add_argument('--coverage', type=float,default=0.03)
parser.add_argument('--max_cn', type=int, default=9)
parser.add_argument('--neutral_cn', type=int, default=2)
parser.add_argument('--bins', type=int, default=1000)
parser.add_argument('--tree_size', type=int, default=20)
parser.add_argument('--clusters', type=int, default=20, help="Number of cell clusters")
parser.add_argument('--cluster_size', type=int, default=20)
parser.add_argument('--random_attachment', type=bool, default=False, help="If set to False clusters will be attached to node in deterministic fashion")
parser.add_argument('--snv_events_per_edge', type=int, default=1,
                    help="average number of SNV events on an edge")
parser.add_argument('--snv_in_cn_proportion', type=float, default=0.1,
                    help="Average proportion of cn nodes which have SNVs sampled so that they overlap with CN event")
parser.add_argument('--simulation_dir', type=str, required=False)
parser.add_argument('--stats_dir', type=str, required=False)
parser.add_argument('--prefix', type=str, default="")
args = parser.parse_args()


if __name__ == "__main__":
    if args.simulation_dir is not None:
        calc = StatisticsCalculator(args.simulation_dir, args.prefix)
        res = calc.calculate() + '\n'
        with open(args.stats_dir, "a") as f:
            f.write(res)
    else:
        class Context(SNVGeneratorContext):

            def __init__(self):
                pass

            def sequencing_error(self) -> float:
                return args.sequencing_error

            def read_success_prob(self) -> float:
                return args.read_success_prob

            def snv_in_cn_proportion(self) -> float:
                return args.snv_in_cn_proportion

            def per_allele_coverage(self) -> float:
                return args.coverage

            def max_cn(self) -> int:
                return args.max_cn

            def neutral_cn(self) -> int:
                return args.neutral_cn

            def number_of_bins(self) -> int:
                return args.bins

            def sample_number_of_snvs_for_edge(self, num_available_snvs: int) -> int:
                return min(num_available_snvs, np.random.poisson(lam=args.snv_events_per_edge, size=None))


        ctxt = Context()
        model_generator = SNVModelGenerator(ctxt)
        model = model_generator.generate_model(args.tree_size)

        while not any([np.min(n) == 0.0 for n in model.node_to_cn_profile.values()]):
            logging.getLogger().error("Failed to generate tree with at least one deletion, trying one more time...")
            model = model_generator.generate_model(args.tree_size)

        cells_data, cc_matrix = sample_cells_data(clusters=args.clusters, cluster_size=args.cluster_size, model=model,
                                                  ctxt=ctxt, random_attachment=args.random_attachment)

        raw_cc_matrix_to_CONET_format(cc_matrix, model.tree).to_csv(f"./cc", index=False)
        cn = create_cn_matrix(cells_data, EventTreeWithCounts(model.tree, model.node_to_cn_profile))
        numpy.savetxt("./cn", cn)
        with open("./cluster_sizes", "w") as f:
            for _ in range(0, args.clusters):
                f.write(f"{args.cluster_size}\n")

        with open("./snvs_data", "w") as f:
            for i in range(0, ctxt.number_of_bins()):
                f.write(f"{i};{i};1\n")

        numpy.savetxt("B", cells_data.b, delimiter=";")
        numpy.savetxt("D", cells_data.d, delimiter=";")
        with open(f"./event_tree", "wb") as f:
            pickle.dump(EventTreeWithCounts(model.tree, model.node_to_cn_profile), f)

        with open(f"./event_tree.txt", "w") as f:
            for edge in model.tree.cn_event_tree.edges:
                f.write(f"{edge[0][0]},{edge[0][1]},{edge[1][0]},{edge[1][1]}\n")
        with open(f"./real_breakpoints.txt", "w") as f:
            breakpoints = list(set([n[0] for n in model.tree.cn_event_tree.nodes if n != (0,0)] + [n[1] for n in model.tree.cn_event_tree.nodes if n != (0,0)]))
            breakpoints.sort()

            f.write(",".join([str(s) for s in breakpoints]))
        with open(f"./attachment", "wb") as f:
            pickle.dump(cells_data.attachment, f)
