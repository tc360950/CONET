import argparse
import math
from logging import warning
from typing import NamedTuple

import networkx as nx
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont

from generator.statistics.conet_reader import ConetReader
from generator.statistics.model_reader import ModelReader
from conet.data_converter.corrected_counts import CorrectedCounts
from conet.data_converter.data_converter import DataConverter
from conet import CONET, CONETParameters, InferenceResult
from conet.snv_inference import MMEstimator, NewtonRhapsonEstimator
from conet.clustering import find_clustering_top_down_cn_normalization, find_clustering_top_down, \
    find_clustering_bottom_up, cluster_array
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.special import logsumexp


class CONETStats(NamedTuple):
    conet_likelihood: float
    prior: float
    counts_penlty: float
    snv_lik_candidates: float
    snv_full: float
    snv_b_lik: float
    snv_d_lik: float

    def get_full(self, snv_constant: float) -> float:
        return self.conet_likelihood + self.prior + self.counts_penlty + snv_constant * self.snv_lik_candidates

    @classmethod
    def from_file(cls, path: Path):
        with open(path, "r") as f:
            line = f.readline()
            values = [float(v) for v in line.split(";")]
        return cls(
            *values
        )


class ComparisonStats(NamedTuple):
    inferred_tree_size: float
    real_tree_size: float
    inferred_snvs: float
    real_snvs: float
    cn_node_recall: float
    cn_node_precision: float
    cn_edge_recall: float
    cn_edge_precision: float
    snv_recall: float
    snv_precision: float
    cell_snv_recall: float
    cell_snv_precision: float
    cn_prob: float
    non_trivial_cns: float

    @classmethod
    def from_file(cls, path: Path):
        with open(path, "r") as f:
            line = f.readline()
            values = [float(v) for v in line.split(";") if v]
        return cls(
            *values
        )


parser = argparse.ArgumentParser(description='Run CONET')
parser.add_argument('--data_dir', type=str, default='/data')
parser.add_argument('--param_inf_iters', type=int, default=30000)
parser.add_argument('--pt_inf_iters', type=int, default=100000)
parser.add_argument('--counts_penalty_s1', type=float, default=0.0)
parser.add_argument('--counts_penalty_s2', type=float, default=0.0)
parser.add_argument('--event_length_penalty_k0', type=float, default=1.0)
parser.add_argument('--tree_structure_prior_k1', type=float, default=0.0)
parser.add_argument('--use_event_lengths_in_attachment', type=bool, default=False)
parser.add_argument('--seed', type=int, default=12312)
parser.add_argument('--mixture_size', type=int, default=4)
parser.add_argument('--num_replicas', type=int, default=5)
parser.add_argument('--threads_likelihood', type=int, default=4)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--neutral_cn', type=float, default=2.0)
parser.add_argument('--output_dir', type=str, default='/data/out')
parser.add_argument('--end_bin_length', type=int, default=150000)
parser.add_argument('--snv_constant', type=float, default=1.0)
parser.add_argument('--add_chr_ends', type=bool, default=False)
parser.add_argument('--tries', type=int, default=1)
parser.add_argument('--snv_candidates', type=int, default=40)
parser.add_argument('--cbs_min_cells', type=int, default=1)
parser.add_argument('--estimate_snv_constant', type=bool, default=False)
parser.add_argument('--min_coverage', type=float, default=5)
parser.add_argument('--max_coverage', type=float, default=12.5)
parser.add_argument('--dont_infer_breakpoints', type=bool, default=False)
parser.add_argument('--sequencing_error', type=float, default=0.00001)
parser.add_argument('--recalculate_cbs', type=bool, default=False)
parser.add_argument('--clusterer', type=int, default=0)
parser.add_argument('--snv_scaling_factor', type=float, default=1.0)
args = parser.parse_args()

if __name__ == "__main__":

    (Path(args.data_dir) / Path("tmp/")).mkdir(parents=False, exist_ok=True)
    data_dir = str(Path(args.data_dir) / Path("tmp/"))
    shutil.copyfile(Path(args.data_dir) / Path("snvs_data"), Path(data_dir) / Path("snvs_data"))
    shutil.copyfile(Path(args.data_dir) / Path("D"), Path(data_dir) / Path("D"))
    shutil.copyfile(Path(args.data_dir) / Path("B"), Path(data_dir) / Path("B"))
    shutil.copyfile(Path(args.data_dir) / Path("cc"), Path(data_dir) / Path("cc"))
    shutil.copyfile(Path(args.data_dir) / Path("event_tree.txt"), Path(data_dir) / Path("event_tree.txt"))
    shutil.copyfile(Path(args.data_dir) / Path("real_breakpoints.txt"), Path(data_dir) / Path("real_breakpoints.txt"))

    print("Inferring CN profiles using CBS+MergeLevels...")
    corrected_counts: pd.DataFrame = pd.read_csv(Path(data_dir) / Path("cc"))
    if not args.recalculate_cbs:
        print("CBS+MegeLevels output files found in output directory, skipping inference...")
    else:
        x = subprocess.run(["Rscript", "CBS_MergeLevels.R", f"--mincells={args.cbs_min_cells}",
                            f"--output={Path(data_dir) / Path('cc_with_candidates')}",
                            f"--cn_output={Path(data_dir) / Path('cn_cbs')}",
                            f"--dataset={Path(data_dir) / Path('cc')}"])

        if x.returncode != 0:
            raise RuntimeError("CBS CN inference failed")

    cn = pd.read_csv(Path(data_dir) / Path('cn_cbs'), header=None)
    if args.dont_infer_breakpoints:
        cc_with_candidates = pd.read_csv(Path(data_dir) / Path("cc"), sep=",")
    else:
        cc_with_candidates = pd.read_csv(Path(data_dir) / Path('cc_with_candidates'))
        print("Breakpoint inference finished")
        print(f"Found {np.sum(cc_with_candidates.candidate_brkp)} breakpoints")

    cn = np.array(cn).T
    D = np.loadtxt(Path(data_dir) / Path("D"), delimiter=";")
    B = np.loadtxt(Path(data_dir) / Path("B"), delimiter=";")
    cluster_sizes = [1 for _ in range(cc_with_candidates.shape[1] - 5)]

    snvs_data = np.loadtxt(Path(data_dir) / Path("snvs_data"), delimiter=";").astype(int)
    cn_for_snvs = np.full(D.shape, args.neutral_cn)
    for cell in range(0, cn_for_snvs.shape[0]):
        for snv in range(0, cn_for_snvs.shape[1]):
            if snvs_data[snv, 1] >= 0:
                cn_for_snvs[cell, snv] = cn[cell, snvs_data[snv, 1]]

    print("Clustering cells...")
    # find_clustering_top_down_cn_normalization, find_clustering_top_down, find_clustering_bottom_up
    if args.clusterer == 0:
        clustering = find_clustering_top_down(D, cn, args.min_coverage, args.max_coverage, Path(data_dir), cn_for_snvs)
    elif args.clusterer == 1:
        clustering = find_clustering_top_down_cn_normalization(D, cn, args.min_coverage, args.max_coverage,
                                                               Path(data_dir), cn_for_snvs)
    elif args.clusterer == 2:
        clustering = find_clustering_bottom_up(D, cn, args.min_coverage, args.max_coverage, Path(data_dir), cn_for_snvs)
    elif args.clusterer == 3:
        print("Creating 1-cell clusters")
        clustering = list(range(cn.shape[0]))
    else:
        print("Taking real clustering")
        mod_reader = ModelReader("/data")
        mod_reader.load()
        node_to_num = {}
        clustering = []
        for n in mod_reader.attachment:
            if n not in node_to_num:
                node_to_num[n] = len(node_to_num)
            clustering.append(node_to_num[n])
    D = cluster_array(D, clustering, function="sum")
    B = cluster_array(B, clustering, function="sum")
    np.savetxt(Path(data_dir) / Path("D"), D, delimiter=";")
    np.savetxt(Path(data_dir) / Path("B"), B, delimiter=";")
    with open(Path(data_dir) / Path("cluster_sizes"), "w") as f:
        clusters = list(set(clustering))
        clusters.sort()
        for c in clusters:
            f.write(f"{sum([cluster_sizes[i] for i, x in enumerate(clustering) if x == c])}\n")
    with open(Path(args.output_dir) / Path("cell_to_cluster"), "w") as f:
        for i, c in enumerate(clustering):
            f.write(f"{cc_with_candidates.columns[i + 5]};cluster_{c}\n")

    cc_ = np.array(cc_with_candidates.iloc[:, 5:]).T
    cc_ = cluster_array(cc_, clustering, function="median").T

    cc_with_candidates = cc_with_candidates.iloc[:, :5]
    for c in range(0, len(set(clustering))):
        cc_with_candidates[f"cluster_{c}"] = cc_[:, c]

    with open(Path(data_dir) / Path("real_breakpoints.txt"), "r") as f:
        line = f.readline()
        breakpoints = [int(x) for x in line.split(",")]

    cc_with_candidates.iloc[:, 4] = 0
    cc_with_candidates.iloc[breakpoints, 4] = 1
    cc_with_candidates.to_csv(Path(data_dir) / Path("clustered_cc"), sep=",", index=False)
    print("Inferring SNV likelihood parameters...")
    cluster_sizes = np.loadtxt(Path(data_dir) / Path("cluster_sizes"))
    cluster_sizes = [int(y) for y in list(cluster_sizes)]
    snvs_data = np.loadtxt(Path(data_dir) / Path("snvs_data"), delimiter=";").astype(int)
    cn_for_snvs = np.full(D.shape, args.neutral_cn)
    for cell in range(0, cn_for_snvs.shape[0]):
        for snv in range(0, cn_for_snvs.shape[1]):
            if snvs_data[snv, 1] >= 0:
                cn_for_snvs[cell, snv] = cn[cell, snvs_data[snv, 1]]

    print(f"Running MM estimator with sequencing error {args.sequencing_error}")
    MMEstimator.DEFAULT_SEQUENCING_ERROR = args.sequencing_error
    params = MMEstimator.estimate(D, cn_for_snvs, cluster_sizes)
    params.e = args.sequencing_error
    print(f"Estimated params: {params}")
    print("Running newton-Rhapson estimator...")
    params = NewtonRhapsonEstimator(D, cn_for_snvs, cluster_sizes).solve(params)
    print(f"Estimated params: {params}")

    with open(Path(args.output_dir) / Path("SNV_params"), "w") as f:
        f.write(f"Estimated params: {params}")

    signals = list(np.mean(B, axis=0))
    signals = [(i, s) for i, s in enumerate(signals) if snvs_data[i, 2] == 1.0]
    print(f"Selecting SNV candidates from {len(signals)} candidates")
    signals.sort(key=lambda s: -s[1])
    snv_indices = [i[0] for i in signals[:args.snv_candidates]]

    snvs_data[:, 2] = 0
    snvs_data[snv_indices, 2] = 1
    snvs_data = snvs_data.astype(int)

    brkp_candidate_bin_to_num = {}
    counter = 0
    for i in range(0, cc_with_candidates.candidate_brkp.shape[0]):
        if cc_with_candidates.candidate_brkp.iloc[i] > 0.0:
            brkp_candidate_bin_to_num[i] = counter
            counter += 1

    for i in range(0, snvs_data.shape[0]):
        bin = snvs_data[i, 1]
        while bin >= 0 and cc_with_candidates.candidate_brkp.iloc[bin] == 0.0:
            bin -= 1
        if bin < 0 or cc_with_candidates.chr.iloc[snvs_data[i, 1]] != cc_with_candidates.chr.iloc[bin]:
            snvs_data[i, 1] = -1
        else:
            snvs_data[i, 1] = brkp_candidate_bin_to_num[bin]
    np.savetxt(str(Path(data_dir) / Path("snvs_data")), snvs_data, delimiter=";")

    cc = CorrectedCounts(cc_with_candidates)
    DataConverter(event_length_normalizer=3095677412).create_CoNET_input_files(out_path=data_dir + "/",
                                                                               corrected_counts=cc)
    conet = CONET("./CONET", "./analyzer", args.output_dir)
    params = CONETParameters(
        data_dir=data_dir + "/",
        param_inf_iters=args.param_inf_iters,
        pt_inf_iters=100,
        counts_penalty_s1=args.counts_penalty_s1,
        counts_penalty_s2=args.counts_penalty_s2,
        event_length_penalty_k0=args.event_length_penalty_k0,
        tree_structure_prior_k1=args.tree_structure_prior_k1,
        use_event_lengths_in_attachment=args.use_event_lengths_in_attachment,
        seed=args.seed,
        mixture_size=args.mixture_size,
        num_replicas=args.num_replicas,
        threads_likelihood=args.threads_likelihood,
        verbose=args.verbose,
        neutral_cn=args.neutral_cn,
        output_dir=args.output_dir,
        snv_constant=args.snv_constant,
        tries=args.tries,
        estimate_snv_constant=args.estimate_snv_constant,
        e=params.e,
        m=params.m,
        q=params.q,
        snv_scaling_factor=args.snv_scaling_factor
    )
    print("Running parameter inference")
    conet.infer_tree(params)
    print("Parameter inference finished")
    reader = ConetReader("/data/out", "/data/cc", "")


    def refresh():
        conet.analyze(params)
        result = InferenceResult(args.output_dir, cc, clustered=True)

        result.dump_results_to_dir(args.output_dir, neutral_cn=args.neutral_cn)
        reader.load()
        if Path("./statistics.tmp").exists():
            Path("./statistics.tmp").unlink()
        x = subprocess.run(
            ["python3", "-m", "generator", "--stats_dir", "./statistics.tmp", "--simulation_dir", "/data"])

        if x.returncode != 0:
            raise RuntimeError("CBS CN inference failed")

        stats = CONETStats.from_file(Path("/data/out/stats"))
        comparison_stats = ComparisonStats.from_file(Path("./statistics.tmp"))
        return stats, comparison_stats

    def calculate_likelihood():
        d_likelihood, b_likelihood = 0.0, 0.0
        # cell x snv
        for snv in range(D.shape[1]):
            for cell in range(D.shape[0]):
                attachment = reader.attachment[cell]
                cn = reader.cn[snv, cell]
                if attachment == (0,0):
                    genotypes = [(0, cn)]
                else:
                    path_to_root = list(nx.all_simple_paths(reader.tree, source=(0,0), target=attachment))[0]
                    snv_idx = -1
                    for i, n in enumerate(path_to_root):
                        if n in reader.snvs and snv in reader.snvs[n]:
                            snv_idx = i
                            break
                    snv_on_path = snv_idx >= 0
                    cn_change_after_snv = snv_on_path and any([
                        n[0] <= snv < n[1]
                        for n in path_to_root[snv_idx:]
                    ])
                    if cn == 0:
                        genotypes = [(0, 0)]
                    elif snv_on_path and cn_change_after_snv:
                        genotypes = [(a, cn) for a in range(0, cn+1)]
                    elif snv_on_path:
                        genotypes = [(1, cn)]
                    else:
                        genotypes = [(0, cn)]

                xs = []
                for g in genotypes:
                    prob = params.e if g[0] == 0 else g[0] / g[1]
                    prob = min(prob, 1.0 - params.e)

                    x = -math.log(len(genotypes)) + (D[cell, snv] - B[cell, snv]) * math.log(1.0 - prob) + B[cell, snv] * math.log(
                        prob)
                    xs.append(x)
                w = logsumexp(xs)
                b_likelihood += w
        print(f"B likelihood: {b_likelihood}")

        alpha = np.zeros(D.shape)
        beta = np.zeros(D.shape)
        coef = math.exp(math.log(1.0 - params.q) - math.log(params.q))
        for cell in range(0, D.shape[0]):
            for snv in range(0, D.shape[1]):
                alpha[cell][snv] = params.e + reader.cn[snv, cell] * params.m
                for i in range(0, int(D[cell, snv])):
                    beta[cell,snv] += math.log(alpha[cell,snv] * coef + i)

                d_likelihood += coef * math.log(1-params.q)*alpha[cell,snv] + math.log(params.q) * D[cell, snv] + beta[cell, snv]
        print(f"D likelihood: {d_likelihood}")


    def display_tree(stats: CONETStats, comp_stats: ComparisonStats, prev_stats, prev_comp):
        tree = reader.tree
        labels = {}
        model_reader = ModelReader("/data")
        model_reader.load()

        edge_labels = {}
        for edge in tree.edges:
            edge_labels[edge] = "SNVs: [" + ",".join([str(x) for x in reader.snvs[edge[1]]]) + "]"

        for n in list(tree.nodes):
            well_attached = 0
            for i, node in enumerate(model_reader.attachment):
                if node == n and reader.attachment[i] == n:
                    well_attached += 1

            attched_cells = len([x for x in reader.attachment if x == n])
            labels[n] = (
                f"({n[0]}, {n[1]}) {attched_cells} cells, {int(well_attached / max(attched_cells,1) * 100)}% well at."
            )
        plt.figure(figsize=(50, 50))
        pos = graphviz_layout(tree, prog="dot")
        nx.draw(tree, pos=pos, labels=labels,
                with_labels=True, node_color="grey", node_size=60, verticalalignment="bottom",
                font_size=20, edge_color="grey", font_color="green")

        nx.draw_networkx_edge_labels(
            tree, pos,
            edge_labels=edge_labels,
            font_color='red'
        )

        plt.savefig('/data/out/tree.png')

        img = Image.new('RGB', (1000, 1000), color=(255, 255, 255))

        d = ImageDraw.Draw(img)
        d.text((10,10), "Likelihood components", fill=(0,0,0))
        d.text((300, 20), "Current:", fill=(0, 0, 0))
        d.text((500, 20), "Previous:", fill=(0, 0, 0))
        GREEN = (0,255,0)
        RED = 'red'

        # conet_likelihood: float
        # prior: float
        # counts_penlty: float
        # snv_lik_candidates: float
        # snv_full: float
        i = 0
        def draw_attr(obj1, obj2, attr, name, positive: bool = True):
            nonlocal i
            d.text((20, 40 + i *20), name, fill=(0, 0, 0))
            if obj2 is None:
                d.text((300, 40 + i *20), str(obj1.__getattribute__(attr)),
                       fill=GREEN)
                d.text((500, 40 + i * 20), "NAN",
                       fill=(0,0,0))
            else:
                at1 = obj1.__getattribute__(attr)
                at2 = obj2.__getattribute__(attr)
                d.text((300, 40 + i * 20), str(at1),
                       fill=GREEN if (positive and at1 > at2) or (not positive and at1 < at2) else RED)
                d.text((500, 40 + i * 20), str(at2),
                       fill=RED if (positive and at1 > at2) or (not positive and at1 < at2) else GREEN)
            i+=1
        attrs = ["conet_likelihood", "prior", "counts_penlty", "snv_lik_candidates", "snv_full", "snv_b_lik", "snv_d_lik"]
        names = ["COnet likelihood", "Tree prior", "Counts penalty", "SNV lik candidates", "SNV lik full",  "snv_b_lik", "snv_d_lik"]
        for at, name in zip(attrs, names):
            draw_attr(stats, prev_stats, at, name)

        d.text((10, 40 + i *20), "Scores", fill=(0,0,0))
        i+=1
        d.text((300, 40 + i *20), "Current:", fill=(0, 0, 0))
        d.text((500, 40 + i *20), "Previous:", fill=(0, 0, 0))
        i += 1
        attrs = ["inferred_tree_size","real_tree_size","inferred_snvs","real_snvs","cn_node_recall","cn_node_precision","cn_edge_recall","cn_edge_precision", "snv_recall","snv_precision","cell_snv_recall","cell_snv_precision","cn_prob","non_trivial_cns"]
        for at in attrs:
            draw_attr(comp_stats, prev_comp, at, at)

        img.save('/data/out/stats_comparison.png')
        calculate_likelihood()

    def draw_real_tree():
        labels = {}
        model_reader = ModelReader("/data")
        model_reader.load()
        tree = model_reader.tree
        edge_labels = {}
        for edge in tree.edges:
            edge_labels[edge] = "SNVs: [" + ",".join([str(x) for x in model_reader.snvs[edge[1]]]) + "]"
        for n in list(tree.nodes):
            attched_cells = len([x for x in model_reader.attachment if x == n])
            labels[n] = (
                f"({n[0]}, {n[1]}) {attched_cells} cells\n"
            )
        plt.figure(figsize=(50, 50))
        pos = graphviz_layout(tree, prog="dot")
        nx.draw(tree, pos=pos, labels=labels,
                with_labels=True, node_color="grey", node_size=60, verticalalignment="bottom",
                font_size=20, edge_color="grey", font_color="green")

        nx.draw_networkx_edge_labels(
            tree, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        plt.savefig('/data/out/real_tree.png')

    def command_prompt_help():
        return (
            f"Breakpoints: {breakpoints}"
            f"Commands:\n ADD_NODE X1 X2 X3 X4\n    Add node (X3, X4) to (X1, X2)\n"
            " DELETE_NODE X1 X2\n    Delete node (X1, X2)\n"
            " RESET\n    Get back to the real tree\n"
        )


    previous_stats = None
    previous_comparison = None
    draw_real_tree()
    while True:
        stats, comparison_stats = refresh()
        if previous_stats is not None:
            print("Acceptance w.r.t to previous tree:")
            print(f"{stats.get_full(args.snv_constant) - previous_stats.get_full(args.snv_constant)}")
        display_tree(stats, comparison_stats, previous_stats, previous_comparison)
        command = input("Waiting for command:")
        command = " ".join(command.split())
        cmd = command.split(" ")
        if not cmd:
            warning("Command is empty")
        elif cmd[0] == "HELP":
            print(command_prompt_help())
        elif cmd[0] == "RESET":
            shutil.copyfile("/data/event_tree.txt", "/data/tmp/event_tree.txt")
            previous_stats = stats
            previous_comparison = comparison_stats
        elif cmd[0] == "DELETE_NODE" and len(cmd) >= 3:
            try:
                node = (int(cmd[1]), int(cmd[2]))
            except Exception as e:
                warning("Error parsing node: e")
            else:
                if node not in set(reader.tree.nodes):
                    warning(f"Node {node} is not on the tree,can't delete it")
                else:
                    previous_stats = stats
                    previous_comparison = comparison_stats
                    tree = reader.tree.copy()
                    tree.remove_node(node)
                    with open("/data/tmp/event_tree.txt", "w") as f:
                        for edge in tree.edges:
                            f.write(f"{edge[0][0]},{edge[0][1]},{edge[1][0]},{edge[1][1]}\n")

        elif cmd[0] == "ADD_NODE" and len(cmd) >= 5:
            fail = False
            try:
                parent = (int(cmd[1]), int(cmd[2]))
                child = (int(cmd[3]), int(cmd[4]))
            except Exception as e:
                fail = True
                warning(f"Cant parse tree nodes {e}")
            else:
                if parent not in set(reader.tree.nodes):
                    warning(f"Node {parent} does not exist")
                    fail = True
                if child in set(reader.tree.nodes):
                    warning(f"Node {child} already is on the tree")
                    fail = True
                if child[0] >= child[1]:
                    warning("First breakpoint must be smaller than the second")
                    fail = True
                if child[0] not in breakpoints or child[1] not in breakpoints:
                    warning(f"You can only use real breakpoits to construct nodes")
                    fail = True
                if not fail:
                    tree = reader.tree.copy()
                    tree.add_edge(parent, child)
                    previous_stats = stats
                    previous_comparison = comparison_stats
                    with open("/data/tmp/event_tree.txt", "w") as f:
                        for edge in tree.edges:
                            f.write(f"{edge[0][0]},{edge[0][1]},{edge[1][0]},{edge[1][1]}\n")

        else:
            warning("Unknown command")