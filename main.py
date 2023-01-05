import argparse

import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
import shutil

from conet.data_converter.corrected_counts import CorrectedCounts
from conet.data_converter.data_converter import DataConverter
from conet import CONET, CONETParameters, InferenceResult
from conet.snv_inference import MMEstimator, NewtonRhapsonEstimator, NewtonRhapsonEstimatorOnlyQ
from conet.clustering import find_clustering, cluster_array

parser = argparse.ArgumentParser(description='Run CONET')
parser.add_argument('--data_dir', type=str, default='/data')
parser.add_argument('--param_inf_iters', type=int, default=100000)
parser.add_argument('--pt_inf_iters', type=int, default=100000)
parser.add_argument('--counts_penalty_s1', type=float, default=0.0)
parser.add_argument('--counts_penalty_s2', type=float, default=0.0)
parser.add_argument('--event_length_penalty_k0', type=float, default=1.0)
parser.add_argument('--tree_structure_prior_k1', type=float, default=1.0)
parser.add_argument('--use_event_lengths_in_attachment', type=bool, default=True)
parser.add_argument('--seed', type=int, default=12312)
parser.add_argument('--mixture_size', type=int, default=4)
parser.add_argument('--num_replicas', type=int, default=5)
parser.add_argument('--threads_likelihood', type=int, default=4)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--neutral_cn', type=float, default=2.0)
parser.add_argument('--output_dir', type=str, default='./')
parser.add_argument('--end_bin_length', type=int, default=150000)
parser.add_argument('--snv_constant', type=float, default=1.0)
parser.add_argument('--add_chr_ends', type=bool, default=False)
parser.add_argument('--tries', type=int, default=1)
parser.add_argument('--snv_candidates', type=int, default=10)
parser.add_argument('--cbs_min_cells', type=int, default=1)
parser.add_argument('--estimate_snv_constant', type=bool, default=False)
parser.add_argument('--min_coverage', type=float, default=10)
parser.add_argument('--cluster', type=bool, default=False)
parser.add_argument('--dont_infer_breakpoints', type=bool, default=False)
parser.add_argument('--default_sequencing_error', type=float, default=0.00001)
parser.add_argument('--nr_only_q', type=bool, default=True)
args = parser.parse_args()

if __name__ == "__main__":

    p = (Path(args.data_dir) / Path("tmp/")).mkdir(parents=False, exist_ok=True)
    data_dir = str(Path(args.data_dir) / Path("tmp/"))
    shutil.copyfile(Path(args.data_dir) / Path("snvs_data"), Path(data_dir) / Path("snvs_data"))
    if (Path(args.data_dir) / Path("cluster_sizes")).exists():
        shutil.copyfile(Path(args.data_dir) / Path("cluster_sizes"), Path(data_dir) / Path("cluster_sizes"))
    shutil.copyfile(Path(args.data_dir) / Path("D"), Path(data_dir) / Path("D"))
    shutil.copyfile(Path(args.data_dir) / Path("B"), Path(data_dir) / Path("B"))
    shutil.copyfile(Path(args.data_dir) / Path("cc"), Path(data_dir) / Path("cc"))

    print("Inferring breakpoints...")
    corrected_counts: pd.DataFrame = pd.read_csv(Path(data_dir) / Path("cc"))
    x = subprocess.run(["Rscript", "CBS_MergeLevels.R", f"--mincells={args.cbs_min_cells}",
                        f"--output=./cc_with_candidates", f"--cn_output=./cn_cbs",
                        f"--dataset={Path(data_dir) / Path('cc')}"])

    if x.returncode != 0:
        raise RuntimeError("Breakpoint inference failed")

    cn = pd.read_csv("./cn_cbs", header=None)
    if args.dont_infer_breakpoints:
        cc_with_candidates = pd.read_csv(Path(data_dir) / Path("cc"), sep=",")
    else:
        cc_with_candidates = pd.read_csv("./cc_with_candidates")
        print("Breakpoint inference finished")
        print(f"Found {np.sum(cc_with_candidates.candidate_brkp)} breakpoints")
    cn = np.array(cn).T
    D = np.loadtxt(Path(data_dir) / Path("D"), delimiter=";")
    B = np.loadtxt(Path(data_dir) / Path("B"), delimiter=";")

    if args.cluster:
        print("Clustering cells...")
        clustering = find_clustering(D, cn, args.min_coverage)
        D= cluster_array(D, clustering, function="sum")
        B= cluster_array(B, clustering, function="sum")
        np.savetxt(Path(data_dir) / Path("D"), D, delimiter=";")
        np.savetxt(Path(data_dir) / Path("B"), B, delimiter=";")
        cn = cluster_array(cn, clustering, function ="median")
        cn = cn.astype(int)
        with open(Path(data_dir) / Path("cluster_sizes"), "w") as f:
            clusters = list(set(clustering))
            clusters.sort()
            for c in clusters:
                f.write(f"{len([x for x in clustering if x == c])}\n")


    print("Inferring SNV likelihood parameters...")
    cluster_sizes = np.loadtxt(Path(data_dir) / Path("cluster_sizes"))
    cluster_sizes = [int(y) for y in list(cluster_sizes)]
    snvs_data = np.loadtxt(Path(data_dir) / Path("snvs_data"), delimiter=";").astype(int)
    cn_for_snvs = np.full(D.shape, args.neutral_cn)
    for cell in range(0, cn_for_snvs.shape[0]):
        for snv in range(0, cn_for_snvs.shape[1]):
            if snvs_data[snv, 1] >= 0:
                cn_for_snvs[cell, snv] = cn[cell, snvs_data[snv, 1]]

    print(f"Running MM estimator with default sequencing error {args.default_sequencing_error}")
    MMEstimator.DEFAULT_SEQUENCING_ERROR = args.default_sequencing_error
    params = MMEstimator.estimate(D, cn_for_snvs, cluster_sizes)
    print(f"Estimated params: {params}")
    print("Running newton-Rhapson estimator...")
    if args.nr_only_q:
        params = NewtonRhapsonEstimatorOnlyQ(D, cn_for_snvs, cluster_sizes).solve(params)
    else:
        params = NewtonRhapsonEstimator(D, cn_for_snvs, cluster_sizes).solve(params)
    print(f"Estimated params: {params}")

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
            snvs_data[i, 1] =  brkp_candidate_bin_to_num[bin]
    np.savetxt(str(Path(data_dir) / Path("snvs_data")), snvs_data, delimiter=";")

    cc = CorrectedCounts(cc_with_candidates)
    if args.add_chr_ends:
        print(f"Adding chromosome ends {args.add_chr_ends}")
        cc.add_chromosome_ends(neutral_cn=args.neutral_cn, end_bin_length=args.end_bin_length)
    DataConverter(event_length_normalizer=3095677412).create_CoNET_input_files(out_path=data_dir + "/",
                                                                               corrected_counts=cc)
    conet = CONET("./CONET", args.output_dir)
    params = CONETParameters(
        data_dir=data_dir + "/",
        param_inf_iters=args.param_inf_iters,
        pt_inf_iters=args.pt_inf_iters,
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
        q=params.q
    )
    conet.infer_tree(params)
    result = InferenceResult(params.output_dir, cc)
    inferred_counts = np.transpose(result.get_inferred_copy_numbers(neutral_cn=int(params.neutral_cn)))
    np.savetxt(f"{params.output_dir}inferred_counts", inferred_counts)
    result = InferenceResult(args.output_dir, cc)

    result.dump_results_to_dir(args.output_dir, neutral_cn=args.neutral_cn)

