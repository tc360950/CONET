#include <chrono>
#include <cstdlib>
#include <fstream>
#include <tuple>
#include <utility>

#include "src/conet_result.h"
#include "src/input_data/csv_reader.h"
#include "src/parallel_tempering_coordinator.h"
#include "src/tree/event_tree.h"
#include "src/tree/tree_formatter.h"
#include "src/tree_sampler_coordinator.h"
#include "src/utils/random.h"

#include <boost/program_options.hpp>

using namespace std;

namespace po = boost::program_options;

int main(int argc, char **argv) {
  po::options_description description("MyTool Usage");

  description.add_options()("data_dir", po::value<string>()->required(),
                            "Path to directory containing input files.")(
      "output_dir", po::value<string>()->required(),
      "Path to output directory. Inference results will be saved there.")(
      "param_inf_iters", po::value<int>()->default_value(100000),
      "Number of MCMC iterations for joint tree and model parameters "
      "inference.")("pt_inf_iters", po::value<int>()->default_value(100000),
                    "Number of MCMC iterations for tree inference.")(
      "counts_penalty_s1", po::value<double>()->default_value(0.0),
      "Constant controlling impact of penalty for large discrepancies between "
      "inferred and real count matrices.")(
      "counts_penalty_s2", po::value<double>()->default_value(0.0),
      "Constant controlling impact of penalty for inferring clusters with "
      "changed copy number equal to basal ploidy.")(
      "event_length_penalty_k0", po::value<double>()->default_value(1.0),
      "Constant controlling impact of penalty for long inferred events.")(
      "tree_structure_prior_k1", po::value<double>()->default_value(1.0),
      "Constant controlling impact of data size part of tree structure prior.")(
      "use_event_lengths_in_attachment", po::value<bool>()->default_value(true),
      "If True cell attachment probability will depend on average event length "
      "in the history, otherwise it will be uniform.")(
      "seed", po::value<int>()->default_value(12312), "Seed for C++ RNG")(
      "mixture_size", po::value<size_t>()->default_value(4),
      "Initial number of components in difference distribution for breakpoint "
      "loci.")("num_replicas", po::value<size_t>()->default_value(5),
               "Number of tempered chain replicas in MAP event tree search.")(
      "threads_likelihood", po::value<size_t>()->default_value(4),
      "Number of threads which will be used for the most demanding likelihood "
      "calculations.")("verbose", po::value<bool>()->default_value(true),
                       "True if CONET should print messages during inference.")(
      "neutral_cn", po::value<double>()->default_value(2.0),
      "Neutral copy number")("e", po::value<double>()->default_value(0.001),
                             "Sequencing error")(
      "m", po::value<double>()->default_value(0.3),
      "Per allele coverage")("q", po::value<double>()->default_value(0.0001),
                             "Read success probability")(
      "snv_constant", po::value<double>()->default_value(1.0),
      "SNV penalty constant")("tries", po::value<size_t>()->default_value(1))(
      "estimate_snv_constant", po::value<bool>()->default_value(true))(
      "snv_scaling_factor", po::value<double>()->default_value(0.01),
      "SNV_SCALING_FACTOR")(
      "infer_attachment", po::value<bool>()->default_value(true))
       (
      "snv_clustered", po::value<int>()->default_value(0), "SNV clustered mode if > 0");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
  po::notify(vm);

  bool infer_attachment = vm["infer_attachment"].as<bool>();
  auto data_dir = vm["data_dir"].as<string>();
  auto output_dir = vm["output_dir"].as<string>();
  if (data_dir.back() != '/')
    data_dir.push_back('/');
  if (output_dir.back() != '/')
    output_dir.push_back('/');
  SNV_CLUSTERED = vm["snv_clustered"].as<int>() > 0;
  auto param_inf_iters = vm["param_inf_iters"].as<int>();
  auto pt_inf_iters = vm["pt_inf_iters"].as<int>();

  ESTIMATE_SNV_CONSTANT = vm["estimate_snv_constant"].as<bool>();

  COUNTS_SCORE_CONSTANT_0 = vm["counts_penalty_s1"].as<double>();
  SNV_SCALING_FACTOR = vm["snv_scaling_factor"].as<double>();
  COUNTS_SCORE_CONSTANT_1 = vm["counts_penalty_s2"].as<double>();
  EVENTS_LENGTH_PENALTY = vm["event_length_penalty_k0"].as<double>();
  DATA_SIZE_PRIOR_CONSTANT = vm["tree_structure_prior_k1"].as<double>();
  USE_EVENT_LENGTHS_IN_ATTACHMENT =
      vm["use_event_lengths_in_attachment"].as<bool>();
  SEED = vm["seed"].as<int>();
  MIXTURE_SIZE = vm["mixture_size"].as<size_t>();
  NUM_REPLICAS = vm["num_replicas"].as<size_t>();
  THREADS_LIKELIHOOD = vm["threads_likelihood"].as<size_t>();
  VERBOSE = vm["verbose"].as<bool>();
  NEUTRAL_CN = vm["neutral_cn"].as<double>();
  P_E = vm["e"].as<double>();
  P_M = vm["m"].as<double>();
  P_Q = vm["q"].as<double>();
  SNV_CONSTANT = vm["snv_constant"].as<double>();
  size_t TRIES = vm["tries"].as<size_t>();
  Random<double> random(SEED);

  log("Input files have been loaded successfully");
  auto snv_constant_backup = SNV_CONSTANT;
  std::vector<CONETInferenceResult<double>> results;
  std::vector<double> snv_likelihoods;

  CONETInputData<double> provider = create_from_file(
      string(data_dir).append("ratios"), string(data_dir).append("counts"),
      string(data_dir).append("counts_squared"), ';');

  log("Loading data for SNV extension...");
  auto B = string_matrix_to_int(
      split_file_by_delimiter(string(data_dir).append("B"), ';'));
  log("Loaded matrix of alternative reads with ", B.size(), " rows and ",
      B[0].size(), " columns");
  auto D = string_matrix_to_int(
      split_file_by_delimiter(string(data_dir).append("D"), ';'));
  log("Loaded matrix of total reads with ", D.size(), " rows and ", D[0].size(),
      " columns");
  auto cluster_sizes = string_matrix_to_int(
      split_file_by_delimiter(string(data_dir).append("cluster_sizes"), ';'));
  log("Loaded cluster sizes info...");
  auto snvs = string_matrix_to_int(
      split_file_by_delimiter(string(data_dir).append("snvs_data"), ';'));
  log("Loaded snv info data");
  if (snvs.size() != D[0].size() || cluster_sizes.size() != D.size()) {
    throw "Data sizes do not match!";
  }
  provider.D = D;
  provider.B = B;
  for (auto &el : cluster_sizes) {
    provider.cluster_sizes.push_back(el[0]);
  }
  for (size_t i = 0; i < snvs.size(); i++) {
    provider.snvs.push_back(SNVEvent(i, snvs[i][1], snvs[i][2]));
  }

  auto edges = string_matrix_to_int(
      split_file_by_delimiter(string(data_dir).append("event_tree.txt"), ','));
  auto breakpoints = string_matrix_to_int(split_file_by_delimiter(
      string(data_dir).append("real_breakpoints.txt"), ','));
  counter = 0;
  std::map<size_t, size_t> bin_to_brkp;
  for (auto el : breakpoints[0]) {
    bin_to_brkp[el] = counter;
    counter++;
    log(bin_to_brkp[el], " ", el);
  }
  log("Edges ", edges.size());

  /*
  Create starting tree
  */
  auto tree = EventTree();
  for (size_t i = 0; i < edges.size(); i++) {
    log(edges[i].size());
    log("Adding edge ", edges[i][0], " ", edges[i][1], " ", edges[i][2], " ",
        edges[i][3]);
    auto nodes = tree.get_descendants(tree.get_root());
    log("Nodes: ", nodes.size());
    for (auto n : nodes) {
      auto label = tree.get_node_label(n);
      if (label.first == bin_to_brkp[edges[i][0]] &&
          label.second == bin_to_brkp[edges[i][1]]) {
        size_t e1 = edges[i][2];
        size_t e2 = edges[i][3];
        size_t b1 = bin_to_brkp[e1];
        size_t b2 = bin_to_brkp[e2];
        auto child = std::make_pair(b1, b2);
        tree.add_leaf(n, child);
      }
    }
  }

  log("Tree size ", tree.get_size());

  auto real_attachment = string_matrix_to_int(
      split_file_by_delimiter(string(data_dir).append("attachment.txt"), ';'));
  log("Loaded real attachment");
  Attachment real_at{std::make_pair(0, 0), real_attachment.size()};
  for (size_t cell = 0; cell < real_attachment.size(); cell++) {
    log("Cell ", cell);
    if (real_attachment[cell][0] == real_attachment[cell][1]) {
        real_at.set_attachment(cell, std::make_pair(0, 0));
    } else
        real_at.set_attachment(cell, std::make_pair(bin_to_brkp[real_attachment[cell][0]], bin_to_brkp[real_attachment[cell][1]]));
  }
    auto max_attachment = real_at;
  if (infer_attachment) {
      auto likelihood_data_raw = string_matrix_to_double(split_file_by_delimiter(
          string(output_dir).append("conet_parameters"), ';'));
      auto no_brkp = Gauss::Gaussian<double>{likelihood_data_raw[0][0],
                                             likelihood_data_raw[0][1], random};
      std::vector<double> weights;
      std::vector<double> means;
      std::vector<double> sds;
      for (size_t i = 1; i < likelihood_data_raw.size(); i++) {
        weights.push_back(likelihood_data_raw[i][0]);
        means.push_back(-likelihood_data_raw[i][1]);
        sds.push_back(likelihood_data_raw[i][2]);
      }
      auto brkp = Gauss::GaussianMixture<double>{weights, means, sds, random};
      LikelihoodData<double> lik_data{no_brkp, brkp};
      const std::map<MoveType, double> move_probabilities = {
          {DELETE_LEAF, 100.0},       {ADD_LEAF, 30.0},     {PRUNE_REATTACH, 30.0},
          {SWAP_LABELS, 30.0},        {CHANGE_LABEL, 30.0}, {SWAP_SUBTREES, 30.0},
          {SWAP_ONE_BREAKPOINT, 30.0}};

      LikelihoodCoordinator<double> lik_coord(lik_data, tree, provider,
                                              random.next_int());
      TreeSamplerCoordinator<double> coord(tree, lik_coord, 123, provider,
                                           move_probabilities);
      lik_coord.calculate_likelihood();
      lik_coord.persist_likelihood_calculation_result();
      coord.recalculate_counts_dispersion_penalty();
      max_attachment = lik_coord.get_max_attachment();
   }
  log("Saving inferred tree");
  std::ofstream tree_file{string(output_dir).append("inferred_tree")};
  tree_file << TreeFormatter::to_string_representation(tree);
  std::ofstream attachment_file{
      string(output_dir).append("inferred_attachment")};
  attachment_file << max_attachment;

  SNV_CONSTANT = 1.0;
  SNVSolver<double> snv_solver(provider);
  auto snv_before = snv_solver.insert_snv_events(
      tree, max_attachment, SNVParams<double>(P_E, P_M, P_Q), true);

  std::cout << "\nSNV likelihood: " << snv_before;
  std::ofstream snv_file{string(output_dir).append("inferred_snvs")};
  for (auto n : snv_solver.likelihood.node_to_snv) {
    for (auto snv : n.second) {
      snv_file << label_to_str(n.first->label) << ";" << snv << "\n";
    }
  }

  return 0;
}
