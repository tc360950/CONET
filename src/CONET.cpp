#include <utility>
#include <chrono>
#include <cstdlib>
#include <tuple>
#include <chrono>
#include <fstream>

#include "src/tree/event_tree.h"
#include "src/input_data/csv_reader.h"
#include "src/tree_sampler_coordinator.h"
#include "src/utils/random.h"
#include "src/parallel_tempering_coordinator.h"
#include "src/tree/tree_formatter.h"
#include "src/conet_result.h"

#include <boost/program_options.hpp>

using namespace std;

namespace po = boost::program_options;

int main(int argc, char **argv) {
	po::options_description description("MyTool Usage");

	description.add_options()
		("data_dir", po::value<string>()->required(),  "Path to directory containing input files.")
		("output_dir",  po::value<string>()->required(), "Path to output directory. Inference results will be saved there.")
		("param_inf_iters",  po::value<int>()->default_value(100000), "Number of MCMC iterations for joint tree and model parameters inference.")
		("pt_inf_iters",  po::value<int>()->default_value(100000), "Number of MCMC iterations for tree inference.")
		("counts_penalty_s1",  po::value<double>()->default_value(0.0), "Constant controlling impact of penalty for large discrepancies between inferred and real count matrices.")
		("counts_penalty_s2",  po::value<double>()->default_value(0.0), "Constant controlling impact of penalty for inferring clusters with changed copy number equal to basal ploidy.")
		("event_length_penalty_k0",  po::value<double>()->default_value(1.0), "Constant controlling impact of penalty for long inferred events.")
		("tree_structure_prior_k1",  po::value<double>()->default_value(1.0), "Constant controlling impact of data size part of tree structure prior.")
		("use_event_lengths_in_attachment",  po::value<bool>()->default_value(true), "If True cell attachment probability will depend on average event length in the history, otherwise it will be uniform.")
		("seed",  po::value<int>()->default_value(12312), "Seed for C++ RNG")
		("mixture_size",  po::value<size_t>()->default_value(4), "Initial number of components in difference distribution for breakpoint loci.")
		("num_replicas",  po::value<size_t>()->default_value(5), "Number of tempered chain replicas in MAP event tree search.")
		("threads_likelihood",  po::value<size_t>()->default_value(4), "Number of threads which will be used for the most demanding likelihood calculations.")
		("verbose",  po::value<bool>()->default_value(true), "True if CONET should print messages during inference.")
		("neutral_cn",  po::value<double>()->default_value(2.0), "Neutral copy number")
		        ("e",  po::value<double>()->default_value(0.001), "Sequencing error")
        ("m",  po::value<double>()->default_value(0.3), "Per allele coverage")
		 ("q",  po::value<double>()->default_value(0.0001), "Read success probability")
        ("snv_constant",  po::value<double>()->default_value(1.0), "SNV penalty constant")
		("use_snv_in_swap",  po::value<bool>()->default_value(false), "SNV penalty constant")
		("snv_batch_size",  po::value<size_t>()->default_value(0))
		("snv_burnin",  po::value<size_t>()->default_value(0))
		("tries",  po::value<size_t>()->default_value(5))

				;
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
	po::notify(vm);

	size_t TRIES = vm["tries"].as<size_t>();
	auto data_dir = vm["data_dir"].as<string>();
	auto output_dir = vm["output_dir"].as<string>();
    if(data_dir.back() != '/') data_dir.push_back('/');
    if(output_dir.back() != '/') output_dir.push_back('/');

	auto param_inf_iters = vm["param_inf_iters"].as<int>();
	auto pt_inf_iters = vm["pt_inf_iters"].as<int>();

	USE_SNV_IN_SWAP = vm["use_snv_in_swap"].as<bool>();
	SNV_BATCH_SIZE = vm["snv_batch_size"].as<size_t>();
	SNV_BURNIN = vm["snv_burnin"].as<size_t>();
	COUNTS_SCORE_CONSTANT_0 = vm["counts_penalty_s1"].as<double>();
	COUNTS_SCORE_CONSTANT_1 = vm["counts_penalty_s2"].as<double>();
	EVENTS_LENGTH_PENALTY = vm["event_length_penalty_k0"].as<double>();
	DATA_SIZE_PRIOR_CONSTANT = vm["tree_structure_prior_k1"].as<double>();
	USE_EVENT_LENGTHS_IN_ATTACHMENT = vm["use_event_lengths_in_attachment"].as<bool>();
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

	Random<double> random(SEED);
    CONETInputData<double> provider = create_from_file(string(data_dir).append("ratios"), string(data_dir).append("counts"), string(data_dir).append("counts_squared"), ';');

    log("SNV batch size: ", SNV_BATCH_SIZE);
	log("Loading data for SNV extension...");
    auto B = string_matrix_to_int(split_file_by_delimiter(string(data_dir).append("B"), ';'));
    log("Loaded matrix of alternative reads with ", B.size(), " rows and ", B[0].size(), " columns");
    auto D = string_matrix_to_int(split_file_by_delimiter(string(data_dir).append("D"), ';'));
    log("Loaded matrix of total reads with ", D.size(), " rows and ", D[0].size(), " columns");
    auto cluster_sizes = string_matrix_to_int(split_file_by_delimiter(string(data_dir).append("cluster_sizes"), ';'));
    log("Loaded cluster sizes info...");
    auto snvs = string_matrix_to_int(split_file_by_delimiter(string(data_dir).append("snvs_data"), ';'));
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
        provider.snvs.push_back(SNVEvent(i, snvs[i][1]));
    }

    log("Input files have been loaded successfully");

    std::vector<CONETInferenceResult<double>> results;
    std::vector<double> snv_likelihoods;
    for (size_t i = 0; i < TRIES; i++)
    {
        log("Starting try ", i);
        SNV_CONSTANT = 0.0;
        ParallelTemperingCoordinator<double> PT(provider, random);
        CONETInferenceResult<double> result = PT.simulate(param_inf_iters, pt_inf_iters);
        log("Tree inference has finished");
        results.push_back(result);
        SNVSolver<double> snv_solver(provider);

        SNV_CONSTANT = 1.0;
        log("Inferring SNVs");
        auto snv_lik = snv_solver.insert_snv_events(result.tree, result.attachment, SNVParams<double>(P_E, P_M, P_Q));
        snv_likelihoods.push_back(snv_lik);
    }


    for(size_t i = 0; i < 6; i++) {
        double constant = i == 0 ? 0.0 : 1.0 / std::pow(10.0, (double) (i - 1));

        bool max_set = false;
        double max_score = 0.0;
        size_t max_indx = 0;

        for (size_t j =0; j < results.size(); j++) {
            if (!max_set || results[j].likelihood + constant * snv_likelihoods[j] > max_score) {
                max_set = true;
                max_indx = j;
                max_score = results[j].likelihood + constant * snv_likelihoods[j] > max_score;
            }
        }


        std::ofstream tree_file{ string(output_dir).append("inferred_tree_").append(std::to_string(i)) };
        tree_file << TreeFormatter::to_string_representation(results[max_indx].tree);

        std::ofstream attachment_file{ string(output_dir).append("inferred_attachment_").append(std::to_string(i)) };
        attachment_file << results[max_indx].attachment;


        SNV_CONSTANT = 1.0;
        SNVSolver<double> snv_solver(provider);
        auto snv_before = snv_solver.insert_snv_events(results[max_indx].tree, results[max_indx].attachment, SNVParams<double>(P_E, P_M, P_Q));
        std::ofstream snv_file{ string(output_dir).append("inferred_snvs_").append(std::to_string(i)) };
        for (auto n : snv_solver.likelihood.node_to_snv) {
            for (auto snv : n.second) {
                snv_file << label_to_str(n.first->label) << ";" <<snv << "\n";
            }
        }

    }
    return 0;
}
