#include "parameters.h"

size_t NUM_REPLICAS = 1;
bool USE_EVENT_LENGTHS_IN_ATTACHMENT = false;
double DATA_SIZE_PRIOR_CONSTANT = 0.01;
double COUNTS_SCORE_CONSTANT_0 = 0.1;
double COUNTS_SCORE_CONSTANT_1 = 0.1;
double EVENTS_LENGTH_PENALTY = 1.0;
size_t PARAMETER_RESAMPLING_FREQUENCY = 10;
size_t NUMBER_OF_MOVES_BETWEEN_SWAPS = 10;
size_t THREADS_LIKELIHOOD = 10;
size_t MIXTURE_SIZE = 8;
long SEED = 12312414;
bool VERBOSE = false;
double NEUTRAL_CN = 2;
double P_E = 0.001;
double P_M = 0.03;
double P_Q = 0.000001;
double SNV_CONSTANT = 1.0;
bool USE_SNV_IN_SWAP = false;
size_t SNV_BATCH_SIZE = 0;
size_t SNV_BURNIN= 0;