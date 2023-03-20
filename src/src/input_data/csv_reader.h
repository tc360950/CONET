#ifndef CSV_READER_H
#define CSV_READER_H
#include "input_data.h"

CONETInputData<double> create_from_file(std::string path,
                                        std::string summed_counts_path,
                                        std::string squared_counts_path,
                                        char delimiter);

std::vector<std::vector<std::string>> split_file_by_delimiter(std::string path, char delimiter);

std::vector<std::vector<int>> string_matrix_to_int(std::vector<std::vector<std::string>> data);
std::vector<std::vector<double>> string_matrix_to_double(std::vector<std::vector<std::string>> data);

#endif // !CSV_READER_H
