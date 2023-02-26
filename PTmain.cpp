#include <iostream>

#include "Pair_Trading.hpp"
int main() {
    Stock_data data;
    vector<double> tmp;
    data = data.read_csv("./check_data/check_data_13.csv");
    data = data.to_log();
    // data.show_print();
    data.convert_to_mat();
    data.bic_tranformed();
    data.JCI_AutoSelection();
    data.Johanson_Mean();
    data.Johanson_Stdev();
    //   data._convert_data.print();
}