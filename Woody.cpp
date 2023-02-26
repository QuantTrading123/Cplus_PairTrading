#include "Woody.hpp"
template <typename T>

std::vector<size_t> argsort(const std::vector<T>& array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });
    cout << array[0] << " " << array[1] << " " << array[2] << endl;
    return indices;
}

double** mat_to_carr(arma::cx_mat& M, std::size_t& n, std::size_t& m) {
    const std::size_t nrows = M.n_rows;
    const std::size_t ncols = M.n_cols;
    double** array = (double**)malloc(nrows * sizeof(double*));

    for (std::size_t i = 0; i < nrows; i++) {
        array[i] = (double*)malloc(ncols * sizeof(double));
        for (std::size_t j = 0; j < ncols; ++j)
            array[i][j] = M(i + j * ncols).real();
    }

    n = nrows;
    m = ncols;

    return array;
}
double** mat_to_val(arma::cx_vec& M, std::size_t& x, std::size_t& y) {
    const std::size_t nrows = M.n_rows;
    const std::size_t ncols = M.n_cols;
    double** array = (double**)malloc(nrows * sizeof(double*));

    for (std::size_t i = 0; i < nrows; i++) {
        array[i] = (double*)malloc(ncols * sizeof(double));
        for (std::size_t j = 0; j < ncols; ++j)
            array[i][j] = M(i + j * ncols).real();
    }

    x = nrows;
    y = ncols;

    return array;
}
Stock_data& Stock_data::operator=(Stock_data const& other) {
    if (this == &other) {
        return *this;
    }
    _data = other._data;
    return *this;
}
Stock_data& Stock_data::read_csv(string _filename) {
    ifstream myFile(_filename);
    if (!myFile.is_open()) throw runtime_error("Could not open file");
    string line, colname;
    char delim = ',';
    string val;
    if (myFile.good()) {
        getline(myFile, line);
    }
    while (getline(myFile, line)) {
        vector<double> v;
        int index;
        char c;
        double n1, n2;
        // stringstream ss(line);
        istringstream(line) >> index >> c >> n1 >> c >> n2;
        v.push_back(n1);
        v.push_back(n2);
        // while (getline(ss, val, delim)) {
        //     v.push_back(stod(val));
        // }
        _data.push_back(v);
    }
    // cout << _data.size() << _data[0].size() << endl;
    // for (const auto& row : _data) {
    //     for (const auto& s : row) std::cout << s << ' ';
    //     std::cout << std::endl;
    // }
    return *this;
}
Stock_data& Stock_data::to_log() {
    vector<vector<double>>::iterator it;
    vector<double>::iterator vit;
    for (it = _data.begin(); it != _data.end(); it++) {
        for (vit = it->begin(); vit != it->end(); vit++) {
            *vit = log(*vit);
        }
    }
    return *this;
}
Stock_data& Stock_data::show_print() {
    for (const auto& row : _data) {
        for (const auto& s : row) std::cout << s << ' ';
        std::cout << std::endl;
    }
    return *this;
}
Stock_data& Stock_data::convert_to_mat() {
    std::vector<double> z_flat;
    size_t m_nrow = _data.size();
    size_t m_ncol = _data[0].size();
    for (size_t i = 0; i < m_ncol; ++i) {
        for (size_t j = 0; j < m_nrow; ++j) {
            z_flat.push_back(_data[j][i]);
        }
    }
    arma::mat tmp(&z_flat.front(), m_nrow, m_ncol);
    cout << "lalalla" << endl;
    _convert_data = tmp;
    _convert_data.print("convert data");
    k = _convert_data.n_cols;
    n = _convert_data.n_rows;
    return *this;
}

Stock_data& Stock_data::bic_tranformed() {
    opt_p = order_select(5);
    cout << opt_p << endl;
    return *this;
}
Stock_data& Stock_data::JCI_AutoSelection() {
    int Numobs = _convert_data.n_rows;
    int k = _convert_data.n_cols;
    int opt_q = opt_p - 1;
    int Tl = Numobs - opt_p;
    double inf = std::numeric_limits<double>::infinity();
    arma::mat TraceTest_table = arma::zeros(5, k);
    double BIC_table[5] = {0};
    double BIC_List[5] = {inf, inf, inf, inf, inf};
    for (int i = 0; i < 5; ++i) {
        cout << "BIC value 111:" << BIC_List[i] << endl;
    }
    int opt_model_num = 0;

    for (int mr = 0; mr < 3; ++mr) {
        double tr_H[2];
        auto ut = JCI_Trace(mr + 1, tr_H);
        TraceTest_table(mr, arma::span(0, 1)) = {1, 0};
        arma::mat tmp(&tr_H[0], 1, 2);
        TraceTest_table(mr, arma::span(0, 1)) = tmp;
        arma::mat eps = ut;
        arma::mat sq_Res_r1 = eps.t() * eps / Tl;
        arma::mat errorRes_r1 = eps * sq_Res_r1.i() * eps.t();
        double trRes_r1 = arma::trace(errorRes_r1);
        double L = (-k * Tl * 0.5 * log(2 * M_PI) - (Tl * 0.5) * log(arma::det(sq_Res_r1))) - 0.5 * trRes_r1;
        double deg_Fred = 0;
        if (mr == 0) {
            deg_Fred = 2 * k + opt_q * (k * k);
        } else if (mr == 1) {
            deg_Fred = 2 * k + 1 + opt_q * (k * k);
        } else if (mr == 2) {
            deg_Fred = 3 * k + 1 + opt_q * (k * k);
        } else if (mr == 3) {
            deg_Fred = 3 * k + 2 + opt_q * (k * k);
        } else if (mr == 4) {
            deg_Fred = 4 * k + 2 + opt_q * (k * k);
        }
        BIC_table[mr] = -2 * log(L) + deg_Fred * log(Numobs * k);
        if (TraceTest_table(mr, 0) == 1 && TraceTest_table(mr, 1) == 0) {
            BIC_List[mr] = BIC_table[mr];
            ++opt_model_num;
        }
    }
    auto it = min_element(begin(BIC_List), end(BIC_List));
    auto Opt_model = std::distance(begin(BIC_List), it);
    cout << "Opt model : " << Opt_model << endl;
    opt_model = Opt_model;
    return *this;
}
Stock_data& Stock_data::Johanson_Mean() {
    std::cout << "======== Johanson mean========" << std::endl;
    int NumDim = 2;
    int lagp = opt_p - 1;
    arma::mat sumgamma = arma::zeros(NumDim, NumDim);
    for (int i = 0; i < lagp; ++i) {
        sumgamma += list_gamma[opt_model][i];
    }
    arma::mat GAMMA = arma::eye(NumDim, NumDim) - sumgamma;
    arma::mat alpha_orthogonal = list_jci_alpha[opt_model];
    alpha_orthogonal.print("alpha orthogonal");
    arma::mat alpha_t = list_jci_alpha[opt_model].t();
    alpha_orthogonal(1, 0) = -(alpha_t(0, 0) * alpha_orthogonal(0, 0)) / alpha_t(0, 1);
    alpha_orthogonal = alpha_orthogonal / arma::accu(arma::abs(alpha_orthogonal));
    arma::mat beta_orthogonal = list_jci_beta[opt_model];
    w1 = beta_orthogonal(1, 0);
    w2 = beta_orthogonal(0, 0);
    beta_orthogonal.print("beta orthogonal");
    arma::mat beta_t = list_jci_beta[opt_model].t();
    beta_orthogonal(1, 0) = -(beta_t(0, 0) * beta_orthogonal(0, 0)) / beta_t(0, 1);
    beta_orthogonal = beta_orthogonal / arma::accu(arma::abs(beta_orthogonal));
    arma::mat temp1 = (alpha_orthogonal.t() * GAMMA * beta_orthogonal);
    arma::mat C = beta_orthogonal * temp1 * alpha_orthogonal.t();
    arma::mat temp2 = (list_jci_alpha[opt_model].t() * list_jci_alpha[opt_model]).i();
    arma::mat alpha_hat = list_jci_alpha[opt_model] * temp2;
    arma::mat temp3 = GAMMA * C - arma::eye(NumDim, NumDim);
    arma::mat C0 = Ct[opt_model][0];
    arma::mat C1 = Ct[opt_model][2];
    arma::mat D0 = Ct[opt_model][1];
    arma::mat D1 = Ct[opt_model][3];
    arma::mat wrong1 = list_jci_alpha[opt_model] * C0;
    C0 = list_jci_alpha[opt_model] * C0 + C1 + list_jci_alpha[opt_model] * D0 + D1;
    arma::mat Ct = list_jci_alpha[opt_model] * D0 + D1;
    arma::mat expect_intercept = alpha_hat.t() * temp3 * C0;
    arma::mat expect_slope = alpha_hat.t() * temp3 * Ct;
    Johanson_mean = expect_intercept(0, 0);
    return *this;
}
Stock_data& Stock_data::Johanson_Stdev() {
    int rank = 1;
    std::cout << "======== Johanson standard corrected ========" << std::endl;
    int NumDim = 2;
    int lag_p = opt_p - 1;
    cout << "lag_p " << lag_p << endl;
    arma::mat alpha = list_jci_alpha[opt_model];
    arma::mat beta = list_jci_beta[opt_model];
    vector<arma::mat> mod_gamma = list_gamma[opt_model];
    arma::mat ut = list_ut[opt_model];
    arma::mat tilde_B = beta;
    arma::mat tilde_A = alpha;
    if (lag_p > 0) {
        //建立～A
        arma::mat tilde_A_11 = alpha;
        arma::mat tilde_A_21 = arma::zeros(NumDim * lag_p, 1);
        arma::mat tilde_A_12 = arma::zeros(NumDim, NumDim * lag_p);
        //建立～B
        arma::mat tilde_B_11 = beta;
        // tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        arma::mat tilde_B_3 = arma::zeros(NumDim + NumDim * lag_p, NumDim * lag_p);
        mod_gamma[0].print();
        //# 用同一個迴圈同時處理～A與～B
        for (int qi = 0; qi < lag_p; ++qi) {
            tilde_A_12(arma::span(0, NumDim - 1), arma::span(qi * NumDim, ((qi + 1) * NumDim) - 1)) = mod_gamma[qi];
            arma::mat v = arma::join_cols(arma::eye(NumDim, NumDim), -arma::eye(NumDim, NumDim));
            tilde_B_3(arma::span(qi * NumDim, NumDim * (2 + qi) - 1), arma::span(qi * NumDim, NumDim * (1 + qi) - 1)) = v;
        }
        tilde_A_12.print("A_12");
        tilde_B_3.print("B_3");

        arma::mat tilde_A_22 = arma::eye(NumDim * lag_p, NumDim * lag_p);
        arma::mat tA1 = arma::join_cols(tilde_A_11, tilde_A_21);
        arma::mat tA2 = arma::join_cols(tilde_A_12, tilde_A_22);
        arma::mat tmpA = arma::join_rows(tA1, tA2);
        tilde_A = tmpA;

        arma::mat tBA = arma::join_cols(tilde_B_11, tilde_A_21);
        tilde_B = arma::join_rows(tBA, tilde_B_3);
        tilde_A.print("tilde_A print");
        tilde_B.print("tilde B print");
    }

    std::cout << ">>>>>>>> <<<<<<<<<" << std::endl;
    arma::mat tilde_sigma = arma::zeros(NumDim * (lag_p + 1), NumDim * (lag_p + 1));
    tilde_sigma.print();
    auto tmp = ut.t() * ut;
    std::cout << ut.n_rows << std::endl;
    tmp.raw_print("UUUUUUUUUTTTTTTTT");
    tilde_sigma(arma::span(0, NumDim - 1), arma::span(0, NumDim - 1)) = (ut.t() * ut) / (ut.n_rows - 1);
    arma::mat tilde_J = arma::zeros(1, 1 + NumDim * lag_p);
    tilde_J(0, 0) = 1;
    std::cout << ">>>>>>>> <<<<<<<<<" << std::endl;
    arma::mat var;
    if (lag_p == 0) {
        arma::mat tmp1 = arma::eye(rank, rank) + beta.t() * alpha;
        std::cout << ">>>>>>>>1 <<<<<<<<<" << std::endl;
        arma::mat tmp2 = arma::kron(tmp1, tmp1);
        std::cout << ">>>>>>>>2 <<<<<<<<<" << std::endl;
        arma::mat tmp = arma::eye(rank, rank);
        std::cout << ">>>>>>>>3 <<<<<<<<<" << std::endl;
        arma::mat tmp3 = (tmp - tmp2).i();
        std::cout << ">>>>>>>>4 <<<<<<<<<" << std::endl;
        arma::mat omega = (ut.t() * ut) / (ut.n_rows - 1);
        std::cout << ">>>>>>>> 5<<<<<<<<<" << std::endl;
        arma::mat tmp4 = beta.t() * omega * beta;
        std::cout << ">>>>>>>>6 <<<<<<<<<" << std::endl;
        var = tmp3 * tmp4;
        std::cout << ">>>>>>>>7 <<<<<<<<<" << std::endl;
        var.raw_print();

    } else {
        tilde_A.print();
        tilde_B.print();
        arma::mat tmp1 = arma::eye(NumDim * (lag_p + 1) - 1, NumDim * (lag_p + 1) - 1) + tilde_B.t() * tilde_A;
        std::cout << ">>>>>>>>1 <<<<<<<<<" << std::endl;
        arma::mat tmp2 = arma::kron(tmp1, tmp1);
        int k = (NumDim * (lag_p + 1) - 1) * (NumDim * (lag_p + 1) - 1);
        std::cout << k << std::endl;
        std::cout << ">>>>>>>>2 <<<<<<<<<" << std::endl;
        arma::mat tmp = arma::eye(k, k);
        arma::mat tmp3 = (tmp - tmp2).i();
        tilde_sigma.print("tilde sigma");
        arma::mat tmp4 = tilde_B.t() * tilde_sigma * tilde_B;
        std::cout << ">>>>>>>>3 <<<<<<<<<" << std::endl;
        tmp4.print("tmp4 ");
        tmp3.print("tmp3");
        arma::mat v = vectorise(tmp4, 0);
        v.print("v");

        std::cout << ">>>>>>>>4 <<<<<<<<<" << std::endl;
        arma::mat tmp5 = tmp3 * v;
        arma::mat sigma_telta_beta = arma::zeros(NumDim * (lag_p + 1) - 1, NumDim * (lag_p + 1) - 1);
        tmp5.print("tmp5");

        std::cout << ">>>>>>>>5 <<<<<<<<<" << std::endl;
        for (int i = 0; i < NumDim * (lag_p + 1) - 1; ++i) {
            for (int j = 0; j < NumDim * (lag_p + 1) - 1; ++j) {
                sigma_telta_beta(i, j) = tmp5(i + j * (NumDim * (lag_p + 1) - 1), 0);
            }
        }

        sigma_telta_beta.print("stb");
        var = sigma_telta_beta;  //* tilde_J.t();
        var.print("var");
    }
    Johanson_std = sqrt(var(0, 0));
    cout << "Johansan std :" << Johanson_std << endl;
    cout << "w1 :" << w2 << "|"
         << "w2 :" << w1 << "|"
         << "MU :" << Johanson_mean << "|"
         << "std :" << Johanson_std << endl;

    return *this;
}
