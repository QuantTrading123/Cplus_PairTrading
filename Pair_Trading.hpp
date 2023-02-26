#include <math.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
using namespace std;
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
class Stock_data {
   public:
    Stock_data() = default;
    ~Stock_data() = default;
    Stock_data& operator=(Stock_data const& other);
    Stock_data& read_csv(string _file);
    Stock_data& to_log();
    Stock_data& show_print();
    Stock_data& drop_stationary();
    Stock_data& convert_to_mat();
    Stock_data& bic_tranformed();
    Stock_data& JCI_AutoSelection();
    Stock_data& JCItestpara_spilCt();
    Stock_data& Johanson_Mean();
    Stock_data& Johanson_Stdev();
    vector<double> operator()(string _id) const;
    arma::mat _estimate_var(int p) {
        arma::mat xt(n - p, (k * p) + 1);
        xt.ones();
        int _k = 0;
        for (int i = 0; i < (n - p); i++) {
            arma::rowvec a(1 + k * p);
            a.zeros();
            _k = 0;
            a.at(_k++) = 1.0;
            for (int j = 0; j < p; j++) {
                a.at(_k++) = _convert_data(i + p - j - 1, 0);
                a.at(_k++) = _convert_data(i + p - j - 1, 1);
            }
            xt.row(i) = a;
        }
        arma::mat zt = _convert_data.rows(p, n - 1);
        arma::mat beta = (xt.t() * xt).i() * xt.t() * zt;

        arma::mat A = zt - (xt * beta);
        arma::mat sigma = ((A.t()) * A) / (double)(n - p);
        return sigma;
    }

    arma::uword order_select(int max_p) {
        cout << "order select lalalla" << endl;
        arma::rowvec bic(max_p);
        bic.zeros();
        for (int p = 1; p < max_p + 1; p++) {
            arma::mat sigma = _estimate_var(p);
            bic.at(p - 1) = log(det(sigma)) + log(n) * (double)p * (double)(k * k) / (double)n;
        }
        cout << "order select lalalla" << endl;
        arma::uword bic_order = bic.index_min() + 1;

        return bic_order;
    }
    auto JCI_Trace(int model_type, double* TraceTest_H) {
        int lag_p = opt_p - 1;
        int NumObs = _convert_data.n_rows;
        int NumDim = _convert_data.n_cols;
        cout << "model type " << model_type << endl;
        arma::mat jci_alpha;
        arma::mat jci_beta;
        arma::mat ut;
        vector<arma::mat> gamma;
        arma::mat dY_ALL = _convert_data(arma::span(1, NumObs - 1), arma::span(0, NumDim - 1)) - _convert_data(arma::span(0, NumObs - 2), arma::span(0, NumDim - 1));
        arma::mat dY = dY_ALL(arma::span(lag_p, dY_ALL.n_rows - 1), arma::span(0, dY_ALL.n_cols - 1));
        arma::mat Ys = _convert_data(arma::span(lag_p, NumObs - 2), arma::span(0, NumDim - 1));
        arma::mat dX;
        if (lag_p == 0) {
            if (model_type == 1) {
                dX = arma::zeros(NumObs - 1, NumDim);
            } else if (model_type == 2) {
                dX = arma::zeros(NumObs - 1, NumDim);
                Ys = arma::join_rows(Ys, arma::ones(NumObs - lag_p - 1, 1));
            } else if (model_type == 3) {
                dX = arma::ones(NumObs - lag_p - 1, 1);
            } else if (model_type == 4) {
                dX = arma::ones(NumObs - lag_p - 1, 1);
                arma::vec tmp = arma::regspace(1, NumObs - lag_p - 1);
                Ys = arma::join_rows(Ys, arma::reshape(tmp, NumObs - lag_p - 1, 1));
            } else if (model_type == 5) {
                arma::vec tmp = arma::regspace(1, NumObs - lag_p - 1);
                dX = arma::join_rows(arma::ones(NumObs - lag_p - 1, 1), arma::reshape(tmp, NumObs - lag_p - 1, 1));
            }
        } else {
            std::cout << "????" << std::endl;
            dX = arma::zeros(NumObs - lag_p - 1, NumDim * lag_p);
            for (int xi = 0; xi < lag_p; ++xi) {
                std::cout << "!!????" << std::endl;
                dX(arma::span(0, dX.n_rows - 1), arma::span(xi * NumDim, (xi + 1) * NumDim - 1)) = dY_ALL(arma::span(lag_p - xi - 1, NumObs - xi - 3), arma::span(0, dY_ALL.n_cols - 1));

                if (model_type == 2) {
                    Ys = arma::join_rows(Ys, arma::ones(NumObs - lag_p - 1, 1));
                    std::cout << "!!!!" << std::endl;
                } else if (model_type == 3) {
                    dX = arma::join_rows(dX, arma::ones(NumObs - lag_p - 1, 1));
                } else if (model_type == 4) {
                    arma::vec tmp = arma::regspace(1, NumObs - lag_p - 1);
                    Ys = arma::join_rows(Ys, arma::reshape(tmp, NumObs - lag_p - 1, 1));
                    dX = arma::join_rows(dX, arma::ones(NumObs - lag_p - 1, 1));
                } else if (model_type == 5) {
                    arma::vec tmp = arma::regspace(1, NumObs - lag_p - 1);
                    dX = arma::join_rows(arma::ones(NumObs - lag_p - 1, 1), arma::reshape(tmp, NumObs - lag_p - 1, 1));
                }
            }
        }
        cout << "-----------------------" << endl;
        arma::mat dX_2 = dX.t() * dX;
        arma::mat M;

        if (arma::accu(dX_2) == 0) {
            M = arma::eye(NumObs - lag_p - 1, NumObs - lag_p - 1) - dX * dX.t();
        } else {
            M = arma::eye(NumObs - lag_p - 1, NumObs - lag_p - 1) - dX * dX_2.i() * dX.t();
        }
        arma::mat R0 = dY.t() * M;
        arma::mat R1 = Ys.t() * M;
        arma::mat S00 = R0 * R0.t() / (NumObs - lag_p - 1);
        arma::mat S01 = R0 * R1.t() / (NumObs - lag_p - 1);
        arma::mat S10 = R1 * R0.t() / (NumObs - lag_p - 1);
        arma::mat S11 = R1 * R1.t() / (NumObs - lag_p - 1);
        S00.print("SO0");
        S01.print("S01");
        S10.print("S10");
        S11.print("S11");
        arma::mat tmp_eig = S11.i() * S10 * S00.i() * S01;
        // arma::vec eigval2;
        // arma::mat eigvec2;
        // arma::eig_sym(eigval2, eigvec2, tmp_eig);
        // eigval2.print("sym eigval");

        arma::cx_vec eigval1;
        arma::cx_mat leigvec1;
        arma::cx_mat reigvec1;
        // arma::cx_mat eigvec2;
        arma::eig_gen(eigval1, leigvec1, reigvec1, tmp_eig, "balance");
        eigval1.print("eigval1");
        leigvec1.print("eigvec1");
        reigvec1.print("reigvec1");
        double first_col_sum = 0;
        std::size_t n, m;
        auto array = mat_to_carr(reigvec1, n, m);
        std::size_t x, y;
        auto val_array = mat_to_val(eigval1, x, y);
        vector<double> vec_eig_val;
        arma::mat reconstruct_eigvec(n, m);
        arma::mat reconstruct_eigval(x, y);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                std::cout << array[i][j] << " ";
                /// reconstruct_eigvec(i, j) = array[i][j];
            }
            std::cout << std::endl;
        }
        for (std::size_t i = 0; i < x; ++i) {
            for (std::size_t j = 0; j < y; ++j) {
                std::cout << val_array[i][j] << " ";
                reconstruct_eigval(i, j) = val_array[i][j];
                vec_eig_val.push_back(-val_array[i][j]);
            }
            std::cout << std::endl;
        }
        std::vector<size_t> index = argsort(vec_eig_val);
        // for (int item : index) {
        //     cout << "index :" << item << "\t";
        // }
        // sort(index.begin(), index.end(), greater<size_t>());
        // for (int item : index) {
        //     cout << "reindex :" << item << "\t";
        // }
        std::cout << "re array \n"
                  << std::endl;
        double reindex_array[n][m];
        int c = 0;
        for (int item : index) {
            for (int r = 0; r < n; ++r) {
                std::cout << array[r][item] << " ";
                reindex_array[r][c] = array[r][item];
            }
            std::cout << std::endl;
            c++;
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                std::cout << reindex_array[i][j] << " ";
            }
            std::cout << std::endl;
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                reconstruct_eigvec(i, j) = reindex_array[i][j];
            }
            std::cout << std::endl;
        }
        reconstruct_eigvec.print("reconstruct eigvec");
        reconstruct_eigval.print("reconstruct eigval");
        for (int i = 0; i < 2; ++i) {
            first_col_sum += abs(reconstruct_eigvec(i, 0));
        }

        cout << " first_col_sum " << first_col_sum << endl;

        arma::mat test_eigvecs_st = reconstruct_eigvec / first_col_sum;
        test_eigvecs_st.print("test eigvecs_st");
        // arma::eig_pair(eigval, eigvec2, show_S, S11);
        // eigval.print("eigenval");
        // eigvec2.print("eigenvector");
        // cout << eigval.n_rows << eigval.n_cols << endl;

        // arma::mat eigvec = {{40.01801448, -454.3070705},
        //                     {-63.52375491, 94.34738988}};

        arma::mat eigvecs_st = test_eigvecs_st;
        // beta
        jci_beta = eigvecs_st(arma::span(0, eigvecs_st.n_rows - 1), 0);
        jci_beta.reshape(NumDim, 1);
        // Alpha
        arma::mat a = eigvecs_st(arma::span(0, eigvecs_st.n_rows - 1), 0);
        cout << " model type :" << model_type << endl;
        cout << "blablalba" << endl;
        a.print("aaaaaa");
        a.reshape(1, a.n_rows);
        cout << "blablalba" << endl;
        arma::mat jci_a = S01 * a.t();
        jci_alpha = jci_a / arma::accu(arma::abs(jci_a));
        jci_alpha.print("jci alpha JCI\n");
        jci_beta.print("jci beta JCI\n");
        arma::mat c0 = {0};
        arma::mat d0 = {0};
        arma::mat c1 = arma::zeros(NumDim, 1);
        arma::mat d1 = arma::zeros(NumDim, 1);
        arma::mat W;
        arma::mat P;
        double cvalue[2] = {0};
        if (model_type == 1) {
            W = dY - Ys * jci_beta * jci_alpha.t();
            P = arma::pinv(dX) * W;
            P = P.t();
            cvalue[0] = 12.3329;
            cvalue[1] = 4.1475;
        } else if (model_type == 2) {
            arma::mat remat(NumObs - lag_p - 1, 1);
            c0 = eigvecs_st(eigvecs_st.n_rows - 1, 0);
            c0.print("c0\n");
            cout << "blablalba" << endl;
            remat.fill(c0(0, 0));
            remat.print("remat");
            W = dY - (Ys(arma::span(0, Ys.n_rows - 1), arma::span(0, 1)) * jci_beta + remat) * jci_alpha.t();
            cout << "blablalb2222a" << endl;
            P = arma::pinv(dX) * W;
            P = P.t();
            cout << "blablalb2222a" << endl;
            cvalue[0] = 20.3032;
            cvalue[1] = 9.1465;
        } else if (model_type == 3) {
            W = dY - Ys * jci_beta * jci_alpha.t();
            P = arma::pinv(dX) * W;
            P = P.t();
            arma::mat c = P(arma::span(0, P.n_rows - 1), P.n_cols - 1);
            // arma::mat tmp = jci_alpha.i() * c;
            c0 = arma::pinv(jci_alpha) * c;
            c1 = c - jci_alpha * c0;
            cvalue[0] = 15.4904;
            cvalue[1] = 3.8509;

        } else if (model_type == 4) {
            d0 = eigvecs_st(-1, 0);
            cvalue[0] = 25.8863;
            cvalue[1] = 12.5142;

        } else if (model_type == 5) {
            cvalue[0] = 18.3837;
            cvalue[1] = 3.8395;
        }
        ut = W - dX * P.t();
        arma::mat Ct_all = jci_alpha * c0 + c1 + jci_alpha * d0 + d1;
        for (int bi = 1; bi < lag_p + 1; ++bi) {
            arma::mat Bq = P(arma::span(0, P.n_rows - 1), arma::span((bi - 1) * NumDim, bi * NumDim - 1));
            gamma.push_back(Bq);
        }
        arma::mat tmp1 = jci_beta.t() * S11(arma::span(0, 1), arma::span(0, 1)) * jci_beta;
        arma::mat omega_hat = S00(arma::span(0, 1), arma::span(0, 1)) - jci_alpha * tmp1 * jci_alpha.t();
        tmp1.print("tmp11111");
        omega_hat.print("omega_haaatt");
        vector<arma::mat> tmp_Ct;
        tmp_Ct.push_back(c0);
        tmp_Ct.push_back(d0);
        tmp_Ct.push_back(c1);
        tmp_Ct.push_back(d1);
        c0.print("C0");
        d0.print("d0");
        c1.print("c1");
        d1.print("d1");
        ut.print("ut");
        Ct_all.print("Ct_ALL");
        Ct.push_back(tmp_Ct);
        list_jci_alpha.push_back(jci_alpha);
        list_jci_beta.push_back(jci_beta);
        list_ut.push_back(ut);
        list_gamma.push_back(gamma);

        double TraceTest_T[2] = {0};
        arma::mat eig_lambda;
        for (int rn = 0; rn < NumDim; ++rn) {
            eig_lambda = arma::cumprod(1 - reconstruct_eigval(arma::span(rn, reconstruct_eigval.n_rows - 1), 0));
            eig_lambda.print("eig_lambda");
            double trace_stat = -2 * log(pow(eig_lambda(eig_lambda.n_rows - 1, 0), (NumObs - lag_p - 1) / 2));
            cout << "trace stat :" << trace_stat << endl;
            TraceTest_H[rn] = cvalue[rn] < trace_stat;
            TraceTest_T[rn] = trace_stat;
        }
        for (int i = 0; i < 2; i++) {
            cout << TraceTest_H[i] << endl;
        }
        return ut;
    }

   public:
    vector<vector<double>> _data;
    arma::mat _convert_data;
    int n, k;
    arma::uword opt_p;
    int opt_model;
    vector<arma::mat> list_jci_alpha;
    vector<arma::mat> list_jci_beta;
    vector<arma::mat> list_ut;
    vector<vector<arma::mat>> list_gamma;
    vector<vector<arma::mat>> Ct;
    double Johanson_slope;
    double Johanson_intercept;
    double Johanson_mean;
    double Johanson_std;
    double w1, w2;
};
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
    w1 = beta_orthogonal(0, 0);
    w2 = beta_orthogonal(1, 0);
    beta_orthogonal.print("beta orthogonal");
    arma::mat beta_t = list_jci_beta[opt_model].t();
    beta_orthogonal(1, 0) = -(beta_t(0, 0) * beta_orthogonal(0, 0)) / beta_t(0, 1);
    beta_orthogonal = beta_orthogonal / arma::accu(arma::abs(beta_orthogonal));
    arma::mat temp1 = (alpha_orthogonal.t() * GAMMA * beta_orthogonal).i();
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
    cout << "============" << endl;
    arma::mat beta = list_jci_beta[opt_model];
    cout << "============" << endl;
    vector<arma::mat> mod_gamma = list_gamma[opt_model];
    cout << "============" << endl;
    arma::mat ut = list_ut[opt_model];
    cout << "============" << endl;
    arma::mat tilde_B = beta;
    arma::mat tilde_A = alpha;
    cout << "============" << endl;
    if (lag_p > 0) {
        //建立～A
        arma::mat tilde_A_11 = alpha;
        arma::mat tilde_A_21 = arma::zeros(NumDim * lag_p, 1);
        arma::mat tilde_A_12 = arma::zeros(NumDim, NumDim * lag_p);
        //建立～B
        arma::mat tilde_B_11 = beta;
        // tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        arma::mat tilde_B_3 = arma::zeros(NumDim + NumDim * lag_p, NumDim * lag_p);
        cout << "stanlalalallala" << endl;
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
        var = tilde_J * sigma_telta_beta * tilde_J.t();
        var.print("var");
    }
    Johanson_std = sqrt(var(0, 0));
    cout << "Johansan std :" << Johanson_std << endl;
    cout << "w1 :" << w1 << "|"
         << "w2 :" << w2 << "|"
         << "MU :" << Johanson_mean << "|"
         << "std :" << Johanson_std << endl;

    return *this;
}
