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
std::vector<size_t> argsort(const std::vector<T>& array);
double** mat_to_carr(arma::cx_mat& M, std::size_t& n, std::size_t& m);
double** mat_to_val(arma::cx_vec& M, std::size_t& x, std::size_t& y);
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
            dX = arma::zeros(NumObs - lag_p - 1, NumDim * lag_p);
            for (int xi = 0; xi < lag_p; ++xi) {
                dX(arma::span(0, dX.n_rows - 1), arma::span(xi * NumDim, (xi + 1) * NumDim - 1)) = dY_ALL(arma::span(lag_p - xi - 1, NumObs - xi - 2), arma::span(0, dY_ALL.n_cols));

                if (model_type == 2) {
                    Ys = arma::join_rows(Ys, arma::ones(NumObs - lag_p - 1, 1));
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
