#if !def JOHANSON_HPP
#define JOHANSON_HPP
#include <map>
#include <string>
#include <tuple>
#include <vector>
#define ARMA_DONT_USE_WRAPPER
#include <math.h>

#include <armadillo>
#include <tuple>

class VAR {
   private:
    arma::mat _data;
    int max_lags;
    int n, k;
    std::vector<std::vector<double>> _v_data;

    /* data */
   public:
    VAR(const std::vector<std::vector<double>> &z) {
        std::vector<double> z_flat;
        size_t m_nrow = z.size();
        size_t m_ncol = z[0].size();
        for (size_t i = 0; i < m_ncol; ++i) {
            for (size_t j = 0; j < m_nrow; ++j) {
                z_flat.push_back(z[j][i]);
            }
        }
        arma::mat tmp(&z_flat.front(), m_nrow, m_ncol);
        _data = tmp;
        k = _data.n_cols;
        n = _data.n_rows;
        _v_data = z;
    };
    ~VAR() = default;

    arma::mat get_var_endog(int lags) {
        arma::mat Z(n - lags, k * lags);
        int _k = 0;
        for (int t = lags; t < n; t++) {
            arma::rowvec a = vectorise(reverse(_data.rows(t - lags, t - 1)), 1);
            Z.row(_k++) = a;
        }
        return Z;
    }

    void fit(int lags) {
        max_lags = int(12 * pow(0.01 * n, 0.25));
        arma::mat tmp = get_var_endog(lags);
    }

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
                a.at(_k++) = _data(i + p - j - 1, 0);
                a.at(_k++) = _data(i + p - j - 1, 1);
            }
            xt.row(i) = a;
        }
        arma::mat zt = _data.rows(p, n - 1);
        arma::mat beta = (xt.t() * xt).i() * xt.t() * zt;

        arma::mat A = zt - (xt * beta);
        arma::mat sigma = ((A.t()) * A) / (double)(n - p);
        return sigma;
    }

    arma::uword order_select(int max_p) {
        arma::rowvec bic(max_p);
        bic.zeros();
        for (int p = 1; p < max_p + 1; p++) {
            arma::mat sigma = _estimate_var(p);
            bic.at(p - 1) = log(det(sigma)) + log(n) * (double)p * (double)(k * k) / (double)n;
        }
        arma::uword bic_order = bic.index_min() + 1;

        return bic_order;
    }
    arma::uword JCI_AutoSelection(int opt_q) {
        int opt_p = opt_q + 1;
        int Tl = n - opt_p;
        std::cout << opt_p << ' ';
        std::out << Tl;
        // auto TraceTest_table;
        arma::mat TraceTest_table(5, n);
        arma::rowvec BIC_table(5);
        arma::rowvec BIC_List(5);
        BIC_List.print();
        int opt_model_num = 0;
        for (int mr = 0; mr < 1; ++mr) {
            JCItest_withTrace(mr + 1, opt_q);
        }
    }
    arma::mat dx_dY_Ys_calculation(arma::mat *dY, arma::mat *Ys, arma::mat *dY_ALL, int lag_p, int model_type) {
        if (lag_p == 0) {
            if (model_type == 1) {
                arma::mat dX(n - 1, k);
                dX.zeros();
                dX.print("model type 1 :");
                return dX;
            } else if (model_type == 2) {
                arma::mat dX(n - 1, k);
                arma::mat B(Ys->n_rows, 1, fill::ones);
                Ys->print("origin YS");
                dX.zeros();
                Ys->insert_cols(2, B);
                Ys->print("YYYSSSS");
                // auto tmp = nc::ones<double>(NumObs-lag_p-1, 1);
                // Ys = nc::hstack({Ys, tmp});
            } else if (model_type == 3) {
                arma::mat dX(n - 1, k);
                dX.ones();
            } else if (model_type == 4) {
                arma::mat dX(n - lag_p - 1, k);
                dX.ones();
                std::vector<int> x(n - lag_p - 1);
                iota(x.begin(), x.end(), 1);
                arma::mat tmp = arma::conv_to<arma::mat>::from(x);
                Ys->insert_cols(2, tmp);
                Ys->print("model type 4");

            } else if (model_type == 5) {
                arma::mat dX(n - lag_p - 1, 1);
                dX.ones();
                std::vector<int> x(n - lag_p - 1);
                iota(x.begin(), x.end(), 1);
                arma::mat tmp = arma::conv_to<arma::mat>::from(x);
                dX.insert_cols(1, tmp);
                dY_ALL->print("DY ALL");
                dX.print("model type 5");
            }
        } else if (lag_p > 0) {
            arma::mat dX(n - lag_p - 1, k * lag_p);
            for (int xi = 0; xi < lag_p; ++xi) {
                dX.submat(arma::span::all, arma::span(xi * k, (xi + 1) * k - 1)) = dY_ALL->submat(span(lag_p - xi - 1, n - xi - 3), span::all);
            }
            if (model_type == 2) {
                arma::mat tmp(n - lag_p - 1, 1);
                tmp.ones();
                Ys->insert_cols(2, tmp);

            } else if (model_type == 3) {
                arma::mat tmp(n - lag_p - 1, 1);
                tmp.ones();
                dX.insert_cols(k * lag_p, tmp);
            } else if (model_type == 4) {
                arma::mat tmp1(n - lag_p - 1, 1);
                tmp1.ones();
                dX.insert_cols(k * lag_p, tmp1);
                std::vector<int> x(n - lag_p - 1);
                iota(x.begin(), x.end(), 1);
                arma::mat tmp2 = arma::conv_to<arma::mat>::from(x);
                Ys->insert_cols(2, tmp2);
            } else if (model_type == 5) {
                arma::mat tmp1(n - lag_p - 1, 1);
                dX.ones();
                std::vector<int> x(n - lag_p - 1);
                iota(x.begin(), x.end(), 1);
                arma::mat tmp2 = arma::conv_to<arma::mat>::from(x);
                dX.insert_cols(k * lag_p, tmp1);
                dX.insert_cols(k * lag_p + 1, tmp2);
                dX.print("model type 5");
            }
        }
    }
    arma::uword JCItest_withTrace(int model_type, int lag_p) {
        std::cout << model_type << ' ' << lag_p << std::endl;
        arma::mat X_data_first = _data.submat(arma::span(1, 119), arma::span::all);
        arma::mat X_data_second = _data.submat(arma::span(0, 118), arma::span::all);

        arma::mat dY_ALL = X_data_first - X_data_second;
        dY_ALL.print("dY_ALL");
        arma::mat dY = dY_ALL.rows(lag_p, 118);
        arma::mat Ys = _data.rows(lag_p, 118);
        arma::mat dX = dx_dY_Ys_calculation(&dY, &Ys, &dY_ALL, lag_p, model_type);
        arma::mat dX_2 = dX.t() * dX;  // python 366
        dX_2.print("DX2 steve");
        double sum_dx2 = accu(dX_2);
        arma::mat M1(n - lag_p + 1, n - lag_p + 1);
        if (sum_dx2 == 0) {
            arma::mat i = eye(n - lag_p - 1, n - lag_p - 1);
            M1 = i - dX * dX.t();

        } else {
            arma::mat B = inv(dX_2);
            arma::mat i = eye(n - lag_p - 1, n - lag_p - 1);
            M1 = i - dX * B * dX.t();
        }
        arma::mat R0 = dY.t() * M1;
        arma::mat R1 = Ys.t() * M1;
        arma::mat S00 = R0 * R0.t() / (n - lag_p - 1);
        arma::mat S01 = R0 * R1.t() / (n - lag_p - 1);
        arma::mat S10 = R1 * R0.t() / (n - lag_p - 1);
        arma::mat S11 = R1 * R1.t() / (n - lag_p - 1);
        arma::mat A = S10 * S00.i() * S01;
        A.print("A");
        S11.print("B");
        // A.print("A");
        arma::cx_vec eigval;
        arma::cx_mat eigvec;
        eig_pair(eigval, eigvec, S11, A);
        eigval.print("eigval");
        eigvec.print("eigvec");
    }
};

std::tuple<float, float> Johanson_mean(arma::mat alpha, arma::mat beta, double gamma, arma::mat mu, int lagp, int NumDim = 2) {
    arma::mat sumgamma(NumDim, NumDim, arma::fill::zeros);
    for (int i = 0; i < lagp; ++i) {
    }
    arma::mat GAMMA = eye(NumDim, NumDim) - sumgamma;
    arma::mat alpha_orthogonal = alpha;
    arma::mat alpha_t = alpha.t();
    alpha_orthogonal(1, 0) = (-(alpha_t(0, 0)) * alpha_orthogonal(0, 0)) / alpha_t(0, 1);
    alpha_orthogonal = alpha_orthogonal / accu(abs(alpha_orthogonal));
    arma::mat beta_orthogonal = beta;
    arma::mat beta_t = beta.t();
    beta_orthogonal(1, 0) = (beta_t(0, 0) * beta_orthogonal(0, 0)) / beta_t(0, 1);
    beta_orthogonal = beta_orthogonal / accu(beta_orthogonal);
    arma::mat tmp1 = ((alpha_orthogonal.t() * GAMMA) * beta_orthogonal).i();
    arma::mat C = (beta_orthogonal * tmp1) * alpha_orthogonal.t();
    arma::mat tmp2 = (alpha.t() * alpha).i();
    arma::mat alpha_hat = alpha * tmp2;
    arma::mat tmp3 = (GAMMA * C) - eye(NumDim, NumDim);
    arma::mat C0 = mu(0, 1);
    arma::mat C1 = mu(2, 1);
    arma::mat D0 = mu(1, 1);
    arma::mat D1 = mu(3, 1);
    arma::mat C1_new = alpha * C0 + C1 + alpha * D0 + D1;
    arma::mat Ct = alpha * D0 + D1;
    float expected_intercept = (alpha_hat.t() * tmp3) * C0;
    float expected_slope = (alpha_hat.t() * tmp3) * Ct;
    return {expected_intercept, expected_slope};
}

double Johansen_std_correct(arma::mat alpha, arma::mat beta, arma::mat ut, arma::mat mod_gamma, int lag_p, int rank = 1) {
    int NumDim = 2;
    if (lag_p > 0) {
        arma::mat tilde_A_11 = alpha;
        arma::mat tilde_A_21 = arma::zeros(NumDim * lag_p, 1);
        arma::mat tilde_A_12 = arma::zeros(NumDim, NumDim * lag_p);
        arma::mat tilde_B_11 = beta;
        arma::mat tilde_B_3 = arma::zeros(NumDim + NumDim * lag_p, NumDim * lag_p);
        for (int qi = 0; qi < lag_p; ++qi) {
            tilde_A_12(arma::span(0, NumDim), arma::span(qi * NumDim, ((qi + 1) * NumDim))) = mod_gamma[qi];
        }
        arma::mat tilde_A_22 = arma::eye(NumDim * lag_p, NumDim * lag_p);

    } else {
        arma::mat tilde_A = alpha;
        arma::mat tilde_B = beta;
    }
    arma::mat tilde_sigma = arma::zeros(NumDim * (lag_p + 1), NumDim * (lag_p + 1));
    tilde_sigma(arma::span(0, NumDim), arma::span(0, NumDim)) = ut.t() * ut / (ut.n_rows - 1);
    arma::mat tilde_J = arma::zeros(1, 1 + NumDim * lag_p);
    tilde_J(0, 0) = 1;
    if (lag_p == 0) {
        arma::mat tmp1 = arma::eye(rank, rank) + beta.t() * alpha;
        arma::mat tmp2 = arma::kron(tmp1, tmp1);
        arma::mat tmp = arma::eye(rank, rank);
        arma::mat tmp3 = (tmp - tmp2).i();
        arma::mat omega = (ut.i() * ut) / (ut.n_rows - 1);
        arma::mat tmp4 = beta.t() * omega * beta;
        arma::mat var = tmp3 * tmp4;
    } else {
        arma::mat tmp1 = arma::eye(NumDim * (lag_p + 1) - 1, NumDim * (lag_p + 1) - 1) + tilde_B.t() * tilde_A;
        arma::mat tmp2 = arma::kron(tmp1, tmp1);
        int k = (NumDim * (lag_p + 1) - 1) * (NumDim * (lag_p + 1) - 1);
        arma::mat tmp = arma::eye(k, k);
        arma::mat tmp3 = (tmp - tmp2).i();
        arma::mat tmp4 = tilde_B.t() * tilde_sigma * tilde_B;

        arma::vec v = arma::vectorise(tmp4, 1);
        arma::mat tmp5 = tmp3 * tmp4;
        arma::mat sigma_telta_beta = arma::zeros(NumDim * (lag_p + 1) - 1, NumDim * (lag_p + 1) - 1);

        for (int i = 0; i < NumDim * (lag_p + 1) - 1; ++i) {
            for (int j = 0; j < NumDim * (lag_p + 1) - 1; ++j) {
                sigma_telta_beta(i, j) = tmp5(0, i + j * (NumDim * (lag_p + 1) - 1));
            }
        }
        arma::mat var = tilde_J * sigma_telta_beta * tilde_J.t();
    }
}
// class Johanson_test {
//    public:
//     Johanson_test() = default;
//     ~Johanson_test() = default;
//     Johanson_test& JCI_AutoSelection(const std::vector<std::vector<double>>& z);

//    public:
//     std::vector<std::vector<double>> _data;
// };

// Johanson_test& Johanson_test::JCI_AutoSelection(const std::vector<std::vector<double>>& z, int opt_q) {
//     size_t NumObs = z[0].size();
//     size_t k = z.size();
//     int opt_p = opt_q + 1;
//     int Tl = NumObs - opt_p;
//     double TraceTest_table[5] = {0};
//     double BIC_talbe[5] = {0};
//     int opt_model_num = 0;
//     for (int i = 0; i < 5; ++i) {
//     }
// }
#endif