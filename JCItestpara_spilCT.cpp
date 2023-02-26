#include <iostream>
using namespace std;
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
double **mat_to_carr(arma::cx_mat &M, std::size_t &n, std::size_t &m) {
    const std::size_t nrows = M.n_rows;
    const std::size_t ncols = M.n_cols;
    double **array = (double **)malloc(nrows * sizeof(double *));

    for (std::size_t i = 0; i < nrows; i++) {
        array[i] = (double *)malloc(ncols * sizeof(double));
        for (std::size_t j = 0; j < ncols; ++j)
            array[i][j] = M(i + j * ncols).real();
    }

    n = nrows;
    m = ncols;

    return array;
}
int JCItestpara_spiltCT(arma::mat rowAS, int model_type, int lag_p) {
    int NumObs = rowAS.n_rows;
    int NumDim = rowAS.n_cols;
    cout << NumDim << " " << NumObs << endl;
    arma::mat dY_ALL = rowAS(arma::span(1, NumObs - 1), arma::span(0, NumDim - 1)) - rowAS(arma::span(0, NumObs - 2), arma::span(0, NumDim - 1));
    dY_ALL.print("DY");
    arma::mat dY = dY_ALL(arma::span(lag_p, dY_ALL.n_rows - 1), arma::span(0, dY_ALL.n_cols - 1));
    arma::mat Ys = rowAS(arma::span(lag_p, NumObs - 2), arma::span(0, NumDim - 1));
    dY.print("dyYYYYYY");
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
    dX.print("DXXXXX");
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
    arma::mat tmp = S10 * S00.i() * S01;
    tmp.print("tmptmptpmp");

    arma::mat show_S = S10 * S00.i() * S01;
    arma::mat tmp_eig = S11.i() * S10 * S00.i() * S01;
    arma::vec eigval2;
    arma::mat eigvec2;
    arma::eig_sym(eigval2, eigvec2, tmp_eig);
    eigval2.print("sym eigval");

    arma::cx_vec eigval1;
    arma::cx_mat eigvec1;
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
    arma::mat reconstruct_eigvec(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            std::cout << array[i][j] << " ";
            reconstruct_eigvec(i, j) = array[i][j];
        }
        std::cout << std::endl;
    }
    reconstruct_eigvec.print("reconstruct eigvec");
    for (int i = 0; i < reconstruct_eigvec.n_rows; ++i) {
        first_col_sum += abs(reconstruct_eigvec(i, 0));
    }

    cout << " first_col_sum " << first_col_sum << endl;

    arma::mat test_eigvecs_st = reconstruct_eigvec / first_col_sum;
    test_eigvecs_st.print("test eigvecs_st");
    // arma::eig_pair(eigval, eigvec2, show_S, S11);
    // eigval.print("eigenval");
    // eigvec2.print("eigenvector");
    // cout << eigval.n_rows << eigval.n_cols << endl;

    arma::mat eigvec = {{40.01801448, -454.3070705},
                        {-63.52375491, 94.34738988}};

    arma::mat eigvecs_st = test_eigvecs_st;
    // beta
    arma::mat jci_beta = eigvecs_st(arma::span(0, eigvecs_st.n_rows - 1), 0);
    jci_beta.reshape(NumDim, 1);
    // Alpha
    arma::mat a = eigvecs_st(arma::span(0, eigvecs_st.n_rows - 1), 0);
    a.reshape(1, NumDim);
    arma::mat jci_a = S01 * a.t();
    arma::mat jci_alpha = jci_a / arma::accu(arma::abs(jci_a));
    arma::mat c0 = {0};
    arma::mat d0 = {0};
    arma::mat c1 = arma::zeros(NumDim, 1);
    arma::mat d1 = arma::zeros(NumDim, 1);
    arma::mat W;
    arma::mat P;
    if (model_type == 1) {
        W = dY - Ys * jci_beta * jci_alpha.t();
        P = arma::pinv(dX) * W;
        P = P.t();
    } else if (model_type == 2) {
        arma::mat remat(NumObs - lag_p - 1, 1);
        c0 = eigvecs_st(eigvecs_st.n_rows - 1, 0);
        remat.fill(c0(0, 0));
        W = dY - (Ys(arma::span(0, Ys.n_rows - 1), arma::span(0, 1)) * jci_beta) + remat * jci_alpha.t();
        P = arma::pinv(dX) * W;
        P = P.t();
    } else if (model_type == 3) {
        W = dY - Ys * jci_beta * jci_alpha.t();
        P = arma::pinv(dX) * W;
        P = P.t();
        arma::mat c = P(arma::span(0, P.n_rows - 1), P.n_cols - 1);
        // arma::mat tmp = jci_alpha.i() * c;
        c0 = arma::pinv(jci_alpha) * c;
        c1 = c - jci_alpha * c0;
    } else if (model_type == 4) {
        d0 = eigvecs_st(-1, 0);
    }
    arma::mat ut = W - dX * P.t();
    arma::mat Ct_all = jci_alpha * c0 + c1 + jci_alpha * d0 + d1;
    vector<arma::mat> gamma;
    for (int bi = 1; bi < lag_p + 1; ++bi) {
        arma::mat Bq = P(arma::span(0, P.n_rows - 1), arma::span((bi - 1) * NumDim, bi * NumDim - 1));
        gamma.push_back(Bq);
    }
    arma::mat tmp1 = jci_beta.t() * S11(arma::span(0, 1), arma::span(0, 1)) * jci_beta;
    arma::mat omega_hat = S00(arma::span(0, 1), arma::span(0, 1)) - jci_alpha * tmp1 * jci_alpha.t();
    tmp1.print("tmp11111");
    omega_hat.print("omega_haaatt");
    vector<arma::mat> Ct;
    Ct.push_back(c0);
    Ct.push_back(d0);
    Ct.push_back(c1);
    Ct.push_back(d1);
    c0.print("C0");
    d0.print("d0");
    c1.print("c1");
    d1.print("d1");
    ut.print("ut");
    Ct_all.print("Ct_ALL");
}

int main() {
    int opt_model = 3;
    int p = 1;
    arma::mat RowAS = {{7.87619624, 3.39534697},
                       {7.87870274, 3.39836003},
                       {7.87607473, 3.39501162},
                       {7.87871789, 3.40069726},
                       {7.87679215, 3.39769124},
                       {7.87753559, 3.39802569},
                       {7.87518579, 3.39501162},
                       {7.87831258, 3.39819287},
                       {7.88039421, 3.4003637},
                       {7.8819205, 3.40169726},
                       {7.87994419, 3.4010307},
                       {7.88020893, 3.40402671},
                       {7.88069663, 3.40435904},
                       {7.8801333, 3.4033617},
                       {7.87902082, 3.40203037},
                       {7.88004631, 3.40402671},
                       {7.87976639, 3.40601907},
                       {7.87942206, 3.40568729},
                       {7.87948639, 3.40302904},
                       {7.88028834, 3.40269626},
                       {7.8818601, 3.40568729},
                       {7.88144095, 3.40535539},
                       {7.87852094, 3.40269626},
                       {7.88044714, 3.4033617},
                       {7.87842624, 3.40069726},
                       {7.87878227, 3.39936237},
                       {7.87893374, 3.39936237},
                       {7.87945612, 3.40136403},
                       {7.87998958, 3.40236337},
                       {7.88090071, 3.40269626},
                       {7.87913061, 3.40269626},
                       {7.88020137, 3.39902836},
                       {7.88031103, 3.39836003},
                       {7.88136919, 3.39936237},
                       {7.88171285, 3.40203037},
                       {7.88206394, 3.40469127},
                       {7.88407752, 3.40734511},
                       {7.8890563, 3.41065254},
                       {7.8941484, 3.4136199},
                       {7.88753335, 3.41131272},
                       {7.88451062, 3.40999193},
                       {7.88701518, 3.41394907},
                       {7.88828761, 3.41985556},
                       {7.88950598, 3.42344799},
                       {7.88913875, 3.43543776},
                       {7.88834388, 3.43769002},
                       {7.88796121, 3.43172659},
                       {7.88908628, 3.43188823},
                       {7.89004534, 3.43350322},
                       {7.89065551, 3.43253454},
                       {7.8889326, 3.42994679},
                       {7.88915749, 3.42800158},
                       {7.8887339, 3.42800158},
                       {7.88764972, 3.42735234},
                       {7.88610208, 3.42377394},
                       {7.88710531, 3.42670267},
                       {7.8868762, 3.42572738},
                       {7.88736066, 3.42962286},
                       {7.88534241, 3.42702756},
                       {7.88534241, 3.43447096},
                       {7.88633513, 3.43543776},
                       {7.88718042, 3.43479333},
                       {7.88778484, 3.43479333},
                       {7.88860266, 3.4351156},
                       {7.8890563, 3.43769002},
                       {7.88930365, 3.43801135},
                       {7.88952471, 3.44121906},
                       {7.89014644, 3.43865372},
                       {7.89182615, 3.44057834},
                       {7.8903224, 3.43865372},
                       {7.8904609, 3.43833259},
                       {7.88919497, 3.43769002},
                       {7.88935986, 3.44057834},
                       {7.88934862, 3.44089875},
                       {7.89028496, 3.44249926},
                       {7.89005283, 3.44473568},
                       {7.88978693, 3.44313875},
                       {7.8897345, 3.44633007},
                       {7.89123539, 3.45014626},
                       {7.89248379, 3.45109804},
                       {7.89376422, 3.45204891},
                       {7.89177755, 3.45204891},
                       {7.88945728, 3.44855795},
                       {7.88890636, 3.44537374},
                       {7.89009402, 3.4444165},
                       {7.89060686, 3.44473568},
                       {7.89062557, 3.44377783},
                       {7.89057692, 3.44377783},
                       {7.88952471, 3.4444165},
                       {7.88775857, 3.44249926},
                       {7.8884564, 3.43447096},
                       {7.88767599, 3.43640364},
                       {7.88815632, 3.43769002},
                       {7.88819758, 3.43897475},
                       {7.88815256, 3.43897475},
                       {7.88893635, 3.43897475},
                       {7.88870015, 3.44185936},
                       {7.88774731, 3.43897475},
                       {7.88802125, 3.44025783},
                       {7.88651176, 3.4338259},
                       {7.88435623, 3.4338259},
                       {7.88333138, 3.4338259},
                       {7.88439389, 3.43769002},
                       {7.88451062, 3.44025783},
                       {7.88530102, 3.44473568},
                       {7.88616974, 3.4460114},
                       {7.88545903, 3.44473568},
                       {7.88776232, 3.44633007},
                       {7.88758591, 3.44233932},
                       {7.8887264, 3.44409722},
                       {7.8887114, 3.44585202},
                       {7.88842639, 3.44185936},
                       {7.88952097, 3.44377783},
                       {7.88769101, 3.43897475},
                       {7.88900757, 3.44217936},
                       {7.88833638, 3.44281906},
                       {7.8886889, 3.44089875},
                       {7.88822384, 3.43897475},
                       {7.88644036, 3.43801135},
                       {7.8856471, 3.43479333}};
    JCItestpara_spiltCT(RowAS, opt_model, p - 1);
    return 0;
}