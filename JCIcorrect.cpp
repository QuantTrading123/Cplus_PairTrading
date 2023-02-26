#include <iostream>
using namespace std;
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

//#include "Johanson.hpp"
double Johansen_std_correct(const arma::mat& alpha, const arma::mat& beta, const arma::mat& ut, vector<arma::mat> mod_gamma, int lag_p, int rank = 1) {
    std::cout << "======== Johanson standard corrected ========" << std::endl;
    int NumDim = 2;
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
        arma::mat var = tmp3 * tmp4;
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
        arma::mat var = tilde_J * sigma_telta_beta * tilde_J.t();
        var.print("var");
    }
}

// template <class T>
// void printArray2D(T&& list, int row, int col) {
//     for (int i = 0; i < row; i++) {
//         for (int j = 0; j < col; j++) {
//             cout << list{i}{j} << " ";
//         }
//         cout << endl;
//     }
// }
int main() {
    arma::mat ut = {{-1.24922125e-03, 2.26971600e-03},
                    {2.20896560e-03, 3.65529153e-03},
                    {4.72045504e-04, 4.25531707e-03},
                    {1.30122515e-03, 3.50027011e-03},
                    {-2.35707920e-05, 1.24522312e-02},
                    {3.14119309e-04, 3.83764688e-03},
                    {-5.87853497e-05, -5.39474126e-03},
                    {7.29739107e-04, -2.63886834e-04},
                    {8.85272357e-04, 1.46531387e-03},
                    {9.97750939e-04, -3.41960549e-04},
                    {-1.17018892e-03, -1.55364014e-03},
                    {8.05525622e-04, -6.94826275e-04},
                    {-2.94911385e-04, 3.31445354e-04},
                    {-7.84165248e-04, -5.55650368e-05},
                    {-1.46093991e-03, -3.31109673e-03},
                    {5.39608367e-04, 2.34475224e-03},
                    {-7.32667690e-04, -1.99011454e-03},
                    {3.71614817e-05, 3.18326645e-03},
                    {-2.24289239e-03, -3.12509262e-03},
                    {-5.56132200e-04, 6.67882314e-03},
                    {3.86492917e-04, -3.54983200e-04},
                    {-1.33340457e-04, -2.43305485e-03},
                    {-1.29712356e-04, -1.28520202e-03},
                    {4.08521609e-04, -3.96482297e-04},
                    {2.82608555e-04, 2.25911660e-03},
                    {4.62232784e-04, 6.26816496e-04},
                    {4.39421897e-04, 3.61319163e-03},
                    {1.07624347e-03, -1.84539046e-03},
                    {1.89050315e-03, 2.40784107e-03},
                    {-7.61878972e-04, -7.09923682e-04},
                    {1.09513743e-03, 1.56547518e-03},
                    {-6.54012078e-04, 4.94271491e-04},
                    {7.45978942e-04, 4.04589702e-03},
                    {3.88790785e-04, 9.57635350e-04},
                    {1.22050019e-03, 2.13518797e-03},
                    {1.57291540e-04, 2.86800496e-03},
                    {4.14745052e-04, -4.16504318e-04},
                    {3.19030173e-04, 3.96523304e-03},
                    {2.06062515e-03, 4.73690756e-03},
                    {1.86410805e-03, 1.87851996e-03},
                    {2.21856024e-03, 2.57183948e-03},
                    {-6.20245405e-04, 2.38385488e-03},
                    {-7.18209228e-04, -5.02777138e-04},
                    {1.55219041e-04, -1.64889693e-03},
                    {1.21577382e-03, -7.23656041e-04},
                    {5.79166464e-04, 4.58646211e-04},
                    {5.46132860e-04, -1.27715826e-05},
                    {5.63702147e-04, 1.16692791e-03},
                    {-3.75946920e-04, 1.88679359e-03},
                    {-1.11936095e-03, -7.00143332e-04},
                    {8.01189683e-04, -7.65281955e-03},
                    {-1.48727607e-03, 9.79581325e-04},
                    {5.58863849e-04, 1.42857092e-03},
                    {-1.47698388e-04, 9.16585932e-04},
                    {-9.92044984e-05, -1.07646894e-04},
                    {6.58869420e-04, -1.81793195e-04},
                    {-3.38370191e-04, 2.70581606e-03},
                    {-6.74519763e-04, -2.44626323e-03},
                    {1.00293047e-04, 1.16360564e-03},
                    {-1.70839792e-03, -6.80833105e-03},
                    {-2.76595290e-03, -7.40721562e-04},
                    {-1.78638815e-03, -1.22220170e-03},
                    {-3.99611820e-04, 1.34041188e-03},
                    {-1.40023697e-03, -2.93280610e-04},
                    {-5.06206308e-04, 2.10149706e-03},
                    {-2.78931300e-04, -9.36969420e-04},
                    {-1.79329195e-03, -3.25578467e-03},
                    {1.30222678e-03, -7.29599346e-05},
                    {-1.12019384e-03, -5.80012982e-03},
                    {5.72525337e-04, 9.40986713e-04},
                    {-2.37577497e-04, 1.27906647e-03},
                    {-1.80099907e-04, -3.83146542e-03},
                    {8.57704557e-04, 1.69741940e-03},
                    {-1.74778125e-03, -4.73889480e-03},
                    {1.26780051e-03, 3.41970447e-03},
                    {-7.38787117e-04, 3.78320091e-04},
                    {4.84392611e-04, -1.63944242e-03},
                    {-6.63097758e-04, -2.18234147e-03},
                    {-1.87775422e-03, -9.99338683e-04},
                    {-1.04111945e-03, -3.50427557e-03},
                    {-1.08639946e-03, -8.30158125e-04},
                    {-6.37197876e-04, -1.98791532e-03},
                    {-7.26460412e-04, -4.06751755e-03},
                    {1.03071889e-03, 5.34606804e-04},
                    {-6.31437628e-04, -4.06946903e-03},
                    {-4.24584218e-04, -5.15829659e-03},
                    {3.78383085e-04, -2.02599310e-03},
                    {1.20891357e-03, 2.07103912e-03},
                    {1.58997778e-04, -4.43923192e-04},
                    {1.32845540e-03, 5.95301812e-03},
                    {-1.19566301e-03, -2.85234403e-03},
                    {1.67197803e-03, -9.65003449e-03},
                    {-1.82545956e-04, 2.60349105e-03},
                    {2.22253578e-04, -6.14004970e-04},
                    {6.47643686e-04, 3.74676047e-03},
                    {8.52406813e-04, 1.09014606e-03},
                    {1.78792390e-03, 5.19718266e-03},
                    {3.57751399e-04, -1.23880621e-03},
                    {3.29929462e-04, -7.55795052e-03},
                    {-2.70197194e-04, 3.27622370e-03},
                    {1.04348535e-03, 1.15848874e-03},
                    {8.57535514e-04, 1.28650419e-03},
                    {2.82801606e-04, -1.84798799e-04},
                    {-1.05119062e-03, -2.07750270e-03},
                    {-9.04441671e-04, -2.11347781e-03},
                    {8.47543971e-05, -1.08446208e-03},
                    {-4.60421754e-04, -1.64449739e-03},
                    {-7.00443400e-04, -2.47999215e-03},
                    {-3.44155327e-04, 7.52231357e-04},
                    {-2.51808612e-03, -3.23276979e-03},
                    {-3.17156026e-04, 2.44565811e-03},
                    {6.12210797e-04, -1.76626903e-03},
                    {2.34851792e-04, 8.73776037e-04},
                    {-8.64391215e-04, -3.32388501e-03},
                    {1.22970302e-03, 1.65408473e-03},
                    {-8.79646707e-04, -1.94480939e-03},
                    {-9.70080967e-04, -1.00559578e-03},
                    {5.36414782e-04, 5.38831596e-04}};
    arma::mat alpha = arma::colvec({0.35912113, 0.64087887});
    arma::mat beta = arma::colvec({-0.97281185, 0.02718815});
    arma::mat gamma = {{0.30294791, -0.06315603}, {0.58598012, -0.07522277}};
    vector<arma::mat> mod_gamma;
    mod_gamma.push_back(gamma);
    // printArray2D(ut, 119, 2);
    // arma::mat a(&alpha{0}{0}, sizeof(alpha) / sizeof(alpha{0}), sizeof(alpha{0}) / sizeof(alpha{0}{0}));
    // arma::mat b(&beta{0}{0}, sizeof(beta) / sizeof(beta{0}), sizeof(beta{0}) / sizeof(beta{0}{0}));
    // arma::mat u(&ut{0}{0}, sizeof(ut) / sizeof(ut{0}), sizeof(ut{0}) / sizeof(ut{0}{0}), true, true);
    // // double gamma{};
    Johansen_std_correct(alpha, beta, ut, mod_gamma, 1);
}
