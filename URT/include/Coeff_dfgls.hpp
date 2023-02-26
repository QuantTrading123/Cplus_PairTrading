//=================================================================================================
//                    Copyright (C) 2016 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#ifndef COEFF_DFGLS_HPP
#define COEFF_DFGLS_HPP

static const std::map<std::string,std::map<float,std::map<int,std::vector<float>>>> coeff_dfgls = 
{
    {"c",{
        {0.001,{
            {0, {-3.281721, -20.10785, 20.54557, 919.4611, -15768.36}},
            {1, {1.214884, -9.190296, 465.547, -4800.389, 18494.58, -24523.48}}}},
        {0.005,{
            {0, {-2.798722, -19.87674, 158.6922, -1361.471, 649.8149}},
            {1, {1.377745, -9.989878, 369.4152, -3615.128, 13677.44, -17888.14}}}},
        {0.01,{
            {0, {-2.566274, -20.06523, 218.4769, -2506.118, 9754.758}},
            {1, {1.367608, -14.98339, 466.491, -4344.437, 16100.97, -20814.35}}}},
        {0.025,{
            {0, {-2.227436, -20.71955, 276.1964, -3313.464, 16034.91}},
            {1, {1.075308, -5.706035, 307.2224, -3143.095, 12178.3, -16217.98}}}},
        {0.05,{
            {0, {-1.941807, -21.8198, 327.6527, -3983.442, 20397.75}},
            {1, {0.9627055, -2.586653, 239.4981, -2548.091, 10013.66, -13436.96}}}},
        {0.1,{
            {0, {-1.618534, -23.81555, 403.463, -5111.848, 27837.04}},
            {1, {0.8819745, -0.5482568, 184.5334, -2036.822, 8119.434, -10998.77}}}},
        {0.2,{
            {0, {-1.236467, -27.13579, 499.8872, -6332.546, 34623.67}},
            {1, {0.7793006, 1.366651, 137.9713, -1615.072, 6560.24, -8970.884}}}},
        {0.5,{
            {0, {-0.5023128, -38.29761, 767.7794, -9009.913, 44651.65}},
            {1, {0.5846996, 8.288074, 25.93693, -785.592, 3753.654, -5481.515}}}},
        {0.8,{
            {0, {0.4066535, -50.03626, 832.9085, -6572.743, 16628.01}},
            {1, {0.6261542, 17.42774, -155.8765, 586.6406, -901.9577, 418.7792}}}},
        {0.9,{
            {0, {0.8933428, -51.3788, 692.0526, -2970.695, -9079.985}},
            {1, {0.5300101, 20.07061, -220.2416, 1188.216, -3179.489, 3487.29}}}},
        {0.95,{
            {0, {1.290573, -51.06435, 508.2149, 1089.122, -36173.9}},
            {1, {0.3381674, 23.35171, -277.6058, 1727.191, -5242.62, 6267.253}}}},
        {0.975,{
            {0, {1.63135, -50.43061, 335.0748, 4694.975, -59130.16}},
            {1, {0.2820731, 20.09923, -232.4119, 1600.805, -5325.437, 6810.411}}}},
        {0.99,{
            {0, {2.02525, -49.52247, 131.545, 8734.051, -83906.98}},
            {1, {0.121659, 17.82018, -184.9223, 1512.869, -5754.938, 7987.295}}}},
        {0.995,{
            {0, {2.289887, -48.58579, -33.96214, 12094.66, -104876.2}},
            {1, {0.3138123, 4.254046, 43.79203, 106.5682, -1929.322, 4154.06}}}},
        {0.999,{
            {0, {2.832592, -45.2852, -454.6888, 20303.27, -152655.0}},
            {1, {0.4460974, -11.79829, 343.2512, -1557.125, 1982.09, 866.629}}}}
        }},
    {"ct",{
        {0.001,{
            {0, {-4.064679, -24.35303, -160.8093, 3417.97, -29721.44}},
            {1, {2.988795, -50.01775, 1285.353, -11102.31, 39779.73, -50648.68}}}},
        {0.005,{
            {0, {-3.617084, -22.33611, 5.425624, 499.2113, -6856.833}},
            {1, {2.212498, -22.46913, 777.0021, -7112.543, 26047.46, -33445.7}}}},
        {0.01,{
            {0, {-3.407817, -21.3686, 49.65953, -17.72955, -3288.767}},
            {1, {2.233841, -25.78092, 773.3825, -6772.984, 24204.3, -30586.87}}}},
        {0.025,{
            {0, {-3.102192, -20.3775, 98.40342, -417.9263, -1395.028}},
            {1, {1.784919, -11.76577, 523.3629, -4900.973, 18035.26, -23151.24}}}},
        {0.05,{
            {0, {-2.846181, -19.94484, 150.1849, -1202.047, 4403.494}},
            {1, {1.496648, -4.331499, 373.3418, -3693.049, 13868.73, -18009.14}}}},
        {0.1,{
            {0, {-2.557861, -20.1136, 217.2022, -2172.267, 10348.15}},
            {1, {1.263156, -0.2713012, 281.1032, -2933.85, 11254.27, -14803.17}}}},
        {0.2,{
            {0, {-2.221751, -20.61889, 276.7115, -2809.184, 13107.6}},
            {1, {1.050483, 1.595909, 222.6227, -2416.858, 9407.894, -12460.15}}}},
        {0.5,{
            {0, {-1.61915, -23.2942, 406.8652, -4234.284, 18540.95}},
            {1, {0.428236, 9.900402, 65.28261, -1217.764, 5435.676, -7654.235}}}},
        {0.8,{
            {0, {-1.069954, -27.7498, 576.8673, -6323.645, 27317.89}},
            {1, {-0.3402695, 19.09789, -71.12733, -235.8644, 2200.882, -3720.224}}}},
        {0.9,{
            {0, {-0.7906876, -30.03265, 661.9614, -7425.294, 32124.67}},
            {1, {-0.9360614, 27.63615, -188.8822, 563.4788, -311.7842, -782.7323}}}},
        {0.95,{
            {0, {-0.5575089, -30.92955, 687.6883, -7680.324, 33082.79}},
            {1, {-1.682314, 34.56686, -263.5979, 1071.46, -1998.443, 1295.644}}}},
        {0.975,{
            {0, {-0.3564748, -31.35741, 695.8376, -7714.068, 33126.65}},
            {1, {-2.585414, 41.35117, -301.7647, 1242.268, -2510.971, 1935.754}}}},
        {0.99,{
            {0, {-0.1274131, -31.69644, 693.3064, -7628.013, 33448.61}},
            {1, {-3.924606, 52.94466, -361.8511, 1435.423, -2908.53, 2306.161}}}},
        {0.995,{
            {0, {0.02660655, -32.1516, 699.7033, -7685.206, 34284.54}},
            {1, {-5.082007, 65.0375, -432.9132, 1686.095, -3478.321, 2928.382}}}},
        {0.999,{
            {0, {0.3354447, -33.57506, 740.3102, -8450.701, 40957.97}},
            {1, {-7.850939, 99.3478, -677.2877, 2666.331, -5583.625, 4786.65}}}}
        }}
};

#endif
