#pragma once

#include <vector>

namespace nnet::net14::constants {
	const double Kb = 1.380658e-16;
	const double Na = 6.022137e23;
	const double e2 = 2.306022645e-19;
	const double Mev_to_cJ = 9.648529392e17;

	/// constant atomic number values
	const std::vector<double> Z = {2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};

	/// constant number of masses values
	const std::vector<double> A = {4, 12, 16, 20, 24, 28, 32,  36, 40, 44, 48, 52, 56, 60};

	namespace fits {
		int inline get_temperature_range(double T) {
			if (T < 1.5e8) return 0;

			if (T < 2e8) return 1;
			if (T < 3e8) return 2;
			if (T < 4e8) return 3;
			if (T < 5e8) return 4;
			if (T < 6e8) return 5;
			if (T < 7e8) return 6;
			if (T < 8e8) return 7;
			if (T < 9e8) return 8;
			if (T < 1e9) return 9;

			if (T < 1.5e9) return 10;
			if (T <   2e9) return 11;
			if (T < 2.5e9) return 12;
			if (T <   3e9) return 13;
			if (T < 3.5e9) return 14;
			if (T <   4e9) return 15;
			if (T < 4.5e9) return 16;
			if (T <   5e9) return 17;

			if (T <  6e9) return 18;
			if (T <  7e9) return 19;
			if (T <  8e9) return 20;
			if (T <  9e9) return 21;
			if (T < 1e10) return 22;

			return 23;
		}

		const double q[14 - 4] = {9.3160e0, 9.9840e0, 6.9480e0, 6.6390e0, 7.0400e0, 5.1270e0, 7.6920e0, 7.9390e0, 7.9950e0, 2.7080e0};

		const double fit[14 - 4][8] = {
			{1.335429e2, -2.504361e0,   7.351683e1, -2.217197e2,  1.314774e1, -7.475602e-1, 9.602703e1,  1.583615e2},
			{1.429069e2, -3.288633e0,   1.042707e2, -2.650548e2,  1.391863e1, -6.999523e-1, 1.216164e2,  1.677677e2},
			{9.710066e1, -3.324446e0,   5.358524e1, -1.656830e2,  7.199997e0, -2.828443e-1, 7.933873e1,  1.219924e2},
			{-1.917005e2, 6.797907e-1, -3.384737e2,  5.501609e2, -3.881261e1,  2.530003e0, -2.432384e2, -1.667851e2},
			{-1.290140e2, 3.252004e-1, -3.322780e2,  4.687703e2, -2.913671e1,  1.765989e0, -2.224539e2, -1.040800e2},
			{-7.480556e2, 1.235853e1,  -1.360082e3,  2.199106e3, -1.330803e2,  7.734556e0, -1.034036e3, -7.231065e2},
			{-9.130837e2, 1.594906e1,  -1.694960e3,  2.711843e3, -1.575353e2,  8.856425e0, -1.292620e3, -8.881222e2},
			{-9.291031e2, 2.057299e1,  -2.039678e3,  3.077998e3, -1.715707e2,  9.388271e0, -1.509299e3, -9.041311e2},
			{-1.051633e3, 2.255988e1,  -2.240776e3,  3.416331e3, -1.925435e2,  1.063188e1, -1.666427e3, -1.026652e3},
			{-1.043410e3, 2.280261e1,  -2.281027e3,  3.453872e3, -1.969194e2,  1.101885e1, -1.685657e3, -1.018421e3}
		};

		const double choose[14 - 4 + 1][24] = {
			{
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000016,
				1.000382, 1.002544, 1.009003, 1.022212, 1.043747, 1.074176, 1.113314, 1.215134, 1.343451, 1.493867, 1.664363, 1.854977
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000001, 1.000126,
				1.001778, 1.008703, 1.025095, 1.053477, 1.094361, 1.146883, 1.209546, 1.359068, 1.532917, 1.725840, 1.936524, 2.166286
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000005, 
				1.000164, 1.001296, 1.005132, 1.013718, 1.028685, 1.050938, 1.080705, 1.161517, 1.266929, 1.392745, 1.536751, 1.699342
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
				1.000012, 1.000160, 1.000897, 1.003083, 1.007809, 1.016163, 1.029098, 1.071801, 1.141522, 1.244530, 1.389797, 1.590952
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000001, 
				1.000054, 1.000533, 1.002449, 1.007289, 1.016565, 1.031512, 1.053057, 1.118848, 1.220214, 1.366498, 1.572476, 1.860793, 
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
				1.000000, 1.000000, 1.000008, 1.000060, 1.000286, 1.000980, 1.002659, 1.012335, 1.038607, 1.094915, 1.199775, 1.379520, 
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000001, 1.000004, 1.000017, 1.001149, 
				1.009353, 1.033083, 1.077523, 1.144356, 1.234002, 1.347212, 1.485621, 1.848303, 2.346120, 3.005278, 3.855614, 4.940249, 
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000002, 1.000019, 1.000091, 1.000307, 1.000809, 1.014850, 
				1.063783, 1.153860, 1.279281, 1.432145, 1.606260, 1.797679, 2.004396, 2.463350, 2.995063, 3.632788, 4.440947, 5.538404, 
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000004, 1.000022, 1.000087, 1.000261, 1.006988, 
				1.036149, 1.097022, 1.187924, 1.302823, 1.435985, 1.583461, 1.743116, 2.097506, 2.508070, 3.004926, 3.650892, 4.565183, 
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
				1.000001, 1.000018, 1.000148, 1.000669, 1.002103, 1.005194, 1.010868, 1.034512, 1.084652, 1.180068, 1.354034, 1.666865, 
			}, {
				1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000002, 1.000012, 1.000043, 1.002113, 
				1.014763, 1.047604, 1.104672, 1.185526, 1.288184, 1.411048, 1.553941, 1.909907, 2.405751, 3.157500, 4.395655, 6.561123
			}
        };
	}
}