#include <iostream>

#include "../src/nuclear-net.hpp"

int main() {
	double value_1 = 0;
	double cv = 1e-1;
	double rho = 1;

	// mass excedents
	std::vector<double> BE = {.3, .2, .7};

	// molar masses
	std::vector<double> m = {2, 2, 4};

	// initial state
	std::vector<double> last_Y = {0.1, 0.2, 0.1};
	double last_T = 1;

	std::cout << last_Y[0] << " " << last_Y[1] << " " << last_Y[2] << ",\t" << last_T << std::endl;

	double  m_tot,  m_tot_0 = eigen::dot(last_Y, m);
	double BE_tot, BE_tot_0 = eigen::dot(last_Y, BE);
	double  E_tot,  E_tot_0 = -BE_tot_0 + m_tot_0 + cv*last_T;

	nnet::constants::theta = 0.6;


	/* -----------------
	reaction list */
	std::vector<nnet::reaction> reactions = {
		// two simple photodesintegration (i -> j)
		{{{0}}, {{1}}},
		{{{1}}, {{0}}},

		// different species fusion (i + j -> k)
		{{{1}, {0}}, {{2}}},

		// two different species "fission" (photodesintegration, i > j + k)
		{{{2}}, {{1}, {0}}},

		// same species fusion (i + i -> j)
		{{{1, 2}}, {{2}}},

		// same species "fission" (photodesintegration, i -> j + j)
		{{{2}}, {{0, 2}}},
	};

	/* -----------------
	construct system function */
	auto construct_system = [&](double T_) {
		std::vector<double> drates = {
			// two simple photodesintegration (i -> j)
			1.,
			0.7,

			// different species fusion (i + j -> k)
			1.1,

			// two different species "fission" (photodesintegration, i > j + k)
			0.9,

			// same species fusion (i + i -> j)
			0.5,

			// same species "fission" (photodesintegration, i -> j + j)
			0.15,
		};

		std::vector<double> rates = {
			// two simple photodesintegration (i -> j)
			0.3  + drates[0]*T_,
			0.2  + drates[1]*T_,

			// different species fusion (i + j -> k)
			0.1  + drates[2]*T_,

			// two different species "fission" (photodesintegration, i > j + k)
			0.15 + drates[3]*T_,

			// same species fusion (i + i -> j)
			1.25 + drates[4]*T_,

			// same species "fission" (photodesintegration, i -> j + j)
			1    + drates[5]*T_,
		};

		return std::pair<std::vector<double>, std::vector<double>>{rates, drates};
	};

	double dt=2e-2, T_max = 2;
	int n_max = 100; //T_max/dt;
	const int n_print = 20;
	for (int i = 0; i < n_max; ++i) {
		auto [rate, drates_dT] = construct_system(last_T);

		// solve the system
		net14_debug = i == 0;
		auto [Y, T] = nnet::solve_system(reactions, rate, drates_dT,
			BE, m, last_Y,
			last_T, cv, rho, value_1, dt);
		net14_debug = false;

		 m_tot = eigen::dot(Y, m);
		BE_tot = eigen::dot(last_Y, BE);
		 E_tot = -BE_tot + m_tot + cv*T;

		if (n_print >= n_max || (n_max - i) % (int)((float)n_max / ((float)n_print)) == 0)
			std::cout << Y[0] << " " << Y[1] << " " << Y[2] << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot - E_tot_0 << ",\tDelta_BE_tot=" << BE_tot - BE_tot_0 << ",\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << (m_tot - m_tot_0)/m_tot_0 << "),\t" << T << "\n";

		last_Y = Y;
		last_T = T;
	}
}


