#include <iostream>

#include "../src/nuclear-net.hpp"

int main() {
	double value_1 = 0;
	double cv = 1e-1;
	double rho = 1;

	// mass excedents
	Eigen::VectorXd BE(3);
	BE(0) = 0;
	BE(1) = 1;
	BE(2) = 4;

	// molar masses
	Eigen::VectorXd m(3);
	m(0) = 2;
	m(1) = 2;
	m(2) = 4;

	// initial state
	Eigen::VectorXd last_Y(3);
	double last_T = 1;
	last_Y(0) = 0.8;
	last_Y(1) = 0.7;
	last_Y(2) = 0.1;

	// normalize Y
	last_Y /= last_Y.dot(m);

	std::cout << last_Y.transpose() << ", " << last_T << std::endl;

	double E_tot, E_tot_0 = last_Y.dot(m + BE) + cv*last_T;
	double m_tot, m_tot_0 = last_Y.dot(m);

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
	auto construct_system = [&](const Eigen::VectorXd &Y_, double T_) {
		std::vector<double> drates = {
			// two simple photodesintegration (i -> j)
			1.,
			0.7,

			// different species fusion (i + j -> k)
			1.1,

			// two different species "fission" (photodesintegration, i > j + k)
			0.9,

			// same species fusion (i + i -> j)
			-0.5,

			// same species "fission" (photodesintegration, i -> j + j)
			-0.15,
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

		return std::tuple<std::vector<double>, std::vector<double>>{rates, drates};
	};

	double dt=2e-2, T_max = 2;
	int n_max = 100; //T_max/dt;
	const int n_print = 20;
	for (int i = 0; i < n_max; ++i) {

		auto [rate, drates_dT] = construct_system(last_Y, last_T);

		// solve the system
		net14_debug = i == 0;
		auto [Y, T, _] = nnet::solve_system(reactions, rate, drates_dT,
			BE, last_Y, m,
			last_T, cv, rho, value_1, dt);
		net14_debug = false;

		E_tot = Y.dot(m + BE) + cv*T;
		m_tot = Y.dot(m);

		if (n_print >= n_max || (n_max - i) % (int)((float)n_max / ((float)n_print)) == 0)
			std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << (E_tot - E_tot_0)/E_tot_0 << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << (m_tot - m_tot_0)/m_tot_0 << "),\t" << T << "\n";

		last_Y = Y;
		last_T = T;
	}
}


