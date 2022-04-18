#include <iostream>

#include "../src/nuclear-net.hpp"

int main() {
	double value_1 = 0;
	double cv = 1;

	// mass excedents
	Eigen::VectorXd BE(3);
	BE(0) = 3;
	BE(1) = 4;
	BE(2) = 9;

	// molar masses
	Eigen::VectorXd m(3);
	m(0) = 2;
	m(1) = 2;
	m(2) = 4;

	// initial state
	Eigen::VectorXd Y(3);
	double T = 1;
	Y(0) = 0.8;
	Y(1) = 0.7;
	Y(2) = 0.1;

	// normalize Y
	Y /= Y.dot(m);

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double E_tot, E_tot_0 = Y.dot(m + BE) + cv*T;
	double m_tot, m_tot_0 = Y.dot(m);

	const double theta = 0.6;

	double dt=5e-3, T_max = 2;
	int n_max = T_max/dt;
	const int n_print = 20;
	for (int i = 0; i < n_max; ++i) {

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
			std::vector<double> rates = {
				// two simple photodesintegration (i -> j)
				0.3 + T_,
				0.2 + 0.7*T_,

				// different species fusion (i + j -> k)
				0.1 + 1.1*T_,

				// two different species "fission" (photodesintegration, i > j + k)
				0.15 + 0.9*T_,

				// same species fusion (i + i -> j)
				1.25 - 0.5*T_,

				// same species "fission" (photodesintegration, i -> j + j)
				1 - 0.15*T_,
			};

			// generate matrix
			Eigen::MatrixXd M = nnet::first_order_from_reactions<double>(reactions, rates, 1., Y_);

			// add temperature to the problem
			Eigen::MatrixXd Mp = nnet::include_temp(M, value_1, cv, BE, Y_);



			if (i == 0)
				std::cout << "\n\n" << Mp << "\n\n";



			return Mp;
		};

		construct_system(Y, T);

		// solve the system
		std::tie(Y, T) = nnet::solve_system(construct_system, Y, T, dt, theta, 1e-18);

		E_tot = Y.dot(m + BE) + cv*T;
		m_tot = Y.dot(m);

		if (n_print >= n_max || (n_max - i) % (int)((float)T_max / (dt*(float)n_print)) == 0)
			std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n";

		last_Y = Y;
		last_T = T;
	}
}


