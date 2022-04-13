#include <iostream>
#include "nuclear-net.hpp"

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

	double dt=5e-3, T_max = 0;
	const int n_print = 20;
	for (int i = 0; i <= T_max/dt; ++i) {
		auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
			std::vector<std::pair<nnet::reaction, double>> reactions_and_rates = {
				// two simple photodesintegration (i -> j)
				//{nnet::reaction({(0)}, {(1)}), 0.4 + T},
				//{nnet::reaction({(1)}, {(0)}), 0.3 + 0.7*T},

				// different species fusion (i + j -> k)
				//{nnet::reaction({(1), (0)}, {(2)}), 0.5 + T},

				// two different species "fission" (photodesintegration, i > j + k)
				//{nnet::reaction({(2)}, {(1), (0)}), 0.3 + 1.1*T},

				// same species fusion (i + i -> j)
				//{nnet::reaction({(1, 2)}, {(2)}), 0.5 + T},

				// same species "fission" (photodesintegration, i -> j + j)
				{nnet::reaction({(2)}, {(0, 2)}), 0.5 + T},
			};

			// generate matrix
			Eigen::MatrixXd M = nnet::first_order_from_reactions<double>(reactions_and_rates, Y);

			// add temperature to the problem
			Eigen::MatrixXd Mp = nnet::include_temp(M, value_1, cv, BE, Y);



			if (i == 0)
				std::cout << "\n\n" << Mp << "\n\n";



			return Mp;
		};

		// solve the system
		std::tie(Y, T) = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-12);

		E_tot = Y.dot(m + BE) + cv*T;
		m_tot = Y.dot(m);

		if (i % (int)((float)T_max / (dt*(float)n_print)) == 0)
			std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n";

		if (i != T_max/dt - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n";
}


