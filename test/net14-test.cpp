#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net14/net14.hpp"

double Adot(Eigen::Vector<double, 14> const &Y) {
	double norm = 0;
	for (int i = 0; i < 14; ++i)
		norm += Y(i)*nnet::net14::constants::A(i);
	return norm;
}

int main() {
	const double value_1 = 0; // typical v1 from net14 fortran
	const double cv = 2e7; // typical cv from net14 fortran
	const double density = 1e9; // density, g/cm^3

	// initial state
	Eigen::VectorXd Y(14), X = Eigen::VectorXd::Zero(14);
	X(1) = 0.5;
	X(2) = 0.5;

	for (int i = 0; i < 14; ++i) Y(i) = X(i) / nnet::net14::constants::A(i);

	double T = 1e9;
	auto last_Y = Y;
	double last_T = T;
	double m_tot, m_in = Adot(Y);

	double dt=1e-16, t_max = 5.;
	int n_max = 10000; //t_max/dt;
	const int n_print = 20;

	const double theta = 0.6;

	net14_debug=true;
	auto construct_system = [&](const Eigen::VectorXd &Y_, double T_) {
		// compute rates
		auto rates = nnet::net14::compute_reaction_rates(T_);
		net14_debug = false;

		auto M = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, rates, density, Y_);

		// include temperature
		Eigen::VectorXd BE = nnet::net14::BE + nnet::net14::ideal_gaz_correction(T_);
		return nnet::include_temp(M, value_1, cv, BE, Y_);
	};

	for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
	std::cout << "\t" << T << std::endl;

	Eigen::VectorXd BE = nnet::net14::BE + nnet::net14::ideal_gaz_correction(T);
	std::cout << "\nBE(T=" << T <<")=" << BE.transpose() << "\n\n";

	double delta_m = 0;
	for (int i = 0; i < n_max; ++i) {
		/* ---------------------
		begin test
		--------------------- */
		/*if (i == 0) {
			auto rates = nnet::net14::compute_reaction_rates(T);

			auto M = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, rates, density, Y);

			if (i  == 0)
				std::cout << "\n" << (M*Y).transpose() << "\n" << Adot(M*Y) << "\n\n";

			// include temperature
			Eigen::VectorXd BE = nnet::net14::BE - nnet::net14::ideal_gaz_correction(T);
			auto Mp = nnet::include_temp(M, value_1, cv, BE, Y);

			if (i == 0)
				std::cout << 
					// M
					//Mp
					// Eigen::MatrixXd::Identity(15, 15) - dt*theta*Mp
				<< "\n\n";
		}
		/* ---------------------
		end test
		--------------------- */




		// solve the system
		std::tie(Y, T) = nnet::solve_system(construct_system, Y, T, dt, theta, 1e-5, 0.);

		m_tot = Adot(Y);


		/*if (i == 0) {
			std::cout << "\n, dY=";
			for (int i = 0; i < 14; ++i) std::cout << (Y(i) - last_Y(i)) / dt << ", ";
			std::cout << "\n";
		}*/

		if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0) {
			for (int i = 0; i < 14; ++i) X(i) = Y(i) * nnet::net14::constants::A(i);
			std::cout << "\n";
			for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
			std::cout << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << (m_tot - m_in)/m_in*100 << "%),\t" << T << "\n";
		}

		last_Y = Y;
		last_T = T;
	}

	return 0;
}