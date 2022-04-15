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
	double value_1 = 3.3587703131823750e-002; // typical v1 from net14 fortran
	double cv = 89668307.725306153; // typical cv from net14 fortran

	// initial state
	Eigen::Vector<double, -1> X(14), Y(14);
	X.setZero();
	X(1) = 0.5;
	X(2) = 0.5;

	for (int i = 0; i < 14; ++i) Y(i) = X(i) / nnet::net14::constants::A(i);

	double T = 1e9;
	auto last_Y = Y;
	double last_T = T;
	double m_tot;

	double dt=2e-3, t_max = 5.;
	int n_max = 10000; //t_max/dt;
	const int n_print = 100; //20;



	auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
		// compute rates
		auto rates = nnet::net14::compute_reaction_rates(T);

		auto M = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, rates, Y);

		// include temperature
		return nnet::include_temp(M, value_1, cv, nnet::net14::BE, Y);
	};




	std::cout << X.transpose() << ", " << T << std::endl;

	double delta_m = 0;
	for (int i = 0; i < n_max; ++i) {
		/* ---------------------
		begin test
		--------------------- */
		if (i /*% (int)((float)n_max/(float)n_print)*/ == 0) {
			auto rates = nnet::net14::compute_reaction_rates(T);

			auto M = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, rates, Y);

			if (i /*% (int)((float)n_max/(float)n_print)*/ == 0)
				std::cout << "\n" << (M*Y).transpose() << "\n" << Adot(M*Y) << "\n\n";

			// include temperature
			auto Mp = nnet::include_temp(M, value_1, cv, nnet::net14::BE, Y);

			if (i /*% (int)((float)n_max/(float)n_print)*/ == 0)
				std::cout << Mp << "\n\n";
		}
		/* ---------------------
		end test
		--------------------- */




		// solve the system
		std::tie(Y, T) = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-30, 1e-30);

		m_tot = Adot(Y);

		if (i % (int)((float)n_max/(float)n_print) == 0) {
			for (int i = 0; i < 14; ++i) X(i) = Y(i) * nnet::net14::constants::A(i);
			std::cout << "\n";
			for (int i = 0; i <= 6; ++i) std::cout << X(i) << ", ";
			std::cout << X(12) << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot - 1. << "),\t" << T << "\n";
		}

		if (i != n_max - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	for (int i = 0; i < 14; ++i) X(i) = Y(i) * nnet::net14::constants::A(i);
	std::cout << "\n";
	for (int i = 0; i <= 6; ++i) std::cout << X(i) << ", ";
	std::cout << X(12) << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot - 1. << "),\t" << T << "\n";

	return 0;
}