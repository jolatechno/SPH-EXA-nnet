#include <iostream>

#include "nuclear-net.hpp"
#include "net14.hpp"

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
	Eigen::Vector<double, -1> Y(14);
	double T = 1.8e9;
	for (int i = 0; i < 14; ++i)
		Y(i) = 0.1;

	// normalize Y
	Y /= Adot(Y);

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double m_tot;

	double dt=1e-9, t_max = 5e-2;
	int n_max = 100000; //t_max/dt;
	const int n_print = 20;



	auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
		/*auto r = nnet::net14::get_photodesintegration_rates(T); 
		auto f = nnet::net14::get_fusion_rates(T);

		//to insure that the equation is dY/dt = r*Y
		auto M = nnet::photodesintegration_to_first_order(r, nnet::net14::n_photodesintegration);

		// add fusion rates to desintegration rates
		M += nnet::fusion_to_first_order(f, nnet::net14::n_fusion, Y);*/

		// compute rates
		auto rates = nnet::net14::compute_reaction_rates(T);

		auto M = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, rates, Y);

		// include temperature
		return nnet::include_temp(M, value_1, cv, nnet::net14::BE, Y);
	};



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
		std::tie(Y, T) = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-20);

		m_tot = Adot(Y);

		// !!!!!!!!
		// cheating by normalizing
		//Y /= m_tot;

		if (i % (int)((float)n_max/(float)n_print) == 0)
			std::cout << "\n" << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot - 1. << "),\t" << T << "\n";

		if (i != n_max - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	std::cout << "\n" << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot - 1. << "),\t" << T << "\n";
}


