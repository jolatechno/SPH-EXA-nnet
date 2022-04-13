#include <iostream>
#include "nuclear-net.hpp"

double Adot(Eigen::Vector<double, 14> const &Y) {
	double norm = 0;
	for (int i = 0; i < 14; ++i)
		norm += Y(i)*nnet::net14::constants::A(i);
	return norm;
}

int main() {
	double cv = 1e-9;

	// initial state
	Eigen::Vector<double, -1> Y(14);
	double T = 1.2e9;
	for (int i = 0; i < 14; ++i)
		Y(i) = 0.1;

	// normalize Y
	Y /= Adot(Y);

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double m_tot;

	double dt=5e-7, t_max = 5e-2;
	int n_max = 100000; //t_max/dt;
	const int n_print = 20;



	auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
		auto r = nnet::net14::get_photodesintegration_rates(T); 
		auto f = nnet::net14::get_fusion_rates(T);

		//to insure that the equation is dY/dt = r*Y
		auto M = nnet::photodesintegration_to_first_order(r, nnet::net14::n_photodesintegration);

		// add fusion rates to desintegration rates
		M += nnet::fusion_to_first_order(f, nnet::net14::n_fusion, Y);

		M *= nnet::net14::constants::UNKNOWN/1.0e12;

		// include temperature
		return nnet::include_temp(M, cv, 0., nnet::net14::BE, Y);
	};



	double delta_m = 0;
	for (int i = 0; i < n_max; ++i) {
		/* ---------------------
		test
		--------------------- */
		if (i /*% (int)((float)n_max/(float)n_print)*/ == 0) {
			auto r = nnet::net14::get_photodesintegration_rates(T); 
			auto f = nnet::net14::get_fusion_rates(T);

			//to insure that the equation is dY/dt = r*Y
			auto M = nnet::photodesintegration_to_first_order(r, nnet::net14::n_photodesintegration);
			
			if (i /*% (int)((float)n_max/(float)n_print)*/ == 0)
				std::cout << "\n\n" << Adot(M*Y) << ", ";

			// add fusion rates to desintegration rates
			M += nnet::fusion_to_first_order(f, nnet::net14::n_fusion, Y);

			if (i /*% (int)((float)n_max/(float)n_print)*/ == 0)
				std::cout << Adot(M*Y) << "\n\n";

			M *= nnet::net14::constants::UNKNOWN/1.0e12;

			// include temperature
			auto Mp = nnet::include_temp(M, cv, 0., nnet::net14::BE, Y);

			if (i /*% (int)((float)n_max/(float)n_print)*/ == 0)
				std::cout << Mp << "\n\n";

			/*auto DY = M*Y*dt;

			/* Y += DY;
			T += nnet::net14::BE.dot(DY)*10000000000;*/
		}
		/* ---------------------
		test
		--------------------- */


		// solve the system
		auto DY_T = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-16);
		std::tie(Y, T) = nnet::add_and_cleanup(Y, T, DY_T);

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

	std::cout << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot - 1. << "),\t" << T << "\n";
}


