#include <iostream>
#include "nuclear-net.hpp"

double Adot(Eigen::Vector<double, 14> const &Y) {
	double norm = 0;
	for (int i = 0; i < 14; ++i)
		norm += Y(i)*nnet::net14::constants::A(i);
	return norm;
}

int main() {
	// initial state
	Eigen::Vector<double, -1> Y(14);
	double T = 1e9;
	Y(0) = 0.8;
	Y(1) = 0.7;
	Y(2) = 0.2;

	// normalize Y
	Y /= Adot(Y);

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double m_tot, m_tot_0 = Adot(Y);

	double dt=5e-10, T_max = 2;
	int n_max = 200; //T_max/dt;
	const int n_print = 20;
	for (int i = 0; i < n_max; ++i) {
		auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
			auto r = nnet::net14::get_photodesintegration_rates(T); 
			auto f = nnet::net14::get_fusion_rates(T);

			//to insure that the equation is dY/dt = r*Y
			auto M = nnet::photodesintegration_to_first_order(r, nnet::net14::n_photodesintegration);

			// add fusion rates to desintegration rates
			M += nnet::fusion_to_first_order(f, nnet::net14::n_fusion, Y);

			/*if (i == 0)
				std::cout << "\n\n\n\n" << M << "\n\n\n\n";*/

			// no derivative
			auto dMdT = M;
			dMdT.setZero();

			// add temperature to the problem
			return nnet::include_temp(M, dMdT, 1., 0., nnet::net14::BE, Y);
		};

		// solve the system
		auto DY_T = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-2);
		Y += DY_T(Eigen::seq(1, 14));
		T += DY_T(0);

		m_tot = Adot(Y);

		if (i % (int)((float)n_max/(float)n_print) == 0)
			std::cout << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n\n";

		if (i != T_max/dt - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	std::cout << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n";
}


