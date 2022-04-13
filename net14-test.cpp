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
	double T = 1.2e9;
	Y(0) = 0.8;
	Y(1) = 0.7;
	for (int i = 2; i < 14; ++i)
		Y(i) = 0.1;

	// normalize Y
	Y /= Adot(Y);

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double m_tot, m_tot_0 = Adot(Y);

	double dt=5e-12, T_max = 1e-9;
	int n_max = T_max/dt;
	const int n_print = 20;
	for (int i = 0; i < n_max; ++i) {
		auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
			auto r = nnet::net14::get_photodesintegration_rates(T); 
			auto f = nnet::net14::get_fusion_rates(T);

			//to insure that the equation is dY/dt = r*Y
			auto M = nnet::photodesintegration_to_first_order(r, nnet::net14::n_photodesintegration);

			if (i == 0)
				std::cout << "\n\n" << Adot(M*Y) << ",";

			// add fusion rates to desintegration rates
			M += nnet::fusion_to_first_order(f, nnet::net14::n_fusion, Y);


			if (i /*% (int)((float)n_max/(float)n_print)*/ == 0)
				std::cout << Adot(M*Y) << "\n\n" << M << "\n\n";


			// no derivative
			Eigen::Matrix<double, -1, -1> dMdT = Eigen::Matrix<double, -1, -1>::Zero(14, 14);

			// add temperature to the problem
			return std::pair<Eigen::Matrix<double, -1, -1>, Eigen::Matrix<double, -1, -1>>{nnet::include_temp(M, 1., 0., nnet::net14::BE, Y), dMdT};
		};

		// solve the system
		auto DY_T = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-23);
		std::tie(Y, T) = nnet::add_and_cleanup(Y, T, DY_T);

		m_tot = Adot(Y);

		// !!!!!!!!
		// cheating by normalizing
		//Y /= m_tot;

		if (i % (int)((float)n_max/(float)n_print) == 0)
			std::cout << "\n" << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n";

		if (i != T_max/dt - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	std::cout << Y.transpose() << "\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n";
}


