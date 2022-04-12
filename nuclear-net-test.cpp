#include <iostream>
#include "nuclear-net.hpp"

int main() {
	std::cout << nnet::net14::BE.transpose() << "\n\n";
	std::cout << nnet::net14::n_photodesintegrations << "\n\n";
	//std::cout << nnet::net14::n_fusions << "\n\n";
	std::cout << nnet::net14::get_photodesintegration_rates((double)2e9) << "\n\n";


	// photodesintegration rates
	Eigen::MatrixXd r(3, 3);
	Eigen::MatrixXd dr(3, 3);
	Eigen::MatrixXd n(3, 3);
	n(1, 0) = 1; n(0, 1) = 1;
	dr(1, 0) = 0.1; dr(0, 1) = -2;

	n(0, 2) = 2; n(1, 2) = 2;
	dr(0, 2) = -0.1; dr(1, 2) = -3;

	// fusion rates
	Eigen::Tensor<double, 3> f(3, 3, 3);
	Eigen::Tensor<double, 3> df(3, 3, 3);
	Eigen::Tensor<int, 3> nf(3, 3, 3);
	nf(2, 0, 0) = 1;
	nf(2, 0, 1) = 1;
	df(2, 0, 0) = 0.2;
	df(2, 0, 1) = 0.1;

	nf(0, 1, 2) = 3;
	nf(0, 2, 2) = 4;
	df(0, 1, 2) = 1;
	df(0, 2, 2) = 2;

	// mass excedents
	Eigen::VectorXd BE(3);
	BE(0) = 0.4;
	BE(1) = 0.3;
	BE(2) = 0.2;

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
	Y(2) = 0.2;

	// normalize Y
	Y /= Y.dot(m);

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double E_tot, E_tot_0 = Y.dot(m + BE) + T;
	double m_tot, m_tot_0 = Y.dot(m);

	double dt=5e-3, T_max = 2;
	const int n_print = 20;
	for (int i = 0; i < T_max/dt; ++i) {
		auto construct_system = [&](const Eigen::VectorXd &Y, double T) {
			f(2, 0, 0) = 0.7 + df(2, 0, 0)*(T - 1);
			f(2, 0, 1) = 0.2 + df(2, 0, 1)*(T - 1);

			f(0, 1, 2) = 0.2 + df(0, 1, 2)*(T - 1);
			f(0, 2, 2) = 0.5 + df(0, 2, 2)*(T - 1);

			r(0, 1) = 0.8 + dr(0, 1)*(T - 1);
			r(1, 0) = 0.7 + dr(1, 0)*(T - 1);

			r(0, 2) = 0.4 + dr(0, 2)*(T - 1);
			r(1, 2) = 0.2 + dr(1, 2)*(T - 1);

			//to insure that the equation is dY/dt = r*Y
			auto M = nnet::photodesintegration_to_first_order(r, n);
			auto dMdT = nnet::photodesintegration_to_first_order(dr, n);

			// add fusion rates to desintegration rates
			M += nnet::fusion_to_first_order(f, nf, Y);
			dMdT += nnet::fusion_to_first_order(df, nf, Y);

			// add temperature to the problem
			return nnet::include_temp(M, dMdT, 1., 0., BE, Y);
		};

		// solve the system
		auto DY_T = nnet::solve_system(construct_system, Y, T, dt, 0.6, 1e-10);
		Y += DY_T(Eigen::seq(1, 3));
		T += DY_T(0);

		E_tot = Y.dot(m + BE) + T;
		m_tot = Y.dot(m);

		if (i % (int)((float)T_max / (dt*(float)n_print)) == 0)
			std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n"; //",\t(" << Q.dot(Y - last_Y) << " ?= "<< (T - last_T) << ")\n";

		if (i != T_max/dt - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n"; //",\t(" << Q.dot(Y - last_Y) << " ?= "<< (T - last_T) << ")\n";
}


