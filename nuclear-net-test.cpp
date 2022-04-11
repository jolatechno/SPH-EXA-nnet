#include <iostream>
#include "nuclear-net-mockup.hpp"

int main() {
	// photodesintegration rates
	Eigen::MatrixXd r(3,3);
	Eigen::MatrixXd dr(3,3);
	dr(1,0) = 0.1;
	dr(0,1) = -0.2;

	// fusion rates
	Eigen::Tensor<float, 3> fu(3, 3, 3);
	Eigen::Tensor<float, 3> dfu(3, 3, 3);
	dfu(2, 0, 0) = 0.2;
	dfu(2, 0, 1) = 0.1;
	dfu(2, 1, 1) = 0.2;

	// fission rates
	Eigen::Tensor<float, 3> fi(3, 3, 3);
	Eigen::Tensor<float, 3> dfi(3, 3, 3);
	dfi(0, 0, 2) = 0.01;
	dfi(0, 1, 2) = 0.03;

	// mass excedents
	Eigen::VectorXd Q(3);
	Q(0) = 0.4;
	Q(1) = 0.3;
	Q(2) = 0.2;

	// molar masses
	Eigen::VectorXd M(3);
	M(0) = 2;
	M(1) = 2;
	M(2) = 4;

	// initial state
	Eigen::VectorXd Y(3);
	double T = 1;
	Y(0) = 0.8;
	Y(1) = 0.7;
	Y(2) = 0.2;

	std::cout << Y.transpose() << ", " << T << std::endl;

	auto last_Y = Y;
	double last_T = T;
	double E_tot, E_tot_0 = Y.dot(Q) + T;
	double m_tot, m_tot_0 = Y.dot(M);

	double dt=1e-3, T_max = 2;
	const int n_print = 20;
	for (int i = 0; i < T_max/dt; ++i) {
		fu(2, 0, 0) = 0.7 + dfu(2, 0, 0)*(T - 1);
		fu(2, 0, 1) = 0.6 + dfu(2, 0, 1)*(T - 1);
		fu(2, 1, 1) = 0.4 + dfu(2, 1, 1)*(T - 1);

		fi(0, 0, 2) = 0.1 + dfi(0, 0, 2)*(T - 1);
		fi(0, 1, 2) = 0.05 + dfi(0, 1, 2)*(T - 1);

		r(0,1) = 0.8 + dr(0,1)*(T - 1);
		r(1,0) = 0.7 + dr(1,0)*(T - 1);

		//to insure that the equation is dY/dt = r*Y
		auto r_included = nnet::desintegration_rate_to_first_order(r);
		auto dr_included = nnet::desintegration_rate_to_first_order(dr);

		// add fission rates to desintegration rates
		r_included += nnet::fission_to_desintegration_rates(fi);
		dr_included += nnet::fission_to_desintegration_rates(dfi);

		// add fusion rates to desintegration rates
		r_included += nnet::fusion_to_desintegration_rates(fu, Y);
		dr_included += nnet::fusion_to_desintegration_rates(dfu, Y);

		// add temperature to the problem
		auto RQ = nnet::include_temp(r_included, Q, 1);

		// solve the system
		std::tie(Y, T) = nnet::solve_first_order(Y, T, RQ, dr_included, dt, 0.6);

		E_tot = Y.dot(Q) + T;
		m_tot = Y.dot(M);

		if (i % (int)((float)T_max / (dt*(float)n_print)) == 0)
			std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n"; //",\t(" << Q.dot(Y - last_Y) << " ?= "<< (T - last_T) << ")\n";

		if (i != T_max/dt - 1) {
			last_Y = Y;
			last_T = T;
		}
	}

	std::cout << Y.transpose() << ",\t(E_tot=" << E_tot << ",\tDelta_E_tot=" << E_tot_0 - E_tot << "),\t(m_tot=" << m_tot << ",\tDelta_m_tot=" << m_tot_0 - m_tot << "),\t" << T << "\n"; //",\t(" << Q.dot(Y - last_Y) << " ?= "<< (T - last_T) << ")\n";
}