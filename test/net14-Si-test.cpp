#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net14/net14.hpp"

int main() {
	const double value_1 = 0; // typical v1 from net14 fortran
	const double cv = 1e30; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
	const double rho = 1e7; // rho, g/cm^3
	double T = 6e9;

	// initial state
	Eigen::VectorXd Y(14), X = Eigen::VectorXd::Zero(14);
	X(5) = 1;

	for (int i = 0; i < 14; ++i) Y(i) = X(i) / nnet::net14::constants::A(i);

	
	auto last_Y = Y;
	double last_T = T;
	double m_in = Y.dot(nnet::net14::constants::A);

	const double max_dt=5e-2, min_dt=1e-17;
	double t = 0, dt=1e-12;
	int n_max = 100000;
	const int n_print = 30, n_save=4000;

	const double theta = 0.5;


	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

	for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
	std::cout << "\t" << T << std::endl;

	const double max_dm = 1e-6;
	double old_dm_m = 0;

	for (int i = 1; i <= n_max; ++i) {
		double m_tot, dm_m, next_T;
		Eigen::VectorXd next_Y(14);

		for (int j = 0;; ++j) {
			// solve the system
			std::tie(next_Y, next_T, dt) = nnet::solve_system(nnet::net14::construct_system(rho, cv, value_1), Y, T, dt, theta, 1e-10, 1e-100, 1e-100);
			t += dt;

			// loop condition
			m_tot = next_Y.dot(nnet::net14::constants::A);
			dm_m = (m_tot - m_in)/m_in;
			if (std::abs(dm_m - old_dm_m) <= max_dm) {
				dt = std::min(max_dt, dt*1.2);
				break;
			}

			// timestep tweeking
			dt = std::max(min_dt, dt*0.7);

			// std::cout << "\t" << dm_m << " > " << old_dm_m << ",    " << dt << "\n";
		}

		Y = next_Y;
		T = next_T;

		// formated print (stderr)
		if (n_save >= n_max || (n_max - i) % (int)((float)n_max/(float)n_save) == 0) {
			double m_tot = Y.dot(nnet::net14::constants::A);
			for (int i = 0; i < 14; ++i) X(i) = Y(i) * nnet::net14::constants::A(i);
			std::cerr << t << "," << dt << ",," << T << ",,";
			for (int i = 0; i < 14; ++i) std::cerr << X(i) << ",";
			std::cerr << "," << dm_m << "\n";
		}

		// debug print
		if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0) {
			
			for (int i = 0; i < 14; ++i) X(i) = Y(i) * nnet::net14::constants::A(i);

			std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
			for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
			std::cout << "\t(m_tot=" << m_tot << ",\tDelta_m_tot/m_tot=" << dm_m << "),\t" << T << "\n";
		}


		old_dm_m = dm_m;

		last_Y = Y;
		last_T = T;
	}


	return 0;
}