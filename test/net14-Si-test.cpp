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

	double t = 0, dt=1e-12;
	int n_max = 100000;
	const int n_print = 30, n_save=4000;


	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

	for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
	std::cout << "\t" << T << std::endl;


	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Mp, dM_dT] = nnet::net14::construct_system(Y, T, rho, cv, value_1);
		std::tie(Y, T, dt) = nnet::solve_system_var_timestep(Mp, dM_dT, Y, T, nnet::net14::constants::A, dt);
		t += dt;

		double m_tot = Y.dot(nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

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

		last_Y = Y;
		last_T = T;
	}


	return 0;
}