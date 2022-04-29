#include <iostream>

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"

int main() {
	const double value_1 = 0; // typical v1 from net14 fortran
	const double cv = 1e9; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
	double rho = 5e5; // rho, g/cm^3
	double last_T = 1e9;

	// initial state
	std::vector<double> last_Y(14, 0), X(14, 0);
	X[0] = 1;


	for (int i = 0; i < 14; ++i) last_Y[i] = X[i]/nnet::net14::constants::A[i];

	double E_in = eigen::dot(last_Y, nnet::net14::BE) + cv*last_T;
	double m_in = eigen::dot(last_Y, nnet::net14::constants::A);

	double t = 0, dt=1e-12;
	int n_max = 400;
	const int n_print = 30, n_save=400;

	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

#ifdef DEBUG
		net14_debug = true;
#endif


	auto const eos = [&](const std::vector<double> &Y_, const double T, const double rho_) {
		return std::tuple<double, double, double>{cv, 0, value_1};
	};


	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Y, T, current_dt] = solve_system_NR(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, eos,
				nnet::net14::constants::A, last_Y, last_T, rho, dt);
		t += current_dt;

		net14_debug = false;


		double E_tot = eigen::dot(Y, nnet::net14::BE) + cv*T;
		double dE_E = (E_tot - E_in)/E_in;

		double m_tot = eigen::dot(Y, nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

		// formated print (stderr)
		if (n_save >= n_max || (n_max - i) % (int)((float)n_max/(float)n_save) == 0) {
			for (int i = 0; i < 14; ++i) X[i] = Y[i]*nnet::net14::constants::A[i]/eigen::dot(Y, nnet::net14::constants::A);
			std::cerr << t << "," << dt << ",," << T << ",,";
			for (int i = 0; i < 14; ++i) std::cerr << X[i] << ",";
			std::cerr << "," << dm_m << "\n";
		}

		// debug print
		if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0) {
			for (int i = 0; i < 14; ++i) X[i] = Y[i]*nnet::net14::constants::A[i]/eigen::dot(Y, nnet::net14::constants::A);
			std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
			for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
			std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\t(E=" << E_tot << ",\tdE_E=" << dE_E << "),\t" << T << "\n";
		}

		last_Y = Y;
		last_T = T;
	}


	return 0;
}