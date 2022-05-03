#include <iostream>

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif
	
	const double value_1 = 0; // typical v1 from net14 fortran
	double rho = 1e9; // rho, g/cm^3
	double last_T = 1e9;

	// extension /* TODO */
	const int start_expansion = 600;
	const double rho_half_life = 0.02;
	const double rho_lim = 1e5;

	// initial state
	std::vector<double> last_Y(14, 0), X(14, 0);
	X[1] = 0.5;
	X[2] = 0.5;

	for (int i = 0; i < 14; ++i) last_Y[i] = X[i]/nnet::net14::constants::A[i];

	// double E_in = eigen::dot(last_Y, nnet::net14::BE) + cv*last_T ;
	double m_in = eigen::dot(last_Y, nnet::net14::constants::A);

	double t = 0, dt=1e-12;
	int n_max = 1000;
	const int n_print = 30, n_save=1000;

	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

#ifdef DEBUG
		nnet::debug = true;
#endif


	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::A, nnet::net14::constants::Z);
	const auto eos = [&](const std::vector<double> &Y_, const double T, const double rho_) {
		const double cv = 3.1e7; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
		struct eos_output {
			double cv, dP_dT;
		} res{cv, 0};
		return res;
	};


	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Y, T, current_dt] = solve_system_NR(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, 
#ifndef DONT_USE_HELM_EOS
			helm_eos,
#else
			eos,
#endif
			last_Y, last_T, rho, 0., dt);
		t += current_dt;

		nnet::debug = false;


		// double E_tot = eigen::dot(Y, nnet::net14::BE) + cv*T;
		// double dE_E = (E_tot - E_in)/E_in;

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
			std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tcv=" << helm_eos(last_Y, last_T, rho).cv << ",\t" << T << "\n";
		}

		last_Y = Y;
		last_T = T;
	}


	return 0;
}