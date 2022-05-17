#include <iostream>
#include <chrono>

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif
	
	double rho = 1e9; // rho, g/cm^3
	double last_T = 1e9;

	// initial state
	std::vector<double> last_Y(14, 0), X(14, 0);
	X[1] = 0.5;
	X[2] = 0.5;

	for (int i = 0; i < 14; ++i) last_Y[i] = X[i]/nnet::net14::constants::A[i];

	// double E_in = eigen::dot(last_Y, nnet::net14::BE) + cv*last_T ;
	double m_in = eigen::dot(last_Y, nnet::net14::constants::A);

	double t = 0, dt=1e-2, small_dt=1e-12;
	int n_max = 500;
	const int n_print = 30;


	std::cerr << "\"t\",\"dt\",,\"T\",,";
	for (auto name : nnet::net14::constants::species_names)
		std::cerr << "\"x(" << name << ")\",";
	std::cerr << ",\"Dm/m\"\n";






	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);
	auto const eos = [&](const std::vector<double> &Y_, const double T, const double rho_) {
		const double cv = 3.1e7; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
		struct eos_output {
			double cv, dP_dT;
		} res{cv, 0};
		return res;
	};

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Y, T] = nnet::solve_system_substep(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>,
#ifndef DONT_USE_HELM_EOS
			helm_eos,
#else
			eos,
#endif
			last_Y, last_T,
			rho, 0., dt, small_dt);
		t += dt;

		// debug print
		if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0) {
			// double E_tot = eigen::dot(Y, nnet::net14::BE) + cv*T;
			// double dE_E = (E_tot - E_in)/E_in;

			double m_tot = eigen::dot(Y, nnet::net14::constants::A);
			double dm_m = (m_tot - m_in)/m_in;
		
			for (int i = 0; i < 14; ++i) X[i] = Y[i]*nnet::net14::constants::A[i]/eigen::dot(Y, nnet::net14::constants::A);
			std::cout << "\n(t=" << t << ", dt=" << small_dt << "):\t";
			for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
			std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\t" << T << "\n";
		}

		last_Y = Y;
		last_T = T;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nexec time:" << ((float)duration.count())/1e3 << "s\n";

	return 0;
}