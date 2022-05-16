#include <iostream>
#include <chrono>

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif

	double rho = 1e7; // rho, g/cm^3
	double last_T = 6e9;

	// initial state
	std::vector<double> last_Y(14, 0), X(14, 0);
	X[5] = 1;


	for (int i = 0; i < 14; ++i) last_Y[i] = X[i]/nnet::net14::constants::A[i];

	// double E_in = eigen::dot(last_Y, nnet::net14::BE) + cv*last_T;
	double m_in = eigen::dot(last_Y, nnet::net14::constants::A);

	double t = 0, dt=1e-12;
	int n_max = 100;
	const int n_print = 30, n_save=100;


	std::cerr << "\"t\",\"dt\",,\"T\",,";
	for (auto name : nnet::net14::constants::species_names)
		std::cerr << "\"x(" << name << ")\",";
	std::cerr << ",\"Dm/m\"\n";


#ifdef DEBUG
		nnet::debug = true;
#endif



	auto const eos = [&](const std::vector<double> &Y_, const double T, const double rho_) {
		const double cv = 1e20; // isothermal
		struct eos_output {
			double cv, dP_dT;
		} res{cv, 0};
		return res;
	};

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Y, T, current_dt] = solve_system_NR(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, eos,
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