#include "../../src/sphexa/nuclear-data.hpp"

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"

#include "../../src/eos/helmholtz.hpp"

using vector = sphexa::sphnnet::NuclearAbundances<14>;

int main() {
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif
	
	double rho = 1e9; // rho, g/cm^3
	double last_T = 1e9;

	// extension /* TODO */
	const int start_expansion = 600;
	const double rho_half_life = 0.02;
	const double rho_lim = 1e5;

	// initial state
	vector last_Y, X;
	for (int i = 0; i < 14; ++i) X[i] = 0;
	X[1] = 0.5;
	X[2] = 0.5;

	for (int i = 0; i < 14; ++i) last_Y[i] = X[i]/nnet::net14::constants::A[i];

	// double E_in = eigen::dot(last_Y, nnet::net14::BE) + cv*last_T ;
	double m_in = eigen::dot(last_Y, nnet::net14::constants::A);

	double t = 0, dt=1e-12;
	int n_max = 1000;
	const int n_print = 30;


#ifdef DEBUG
		nnet::debug = true;
#endif


	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);


	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Y, T, current_dt] = solve_system_NR(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double, vector>, helm_eos,
			last_Y, last_T, rho, 0., dt);
		t += current_dt;

		nnet::debug = false;


		// double E_tot = eigen::dot(Y, nnet::net14::BE) + cv*T;
		// double dE_E = (E_tot - E_in)/E_in;

		double m_tot = eigen::dot(Y, nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

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