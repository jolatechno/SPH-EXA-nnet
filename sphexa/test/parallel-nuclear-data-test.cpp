#include "../src/nuclear-data.hpp"
#include "../src/nuclear-net.hpp"

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"

#include "../../src/eos/helmholtz.hpp"

using vector = sphexa::sphnnet::NuclearAbundances<14>;

int main() {
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif
	
	double rho_left = 1e9, rho_right = 7e8; // rho, g/cm^3
	double T_left = 1e9, T_right = 2e9; // rho, g/cm^3

	// extension /* TODO */
	const int start_expansion = 600;
	const double rho_half_life = 0.02;
	const double rho_lim = 1e5;

	// initial state
	const int n_particles = 1000;
	sphexa::sphnnet::NuclearDataType<14, double>  nuclear_data;
	nuclear_data.resize(n_particles);
	for (int i = 0; i < n_particles; ++i) {
		nuclear_data.Y[i][1] = 0.5/nnet::net14::constants::A[1];
		nuclear_data.Y[i][2] = 0.5/nnet::net14::constants::A[2];

		nuclear_data.T[i]   = T_left   + (T_right   - T_left  )*(float)i/(float)(n_particles - 1);
		nuclear_data.rho[i] = rho_left + (rho_right - rho_left)*(float)i/(float)(n_particles - 1);

		//nuclear_data.drho_dt[i] = 0.;
		nuclear_data.previous_rho[i] = nuclear_data.rho[i];

		nuclear_data.dt[i] = 1e-12;
	}

	// double E_in = eigen::dot(nuclear_data.Y[0], nnet::net14::BE) + cv*last_T ;
	double m_in = eigen::dot(nuclear_data.Y[0], nnet::net14::constants::A);

	double t = 0, dt=2e-3;
	int n_max = 1000;
	const int n_print = 30;


#ifdef DEBUG
		nnet::debug = true;
#endif


	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);


	// debug print
	std::vector<double> X(14);
	for (int i = 0; i < 14; ++i) X[i] = nuclear_data.Y[0][i]*nnet::net14::constants::A[i]/eigen::dot(nuclear_data.Y[0], nnet::net14::constants::A);
	std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
	for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
	std::cout << "\tT_left=" << nuclear_data.T[0] << "\tT_right=" << nuclear_data.T[n_particles - 1] << "\n";


	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		sphexa::sphnnet::compute_nuclear_reactions(nuclear_data, dt,
			nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double, vector>, helm_eos);
		t += dt;

		nnet::debug = false;


		// double E_tot = eigen::dot(Y, nnet::net14::BE) + cv*T;
		// double dE_E = (E_tot - E_in)/E_in;

		double m_tot = eigen::dot(nuclear_data.Y[0], nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

		// debug print
		if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0) {
			std::vector<double> X(14);
			for (int i = 0; i < 14; ++i) X[i] = nuclear_data.Y[0][i]*nnet::net14::constants::A[i]/eigen::dot(nuclear_data.Y[0], nnet::net14::constants::A);
			std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
			for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
			std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tT_left=" << nuclear_data.T[0] << "\tT_right=" << nuclear_data.T[n_particles - 1] << "\n";
		}
	}

	return 0;
}