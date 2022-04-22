#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net14/net14.hpp"

int main() {
	const double value_1 = 0; // typical v1 from net14 fortran
	const double cv = 1e200; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
	const double rho0 = 1e7; // rho, g/cm^3
	double last_T = 6e9;

	// initial state
	Eigen::VectorXd last_Y(14), X = Eigen::VectorXd::Zero(14);
	X(5) = 1;


	for (int i = 0; i < 14; ++i) last_Y(i) = X(i) / nnet::net14::constants::A(i);

	double m_in = last_Y.dot(nnet::net14::constants::A);

	double t = 0, dt=1e-12;
	int n_max = 400;
	const int n_print = 30, n_save=4000;

	nnet::constants::theta = 0.55;

	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

	double dm_tot = 0, last_m_tot = 0;
	for (int i = 1; i <= n_max; ++i) {
		// normalize rho
		double rho = rho0/last_Y.dot(nnet::net14::constants::A);

#ifdef DEBUG
		net14_debug = i == 1;
#endif

		// construct system
		auto BE = nnet::net14::get_corrected_BE(last_T);
		auto [rate, drates_dT] = nnet::net14::compute_reaction_rates(last_T);

		// solve the system
		auto [Y, T, actual_dt, dm] = nnet::solve_system_var_timestep(nnet::net14::reaction_list, rate, drates_dT,
			BE, nnet::net14::constants::A, last_Y, 
			last_T, cv, rho, value_1, dt);
		t += actual_dt;
		dm_tot += dm;

		net14_debug = false;

		double m_tot = Y.dot(nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;
		double dm_m_dt = std::abs(1 - m_tot/last_m_tot)/actual_dt;

		// formated print (stderr)
		if (n_save >= n_max || (n_max - i) % (int)((float)n_max/(float)n_save) == 0) {
			for (int i = 0; i < 14; ++i) X(i) = Y(i)*nnet::net14::constants::A(i)/X.sum();
			std::cerr << t << "," << dt << ",," << T << ",,";
			for (int i = 0; i < 14; ++i) std::cerr << X(i) << ",";
			std::cerr << "," << dm_m << "\n";
		}

		// debug print
		if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0) {
			for (int i = 0; i < 14; ++i) X(i) = Y(i)*nnet::net14::constants::A(i)/X.sum();
			std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
			for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
			std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << ", dmth_m0=" << dm_tot/m_in << ", dm_m/dt=" << dm_m_dt << "),\t" << T << "\n";
		}

		last_Y = Y;
		last_T = T;
		last_m_tot = m_tot;
	}


	return 0;
}