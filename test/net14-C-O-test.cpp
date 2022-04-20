#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net14/net14.hpp"

int main() {
	const double value_1 = 0; // typical v1 from net14 fortran
	const double cv = 2e7; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
	const double rho = 1e9; // rho, g/cm^3
	double T = 1e9;

	// initial state
	Eigen::VectorXd Y(14), X = Eigen::VectorXd::Zero(14);
	X(1) = 0.5;
	X(2) = 0.5;

	for (int i = 0; i < 14; ++i) Y(i) = X(i) / nnet::net14::constants::A(i);

	
	auto last_Y = Y;
	double last_T = T;
	double m_in = Y.dot(nnet::net14::constants::A);

	const double max_dt=5e-2, min_dt=1e-17;
	double t = 0, dt=1e-12;
	int n_max = 100000;
	const int n_print = 30, n_save=4000;

	nnet::constants::theta = 0.5;





	/* ---------------------
	begin test
	--------------------- */ /*
	{
		net14_debug = true;
		auto rates = nnet::net14::compute_reaction_rates(T);
		auto M = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, rates, rho, Y);
		net14_debug = false;

		// include temperature
		Eigen::VectorXd BE = nnet::net14::BE + nnet::net14::ideal_gaz_correction(T);
		auto Mp = nnet::include_temp(M, cv, value_1, BE, Y);

		// construct vector
		Eigen::VectorXd Y_T(14 + 1);
		Y_T << T, Y;

		Eigen::VectorXd RHS = Mp*Y_T*dt;
		Eigen::MatrixXd Mpp = Eigen::MatrixXd::Identity(14 + 1, 14 + 1) - nnet::constants::theta*dt*Mp;


		std::cout << "\nBE(T=" << T <<") =\t" << BE.transpose() << "\n\n";
		std::cout << "phi =\n" << Mpp << "\n\n";
		std::cout << "RHS =\t\t" << RHS.transpose() << "\n\n";
		std::cout << "Mp*{T,Y} =\t" << (Mp*Y_T).transpose() << "\n\n\n"; 
	}
	/* ---------------------
	end test
	--------------------- */






	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

	for (int i = 0; i < 14; ++i) std::cout << X(i) << ", ";
	std::cout << "\t" << T << std::endl;

	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		std::tie(Y, T, dt) = nnet::solve_system_var_timestep(nnet::net14::constants::A, nnet::net14::construct_system(rho, cv, value_1), Y, T, dt);
		t += dt;

		double m_tot = Y.dot(nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

		// formated print (stderr)
		if (n_save >= n_max || (n_max - i) % (int)((float)n_max/(float)n_save) == 0) {
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