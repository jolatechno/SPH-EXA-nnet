#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net14/net14.hpp"

int main() {
	const double value_1 = 0; // typical v1 from net14 fortran
	const double rho = 1e9; // rho, g/cm^3
	const double cv = 2e7; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
	double last_T = 1e9;

	// initial state
	Eigen::VectorXd last_Y(14), X = Eigen::VectorXd::Zero(14);
	X(1) = 0.5;
	X(2) = 0.5;

	for (int i = 0; i < 14; ++i) last_Y(i) = X(i) / nnet::net14::constants::A(i);

	double m_in = last_Y.dot(nnet::net14::constants::A);

	double t = 0, dt=1e-12;
	int n_max = 100000;
	const int n_print = 30, n_save=4000;





	/* ---------------------
	begin test
	--------------------- */
#ifdef DEBUG
	{		
		net14_debug = true;
		auto [rates, drates] = nnet::net14::compute_reaction_rates(last_T);
		Eigen::MatrixXd M =     nnet::first_order_from_reactions<double>(nnet::net14::reaction_list,  rates, last_Y, nnet::net14::constants::A, rho);
		net14_debug = false;
		Eigen::MatrixXd dM_dT = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, drates, last_Y, nnet::net14::constants::A, rho);

		// include temperature
		Eigen::VectorXd BE = nnet::net14::BE + nnet::net14::ideal_gaz_correction(last_T);
		auto Mp =  nnet::include_temp(M, last_Y, BE, cv, value_1);
		auto MpT = nnet::include_rate_derivative(Mp, dM_dT, last_Y);

		// construct vector
		Eigen::VectorXd Y_T(14 + 1);
		Y_T << last_T, last_Y;

		Eigen::VectorXd RHS = Mp*Y_T*0.01;

		std::cout << "\nBE(T=" << last_T <<") =\t" << BE.transpose() << "\n\n";
		std::cout << "M.T*A =" << (M.transpose()*nnet::net14::constants::A).transpose() << "\t-> sum=\t" << (M.transpose()*nnet::net14::constants::A).sum() << "\n\n";
		std::cout << "phi =\n" << MpT << "\n\n";
		std::cout << "RHS =\t\t" << RHS.transpose() << "\n\n";
		std::cout << "Mp*{T,Y} =\t" << (Mp*Y_T).transpose() << "\n\n\n"; 
	}
#endif
	/* ---------------------
	end test
	--------------------- */






	std::cerr << "\"t\",\"dt\",,\"T\",,\"x(He)\",\"x(C)\",\"x(O)\",\"x(Ne)\",\"x(Mg)\",\"x(Si)\",\"x(S)\",\"x(Ar)\",\"x(Ca)\",\"x(Ti)\",\"x(Cr)\",\"x(Fe)\",\"x(Ni)\",\"x(Zn)\",,\"Dm/m\"\n";

	for (int i = 1; i <= n_max; ++i) {
		// solve the system
		auto [Mp, dM_dT] = nnet::net14::construct_system(last_Y, last_T, rho, cv, value_1);
		auto [Y, T, actual_dt] = nnet::solve_system_var_timestep(Mp, dM_dT, last_Y, last_T, nnet::net14::constants::A, dt);
		t += actual_dt;

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

			auto [rates, drates] = nnet::net14::compute_reaction_rates(T);
			Eigen::MatrixXd M =     nnet::first_order_from_reactions<double>(nnet::net14::reaction_list,  rates, Y, nnet::net14::constants::A, rho);
			Eigen::MatrixXd dM_dT = nnet::first_order_from_reactions<double>(nnet::net14::reaction_list, drates, Y, nnet::net14::constants::A, rho);

			std::cout << (M*Y).dot(nnet::net14::constants::A) << "=(m*Y).A, " << (dM_dT*Y).dot(nnet::net14::constants::A) << "=(dM_dT*Y).A\n";
		}

		last_Y = Y;
		last_T = T;
	}


	return 0;
}