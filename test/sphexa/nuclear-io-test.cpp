#include <iostream>

#include "../../src/sphexa/nuclear-data.hpp"
#include "../../src/sphexa/nuclear-net.hpp"
#include "../../src/sphexa/nuclear-io.hpp"

#include "../../src/net14/net14-constants.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif
	
	double rho_left = 1e9, rho_right = 7e8;
	double T_left = 1e9, T_right = 2e9;
	double C_left = 0.4, C_right = 0.7;

	// extension /* TODO */
	const int start_expansion = 600;
	const double rho_half_life = 0.02;
	const double rho_lim = 1e5;

	// initial state
	const int n_particles = 20;
	sphexa::sphnnet::NuclearDataType<14, double>  nuclear_data;
	nuclear_data.resize(n_particles);
	for (int i = 0; i < n_particles; ++i) {
		double X_C = C_left + (C_right - C_left)*(float)i/(float)(n_particles - 1);
		double X_O = 1 - X_C;

		nuclear_data.Y[i][1] = X_C/nnet::net14::constants::A[1];
		nuclear_data.Y[i][2] = X_O/nnet::net14::constants::A[2];

		nuclear_data.T[i]   = T_left   + (T_right   - T_left  )*(float)i/(float)(n_particles - 1);
		nuclear_data.rho[i] = rho_left + (rho_right - rho_left)*(float)i/(float)(n_particles - 1);

		//nuclear_data.drho_dt[i] = 0.;
		nuclear_data.previous_rho[i] = nuclear_data.rho[i];

		nuclear_data.dt[i] = 1e-12;
	}

	sphexa::sphnnet::NuclearIoDataSet dataset(nuclear_data);

	std::cout << "T: ";
	for (int i = 0; i < n_particles; ++i)
		std::cout << dataset.T[i] << ", ";
	std::cout << "\nrho: ";
	for (int i = 0; i < n_particles; ++i)
		std::cout << dataset.rho[i] << ", ";

	std::cout << "\n\nx(4He): ";
	for (int i = 0; i < n_particles; ++i)
		std::cout << dataset.Y[0][i]*nnet::net14::constants::A[0] << ", ";
	std::cout << "\nx(C): ";
	for (int i = 0; i < n_particles; ++i)
		std::cout << dataset.Y[1][i]*nnet::net14::constants::A[1] << ", ";
	std::cout << "\nx(O): ";
	for (int i = 0; i < n_particles; ++i)
		std::cout << dataset.Y[2][i]*nnet::net14::constants::A[2] << ", ";
	std::cout << "\n";


	MPI_Finalize();

	return 0;
}