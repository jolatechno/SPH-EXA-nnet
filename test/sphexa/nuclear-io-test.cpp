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
	auto data = dataset.data();

	std::vector<std::string> outFields;
	dataset.setOutputFields(outFields, nnet::net14::constants::species_names);

	for (int i = 0; i < dataset.outputFieldNames.size(); ++i) {
		std::cout << dataset.outputFieldNames[i] << ":\t";

		for (int j = 0; j < n_particles; ++j)
			std::visit([j](auto& arg) {
				std::cout << (*arg)[j] << ", ";
			}, data[i]);

		std::cout << "\n";
	}

	MPI_Finalize();

	return 0;
}