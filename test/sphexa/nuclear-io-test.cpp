#include <iostream>
#include <fstream>

#include "utils/sphexa_utils.hpp"


// physical parameters
#include "../../src/net14/net14-constants.hpp"

// base datatype
#include "../../src/sphexa/nuclear-data.hpp"


/*
function stolen from SPH-EXA and retrofited for testing
*/
template<class Dataset>
void dump(Dataset& d, size_t firstIndex, size_t lastIndex, /*const cstone::Box<typename Dataset::RealType>& box,*/ std::string path) {
    const char separator = ' ';
    // path += std::to_string(d.iteration) + ".txt";

    int rank, numRanks;
    MPI_Comm_rank(d.comm, &rank);
    MPI_Comm_size(d.comm, &numRanks);

    for (int turn = 0; turn < numRanks; turn++)
    {
        if (turn == rank)
        {
            try
            {
                auto fieldPointers = sphexa::getOutputArrays(d);

                bool append = rank != 0;
                sphexa::fileutils::writeAscii(firstIndex, lastIndex, path, append, fieldPointers, separator);
            }
            catch (std::runtime_error& ex)
            {
                if (rank == 0) fprintf(stderr, "ERROR: %s Terminating\n", ex.what());
                MPI_Abort(d.comm, 1);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}











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

	std::vector<std::string> outFields = {"node_id", "nuclear_particle_id", "T", "rho", "Y(4He)", "Y(12C)", "Y(16O)"};
	nuclear_data.setOutputFields(outFields, nnet::net14::constants::species_names);

	dump(nuclear_data, 0, n_particles, "/dev/stdout");

	MPI_Finalize();

	return 0;
}