#include <iostream>
#include <fstream>

#include "../../src/sphexa/nuclear-data.hpp"
#include "../../src/sphexa/nuclear-net.hpp"

#include "../../src/net14/net14-constants.hpp"

namespace sphexa {
	/*! @brief look up indices of field names
	 *
	 * @tparam     Array
	 * @param[in]  allNames     array of strings with names of all fields
	 * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
	 * @return                  the indices of @p subsetNames in @p allNames
	 */
	template<class Array>
	std::vector<int> fieldStringsToInt(const Array& allNames, const std::vector<std::string>& subsetNames)
	{
	    std::vector<int> subsetIndices;
	    subsetIndices.reserve(subsetNames.size());
	    for (const auto& field : subsetNames)
	    {
	        auto it = std::find(allNames.begin(), allNames.end(), field);
	        if (it == allNames.end()) { throw std::runtime_error("Field " + field + " does not exist\n"); }

	        size_t fieldIndex = it - allNames.begin();
	        subsetIndices.push_back(fieldIndex);
	    }
	    return subsetIndices;
	}
}

#include "../../src/sphexa/nuclear-io.hpp"


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

	sphexa::sphnnet::NuclearIoDataSet dataset(nuclear_data);
	auto data = dataset.data();

	std::vector<std::string> outFields = {"node_id", "nuclear_particle_id", "T", "rho", "Y(4He)", "Y(12C)", "Y(16O)"};
	dataset.setOutputFields(outFields, nnet::net14::constants::species_names);

	dump(dataset, 0, n_particles, "/dev/stdout");

	/*for (int i : dataset.outputFieldIndices) {
		std::cout << dataset.outputFieldNames[i] << ":\t";

		for (int j = 0; j < n_particles; ++j)
			std::visit([j](auto& arg) {
				std::cout << (*arg)[j] << ", ";
			}, data[i]);

		std::cout << "\n";
	}*/

	MPI_Finalize();

	return 0;
}