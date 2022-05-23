#include <vector>


#include "utils/arg_parser.hpp"
#include "utils/sphexa_utils.hpp"


// physical parameters
#include "../src/net14/net14.hpp"
#include "../src/eos/helmholtz.hpp"

// base datatype
#include "../src/sphexa/nuclear-data.hpp"

// nuclear reaction wrappers
#include "../src/sphexa/nuclear-net.hpp"
#include "../src/sphexa/initializers.hpp"




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






// mockup of the ParticlesDataType
class ParticlesDataType {
public:
	// communicator
	MPI_Comm comm=MPI_COMM_WORLD;

	// pointers
	std::vector<int> node_id;
	std::vector<std::size_t> particle_id;
	std::vector<double> x, y, z;

	// hydro data
	std::vector<double> rho, T; //...

	void resize(const size_t N) {
		node_id.resize(N);
		particle_id.resize(N);

		x.resize(N);
		y.resize(N);
		z.resize(N);

		rho.resize(N);
		T.resize(N);
	}
};









// mockup of the step function 
template<class func_rate, class func_BE, class func_eos>
void step(ParticlesDataType &d, sphexa::sphnnet::NuclearDataType<14, double>  &n, const double dt,
	const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {

	// domain redecomposition

	sphexa::sphnnet::initializePartition(d, n);

	// do hydro stuff

	sphexa::sphnnet::sendHydroData(d, n);
	sphexa::sphnnet::compute_nuclear_reactions(n, dt,
		reactions, construct_rates, construct_BE, eos);
	sphexa::sphnnet::recvHydroData(d, n);

	// do hydro stuff
}

void printHelp(char* name, int rank) {
	if (rank == 0) {
		std::cout << "\nUsage:\n\n";
		std::cout << name << " [OPTION]\n";

		std::cout << "\nWhere possible options are:\n\n";
	}
}

int main(int argc, char* argv[]) {
	int size, rank;
    MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);






	const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")) {
        printHelp(argv[0], rank);
        MPI_Finalize();
        return 0;
    }

    const double hydro_dt                   = parser.get("--dt", 1e-2);
    const int n_max                         = parser.get("-n", 50);
    const int n_print                       = parser.get("--n-particle-print", 5);
    const size_t total_n_particles          = parser.get("--n-particle", 1000);

    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");

    nnet::net14::skip_coulombian_correction = parser.exists("--skip-coulomb-corr");
    nnet::constants::NR::max_dt             = parser.get("--max_dt",      nnet::constants::NR::max_dt);
    nnet::constants::NR::dT_T_target        = parser.get("--dT_T_target", nnet::constants::NR::dT_T_target);
    nnet::constants::NR::dT_T_tol           = parser.get("--dT_T_tol",    nnet::constants::NR::dT_T_tol);
    nnet::constants::NR::it_tol             = parser.get("--NR_tol",      nnet::constants::NR::it_tol);
    nnet::constants::NR::min_it             = parser.get("--min_NR_it",   nnet::constants::NR::min_it);
    nnet::constants::NR::max_it             = parser.get("--max_NR_it",   nnet::constants::NR::max_it);

	std::array<double, 14> Y0, X;
    for (int i = 0; i < 14; ++i) X[i] = 0;
    if  (      test_case == "C-O-burning") {
    	X[1] = 0.5;
		X[2] = 0.5;
    } else if (test_case == "He-burning") {
    	X[0] = 1;
    } else if (test_case == "Si-burning") {
    	X[5] = 1;
    } else {
    	printHelp(argv[0], rank);
    	throw std::runtime_error("unknown nuclear test case!\n");
    }
    for (int i = 0; i < 14; ++i) Y0[i] = X[i]/nnet::net14::constants::A[i];






    /* initial hydro data */
	double rho_left = 1.2e9, rho_right = 1e9;
	double T_left = 0.8e9, T_right = 1.1e9;

	/* !!!!!!!!!!!!
	initialize the hydro state
	!!!!!!!!!!!! */
	ParticlesDataType particle_data;
	const size_t n_particles = total_n_particles*(rank + 1)/size - total_n_particles*rank/size;
	particle_data.resize(n_particles);
	for (int i = 0; i < n_particles; ++i) {
		particle_data.T[i]   = T_left   + (T_right   - T_left  )*(float)(rank*n_particles + i)/(float)(size*n_particles - 1);
		particle_data.rho[i] = rho_left + (rho_right - rho_left)*(float)(rank*n_particles + i)/(float)(size*n_particles - 1);
	}







	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);
	const auto isotherm_eos = [&](const std::array<double, 14> &Y_, const double T, const double rho_) {
		const double cv = 1e20; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
		struct eos_output {
			double cv, dP_dT;
		} res{cv, 0};
		return res;
	};



	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!!!!!!!!!! */
	sphexa::mpi::initializePointers(particle_data.node_id, particle_data.particle_id, n_particles);
	auto nuclear_data = sphexa::sphnnet::initNuclearDataFromConst<14>(particle_data, Y0);


	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0;
	for (int i = 0; i < n_max; ++i) {
		if (rank == 0)
			std::cout << i << "th iteration...\n";

		if (isotherm) {
			step(particle_data, nuclear_data, hydro_dt,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, isotherm_eos);
		} else
			step(particle_data, nuclear_data, hydro_dt,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, helm_eos);
		t += hydro_dt;

		if (rank == 0)
			std::cout << "\t...Ok\n";
	}

	std::vector<std::string> outFields = {"node_id", "nuclear_particle_id", "T", "rho", "Y(4He)", "Y(12C)", "Y(16O)"};
	nuclear_data.setOutputFields(outFields, nnet::net14::constants::species_names);

	dump(nuclear_data, 0,                     n_print,     "/dev/stdout");
	dump(nuclear_data, n_particles - n_print, n_particles, "/dev/stdout");

	MPI_Finalize();

	return 0;
}