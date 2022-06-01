#include <vector>
#include <chrono>


#include "util/arg_parser.hpp"
#include "util/sphexa_utils.hpp"


// physical parameters
#include "../src/net86/net86.hpp"
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
	std::vector<double> rho, temp; //...

	void resize(const size_t N) {
		node_id.resize(N);
		particle_id.resize(N);

		x.resize(N);
		y.resize(N);
		z.resize(N);

		rho.resize(N);
		temp.resize(N);
	}

	const std::vector<std::string> fieldNames = {
		"nid", "pid", "rho", "temp", "x", "y", "z"
	};

	auto data() {
	    using FieldType = std::variant<
	    	std::vector<size_t>*,
	    	std::vector<int>*,
	    	std::vector<uint8_t/*bool*/>*,
	    	std::vector<double>*>;

	    std::array<FieldType, 7> ret = {
	    	&node_id, &particle_id, &rho, &temp, &x, &y, &z};

	    return ret;
	}

	std::vector<int>         outputFieldIndices;
	std::vector<std::string> outputFieldNames;

	void setOutputFields(const std::vector<std::string>& outFields) {
	    outputFieldNames = fieldNames;
		outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
    }

    bool isAllocated(int i) const {
    	/* TODO */
    	return true;
    }
};







void printHelp(char* name, int rank);

// mockup of the step function 
template<class func_rate, class func_BE, class func_eos, int n_species>
void step(size_t firstIndex, size_t lastIndex,
	ParticlesDataType &d, sphexa::sphnnet::NuclearDataType<n_species, double>  &n, const double dt,
	const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {

	// domain redecomposition

	sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n);
	sphexa::sphnnet::hydroToNuclearUpdate(d, n, {"previous_rho"});

	// do hydro stuff

	std::swap(n.rho, n.previous_rho);
	sphexa::sphnnet::hydroToNuclearUpdate(d, n, {"rho", "temp"});
	sphexa::sphnnet::computeHelmEOS(n, nnet::net14::constants::Z);

	sphexa::sphnnet::computeNuclearReactions(n, dt, dt,
		reactions, construct_rates, construct_BE, eos);
	
	sphexa::sphnnet::nuclearToHydroUpdate(d, n, {"temp"});
	
	// do hydro stuff
}

int main(int argc, char* argv[]) {
	/* initial hydro data */
	double rho_left = 1e9, rho_right = 0.8e9;
	double T_left = 0.95e9, T_right = 1.1e9;




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

    const bool use_net86                    = parser.exists("--use-net86");

    const double hydro_dt                   = parser.get("--dt", 1e-1);
    const int n_max                         = parser.get("-n", 100);
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

    nnet::constants::min_temp               = parser.get("--min-temp",    std::min(T_left, T_right)*0.9 + 0.1*std::max(T_left, T_right));

	util::array<double, 86> Y0_86, X_86;
	util::array<double, 14> Y0_14, X_14;
    if (use_net86) {
   		for (int i = 0; i < 86; ++i) X_86[i] = 0;

	    if  (      test_case == "C-O-burning") {
	    	X_86[nnet::net86::constants::net14_species_order[1]] = 0.5;
			X_86[nnet::net86::constants::net14_species_order[2]] = 0.5;
	    } else if (test_case == "He-burning") {
	    	X_86[nnet::net86::constants::net14_species_order[0]] = 1;
	    } else if (test_case == "Si-burning") {
	    	X_86[nnet::net86::constants::net14_species_order[5]] = 1;
	    } else {
	    	printHelp(argv[0], rank);
	    	throw std::runtime_error("unknown nuclear test case!\n");
	    }

	    for (int i = 0; i < 86; ++i) Y0_86[i] = X_86[i]/nnet::net86::constants::A[i];
    } else {
   		for (int i = 0; i < 14; ++i) X_14[i] = 0;

    	if  (      test_case == "C-O-burning") {
	    	X_14[1] = 0.5;
			X_14[2] = 0.5;
	    } else if (test_case == "He-burning") {
	    	X_14[0] = 1;
	    } else if (test_case == "Si-burning") {
	    	X_14[5] = 1;
	    } else {
	    	printHelp(argv[0], rank);
	    	throw std::runtime_error("unknown nuclear test case!\n");
	    }

	    for (int i = 0; i < 14; ++i) Y0_14[i] = X_14[i]/nnet::net14::constants::A[i];
    }
    






	/* !!!!!!!!!!!!
	initialize the hydro state
	!!!!!!!!!!!! */
	ParticlesDataType particle_data;
	const size_t n_particles = total_n_particles*(rank + 1)/size - total_n_particles*rank/size;
	const size_t offset = 10*rank;
	const size_t first = offset, last = n_particles + offset;

	particle_data.resize(last);
	for (int i = first; i < last; ++i) {
		particle_data.temp[i] = T_left   + (T_right   - T_left  )*(float)(rank*n_particles + i)/(float)(size*n_particles - 1);
		particle_data.rho[i]  = rho_left + (rho_right - rho_left)*(float)(rank*n_particles + i)/(float)(size*n_particles - 1);
	}




	std::vector<std::string> outFields = {/*"nid", "pid",*/ "temp", "rho"};
	particle_data.setOutputFields(outFields);




	const nnet::eos::helmholtz helm_eos_86 = nnet::eos::helmholtz(nnet::net86::constants::Z);
	const nnet::eos::helmholtz helm_eos_14 = nnet::eos::helmholtz(nnet::net14::constants::Z);
	const struct eos_output {
		double cv, dP_dT;
	} isotherm_res{1e20, 0};
	const auto isotherm_eos = [&](const eigen::Vector<double> &Y_, const double T, const double rho_) {
		return isotherm_res;
	};



	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!!!!!!!!!! */
	sphexa::mpi::initializePointers(first, last, particle_data.node_id, particle_data.particle_id, particle_data.comm);
	auto nuclear_data_86 = sphexa::sphnnet::initNuclearDataFromConst<86>(first, last, particle_data, Y0_86);
	auto nuclear_data_14 = sphexa::sphnnet::initNuclearDataFromConst<14>(first, last, particle_data, Y0_14);



	auto start = std::chrono::high_resolution_clock::now();
	double min_time = 3600, max_time = 0;

	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0;
	for (int i = 0; i < n_max; ++i) {
		if (rank == 0)
			std::cout << i << "th iteration...\n";

		MPI_Barrier(MPI_COMM_WORLD);
		auto start_it = std::chrono::high_resolution_clock::now();

		if (use_net86) {
			if (isotherm) {
				step(first, last,
					particle_data, nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates<double>, nnet::net86::compute_BE<double>, isotherm_eos);
			} else
				step(first, last,
					particle_data, nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates<double>, nnet::net86::compute_BE<double>, helm_eos_86);
		} else
			if (isotherm) {
				step(first, last,
					particle_data, nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, isotherm_eos);
			} else
				step(first, last,
					particle_data, nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, helm_eos_14);
		
		t += hydro_dt;

		MPI_Barrier(MPI_COMM_WORLD);
		auto end_it = std::chrono::high_resolution_clock::now();
		auto duration_it = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(end_it - start_it).count())/1e3;
		min_time = std::min(min_time, duration_it);
		max_time = std::max(max_time, duration_it);

		if (rank == 0)
			std::cout << "\t...Ok\n";
	}


	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count())/1e3;
		auto avg_duration = duration/n_max;
		std::cout << "\nexec time: " << duration << "s (avg=" << avg_duration << "s/it, max=" << max_time << "s/it, min=" << min_time  << "s/it)\n\n";
	}
	

	if (rank == 0) {
		for (auto name : outFields)
			std::cout << name << " ";
		std::cout << "\n";
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	dump(particle_data, first, first + n_print, "/dev/stdout");
	dump(particle_data, last - n_print,   last, "/dev/stdout");

	MPI_Finalize();

	return 0;
}

void printHelp(char* name, int rank) {
	if (rank == 0) {
		std::cout << "\nUsage:\n\n";
		std::cout << name << " [OPTION]\n";

		std::cout << "\nWhere possible options are:\n\n";

		std::cout << "\t'-n': number of iterations (default = 50)\n\n";
		std::cout << "\t'--dt': timestep (default = 1e-2s)\n\n";

		std::cout << "\t'--n-particle': total number of particles shared across nodes (default = 1000)\n\n";
		std::cout << "\t'--n-particle-print': number of particle to serialize at the end and begining of each node (default = 5)\n\n";

		std::cout << "\t'--test-case': represent nuclear initial state, can be:\n\n";
		std::cout << "\t\t'C-O-burning: x(12C) = x(16O) = 0.5\n\n";
		std::cout << "\t\t'He-burning: x(4He) = 1\n\n";
		std::cout << "\t\t'Si-burning: x(28Si) = 1\n\n";

		std::cout << "\t'--isotherm': if exists cv=1e20, else use Helmholtz EOS\n\n";
		std::cout << "\t'--skip-coulomb-corr': if exists skip coulombian corrections\n\n";

		std::cout << "\t'--output-net14': if exists output results only for net14 species\n\n";
		std::cout << "\t'--debug-net86': if exists output debuging prints for net86 species\n\n";
	}
}