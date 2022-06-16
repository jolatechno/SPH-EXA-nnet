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




#ifdef USE_CUDA
	using AccType = cstone::GpuTag;
#else
	using AccType = cstone::CpuTag;
#endif




/************************************************************************/
/* non-MPI test to test GPU implementation, mostly using OMP offloading */
/* compile:  clang++ -std=c++17 -fopenmp -DNOT_FROM_SPHEXA -DCUDA_DEBUG_ -DOMP_TARGET_SOLVER -DCPU_BATCH_SOLVER_ -DUSE_CUDA_ -omptargets=nvptx-none parallel-perftest.cpp -o parallel-perftest.out
/* launch:   ./parallel-perftest.out --test-case C-O-burning
/************************************************************************/




/*
function stolen from SPH-EXA and retrofited for testing
*/
template<class Dataset>
void dump(Dataset& d, size_t firstIndex, size_t lastIndex, /*const cstone::Box<typename Dataset::RealType>& box,*/ std::string path) {
    const char separator = ' ';
    // path += std::to_string(d.iteration) + ".txt";

    /*int rank, numRanks;
    MPI_Comm_rank(d.comm, &rank);
    MPI_Comm_size(d.comm, &numRanks);*/

    /*for (int turn = 0; turn < numRanks; turn++)
    {
        if (turn == rank)
        {*/
            try
            {
                auto fieldPointers = sphexa::getOutputArrays(d);

                bool append = true; // rank != 0;
                sphexa::fileutils::writeAscii(firstIndex, lastIndex, path, append, fieldPointers, separator);
            }
            catch (std::runtime_error& ex)
            {
                /* if (rank == 0) */ fprintf(stderr, "ERROR: %s Terminating\n", ex.what());
                /* MPI_Abort(d.comm, 1); */
            }
        /*}

        MPI_Barrier(MPI_COMM_WORLD);
    }*/
}




void printHelp(char* name);

// mockup of the step function 
template<class func_type, class func_eos, size_t n_species, class AccType>
void step(
	sphexa::sphnnet::NuclearDataType<n_species, double, AccType>  &n, const double dt,
	const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos)
{
	sphexa::sphnnet::transferToDevice(n, {"previous_rho", "rho", "temp"});
	sphexa::sphnnet::computeNuclearReactions(n, dt, dt,
		reactions, construct_rates_BE, eos);
	sphexa::sphnnet::transferToHost(n, {"temp"});
}


inline static constexpr struct eos_output {
	double cv, dP_dT, dU_dYe;
} isotherm_res{1e20, 0, 0};
CUDA_FUNCTION_DECORATOR eos_output inline isotherm_eos(const double *Y_, const double T, const double rho_) {
	return isotherm_res;
};


int main(int argc, char* argv[]) {
	/* initial hydro data */
	const double rho_left = 1.1e9, rho_right = 0.8e9;
	const double T_left = 0.5e9, T_right = 1.5e9;





	const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")) {
        printHelp(argv[0]);
        return 0;
    }

    const bool use_net86                    = parser.exists("--use-net86");

    const double hydro_dt                   = parser.get("--dt", 1e-1);
    const int n_max                         = parser.get("-n", 10);
    const int n_print                       = parser.get("--n-particle-print", 5);
    const size_t n_particles                = parser.get("--n-particle", 1000);

    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");

    nnet::net14::skip_coulombian_correction = parser.exists("--skip-coulomb-corr");

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
	    	printHelp(argv[0]);
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
	    	printHelp(argv[0]);
	    	throw std::runtime_error("unknown nuclear test case!\n");
	    }

	    for (int i = 0; i < 14; ++i) Y0_14[i] = X_14[i]/nnet::net14::constants::A[i];
    }
    




	sphexa::sphnnet::NuclearDataType<86, double, AccType> nuclear_data_86;
	sphexa::sphnnet::NuclearDataType<14, double, AccType> nuclear_data_14;

	if (use_net86) {
		nuclear_data_86.setConserved(/*"nid", "pid",*/ "dt", "c", "p", "cv", "temp", "rho", "previous_rho", "Y");
		nuclear_data_86.devData.setConserved("temp", "rho", "previous_rho", "Y", "dt");
		nuclear_data_86.resize(n_particles);
	} else {
		nuclear_data_14.setConserved(/*"nid", "pid",*/ "dt", "c", "p", "cv", "temp", "rho", "previous_rho", "Y");
		nuclear_data_14.devData.setConserved("temp", "rho", "previous_rho", "Y", "dt");
		nuclear_data_14.resize(n_particles);
	}





	/* !!!!!!!!!!!!
	initialize the state
	!!!!!!!!!!!! */
	if (use_net86) {
		for (size_t i = 0; i < n_particles; ++i) {
			nuclear_data_86.Y[i]    = Y0_86;
			nuclear_data_86.temp[i] = T_left   + (T_right   - T_left  )*((float)i)/((float)(n_particles - 1));
			nuclear_data_86.rho[i]  = rho_left + (rho_right - rho_left)*((float)i)/((float)(n_particles - 1));
		}
		sphexa::sphnnet::transferToDevice(nuclear_data_86, {"Y", "dt"});
	} else {
		for (size_t i = 0; i < n_particles; ++i) {
			nuclear_data_14.Y[i]    = Y0_14;
			nuclear_data_14.temp[i] = T_left   + (T_right   - T_left  )*((float)i)/((float)(n_particles - 1));
			nuclear_data_14.rho[i]  = rho_left + (rho_right - rho_left)*((float)i)/((float)(n_particles - 1));
		}
		sphexa::sphnnet::transferToDevice(nuclear_data_14, {"Y", "dt"});
	}
	




	const nnet::eos::helmholtz_functor helm_eos_86 = nnet::eos::helmholtz_functor(nnet::net86::constants::Z, 86);
	const nnet::eos::helmholtz_functor helm_eos_14 = nnet::eos::helmholtz_functor(nnet::net14::constants::Z);




	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!!!!!!!!!! */



	std::vector<std::string> nuclearOutFields = {"temp", "rho", "Y(4He)", "Y(12C)", "Y(16O)"};
	if (use_net86) {
		nuclear_data_86.setOutputFields(nuclearOutFields, nnet::net86::constants::species_names);
	} else
		nuclear_data_14.setOutputFields(nuclearOutFields, nnet::net14::constants::species_names);



	// "warm-up" (first allocation etc...)
	if (use_net86) {
			if (isotherm) {
				step(nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates<double, eos_output>, isotherm_eos);
			} else
				step(nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates<double, nnet::eos::helm_eos_output<double>>, helm_eos_86);
		} else
			if (isotherm) {
				step(nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double, eos_output>, isotherm_eos);
			} else
				step(nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double, nnet::eos::helm_eos_output<double>>, helm_eos_14);





	auto start = std::chrono::high_resolution_clock::now();
	double min_time = 3600, max_time = 0;

	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0;
	for (int i = 0; i < n_max; ++i) {
		std::cout << i << "th iteration...\n";

		auto start_it = std::chrono::high_resolution_clock::now();

		if (use_net86) {
			if (isotherm) {
				step(nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates<double, eos_output>, isotherm_eos);
			} else
				step(nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates<double, nnet::eos::helm_eos_output<double>>, helm_eos_86);
		} else
			if (isotherm) {
				step(nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double, eos_output>, isotherm_eos);
			} else
				step(nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double, nnet::eos::helm_eos_output<double>>, helm_eos_14);
		
		t += hydro_dt;

		auto end_it = std::chrono::high_resolution_clock::now();
		auto duration_it = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(end_it - start_it).count())/1e3;
		min_time = std::min(min_time, duration_it);
		max_time = std::max(max_time, duration_it);

		std::cout << "\t...Ok\n";
	}


	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count())/1e3;
	auto avg_duration = duration/n_max;
	std::cout << "\nexec time: " << duration << "s (avg=" << avg_duration << "s/it, max=" << max_time << "s/it, min=" << min_time  << "s/it)\n\n";




	std::cout << "\n";
	for (auto name : nuclearOutFields)
		std::cout << name << " ";
	std::cout << "\n";
	if (use_net86) {
		sphexa::sphnnet::transferToHost(nuclear_data_86, {"Y"});

		dump(nuclear_data_86, 0,                     n_print,     "/dev/stdout");
		dump(nuclear_data_86, n_particles - n_print, n_particles, "/dev/stdout");
	} else {
		sphexa::sphnnet::transferToHost(nuclear_data_14, {"Y"});

		dump(nuclear_data_14, 0,                     n_print,     "/dev/stdout");
		dump(nuclear_data_14, n_particles - n_print, n_particles, "/dev/stdout");
	}

	return 0;
}

void printHelp(char* name) {
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