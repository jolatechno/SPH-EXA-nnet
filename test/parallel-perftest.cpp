#include <vector>
#include <chrono>


#include "util/arg_parser.hpp"
#include "util/sphexa_utils.hpp"


// physical parameters
#include "../src/net87/net87.hpp"
#include "../src/net14/net14.hpp"
#include "../src/eos/helmholtz.hpp"

// base datatype
#include "../src/sphexa/nuclear-data.hpp"

// nuclear reaction wrappers
#include "../src/sphexa/nuclear-net.hpp"
#include "../src/sphexa/observables.hpp"




#if defined(USE_CUDA) && !defined(CPU_CUDA_TEST)
	using AccType = cstone::GpuTag;
#else
	using AccType = cstone::CpuTag;
#endif




/************************************************************************/
/*              non-MPI test to test GPU implementation                 */
// compile:  nvcc -x cu -Xcompiler="-DNOT_FROM_SPHEXA -fopenmp -DUSE_CUDA -W -std=c++17 -DCPU_CUDA_TEST_" parallel-perftest.cpp -o parallel-perftest.out -std=c++17 --expt-relaxed-constexpr
// launch:   ./parallel-perftest.out --test-case C-O-burning --n-particle 1000000 --dt 1e-4 -n 10 &> res.out &
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
template<class func_type, class func_eos, size_t n_species, typename Float, class AccType>
void step(
	sphexa::sphnnet::NuclearDataType<n_species, double, AccType>  &n, const double dt,
	const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
	const Float* BE)
{
	sphexa::sphnnet::transferToDevice(n, {"previous_rho", "rho", "temp"});
	sphexa::sphnnet::computeNuclearReactions(n, dt, dt,
		reactions, construct_rates_BE, eos);
	sphexa::sphnnet::transferToHost(n, {"temp"});

	/* !! needed for now !! */
	sphexa::sphnnet::transferToHost(n, {"Y"});
	// print total nuclear energy
	Float total_nuclear_energy = sphexa::sphnnet::totalNuclearEnergy(n, BE);
	std::cout << "total nuclear energy = " << total_nuclear_energy << "\n";
}


struct eos_output {
	CUDA_FUNCTION_DECORATOR eos_output(double cv_=0., double dpdT_=0., double dudYe_=0.) :
		cv(cv_),
		dpdT(dpdT_),
		dudYe(dudYe_) {}
	CUDA_FUNCTION_DECORATOR ~eos_output() {}

	double cv, dpdT, dudYe;
};
struct isotherm_eos_struct {
	CUDA_FUNCTION_DECORATOR isotherm_eos_struct() {}

	CUDA_FUNCTION_DECORATOR eos_output inline operator()(const double *Y_, const double T, const double rho_) const {
		return eos_output{1e20, 0, 0};;
	}
} isotherm_eos;


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
    const bool use_net87                    = parser.exists("--electrons") && use_net86;

    const double hydro_dt                   = parser.get("--dt", 1e-1);
    const int n_max                         = parser.get("-n", 10);
    const int n_print                       = parser.get("--n-particle-print", 5);
    const size_t n_particles                = parser.get("--n-particle", 1000);

    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");

    nnet::net14::skip_coulombian_correction = parser.exists("--skip-coulomb-corr");

	util::array<double, 87> Y0_87, X_87;
	util::array<double, 14> Y0_14, X_14;
    if (use_net86) {
   		for (int i = 0; i < 86; ++i) X_87[i] = 0;

	    if  (      test_case == "C-O-burning") {
	    	X_87[nnet::net86::constants::net14_species_order[1]] = 0.5;
			X_87[nnet::net86::constants::net14_species_order[2]] = 0.5;
	    } else if (test_case == "He-burning") {
	    	X_87[nnet::net86::constants::net14_species_order[0]] = 1;
	    } else if (test_case == "Si-burning") {
	    	X_87[nnet::net86::constants::net14_species_order[5]] = 1;
	    } else {
	    	printHelp(argv[0]);
	    	throw std::runtime_error("unknown nuclear test case!\n");
	    }

	    for (int i = 0; i < 86; ++i) Y0_87[i] = X_87[i]/nnet::net86::constants::A[i];

	    Y0_87[nnet::net87::constants::electron] = 1;
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
    

	
	if (!nnet::eos::helmholtz_constants::initalized)
		nnet::eos::helmholtz_constants::initalized    = nnet::eos::helmholtz_constants::read_table();
	if (!nnet::net87::electrons::constants::initalized)
		nnet::net87::electrons::constants::initalized = nnet::net87::electrons::constants::read_table();



	sphexa::sphnnet::NuclearDataType<87, double, AccType> nuclear_data_87;
	sphexa::sphnnet::NuclearDataType<86, double, AccType> nuclear_data_86;
	sphexa::sphnnet::NuclearDataType<14, double, AccType> nuclear_data_14;

	if (use_net87) {
		nuclear_data_87.setDependent(/*"nid", "pid",*/ "dt", "c", "p", "cv", "u", "dpdT", "temp", "rho", "previous_rho", "Y");
		nuclear_data_87.devData.setDependent("c", "p", "cv", "u", "dpdT", "temp", "rho", "previous_rho", "Y", "dt");
		nuclear_data_87.resize(n_particles);
	} else if (use_net86) {
		nuclear_data_86.setDependent(/*"nid", "pid",*/ "dt", "c", "p", "cv", "u", "dpdT", "temp", "rho", "previous_rho", "Y");
		nuclear_data_86.devData.setDependent("c", "p", "cv", "u", "dpdT", "temp", "rho", "previous_rho", "Y", "dt");
		nuclear_data_86.resize(n_particles);
	} else {
		nuclear_data_14.setDependent(/*"nid", "pid",*/ "dt", "c", "p", "cv", "u", "dpdT", "temp", "rho", "previous_rho", "Y");
		nuclear_data_14.devData.setDependent("c", "p", "cv", "u", "dpdT", "temp", "rho", "previous_rho", "Y", "dt");
		nuclear_data_14.resize(n_particles);
	}





	/* !!!!!!!!!!!!
	initialize the state
	!!!!!!!!!!!! */
	if (use_net87) {
		for (size_t i = 0; i < n_particles; ++i) {
			nuclear_data_87.Y[i]    = Y0_87;
			nuclear_data_87.temp[i] = T_left   + (T_right   - T_left  )*((float)i)/((float)(n_particles - 1));
			nuclear_data_87.rho[i]  = rho_left + (rho_right - rho_left)*((float)i)/((float)(n_particles - 1));
		}
		sphexa::sphnnet::transferToDevice(nuclear_data_87, {"Y", "dt"});
	} else if (use_net86) {
		for (size_t i = 0; i < n_particles; ++i) {
			for (int j = 0; j < 86; ++j)
				nuclear_data_86.Y[i][j] = Y0_87[j];
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
	




	const nnet::eos::helmholtz_functor helm_eos_87 = nnet::eos::helmholtz_functor(nnet::net87::constants::Z, 87);
	const nnet::eos::helmholtz_functor helm_eos_86 = nnet::eos::helmholtz_functor(nnet::net86::constants::Z, 86);
	const nnet::eos::helmholtz_functor helm_eos_14 = nnet::eos::helmholtz_functor(nnet::net14::constants::Z);




	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!!!!!!!!!! */



	std::vector<std::string> nuclearOutFields = {"temp", "rho", "Y(4He)", "Y(12C)", "Y(16O)"};
	if (use_net87) {
		nuclear_data_87.setOutputFields(nuclearOutFields, nnet::net86::constants::species_names);
	} else if (use_net86) {
		nuclear_data_86.setOutputFields(nuclearOutFields, nnet::net86::constants::species_names);
	} else
		nuclear_data_14.setOutputFields(nuclearOutFields, nnet::net14::constants::species_names);



	// "warm-up" (first allocation etc...)
	if (use_net87) {
		if (isotherm) {
			step(nuclear_data_87, 1e-10,
				nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, isotherm_eos,
				nnet::net87::BE.data());
		} else
			step(nuclear_data_87, 1e-10,
				nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, helm_eos_87,
				nnet::net87::BE.data());
	} else if (use_net86) {
		if (isotherm) {
			step(nuclear_data_86,  1e-10,
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, isotherm_eos,
				nnet::net86::BE.data());
		} else
			step(nuclear_data_86,  1e-10,
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, helm_eos_86,
				nnet::net86::BE.data());
	} else
		if (isotherm) {
			step(nuclear_data_14,  1e-10,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, isotherm_eos,
				nnet::net14::BE.data());
		} else
			step(nuclear_data_14,  1e-10,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, helm_eos_14,
				nnet::net14::BE.data());




	auto start = std::chrono::high_resolution_clock::now();
	double min_time = 3600, max_time = 0;

	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0;
	for (int i = 0; i < n_max; ++i) {
		std::cout << i << "th iteration...\n";

		auto start_it = std::chrono::high_resolution_clock::now();

		if (use_net87) {
			if (isotherm) {
				step(nuclear_data_87, hydro_dt,
					nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, isotherm_eos,
					nnet::net87::BE.data());
			} else
				step(nuclear_data_87, hydro_dt,
					nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, helm_eos_87,
					nnet::net87::BE.data());
		} else if (use_net86) {
			if (isotherm) {
				step(nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, isotherm_eos,
					nnet::net86::BE.data());
			} else
				step(nuclear_data_86, hydro_dt,
					nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, helm_eos_86,
					nnet::net86::BE.data());
		} else
			if (isotherm) {
				step(nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, isotherm_eos,
					nnet::net14::BE.data());
			} else
				step(nuclear_data_14, hydro_dt,
					nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, helm_eos_14,
					nnet::net14::BE.data());
		
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
	if (use_net87) {
		sphexa::sphnnet::transferToHost(nuclear_data_87, {"Y"});

		dump(nuclear_data_87, 0,                     n_print,     "/dev/stdout");
		dump(nuclear_data_87, n_particles - n_print, n_particles, "/dev/stdout");
	} else if (use_net86) {
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