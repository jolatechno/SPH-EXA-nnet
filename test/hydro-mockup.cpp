#include <vector>
#include <chrono>


#include "util/arg_parser.hpp"
#include "util/sphexa_utils.hpp"


// physical parameters
#include "../src/net87/net87.hpp"
#include "../src/net14/net14.hpp"
#include "../src/eos/helmholtz.hpp"
#include "../src/eos/ideal_gas.hpp"

// base datatype
#include "../src/sphexa/nuclear-data.hpp"

// nuclear reaction wrappers
#include "../src/sphexa/nuclear-net.hpp"
#include "../src/sphexa/observables.hpp"
#include "../src/sphexa/initializers.hpp"




#if defined(COMPILE_DEVICE) && !defined(CPU_CUDA_TEST)
	using AccType = cstone::GpuTag;
#else
	using AccType = cstone::CpuTag;
#endif




/************************************************************************/
/*           MPI test that can test the GPU implementation              */
// compile CUDA: nvcc -ccbin mpic++ -Xcompiler="-DNOT_FROM_SPHEXA -fopenmp -std=c++17 -DUSE_MPI -DCPU_CUDA_TEST_" -dc  ../src/sphexa/CUDA/nuclear-net.cu -o nuclear-net.o -std=c++17 --expt-relaxed-constexpr
// compile CUDA: nvcc -ccbin mpic++ -Xcompiler="-DNOT_FROM_SPHEXA -fopenmp -std=c++17 -DUSE_MPI -DCPU_CUDA_TEST_" -dc  ../src/sphexa/CUDA/nuclear-data-gpu.cu -o nuclear-data.o -std=c++17 --expt-relaxed-constexpr
// compile CPU:  nvcc -ccbin mpic++ -Xcompiler="-DNOT_FROM_SPHEXA -fopenmp -std=c++17 -DUSE_MPI -DCPU_CUDA_TEST_" -c hydro-mockup.cpp -o hydro-mockup-gpu.o -std=c++17 --expt-relaxed-constexpr
// link:         g++ nuclear-net.o hydro-mockup-gpu.o nuclear-data.o -o hydro-mockup-gpu.out -lcudart

// compile: nvcc -x cu -ccbin mpic++ -Xcompiler="-DNOT_FROM_SPHEXA -fopenmp -std=c++17 -DUSE_MPI -DIMPORT_DOT_CU -DCPU_CUDA_TEST_" hydro-mockup.cpp -o hydro-mockup.out -std=c++17 --expt-relaxed-constexpr
/*                                                                      */
/*             Launch on a system with 2 GPUs:                          */
// mpirun --quiet --bind-to hwthread --oversubscribe -n 2 hydro-mockup.out --test-case C-O-burning --n-particle 10000000 --dt 1e-4 -n 10 > res_mpi_big.out 2> err.out &
/*                                                                      */
/*           Comparison between networks on GPU:                        */
/*             launch on a system with 2 GPUs:                          */
// net14: mpirun --quiet --bind-to hwthread --oversubscribe -n 2 hydro-mockup.out --test-case C-O-burning --n-particle 500000 --dt 1e-5 -n 10 > res_mpi_net14.out 2> err.out &
// net86: mpirun --quiet --bind-to hwthread --oversubscribe -n 2 hydro-mockup.out --test-case C-O-burning --n-particle 500000 --dt 1e-5 -n 10 --use-net86 > res_mpi_net86.out 2> err.out &
// net87: mpirun --quiet --bind-to hwthread --oversubscribe -n 2 hydro-mockup.out --test-case C-O-burning --n-particle 500000 --dt 1e-5 -n 10 --use-net86 --electrons > res_mpi_net87.out 2> err.out &
// net87 big: mpirun --quiet --bind-to hwthread --oversubscribe -n 2 hydro-mockup.out --test-case C-O-burning --n-particle 2000000 --dt 1e-6 -n 10 --use-net86 --electrons > res_mpi_net87_big.out 2> err.out &
/*                                                                      */
/*          Comparison with the CPU-only version:                       */
// compile: nvcc -x cu -ccbin mpic++ -Xcompiler="-DNOT_FROM_SPHEXA -fopenmp -DUSE_CUDA -std=c++17 -DUSE_MPI -DIMPORT_DOT_CU -DCPU_CUDA_TEST" hydro-mockup.cpp -o hydro-mockup-cpu.out -std=c++17 --expt-relaxed-constexpr
/*                                                                      */
/*            Launch on a system with 2x32 core:                        */
// mpirun --quiet --bind-to hwthread --oversubscribe --map-by ppr:2:node:PE=32 -n 2 -x OMP_NUM_THREADS=32 hydro-mockup-cpu.out --test-case C-O-burning --n-particle 10000000 --dt 1e-4 -n 10 > res_mpi_cpu_big.out 2> err.out &
/*                                                                      */
/*           Comparison between networks on CPU:                        */
/*           Launch on a system with 2x32 core:                         */
// net14: mpirun --quiet --bind-to hwthread --oversubscribe --map-by ppr:2:node:PE=32 -n 2 -x OMP_NUM_THREADS=32 hydro-mockup-cpu.out --test-case C-O-burning --n-particle 500000 --dt 1e-5 -n 10 > res_mpi_cpu_net14.out 2> err.out &
// net86: mpirun --quiet --bind-to hwthread --oversubscribe --map-by ppr:2:node:PE=32 -n 2 -x OMP_NUM_THREADS=32 hydro-mockup-cpu.out --test-case C-O-burning --n-particle 500000 --dt 1e-5 -n 10 --use-net86 > res_mpi_cpu_net86.out 2> err.out &
// net87: mpirun --quiet --bind-to hwthread --oversubscribe --map-by ppr:2:node:PE=32 -n 2 -x OMP_NUM_THREADS=32 hydro-mockup-cpu.out --test-case C-O-burning --n-particle 500000 --dt 1e-5 -n 10 --use-net86 --electrons > res_mpi_cpu_net87.out 2> err.out &
// net87 big: mpirun --quiet --bind-to hwthread --oversubscribe --map-by ppr:2:node:PE=32 -n 2 -x OMP_NUM_THREADS=32 hydro-mockup-cpu.out --test-case C-O-burning --n-particle 2000000 --dt 1e-6 -n 10 --use-net86 --electrons > res_mpi_cpu_net87_big.out 2> err.out &
/************************************************************************/



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
	std::vector<double> rho, temp, cv; //...

	void resize(const size_t N) {
		node_id.resize(N);
		particle_id.resize(N);

		x.resize(N);
		y.resize(N);
		z.resize(N);
		z.resize(N);

		rho.resize(N);
		temp.resize(N);
	}

	const std::vector<std::string> fieldNames = {
		"nid", "pid", "rho", "temp", "x", "y", "z", "cv"
	};

	auto data() {
	    using FieldType = std::variant<
	    	std::vector<size_t>*,
	    	std::vector<int>*,
	    	std::vector<double>*>;

	    std::array<FieldType, 8> ret = {
	    	&node_id, &particle_id, &rho, &temp, &x, &y, &z, &cv};

	    return ret;
	}

	std::vector<int>         outputFieldIndices;
	std::vector<std::string> outputFieldNames;

	void setOutputFields(const std::vector<std::string>& outFields) {
	    outputFieldNames   = fieldNames;
		outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
    }

    bool isAllocated(int i) const {
    	/* TODO */
    	return true;
    }
};

template<class Data>
double totalInternalEnergy(Data const &n) {
	const size_t n_particles = n.temp.size();
	double total_energy = 0;
	#pragma omp parallel for schedule(static) reduction(+:total_energy)
	for (size_t i = 0; i < n_particles; ++i)
		total_energy += n.u[i]*n.m[i];

#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &total_energy, 1, MPI_DOUBLE, MPI_SUM, n.comm);
#endif

	return total_energy;
}






void printHelp(char* name, int rank);

// mockup of the step function 
template<class func_type, class func_eos, size_t n_species, typename Float, class AccType>
void step(int rank,
	size_t firstIndex, size_t lastIndex,
	ParticlesDataType &d, sphexa::sphnnet::NuclearDataType<n_species, double, AccType>  &n, const double dt,
	const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
	const Float *BE)
{
	size_t n_nuclear_particles = n.Y.size();

	// domain redecomposition

	sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n);

	sphexa::sphnnet::hydroToNuclearUpdate(d, n, {"previous_rho"}); // useless, just for testing

	// do hydro stuff

	// std::swap(n.rho, n.previous_rho); // the way it should be done instead of the first "hydroToNuclearUpdate"
	sphexa::sphnnet::hydroToNuclearUpdate(d, n, {"rho", "temp"});
	sphexa::transferToDevice(n, 0, n_nuclear_particles, {"previous_rho", "rho", "temp"});

	sphexa::sphnnet::computeNuclearReactions(n, dt, dt,
		reactions, construct_rates_BE, eos);
	sphexa::sphnnet::computeHelmEOS(n, nnet::net14::constants::Z);

	sphexa::transferToHost(n, 0, n_nuclear_particles, {"temp",
		"c", "p", "cv", "u", "dpdT"});
	sphexa::sphnnet::nuclearToHydroUpdate(d, n, {"temp"});
	
	// do hydro stuff

	/* !! needed for now !! */
	sphexa::transferToHost(n, 0, n_nuclear_particles, {"Y"});
	// print total nuclear energy
	Float total_nuclear_energy  = sphexa::sphnnet::totalNuclearEnergy(n, BE);
	Float total_internal_energy = totalInternalEnergy(n);
	if (rank == 0)
		std::cout << "etot=" << total_nuclear_energy+total_internal_energy << " (nuclear=" << total_nuclear_energy << ", internal=" << total_internal_energy << ")\n";
}



int main(int argc, char* argv[]) {
	/* initial hydro data */
	const double rho_left = 1.1e9, rho_right = 0.8e9;
	const double T_left = 0.5e9, T_right = 1.5e9;




	int size, rank;
    MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef COMPILE_DEVICE
	cuda_util::initCudaMpi(MPI_COMM_WORLD);
#endif

	if (!nnet::eos::helmholtz_constants::initalized)
		nnet::eos::helmholtz_constants::initalized    = nnet::eos::helmholtz_constants::read_table<AccType>();
	if (!nnet::net87::electrons::constants::initalized)
		nnet::net87::electrons::constants::initalized = nnet::net87::electrons::constants::read_table<AccType>();




	const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")) {
        printHelp(argv[0], rank);
        MPI_Finalize();
        return 0;
    }

    const bool use_net86                    = parser.exists("--use-net86");
    const bool use_net87                    = parser.exists("--electrons") && use_net86;

    const double hydro_dt                   = parser.get("--dt", 1e-1);
    const int n_max                         = parser.get("-n", 10);
    const int n_print                       = parser.get("--n-particle-print", 5);
    const size_t total_n_particles          = parser.get("--n-particle", 1000);

    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");
    const bool idealGas                     = parser.exists("--ideal-gas") || isotherm;

	util::array<double, 86> Y0_87, X_87;
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
	    	printHelp(argv[0], rank);
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
	for (size_t i = first; i < last; ++i) {
		particle_data.temp[i] = T_left   + (T_right   - T_left  )*((float)(total_n_particles*rank/size + i-first))/((float)(total_n_particles - 1));
		particle_data.rho[i]  = rho_left + (rho_right - rho_left)*((float)(total_n_particles*rank/size + i-first))/((float)(total_n_particles - 1));
	}




	const nnet::eos::ideal_gas_functor idea_gas_eos = nnet::eos::ideal_gas_functor(isotherm ? 1e-20 : 10.0);
	const nnet::eos::helmholtz_functor helm_eos_87  = nnet::eos::helmholtz_functor(nnet::net87::constants::Z, 87);
	const nnet::eos::helmholtz_functor helm_eos_86  = nnet::eos::helmholtz_functor(nnet::net86::constants::Z, 86);
	const nnet::eos::helmholtz_functor helm_eos_14  = nnet::eos::helmholtz_functor(nnet::net14::constants::Z);


	sphexa::sphnnet::NuclearDataType<87, double, AccType> nuclear_data_87;
	sphexa::sphnnet::NuclearDataType<86, double, AccType> nuclear_data_86;
	sphexa::sphnnet::NuclearDataType<14, double, AccType> nuclear_data_14;

	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!!!!!!!!!! */
	size_t n_nuclear_particles;
	sphexa::mpi::initializePointers(first, last, particle_data.node_id, particle_data.particle_id, particle_data.comm);
	if (use_net87) {
		nuclear_data_87.setDependent("nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "previous_rho", "Y");
			//"nid", "pid", "temp", "rho", "previous_rho", "Y");
		nuclear_data_87.devData.setDependent("temp", "rho", "previous_rho", "Y", "dt", "c", "p", "cv", "u", "dpdT");

		sphexa::sphnnet::initNuclearDataFromConst(first, last, particle_data, nuclear_data_87, Y0_87);

		n_nuclear_particles = nuclear_data_87.Y.size();
		sphexa::transferToDevice(nuclear_data_87, 0, n_nuclear_particles, {"Y", "dt"});

		std::fill(nuclear_data_87.m.begin(), nuclear_data_87.m.end(), 1.);
	} else if (use_net86) {
		nuclear_data_86.setDependent("nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "previous_rho", "Y");
			//"nid", "pid", "temp", "rho", "previous_rho", "Y");
		nuclear_data_86.devData.setDependent("temp", "rho", "previous_rho", "Y", "dt", "c", "p", "cv", "u", "dpdT");

		sphexa::sphnnet::initNuclearDataFromConst(first, last, particle_data, nuclear_data_86, Y0_87);

		n_nuclear_particles = nuclear_data_86.Y.size();
		sphexa::transferToDevice(nuclear_data_86, 0, n_nuclear_particles, {"Y", "dt"});

		std::fill(nuclear_data_86.m.begin(), nuclear_data_86.m.end(), 1.);
	} else {
		nuclear_data_14.setDependent("nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "previous_rho", "Y", "dt");
			//"nid", "pid", "temp", "rho", "previous_rho", "Y");
		nuclear_data_14.devData.setDependent("temp", "rho", "previous_rho", "Y", "dt", "c", "p", "cv", "u", "dpdT");

		sphexa::sphnnet::initNuclearDataFromConst(first, last, particle_data, nuclear_data_14, Y0_14);

		n_nuclear_particles = nuclear_data_14.Y.size();
		sphexa::transferToDevice(nuclear_data_14, 0, n_nuclear_particles, {"Y", "dt"});

		std::fill(nuclear_data_14.m.begin(), nuclear_data_14.m.end(), 1.);
	}



	std::vector<std::string> hydroOutFields   = {"nid", "pid", "temp", "rho"};
	std::vector<std::string> nuclearOutFields = {"nid", "pid", "temp", "rho", "cv", "u", "dpdT", "Y(4He)", "Y(12C)", "Y(16O)"};
	particle_data.setOutputFields(hydroOutFields);
	if (use_net87) {
		nuclear_data_87.setOutputFields(nuclearOutFields, nnet::net87::constants::species_names);
	} else if (use_net86) {
		nuclear_data_86.setOutputFields(nuclearOutFields, nnet::net86::constants::species_names);
	} else
		nuclear_data_14.setOutputFields(nuclearOutFields, nnet::net14::constants::species_names);





	// "warm-up" (first allocation etc...)
	if (use_net87) {
		if (idealGas) {
			step(rank,
				first, last,
				particle_data, nuclear_data_87, 1e-10,
				nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, idea_gas_eos,
				nnet::net87::BE.data());
		} else
			step(rank,
				first, last,
				particle_data, nuclear_data_87, 1e-10,
				nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, helm_eos_87,
				nnet::net87::BE.data());
	} else if (use_net86) {
		if (idealGas) {
			step(rank,
				first, last,
				particle_data, nuclear_data_86, 1e-10,
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, idea_gas_eos,
				nnet::net86::BE.data());
		} else
			step(rank,
				first, last,
				particle_data, nuclear_data_86, 1e-10,
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, helm_eos_86,
				nnet::net86::BE.data());
	} else
		if (idealGas) {
			step(rank,
				first, last,
				particle_data, nuclear_data_14, 1e-10,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, idea_gas_eos,
				nnet::net14::BE.data());
		} else
			step(rank,
				first, last,
				particle_data, nuclear_data_14, 1e-10,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, helm_eos_14,
				nnet::net14::BE.data());






	MPI_Barrier(MPI_COMM_WORLD);
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

	if (use_net87) {
		if (idealGas) {
			step(rank,
				first, last,
				particle_data, nuclear_data_87, hydro_dt,
				nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, idea_gas_eos,
				nnet::net87::BE.data());
		} else
			step(rank,
				first, last,
				particle_data, nuclear_data_87, hydro_dt,
				nnet::net87::reaction_list, nnet::net87::compute_reaction_rates, helm_eos_87,
				nnet::net87::BE.data());
	} else if (use_net86) {
		if (idealGas) {
			step(rank,
				first, last,
				particle_data, nuclear_data_86, hydro_dt,
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, idea_gas_eos,
				nnet::net86::BE.data());
		} else
			step(rank,
				first, last,
				particle_data, nuclear_data_86, hydro_dt,
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, helm_eos_86,
				nnet::net86::BE.data());
	} else
		if (idealGas) {
			step(rank,
				first, last,
				particle_data, nuclear_data_14, hydro_dt,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, idea_gas_eos,
				nnet::net14::BE.data());
		} else
			step(rank,
				first, last,
				particle_data, nuclear_data_14, hydro_dt,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, helm_eos_14,
				nnet::net14::BE.data());

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


		for (auto name : hydroOutFields)
			std::cout << name << " ";
		std::cout << "\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);
	dump(particle_data, first, first + n_print, "/dev/stdout");
	dump(particle_data, last - n_print,   last, "/dev/stdout");


	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		std::cout << "\n";
		for (auto name : nuclearOutFields)
			std::cout << name << " ";
		std::cout << "\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (use_net87) {
		sphexa::transferToHost(nuclear_data_87, 0, n_nuclear_particles, {"cv"});

		dump(nuclear_data_87, 0,                             n_print,             "/dev/stdout");
		dump(nuclear_data_87, n_nuclear_particles - n_print, n_nuclear_particles, "/dev/stdout");
	} else if (use_net86) {
		sphexa::transferToHost(nuclear_data_86, 0, n_nuclear_particles, {"cv"});

		dump(nuclear_data_86, 0,                             n_print,             "/dev/stdout");
		dump(nuclear_data_86, n_nuclear_particles - n_print, n_nuclear_particles, "/dev/stdout");
	} else {
		sphexa::transferToHost(nuclear_data_14, 0, n_nuclear_particles, {"cv"});

		dump(nuclear_data_14, 0,                             n_print,             "/dev/stdout");
		dump(nuclear_data_14, n_nuclear_particles - n_print, n_nuclear_particles, "/dev/stdout");
	}

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