/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief More representative mockup of full SPH-EXA + nuclear-nets multi-particle simulation (net14 only).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#include <vector>
#include <chrono>


#include "util/arg_parser.hpp"
#include "util/sphexa_utils.hpp"


// physical parameters
#include "nnet/parameterization/net14/net14.hpp"
#include "nnet/parameterization/eos/helmholtz.hpp"
#include "nnet/parameterization/eos/ideal_gas.hpp"

// base datatype
#include "nnet/sphexa/nuclear-data.hpp"

// nuclear reaction wrappers
#include "nnet/sphexa/nuclear-net.hpp"
#include "nnet/sphexa/observables.hpp"
#include "nnet/sphexa/initializers.hpp"




#if !defined(CUDA_CPU_TEST) && defined(USE_CUDA)
	using AccType = cstone::GpuTag;
#else
	using AccType = cstone::CpuTag;
#endif





/*
function stolen from SPH-EXA and retrofited for testing
*/
template<class Dataset>
void dump(Dataset& d, size_t firstIndex, size_t lastIndex, /*const cstone::Box<typename Dataset::RealType>& box,*/ std::string path) {
    const char separator = ' ';
    // path += std::to_string(d.iteration) + ".txt";

    int rank = 0, numRanks = 1;
#ifdef USE_MPI
    MPI_Comm_rank(d.comm, &rank);
    MPI_Comm_size(d.comm, &numRanks);
#endif

    for (int turn = 0; turn < numRanks; turn++)
    {
        if (turn == rank)
        {
            try
            {
                auto fieldPointers = cstone::getOutputArrays(d);

                bool append = rank != 0;
                sphexa::fileutils::writeAscii(firstIndex, lastIndex, path, append, fieldPointers, separator);
            }
            catch (std::runtime_error& ex)
            {
                throw std::runtime_error("ERROR: Terminating\n"/*, ex.what()*/);
            }
        }

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}







// mockup of the ParticlesDataType
	template<size_t n_species, typename Float, typename Int, class AccType>
class ParticlesDataType {
public:
	sphexa::sphnnet::NuclearDataType<n_species, Float, Int, AccType> nuclearData;

#ifdef USE_MPI
	// communicator
	MPI_Comm comm=MPI_COMM_WORLD;
#endif

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
		outputFieldIndices = cstone::fieldStringsToInt(outFields, outputFieldNames);
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
template<class func_type, class func_eos, class Zvector, size_t n_species, typename Float, typename KeyType, class AccType>
void step(int rank,
	size_t firstIndex, size_t lastIndex,
	ParticlesDataType<n_species, Float, KeyType, AccType> &d, const double dt,
	const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
	const Float *BE, const Zvector &Z)
{
	size_t n_nuclear_particles = d.nuclearData.temp.size();

	// domain redecomposition

	sphexa::sphnnet::computeNuclearPartition(firstIndex, lastIndex, d);

	// do hydro stuff

	std::swap(d.nuclearData.rho, d.nuclearData.rho_m1);
	sphexa::sphnnet::syncHydroToNuclear(d, {"rho", "temp"});
	sphexa::transferToDevice(d.nuclearData, 0, n_nuclear_particles, {"rho_m1", "rho", "temp"});

	sphexa::sphnnet::computeNuclearReactions(d.nuclearData, 0, n_nuclear_particles, dt, dt,
		reactions, construct_rates_BE, eos,
		/*considering expansion:*/true);
	sphexa::sphnnet::computeHelmEOS(d.nuclearData, 0, n_nuclear_particles, Z);

	sphexa::transferToHost(d.nuclearData, 0, n_nuclear_particles, {"temp",
		"c", "p", "cv", "u", "dpdT"});
	sphexa::sphnnet::syncNuclearToHydro(d, {"temp"});
	
	// do hydro stuff

	/* !! needed for now !! */
	sphexa::transferToHost(d.nuclearData, 0, n_nuclear_particles, {"Y"});
	// print total nuclear energy
	Float total_nuclear_energy  = sphexa::sphnnet::totalNuclearEnergy(d.nuclearData, BE);
	Float total_internal_energy = totalInternalEnergy(d.nuclearData);
	if (rank == 0)
		std::cout << "etot=" << total_nuclear_energy+total_internal_energy << " (nuclear=" << total_nuclear_energy << ", internal=" << total_internal_energy << ")\n";
}



int main(int argc, char* argv[]) {
	/* initial hydro data */
	const double rho_left = 1.1e9, rho_right = 0.8e9;
	const double T_left = 0.5e9, T_right = 1.5e9;




    int size = 1, rank = 0;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	

#if COMPILE_DEVICE && defined(USE_MPI)
	cuda_util::initCudaMpi(MPI_COMM_WORLD);
#endif
	
#if !defined(CUDA_CPU_TEST) && defined(USE_CUDA)
	nnet::eos::helmholtz_constants::copy_table_to_gpu();
#endif
	


	const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")) {
        printHelp(argv[0], rank);

#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    const double hydro_dt                   = parser.get("--dt", 1e-1);
    const int n_max                         = parser.get("-n", 10);
    const int n_print                       = parser.get("--n-particle-print", 5);
    const size_t total_n_particles          = parser.get("--n-particle", 1000);

    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");
    const bool idealGas                     = parser.exists("--ideal-gas") || isotherm;

	util::array<double, 14> Y0_14, X_14;
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
    




	/* !!!!!!!!!!!!
	initialize the hydro state
	!!!!!!!!!!!! */
	ParticlesDataType<14, double, size_t, AccType> particle_data;
	const size_t n_particles = total_n_particles*(rank + 1)/size - total_n_particles*rank/size;
	const size_t offset = 10*rank;
	const size_t first = offset, last = n_particles + offset;

	particle_data.resize(last);
	for (size_t i = first; i < last; ++i) {
		particle_data.temp[i] = T_left   + (T_right   - T_left  )*((float)(total_n_particles*rank/size + i-first))/((float)(total_n_particles - 1));
		particle_data.rho[i]  = rho_left + (rho_right - rho_left)*((float)(total_n_particles*rank/size + i-first))/((float)(total_n_particles - 1));
	}




	const nnet::eos::ideal_gas_functor idea_gas_eos = nnet::eos::ideal_gas_functor(isotherm ? 1e-20 : 10.0);
	const nnet::eos::helmholtz_functor helm_eos_14  = nnet::eos::helmholtz_functor(nnet::net14::constants::Z);



	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!!!!!!!!!! */
	sphexa::sphnnet::initializeNuclearPointers(first, last, particle_data);

	particle_data.nuclearData.setDependent("nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "rho_m1", "dt");
	particle_data.nuclearData.devData.setDependent("temp", "rho", "rho_m1", "dt", "c", "p", "cv", "u", "dpdT");

	for (int i = 0; i < 14; ++i) {
		particle_data.nuclearData.setDependent("Y" + std::to_string(i));
		particle_data.nuclearData.devData.setDependent("Y" + std::to_string(i));
	}

	sphexa::sphnnet::initNuclearDataFromConst(first, last, particle_data, Y0_14);

	size_t n_nuclear_particles = particle_data.nuclearData.temp.size();
	sphexa::transferToDevice(particle_data.nuclearData, 0, n_nuclear_particles, {"Y", "dt"});

	std::fill(particle_data.nuclearData.m.begin(), particle_data.nuclearData.m.end(), 1.);




	std::vector<std::string> hydroOutFields   = {"nid", "pid", "temp", "rho"};
	std::vector<std::string> nuclearOutFields = {"nid", "pid", "temp", "rho", "cv", "u", "dpdT", "Y0", "Y2", "Y1"};
	particle_data.setOutputFields(hydroOutFields);
	particle_data.nuclearData.setOutputFields(nuclearOutFields);




#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	auto start = std::chrono::high_resolution_clock::now();
	double min_time = 3600, max_time = 0;

	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0;
	for (int i = 0; i < n_max; ++i) {
		if (rank == 0)
			std::cout << i << "th iteration...\n";

#ifdef USE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		auto start_it = std::chrono::high_resolution_clock::now();

		if (idealGas) {
			step(rank,
				first, last,
				particle_data, hydro_dt,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, idea_gas_eos,
				nnet::net14::BE.data(), nnet::net14::constants::Z);
		} else
			step(rank,
				first, last,
				particle_data, hydro_dt,
				nnet::net14::reaction_list, nnet::net14::compute_reaction_rates, helm_eos_14,
				nnet::net14::BE.data(), nnet::net14::constants::Z);

		t += hydro_dt;

#ifdef USE_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		auto end_it = std::chrono::high_resolution_clock::now();
		auto duration_it = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(end_it - start_it).count())/1e3;
		min_time = std::min(min_time, duration_it);
		max_time = std::max(max_time, duration_it);

		if (rank == 0)
			std::cout << "\t...Ok\n";
	}


#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if (rank == 0) {
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = ((float)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count())/1e3;
		auto avg_duration = duration/n_max;
		std::cout << "\nexec time: " << duration << "s (avg=" << avg_duration << "s/it, max=" << max_time << "s/it, min=" << min_time  << "s/it)\n\n";


		for (auto name : hydroOutFields)
			std::cout << name << " ";
		std::cout << "\n";
	}
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	dump(particle_data, first, first + n_print, "/dev/stdout");
	dump(particle_data, last - n_print,   last, "/dev/stdout");


#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if (rank == 0) {
		std::cout << "\n";
		for (auto name : nuclearOutFields)
			std::cout << name << " ";
		std::cout << "\n";
	}
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	sphexa::transferToHost(particle_data.nuclearData, 0, n_nuclear_particles, {"cv"});

	dump(particle_data.nuclearData, 0,                             n_print,             "/dev/stdout");
	dump(particle_data.nuclearData, n_nuclear_particles - n_print, n_nuclear_particles, "/dev/stdout");

#ifdef USE_MPI
	MPI_Finalize();
#endif

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
	}
}