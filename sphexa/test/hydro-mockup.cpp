#include <vector>

#include "../src/mpi-wrapper.hpp"
#include "../src/nuclear-data.hpp"
#include "../src/nuclear-net.hpp"

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"

#include "../../src/eos/helmholtz.hpp"






// mockup of the ParticlesDataType
class ParticlesDataType {
public:
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
void step(ParticlesDataType &p, sphexa::sphnnet::NuclearDataType<14, double>  &n, const double dt,
	const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {

	// domain redecomposition

	auto partition = sphexa::mpi::partitionFromPointers(p.node_id, p.particle_id);

	// do hydro stuff

	sphexa::sphnnet::sendHydroData(p, n, partition, MPI_DOUBLE);
	sphexa::sphnnet::compute_nuclear_reactions(n, dt,
		reactions, construct_rates, construct_BE, eos);
	sphexa::sphnnet::recvHydroData(p, n, partition, MPI_DOUBLE);

	// do hydro stuff
}






using vector = sphexa::sphnnet::NuclearAbundances<14>;

int main(int argc, char* argv[]) {
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif

	int size, rank;
    MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* initial Y value */
	vector Y0, X;
	for (int i = 0; i < 14; ++i) X[i] = 0;
	X[1] = 0.5;
	X[2] = 0.5;
	for (int i = 0; i < 14; ++i) Y0[i] = X[i]/nnet::net14::constants::A[i];
	double m_in = eigen::dot(Y0, nnet::net14::constants::A);
	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);


	/* initial hydro data */
	double rho_left = 1e9, rho_right = 0.8e9; // rho, g/cm^3
	double T_left = 0.8e9, T_right = 1.3e9; // rho, g/cm^3

	ParticlesDataType p;

	const size_t total_n_particles = 1000;
	const size_t n_particles = total_n_particles/size + size - (total_n_particles/size)%size;
	p.resize(n_particles);


	/* !!!!!!!!!!!!
	initialize the hydro state
	!!!!!!!!!!!! */
	for (int i = 0; i < n_particles; ++i) {
		p.T[i]   = T_left   + (T_right   - T_left  )*(float)(rank*n_particles + i)/(float)(size*n_particles - 1);
		p.rho[i] = rho_left + (rho_right - rho_left)*(float)(rank*n_particles + i)/(float)(size*n_particles - 1);
	}


#if defined(NO_COMM) || defined(NO_MIX)
#ifdef NO_COMM
	/* !!!!!!!!!!!!
	initialize pointers with some simple "mixing"
	!!!!!!!!!!!! */
	for (int i = 0; i < n_particles; ++i) {
		p.node_id[i] = rank;
		p.particle_id[i] = i;
	}
#else //ifdef NO_MIX
	/* !!!!!!!!!!!!
	initialize pointers with a lot of communication but no "mixing"
	!!!!!!!!!!!! */
	for (int i = 0; i < n_particles; ++i) {
		p.node_id[i] = (rank + 1)%size;
		p.particle_id[i] = i;
	}
#endif

	/* !!!!!!!!!!!!
	initialize the nuclear data with homogenous abundances
	!!!!!!!!!!!! */
	{
		auto partition = sphexa::mpi::partitionFromPointers(p.node_id, p.particle_id);
		sphexa::mpi::directSyncDataFromPartition(partition, d.rho, n.previous_rho, datatype);
		const size_t nuclear_n_particles = partition.recv_disp[size];
		std::cout << nuclear_n_particles << "\n";
		n.resize(nuclear_n_particles);
		for (size_t i = 0; i < nuclear_n_particles; ++i) {
			n.Y[i] = Y0;
		}
	}
	
#else
	/* !!!!!!!!!!!!
	initialize pointers
	!!!! THE WAY IT SHOULD BE DONE !!!!
	!!!!!!!!!!!! */
	sphexa::mpi::initializePointers(p.node_id, p.particle_id, n_particles);

	/* !!!!!!!!!!!!
	initialize nuclear data
	!!!! THE WAY IT SHOULD BE DONE !!!!
	!!!!!!!!!!!! */
	auto partition = sphexa::mpi::partitionFromPointers(p.node_id, p.particle_id);
	sphexa::sphnnet::NuclearDataType<14> n = sphexa::sphnnet::initNuclearData<14>(p,
		[&](const double x, const double y, const double z) {
			return Y0;
		}, partition, MPI_DOUBLE);
#endif


	MPI_Barrier(MPI_COMM_WORLD);
	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0, dt = 1e-4;
	int n_max = 50;
	for (int i = 0; i < n_max; ++i) {
		if (rank == 0)
			std::cout << i << "th iteration...\n";

		step(p, n, dt,
			nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double, vector>, helm_eos);
		t += dt;

		if (rank == 0)
			std::cout << "\t...Ok\n";
	}
 
	

	MPI_Barrier(MPI_COMM_WORLD);
	if ((0 + 0)%size == rank) {
		double m_tot = eigen::dot(n.Y[0], nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

		for (int i = 0; i < 14; ++i) X[i] = n.Y[0][i]*nnet::net14::constants::A[i]/eigen::dot(n.Y[0], nnet::net14::constants::A);
		std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
		for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
		std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tT_left=" << n.T[0] << "\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if ((size - 1 + n_particles - 1)%size == rank) {
		double m_tot = eigen::dot(n.Y[n.Y.size() - 1], nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

		for (int i = 0; i < 14; ++i) X[i] = n.Y[n.Y.size() - 1][i]*nnet::net14::constants::A[i]/eigen::dot(n.Y[0], nnet::net14::constants::A);
		std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
		for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
		std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tT_right=" << n.T[n.Y.size() - 1] << "\n";
	}


	MPI_Finalize();

	return 0;
}