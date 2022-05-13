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

	// hydro data
	std::vector<double> rho, T; //...

	void resize(const size_t N) {
		node_id.resize(N);
		particle_id.resize(N);

		rho.resize(N);
		T.resize(N);
	}
};





// mockup of the step function 
template<class func_rate, class func_BE, class func_eos>
void step(ParticlesDataType &p, sphexa::sphnnet::NuclearDataType<14, double>  &n, const double dt,
	const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {

	auto partition = sphexa::mpi::partition_from_pointers(p.node_id, p.particle_id);

	sphexa::sphnnet::sendHydroPreviousData(p, n, partition, MPI_DOUBLE);

	// do hydro stuff

	sphexa::sphnnet::sendHydroData(p, n, partition, MPI_DOUBLE);
	sphexa::sphnnet::compute_nuclear_reactions(n, dt,
		reactions, construct_rates, construct_BE, eos);
	sphexa::sphnnet::recvHydroData(p, n, partition, MPI_DOUBLE);

	// do hydro stuff
}






using vector = sphexa::sphnnet::NuclearAbundances<14>;

int main(int argc, char* argv[]) {
	int size, rank;
    MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ParticlesDataType p;
	sphexa::sphnnet::NuclearDataType<14> n;

	const size_t n_particles = 200;
	p.resize(n_particles);
	n.resize(n_particles);


	/* !!!!!!!!!!!!
	initialize the state
	!!!!!!!!!!!! */
#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif
	double rho_left = 1e9, rho_right = 0.5e9; // rho, g/cm^3
	double T_left = 0.8e9, T_right = 2e9; // rho, g/cm^3

	for (int i = 0; i < n_particles; ++i) {
		// nuclear datas
		n.Y[i][1] = 0.5/nnet::net14::constants::A[1];
		n.Y[i][2] = 0.5/nnet::net14::constants::A[2];

		n.dt[i] = 1e-12;

		// hydro data
		p.T[i]   = T_left   + (T_right   - T_left  )*(float)(size*n_particles + i)/(float)(size*n_particles - 1);
		p.rho[i] = rho_left + (rho_right - rho_left)*(float)(size*n_particles + i)/(float)(size*n_particles - 1);

		// pointers
		p.node_id[i] = (rank + 1)%size;
		p.particle_id[i] = i;
	}



	/* !!!!!!!!!!!!
	do simulation
	!!!!!!!!!!!! */
	double t = 0, dt = 1e-3;
	int n_max = 200;
 	double m_in = eigen::dot(n.Y[0], nnet::net14::constants::A);
	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);
	for (int i = 0; i < n_max; ++i) {
		if (rank == 0)
			std::cout << i << "th iteration...\n";

		step(p, n, dt,
			nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double, vector>, helm_eos);
		t += dt;

		if (rank == 0)
			std::cout << "\t...Ok\n";
	}
 
	double m_tot = eigen::dot(n.Y[0], nnet::net14::constants::A);
	double dm_m = (m_tot - m_in)/m_in;

	if (rank == 0) {
		std::vector<double> X(14);
		for (int i = 0; i < 14; ++i) X[i] = n.Y[0][i]*nnet::net14::constants::A[i]/eigen::dot(n.Y[0], nnet::net14::constants::A);
		std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
		for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
		std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tT_left=" << p.T[0] << "\tT_right=" << p.T[n_particles - 1] << "\n";
	}


	MPI_Finalize();

	return 0;
}