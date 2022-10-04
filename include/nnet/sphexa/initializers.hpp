
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
 * @brief Initializer functions for nuclear data.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include <numeric>

#include "nuclear-data.hpp"
#include "mpi/mpi-wrapper.hpp"

namespace sphexa::sphnnet {
	/*! @brief intialize nuclear data, from a function of positions. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType (contains positions) 
	 * @param n            nuclearDataType (to be populated)
	 * @param initializer  function initializing nuclear abundances from position
	 */
	template<size_t n_species, typename Float, typename KeyType, class AccType, class initFunc, class ParticlesDataType>
	void initNuclearDataFromPos(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, KeyType, AccType> &n, const initFunc initializer) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::computePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		

		// receiv position for initializer
#ifdef USE_MPI
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::sphnnet::syncDataToStaticPartition(n.partition, d.x.data(), x.data(), d.comm);
		sphexa::sphnnet::syncDataToStaticPartition(n.partition, d.y.data(), y.data(), d.comm);
		sphexa::sphnnet::syncDataToStaticPartition(n.partition, d.z.data(), z.data(), d.comm);
#else
		std::vector<Float> &x = d.x, &y = d.y, &z = d.z;
#endif

		util::array<Float, n_species> Y;

		// intialize nuclear data
		#pragma omp parallel for firstprivate(Y) schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i) {
			Y = initializer(x[i], y[i], z[i]);
			for (int j = 0; j < n_species; ++j)
				n.Y[j][i] = Y[j];
		}
	}


	/*! @brief intialize nuclear data, from a function of radius. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType (contains positions) 
	 * @param n            nuclearDataType (to be populated)
	 * @param initializer  function initializing nuclear abundances from radius
	 */
	template<size_t n_species, typename Float, typename KeyType, class AccType, class initFunc, class ParticlesDataType>
	void initNuclearDataFromRadius(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, KeyType, AccType> &n, const initFunc initializer) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::computePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif
		const size_t local_n_particles = d.x.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);

		// receiv position for initializer
#ifdef USE_MPI
		std::vector<Float> send_r(local_n_particles), r(local_nuclear_n_particles, d.comm);
#else
		std::vector<Float> r(local_n_particles), &send_r = r;
#endif
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i)
			send_r[i] = std::sqrt(d.x[i]*d.x[i] + d.y[i]*d.y[i] + d.z[i]*d.z[i]);
#ifdef USE_MPI
		sphexa::sphnnet::syncDataToStaticPartition(n.partition, send_r.data(), r.data(), d.comm);
#endif

		util::array<Float, n_species> Y;

		// intialize nuclear data
		#pragma omp parallel for firstprivate(Y) schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i) {
			Y = initializer(r[i]);
			for (int j = 0; j < n_species; ++j)
				n.Y[j][i] = Y[j];
		}
	}


	/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType (contains density) 
	 * @param n            nuclearDataType (to be populated)
	 * @param initializer  function initializing nuclear abundances from radius
	 */
	template<size_t n_species, typename Float, typename KeyType, class AccType, class initFunc, class ParticlesDataType>
	void initNuclearDataFromRho(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, KeyType, AccType> &n, const initFunc initializer) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::computePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif

		// share the initial rho
		n.resize(local_nuclear_n_particles);
#ifdef USE_MPI
		sphexa::sphnnet::syncDataToStaticPartition(n.partition, d.rho.data(), n.rho.data(), d.comm);
#else
		n.rho = d.rho;
#endif

		util::array<Float, n_species> Y;

		// intialize nuclear data
		#pragma omp parallel for firstprivate(Y) schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i) {
			Y = initializer(n.rho[i]);
			for (int j = 0; j < n_species; ++j)
				n.Y[j][i] = Y[j];
		}
	}


	/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex  first (included) particle considered in d
	 * @param lastIndex   last (excluded) particle considered in d
	 * @param d           ParticlesDataType (not used) 
	 * @param n           nuclearDataType (to be populated)
	 * @param Y0          constant abundances vector to be copied
	 */
	template<size_t n_species, typename Float, typename KeyType, class AccType, class Vector, class ParticlesDataType>
	void initNuclearDataFromConst(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, KeyType, AccType> &n, const Vector &Y0) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::computePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif

		// share the initial rho
		n.resize(local_nuclear_n_particles);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < n_species; ++j)
			std::fill(n.Y[j].begin(), n.Y[j].end(), Y0[j]);
	}



	/*! @brief intialize nuclear data, from a function of positions. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
	 * @param initializer  function initializing nuclear abundances from position
	 */
	template<class initFunc, class ParticlesDataType>
	void inline initNuclearDataFromPos(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const initFunc initializer) {
		initNuclearDataFromPos(firstIndex, lastIndex, d, d.nuclearData, initializer);
	}

	/*! @brief intialize nuclear data, from a function of radius. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
	 * @param initializer  function initializing nuclear abundances from radius
	 */
	template<class initFunc, class ParticlesDataType>
	void inline initNuclearDataFromRadius(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const initFunc initializer) {
		initNuclearDataFromRadius(firstIndex, lastIndex, d, d.nuclearData, initializer);
	}

	/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
	 * @param initializer  function initializing nuclear abundances from radius
	 */
	template<class initFunc, class ParticlesDataType>
	void inline initNuclearDataFromRho(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const initFunc initializer) {
		initNuclearDataFromRadius(firstIndex, lastIndex, d, d.nuclearData, initializer);
	}

	/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and detached data.
	 * 
	 * @param firstIndex   first (included) particle considered in d
	 * @param lastIndex    last (excluded) particle considered in d
	 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
	 * @param initializer  function initializing nuclear abundances from radius
	 */
	template<class Vector, class ParticlesDataType>
	void inline initNuclearDataFromConst(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const Vector &Y0) {
		initNuclearDataFromConst(firstIndex, lastIndex, d, d.nuclearData, Y0);
	}
}