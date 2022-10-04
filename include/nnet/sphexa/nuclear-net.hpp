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
 * @brief Simple callback to parallel/parallel-nuclear-net.hpp
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include "../parallel/parallel-nuclear-net.hpp"

#include "mpi/mpi-wrapper.hpp"

#include "sph/data_util.hpp"

#include "../../util/algorithm.hpp"

namespace sphexa::sphnnet {
	/*! @brief function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y
	 * 
	 * @param n                   nuclearDataType including a field of nuclear abundances "Y"
     * @param firstIndex          first (included) particle considered in n
	 * @param lastIndex           last (excluded) particle considered in n
     * @param hydro_dt            integration timestep
     * @param previous_dt         previous integration timestep
     * @param reactions           reaction list
     * @param construct_rates_BE  function constructing rates, rate derivatives and binding energies
     * @param eos                 equation of state
     * @param use_drhodt          if true considers drho/dt in eos
     * @param jumpToNse           function to jump to nuclear statistical equilibrium
     */
	template<class Data, class func_type, class func_eos, typename Float, class nseFunction=void*>
	void inline computeNuclearReactions(Data &n, size_t firstIndex, size_t lastIndex, const Float hydro_dt, const Float previous_dt,
		const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		bool use_drhodt,
		const nseFunction jumpToNse=NULL)
	{
		nnet::parallel_nnet::computeNuclearReactions(n, firstIndex, lastIndex, hydro_dt, previous_dt,
			reactions, construct_rates_BE, eos,
			use_drhodt,
			jumpToNse);
	}


	/*! @brief function to copute the helmholtz eos
	 * 
	 * @param n           nuclearDataType including a field of nuclear abundances "Y"
     * @param firstIndex  first (included) particle considered in n
	 * @param lastIndex   last (excluded) particle considered in n
	 * @param Z           vector of number of charge (used in eos)
	 */
	template<class Data, class Vector>
	void inline computeHelmEOS(Data &n, size_t firstIndex, size_t lastIndex, const Vector &Z) {
		nnet::parallel_nnet::computeHelmEOS(n, firstIndex, lastIndex, Z);
	}


	/*! @brief function that updates the nuclear partition
	 * 
	 * @param firstIndex  first (included) particle considered in d
	 * @param lastIndex   last (excluded) particle considered in d
	 * @param d           ParticlesDataType (contains node_id and particle_id to be populated)
	 * @param n           nuclearDataType (contains mpi_partition to be populated)
	 */
	template<class ParticlesDataType,class nuclearDataType>
	void computePartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, nuclearDataType &n) {
#ifdef USE_MPI
		n.partition = sphexa::mpi::partitionFromPointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.comm);
#endif
	}


	/*! @brief function that updates the nuclear partition
	 * 
	 * @param firstIndex  first (included) particle considered in d
	 * @param lastIndex   last (excluded) particle considered in d
	 * @param d           ParticlesDataType, where d.nuclearData is the nuclear data container
	 */
	template<class ParticlesDataType>
	void computeNuclearPartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d) {
		initializePartition(firstIndex, lastIndex, d, d.nuclearData);
	}


	/*! @brief function that initialize node_id and particle_id for "maximal mixing".
	 * 
	 * @param firstIndex  index of the first particle to be considered (included) when populating the node_id and particle_id vectors
	 * @param lastIndex   index of the last particle to be considered (excluded)  when populating the node_id and particle_id vectors
	 * @param d           ParticlesDataType (contains node_id and particle_id to be populated)
	 */
	template<class ParticlesDataType>
	void initializeNuclearPointers(size_t firstIndex, size_t lastIndex, ParticlesDataType &d) {
#ifdef USE_MPI
		sphexa::mpi::initializePointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.comm);
#endif
	}


	/*! @brief function sending requiered hydro data from ParticlesDataType (attached data) to NuclearDataType (detached data)
	 * 
	 * @param d            ParticlesDataType (attached data)
	 * @param n            NuclearDataType (detached data)
	 * @param sync_fields  names of field to be synced
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void syncDataToStaticPartition(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
		// get data
		auto nuclearData  = n.data();
		auto particleData = d.data();

		// send fields
		for (auto field : sync_fields) {
			// find field
			int nuclearFieldIdx = std::distance(n.fieldNames.begin(), 
				std::find(n.fieldNames.begin(), n.fieldNames.end(), field));
			int particleFieldIdx = std::distance(d.fieldNames.begin(), 
				std::find(d.fieldNames.begin(), d.fieldNames.end(), field));

			// send
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
#ifdef USE_MPI
					syncDataToStaticPartition(n.partition, send->data(), recv->data(), d.comm);
#else
					if constexpr (std::is_same<decltype(send), decltype(recv)>::value)
						*recv = *send;
#endif
				}, particleData[particleFieldIdx], nuclearData[nuclearFieldIdx]);
		}
	}

	/*! @brief function sending requiered hydro data from ParticlesDataType (attached data) to NuclearDataType (detached data)
	 * 
	 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
	 * @param sync_fields  names of field to be synced
	 */
	template<class ParticlesDataType>
	void syncHydroToNuclear(ParticlesDataType &d, const std::vector<std::string> &sync_fields) {
		syncDataToStaticPartition(d, d.nuclearData, sync_fields);
	}


	/*! @brief function sending requiered hydro data from NuclearDataType (detached data) to ParticlesDataType (attached data)
	 * 
	 * @param d            ParticlesDataType (attached data)
	 * @param n            NuclearDataType (detached data)
	 * @param sync_fields  names of field to be synced
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void syncDataFromStaticPartition(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
		auto nuclearData  = n.data();
		auto particleData = d.data();

		// send fields
		for (auto field : sync_fields) {
			// find field
			int nuclearFieldIdx = std::distance(n.fieldNames.begin(), 
				std::find(n.fieldNames.begin(), n.fieldNames.end(), field));
			int particleFieldIdx = std::distance(d.fieldNames.begin(), 
				std::find(d.fieldNames.begin(), d.fieldNames.end(), field));

			std::visit(
				[&d, &n](auto&& send, auto &&recv){
#ifdef USE_MPI
					syncDataFromStaticPartition(n.partition, send->data(), recv->data(), d.comm);
#else
					if constexpr (std::is_same<decltype(send), decltype(recv)>::value)
						*recv = *send;
#endif
				}, nuclearData[nuclearFieldIdx], particleData[particleFieldIdx]);
		}
	}

	/*! @brief function sending requiered hydro data from NuclearDataType (detached data) to ParticlesDataType (attached data)
	 * 
	 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
	 * @param sync_fields  names of field to be synced
	 */
	template<class ParticlesDataType>
	void syncNuclearToHydro(ParticlesDataType &d, const std::vector<std::string> &sync_fields) {
		syncDataFromStaticPartition(d, d.nuclearData, sync_fields);
	}
}