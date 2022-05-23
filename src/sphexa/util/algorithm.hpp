#pragma once

#include <vector>
#include <parallel/algorithm>
#include <parallel/numeric>

#include <omp.h>

/*
Simple utilities
stolen from QuIDS (https://github.com/jolatechno/QuIDS)
*/

namespace utils {
	/// parallel iota
	template <class iteratorType, class valueType>
	void parallel_iota(iteratorType begin, iteratorType end, const valueType value_begin) {
		size_t distance = std::distance(begin, end);

		if (value_begin == 0) {
			#pragma omp parallel for 
			for (size_t i = 0; i < distance; ++i)
				begin[i] = i;
		} else
			#pragma omp parallel for 
			for (size_t i = 0; i < distance; ++i)
				begin[i] = value_begin + i;
	}

	/// linear partitioning algorithm into n partitions without an initialized index list in parallel
	template <class idIteratorType, class idType, class countIteratorType, class functionType>
	void parallel_generalized_partition_from_iota(idIteratorType idx_in, idIteratorType idx_in_end, idType const iotaOffset,
		countIteratorType offset, countIteratorType offset_end,
		functionType const partitioner) {

		int const n_segment = std::distance(offset, offset_end) - 1;
		long long int const id_end = std::distance(idx_in, idx_in_end);

		// limit values
		offset[0] = 0;
		offset[n_segment] = id_end;

		if (n_segment == 1) {
			parallel_iota(idx_in, idx_in_end, iotaOffset);
			return;
		}
		if (id_end == 0) {
			std::fill(offset, offset_end, 0);
			return;
		}

		// number of threads
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		std::vector<size_t> count(n_segment*num_threads, 0);

		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();

			long long int begin = id_end*thread_id/num_threads;
			long long int end = id_end*(thread_id + 1)/num_threads;
			for (long long int i = end + iotaOffset - 1; i >= begin + iotaOffset; --i) {
				auto key = partitioner(i);
				++count[key*num_threads + thread_id];
			}
		}
		
		__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			
			long long int begin = id_end*thread_id/num_threads;
			long long int end = id_end*(thread_id + 1)/num_threads;
			for (long long int i = begin + iotaOffset; i < end + iotaOffset; ++i) {
				auto key = partitioner(i);
				idx_in[--count[key*num_threads + thread_id]] = i;
			}
		}

		#pragma omp parallel for 
		for (int i = 1; i < n_segment; ++i)
			offset[i] = count[i*num_threads];
	}
}