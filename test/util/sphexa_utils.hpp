#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <variant>
#include <type_traits>
#include <array>

#include "sph/traits.hpp"
#include "cstone/util/array.hpp"

#include <iostream>

#include "nnet/CUDA/cuda.inl"

#if COMPILE_DEVICE
    #include <thrust/device_vector.h>
#endif

namespace sphexa {
	namespace fileutils {
		/*! @brief write fields as columns to an ASCII file
		 *
		 * @tparam  T              field type
		 * @tparam  Separators
		 * @param   firstIndex     first field index to write
		 * @param   lastIndex      last field index to write
		 * @param   path           the file name to write to
		 * @param   append         append or overwrite if file already exists
		 * @param   fields         pointers to field array, each field is a column
		 * @param   separators     arbitrary number of separators to insert between columns, eg '\t', std::setw(n), ...
		 */
		template<class... T, class... Separators>
		void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append, const std::vector<std::variant<T*...>>& fields, Separators&&... separators) {
		    std::ios_base::openmode mode;
		    if (append) { mode = std::ofstream::app; }
		    else { mode = std::ofstream::out; }

		    std::ofstream dumpFile(path, mode);

		    if (dumpFile.is_open())
		    {
		        for (size_t i = firstIndex; i < lastIndex; ++i)
		        {
		            for (auto field : fields)
		            {
		                [[maybe_unused]] std::initializer_list<int> list{(dumpFile << separators, 0)...};
		                std::visit([&dumpFile, i](auto& arg) { 
			                	dumpFile << arg[i];
			                }, field);
		            }
		            dumpFile << std::endl;
		        }
		    }
		    else { throw std::runtime_error("Can't open file at path: " + path); }

		    dumpFile.close();
		}
	}


#if COMPILE_DEVICE
    template<class ThrustVec>
    typename ThrustVec::value_type* rawPtr(ThrustVec& p)
    {
        assert(p.size() && "cannot get pointer to unallocated device vector memory");
        return thrust::raw_pointer_cast(p.data());
    }

    template<class ThrustVec>
    const typename ThrustVec::value_type* rawPtr(const ThrustVec& p)
    {
        assert(p.size() && "cannot get pointer to unallocated device vector memory");
        return thrust::raw_pointer_cast(p.data());
    }
#endif



    template<class Dataset, std::enable_if_t<not HaveGpu<typename Dataset::AcceleratorType>{}, int> = 0>
    void transferToDevice(Dataset&, size_t, size_t, const std::vector<std::string>&)
    {
    }

    template<class Dataset, std::enable_if_t<not HaveGpu<typename Dataset::AcceleratorType>{}, int> = 0>
    void transferToHost(Dataset&, size_t, size_t, const std::vector<std::string>&)
    {
    }



    template<class DataType, std::enable_if_t<HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
    void transferToDevice(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
    {
#if COMPILE_DEVICE
        auto hostData = d.data();
        auto deviceData = d.devData.data();

        auto launchTransfer = [first, last](const auto* hostField, auto* deviceField) {
            using Type1 = std::decay_t<decltype(*hostField)>;
            using Type2 = std::decay_t<decltype(*deviceField)>;

            assert(hostField->size() > 0);
            assert(deviceField->size() > 0);
            
            if constexpr (std::is_same_v<typename Type1::value_type, typename Type2::value_type>) {
                size_t transferSize = (last - first) * sizeof(typename Type1::value_type);
                // CHECK_CUDA_ERR(
                gpuErrchk(
                    cudaMemcpy(
                    rawPtr(*deviceField) + first, hostField->data() + first, transferSize, cudaMemcpyHostToDevice));
            } else 
                throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device");
        };

        for (const auto& field : fields)
        {
            for (auto it = DataType::fieldNames.begin(); it != DataType::fieldNames.end(); ++it) 
                if (*it == field) {
                    int fieldIdx = std::distance(DataType::fieldNames.begin(), it);
                    std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
                }
        }
        // replacing :
        /*
        for (const auto& field : fields)
        {
            int fieldIdx =
                std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
            std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
        }
        */
#endif
    }

    template<class DataType, std::enable_if_t<HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
    void transferToHost(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
    {
#if COMPILE_DEVICE
        auto hostData   = d.data();
        auto deviceData = d.devData.data();

        auto launchTransfer = [first, last](auto* hostField, const auto* deviceField) {
            using Type1 = std::decay_t<decltype(*hostField)>;
            using Type2 = std::decay_t<decltype(*deviceField)>;

            assert(hostField->size() > 0);
            assert(deviceField->size() > 0);
        
            if constexpr (std::is_same_v<typename Type1::value_type, typename Type2::value_type>) {
                size_t transferSize = (last - first) * sizeof(typename Type1::value_type);
                // CHECK_CUDA_ERR(
                gpuErrchk(
                    cudaMemcpy(
                    hostField->data() + first, rawPtr(*deviceField) + first, transferSize, cudaMemcpyDeviceToHost));
            } else 
                throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device");
        };

        for (const auto& field : fields)
        {
            for (auto it = DataType::fieldNames.begin(); it != DataType::fieldNames.end(); ++it) 
                if (*it == field) {
                    int fieldIdx = std::distance(DataType::fieldNames.begin(), it);
                    std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
                }
        }
        // replacing :
        /*
        for (const auto& field : fields)
        {
            int fieldIdx =
                std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
            std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
        }
        */
#endif
    }

} // namespace sphexa