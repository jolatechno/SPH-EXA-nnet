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
 * @brief Definition of CUDA GPU data.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include "../../../util/CUDA/cuda.inl"

#include <vector>
#include <array>
#include <memory>

#include "sph/traits.hpp"

#include "cstone/fields/field_states.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/fields/concatenate.hpp"

#include "cstone/util/reallocate.hpp"
#include "cstone/util/array.hpp"

#include <thrust/device_vector.h>

namespace sphexa::sphnnet
{
/*! @brief device nuclear data class for nuclear network */
template<typename Float, typename Int>
class DeviceNuclearDataType : public cstone::FieldStates<DeviceNuclearDataType<Float, Int>>
{
public:
    //! maximum number of nuclear species
    static const int maxNumSpecies = 100;
    //! actual number of nuclear species
    int numSpecies = 0;

    template<class FType>
    using DevVector = thrust::device_vector<FType>;

    // types
    using RealType = Float;
    using KeyType  = Int;
    using Tmass    = float;
    using XM1Type  = float;

    DevVector<RealType>                             c;      // speed of sound
    DevVector<RealType>                             p;      // pressure
    DevVector<RealType>                             cv;     // cv
    DevVector<RealType>                             u;      // internal energy
    DevVector<RealType>                             dpdT;   // dP/dT
    DevVector<RealType>                             rho;    // density
    DevVector<RealType>                             temp;   // temperature
    DevVector<RealType>                             rho_m1; // previous density
    DevVector<Tmass>                                m;      // mass
    DevVector<RealType>                             dt;     // timesteps
    util::array<DevVector<RealType>, maxNumSpecies> Y;      // vector of nuclear abundances
    mutable thrust::device_vector<RealType>         buffer; // solver buffer

    //! base hydro fieldNames (every nuclear species is named "Yn")
    inline static constexpr auto fieldNames =
        concat(enumerateFieldNames<"Y", maxNumSpecies>(), std::array<const char*, 10>{
                                                              "dt",
                                                              "c",
                                                              "p",
                                                              "cv",
                                                              "u",
                                                              "dpdT",
                                                              "m",
                                                              "temp",
                                                              "rho",
                                                              "rho_m1",
                                                          });

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        return std::tuple_cat(dataTuple_helper(std::make_index_sequence<maxNumSpecies>{}),
                              std::tie(dt, c, p, cv, u, dpdT, m, temp, rho, rho_m1));
    }

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        using FieldType = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                       FieldVector<int>*, FieldVector<uint64_t>*>;

        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    //!  resize the number of particules
    void resize(size_t size)
    {
        double growthRate = 1;
        auto   data_      = data();

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                // actually resize
                std::visit([&](auto& arg) { reallocate(*arg, size, growthRate); }, data_[i]);
            }
        }
    }

private:
    template<size_t... Is>
    auto dataTuple_helper(std::index_sequence<Is...>)
    {
        return std::tie(Y[Is]...);
    }
};

// used templates:
template class DeviceNuclearDataType<double, size_t>;
template class DeviceNuclearDataType<float, size_t>;
} // namespace sphexa::sphnnet