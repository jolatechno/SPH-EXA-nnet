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
 * @brief net86 constants definition.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

#include "nnet_util/CUDA/cuda.inl"

#include "../net14/net14_constants.hpp"

namespace nnet::net86::constants
{
static const int proton  = 0;
static const int neutron = 1;
static const int alpha   = 2;

// for net87
static const int electron = 86;

const static double pi =
    3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198938095257201065485863278865936153381827968230301952035301852968995773622599413891249721775283479131515574857242454150695950829533116861727855889075098381754637464939319;
const static double Kb       = 1.380658e-16;
const static double Na       = 6.022137e23;
const static double e2       = 2.306022645e-19;
const static double MevToErg = 9.648529392e17;

/*! @brief constant atomic number values */
DEVICE_DEFINE(inline static const std::array<double COMMA 87>,
              Z, =
                     {1 COMMA 0 COMMA 2 COMMA 6 COMMA 8 COMMA 10 COMMA 11 COMMA 12 COMMA 12 COMMA 10 COMMA 11 COMMA 12 COMMA 11 COMMA 10 COMMA 13 COMMA 14 COMMA 14 COMMA 12 COMMA 13 COMMA 14 COMMA 13 COMMA 12 COMMA 15 COMMA 16 COMMA 16 COMMA 14 COMMA 15 COMMA 16 COMMA 15 COMMA 14 COMMA 17 COMMA 18 COMMA 18 COMMA 16 COMMA 17 COMMA 18 COMMA 17 COMMA 16 COMMA 19 COMMA 20 COMMA 20 COMMA 18 COMMA 19 COMMA 20 COMMA 19 COMMA 18 COMMA 21 COMMA 22 COMMA 22 COMMA 20 COMMA 21 COMMA 22 COMMA 21 COMMA 20 COMMA 23 COMMA 24 COMMA 24 COMMA 22 COMMA 23 COMMA 24 COMMA 23 COMMA 22 COMMA 25 COMMA 26 COMMA 26 COMMA 24 COMMA 25 COMMA 26 COMMA 25 COMMA 24 COMMA 27 COMMA 28 COMMA 28 COMMA 26 COMMA 27 COMMA 28 COMMA 27 COMMA 26 COMMA 29 COMMA 30 COMMA 30 COMMA 28 COMMA 29 COMMA 30 COMMA 29 COMMA 28 COMMA /*Z=-1 for electrons*/
                      - 1};)

/*! @brief constant number of masses values */
DEVICE_DEFINE(inline static const std::array<double COMMA 87>,
              A, = {1 COMMA 1 COMMA 4 COMMA 12 COMMA 16 COMMA 20 COMMA 21 COMMA 24 COMMA 23 COMMA 21 COMMA 23 COMMA 22 COMMA 22 COMMA 22 COMMA 25 COMMA 28 COMMA 27 COMMA 25 COMMA 27 COMMA 26 COMMA 26 COMMA 26 COMMA 29 COMMA 32 COMMA 31 COMMA 29 COMMA 31 COMMA 30 COMMA 30 COMMA 30 COMMA 33 COMMA 36 COMMA 35 COMMA 33 COMMA 35 COMMA 34 COMMA 34 COMMA 34 COMMA 37 COMMA 40 COMMA 39 COMMA 37 COMMA 39 COMMA 38 COMMA 38 COMMA 38 COMMA 41 COMMA 44 COMMA 43 COMMA 41 COMMA 43 COMMA 42 COMMA 42 COMMA 42 COMMA 45 COMMA 48 COMMA 47 COMMA 45 COMMA 47 COMMA 46 COMMA 46 COMMA 46 COMMA 49 COMMA 52 COMMA 51 COMMA 49 COMMA 51 COMMA 50 COMMA 50 COMMA 50 COMMA 53 COMMA 56 COMMA 55 COMMA 53 COMMA 55 COMMA 54 COMMA 54 COMMA 54 COMMA 57 COMMA 60 COMMA 59 COMMA 57 COMMA 59 COMMA 58 COMMA 58 COMMA 58 COMMA /*A=0 for electrons*/
                    0};)

/*! @brief order of nuclear species */
const std::vector<int> speciesOrder = []()
{
    std::vector<int> speciesOrder_(A.size());
    std::iota(speciesOrder_.begin(), speciesOrder_.end(), 0);
    std::sort(speciesOrder_.begin(), speciesOrder_.end() - /*don't sort electrons*/ 1,
              [&](const int idx1, const int idx2)
              {
                  if (Z[idx1] < Z[idx2]) return true;
                  if (Z[idx1] > Z[idx2]) return false;

                  return A[idx1] < A[idx2];
              });

    return speciesOrder_;
}();

/*! @brief nuclear species names */
const std::vector<std::string> speciesNames = []()
{
    // unsorted nuclear species names
    const std::vector<std::string> unsortedSpeciesNames = {"p",    "n",    "4He",
                                                           "12C",  "16O",  "20Ne",
                                                           "21Na", "24Mg", "23Mg",
                                                           "21Ne", "23Na", "22Mg",
                                                           "22Na", "22Ne", "25Al",
                                                           "28Si", "27Si", "25Mg",
                                                           "27Al", "26Si", "26Al",
                                                           "26Mg", "29P",  "32S",
                                                           "31S",  "29Si", "31P",
                                                           "30S",  "30P",  "30Si",
                                                           "33Cl", "36Ar", "35Ar",
                                                           "33S",  "35Cl", "34Ar",
                                                           "34Cl", "34S",  "37K",
                                                           "40Ca", "39Ca", "37Ar",
                                                           "39K",  "38Ca", "38K",
                                                           "38Ar", "41Sc", "44Ti",
                                                           "43Ti", "41Ca", "43Sc",
                                                           "42Ti", "42Sc", "42Ca",
                                                           "45V",  "48Cr", "47Cr",
                                                           "45Ti", "47V",  "46Cr",
                                                           "46V",  "46Ti", "49Mn",
                                                           "52Fe", "51Fe", "49Cr",
                                                           "51Mn", "50Fe", "50Mn",
                                                           "50Cr", "53Co", "56Ni",
                                                           "55Ni", "53Fe", "55Co",
                                                           "54Ni", "54Co", "54Fe",
                                                           "57Cu", "60Zn", "59Zn",
                                                           "57Ni", "59Cu", "58Zn",
                                                           "58Cu", "58Ni", /*electrons*/ "e-"};

    std::vector<std::string> speciesNames_(unsortedSpeciesNames.size());

    for (size_t i = 0; i < unsortedSpeciesNames.size(); ++i)
        speciesNames_[i] = unsortedSpeciesNames[speciesOrder[i]];

    return speciesNames_;
}();

/*! @brief nuclear species index corresponding to net14 species */
const std::vector<int> net14SpeciesOrder = []()
{
    const int net14_n_species = nnet::net14::constants::A.size();
    const int net86_n_species = A.size();

    std::vector<int> net14SpeciesOrder_(net14_n_species);

    for (int i = 0; i < net14_n_species; ++i)
        for (int j = 0; j < net86_n_species; ++j)
            if (nnet::net14::constants::A[i] == A[j] && nnet::net14::constants::Z[i] == Z[j])
            {

                net14SpeciesOrder_[i] = j;

                break;

                if (j == net86_n_species - 1)
                    throw std::runtime_error(
                        "Couldn't find the species in net86 corresponding to a specific species in net14\n");
            }

    return net14SpeciesOrder_;
}();

/*! @brief nuclear species index corresponding to net14 ordered species */
const std::vector<std::vector<int>> net14AccumulatedSpeciesOrder = []()
{
    const int net14_n_species = nnet::net14::constants::A.size();
    const int net86_n_species = A.size();

    std::vector<std::vector<int>> net14AccumulatedSpeciesOrder_(net14_n_species);

    for (int i = 0; i < net14_n_species; ++i)
        for (int j = 0; j < net86_n_species; ++j)
            if (nnet::net14::constants::Z[i] == Z[j]) net14AccumulatedSpeciesOrder_[i].push_back(j);

    return net14AccumulatedSpeciesOrder_;
}();

// function for coulombian correction
template<typename Float>
HOST_DEVICE_FUN Float inline ggt1(const Float x)
{
    const Float a1 = -.898004;
    const Float b1 = .96786;
    const Float c1 = .220703;
    const Float d1 = -.86097;

    Float sqroot2x = std::sqrt(std::sqrt(x));
    return a1 * x + b1 * sqroot2x + c1 / sqroot2x + d1;
}
// function for coulombian correction
template<typename Float>
HOST_DEVICE_FUN Float inline glt1(const Float x)
{
    const Float a1 = -.5 * std::sqrt(3.);
    const Float b1 = .29561;
    const Float c1 = 1.9885;

    return a1 * x * std::sqrt(x) + b1 * std::pow(x, c1);
}

// reactant and products indices
DEVICE_DEFINE(
    static const inline int, main_reactant[157], = {// (-1 applied)
                                                    3 COMMA 3 COMMA 4 COMMA 0 COMMA 2 COMMA 3 COMMA 4 COMMA 5 COMMA 8 COMMA 11 COMMA 5 COMMA 12 COMMA 6 COMMA 6 COMMA 9 COMMA 7 COMMA 16 COMMA 19 COMMA 7 COMMA 20 COMMA 14 COMMA 14 COMMA 17 COMMA 15 COMMA 24 COMMA 27 COMMA 15 COMMA 28 COMMA 22 COMMA 22 COMMA 25 COMMA 23 COMMA 32 COMMA 35 COMMA 23 COMMA 36 COMMA 30 COMMA 30 COMMA 33 COMMA 31 COMMA 40 COMMA 43 COMMA 31 COMMA 44 COMMA 38 COMMA 38 COMMA 41 COMMA 39 COMMA 48 COMMA 51 COMMA 39 COMMA 52 COMMA 46 COMMA 46 COMMA 49 COMMA 47 COMMA 56 COMMA 59 COMMA 47 COMMA 60 COMMA 54 COMMA 54 COMMA 57 COMMA 55 COMMA 64 COMMA 67 COMMA 55 COMMA 68 COMMA 62 COMMA 62 COMMA 65 COMMA 63 COMMA 72 COMMA 75 COMMA 63 COMMA 76 COMMA 70 COMMA 70 COMMA 73 COMMA 71 COMMA 80 COMMA 83 COMMA 71 COMMA 84 COMMA 78 COMMA 78 COMMA 81 COMMA 9 COMMA 12 COMMA 13 COMMA 10 COMMA 17 COMMA 20 COMMA 21 COMMA 18 COMMA 25 COMMA 28 COMMA 29 COMMA 26 COMMA 33 COMMA 36 COMMA 37 COMMA 34 COMMA 41 COMMA 44 COMMA 45 COMMA 42 COMMA 49 COMMA 52 COMMA 53 COMMA 50 COMMA 57 COMMA 60 COMMA 61 COMMA 58 COMMA 65 COMMA 68 COMMA 69 COMMA 66 COMMA 73 COMMA 76 COMMA 77 COMMA 74 COMMA 81 COMMA 84 COMMA 85 COMMA 82 COMMA 5 COMMA 7 COMMA 15 COMMA 23 COMMA 31 COMMA 39 COMMA 47 COMMA 55 COMMA 63 COMMA 71 COMMA 5 COMMA 7 COMMA 15 COMMA 23 COMMA 31 COMMA 39 COMMA 47 COMMA 55 COMMA 63 COMMA 71 COMMA 5 COMMA 7 COMMA 15 COMMA 23 COMMA 31 COMMA 39 COMMA 47 COMMA 55 COMMA 63 COMMA 71};)
DEVICE_DEFINE(
    static const inline int, secondary_reactant[157],
    = {// (-1 applied)
       3 COMMA 4 COMMA 4 COMMA 0 COMMA 2 COMMA 2 COMMA 2 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 1 COMMA 1 COMMA 0 COMMA 1 COMMA 1 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 0 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2 COMMA 2};)
DEVICE_DEFINE(static const inline int, main_product[157], =
                                                              {// (-1 applied)
                                                               5 COMMA 7 COMMA 15 COMMA 0 COMMA 3 COMMA 4 COMMA 5 COMMA 6 COMMA 7 COMMA 8 COMMA 9 COMMA 10 COMMA 11 COMMA 12 COMMA 13 COMMA 14 COMMA 15 COMMA 16 COMMA 17 COMMA 18 COMMA 19 COMMA 20 COMMA 21 COMMA 22 COMMA 23 COMMA 24 COMMA 25 COMMA 26 COMMA 27 COMMA 28 COMMA 29 COMMA 30 COMMA 31 COMMA 32 COMMA 33 COMMA 34 COMMA 35 COMMA 36 COMMA 37 COMMA 38 COMMA 39 COMMA 40 COMMA 41 COMMA 42 COMMA 43 COMMA 44 COMMA 45 COMMA 46 COMMA 47 COMMA 48 COMMA 49 COMMA 50 COMMA 51 COMMA 52 COMMA 53 COMMA 54 COMMA 55 COMMA 56 COMMA 57 COMMA 58 COMMA 59 COMMA 60 COMMA 61 COMMA 62 COMMA 63 COMMA 64 COMMA 65 COMMA 66 COMMA 67 COMMA 68 COMMA 69 COMMA 70 COMMA 71 COMMA 72 COMMA 73 COMMA 74 COMMA 75 COMMA 76 COMMA 77 COMMA 78 COMMA 79 COMMA 80 COMMA 81 COMMA 82 COMMA 83 COMMA 84 COMMA 85 COMMA 12 COMMA 8 COMMA 10 COMMA 7 COMMA 20 COMMA 16 COMMA 18 COMMA 15 COMMA 28 COMMA 24 COMMA 26 COMMA 23 COMMA 36 COMMA 32 COMMA 34 COMMA 31 COMMA 44 COMMA 40 COMMA 42 COMMA 39 COMMA 52 COMMA 48 COMMA 50 COMMA 47 COMMA 60 COMMA 56 COMMA 58 COMMA 55 COMMA 68 COMMA 64 COMMA 66 COMMA 63 COMMA 76 COMMA 72 COMMA 74 COMMA 71 COMMA 84 COMMA 80 COMMA 82 COMMA 79 COMMA 7 COMMA 15 COMMA 23 COMMA 31 COMMA 39 COMMA 47 COMMA 55 COMMA 63 COMMA 71 COMMA 79 COMMA 8 COMMA 16 COMMA 24 COMMA 32 COMMA 40 COMMA 48 COMMA 56 COMMA 64 COMMA 72 COMMA 80 COMMA 10 COMMA 18 COMMA 26 COMMA 34 COMMA 42 COMMA 50 COMMA 58 COMMA 66 COMMA 74 COMMA 82};)
} // namespace nnet::net86::constants