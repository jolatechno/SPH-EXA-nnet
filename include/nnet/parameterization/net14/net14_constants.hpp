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
 * @brief net14 constants definition.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>

#include "nnet_util/CUDA/cuda.inl"

namespace nnet::net14::constants
{
const static double pi =
    3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198938095257201065485863278865936153381827968230301952035301852968995773622599413891249721775283479131515574857242454150695950829533116861727855889075098381754637464939319;
const static double Kb       = 1.380658e-16;
const static double Na       = 6.022137e23;
const static double e2       = 2.306022645e-19;
const static double MevToErg = 9.648529392e17;

/*! @brief constant atomic number values */
DEVICE_DEFINE(
    inline static const std::array<double COMMA 14>, Z,
    = {2 COMMA 6 COMMA 8 COMMA 10 COMMA 12 COMMA 14 COMMA 16 COMMA 18 COMMA 20 COMMA 22 COMMA 24 COMMA 26 COMMA 28 COMMA 30};)

/*! @brief constant number of masses values */
DEVICE_DEFINE(
    inline static const std::array<double COMMA 14>, A,
    = {4 COMMA 12 COMMA 16 COMMA 20 COMMA 24 COMMA 28 COMMA 32 COMMA 36 COMMA 40 COMMA 44 COMMA 48 COMMA 52 COMMA 56 COMMA 60};)

/*! @brief order of nuclear species */
const std::vector<double> speciesOrder = []()
{
    std::vector<double> speciesOrder_(A.size());
    std::iota(speciesOrder_.begin(), speciesOrder_.end(), 0);
    std::sort(speciesOrder_.begin(), speciesOrder_.end(),
              [&](const int idx1, const int idx2)
              {
                  if (Z[idx1] < Z[idx2]) return true;
                  if (Z[idx1] > Z[idx2]) return false;

                  return A[idx1] < A[idx2];
              });

    return speciesOrder_;
}();

/*! @brief nuclear species names */
const std::vector<std::string> speciesNames = {"4He",  "12C",  "16O",  "20Ne", "24Mg", "28Si", "32S",
                                               "36Ar", "40Ca", "44Ti", "48Cr", "52Fe", "56Ni", "60Zn"};
} // namespace nnet::net14::constants