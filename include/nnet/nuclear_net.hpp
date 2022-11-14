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
 * @brief Main nuclear-net header, containing utility and integration functions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include <cmath> // factorial

#include "nnet_util/CUDA/cuda.inl"
#include "nnet_util/eigen.hpp"

namespace nnet
{
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
constants :
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
namespace constants
{
//! initial nuclear timestep
const double initialDt = 1e-5;

//! theta for the implicit method
const double theta = 0.8;

//! minimum temperature at which we compute the nuclear network
const double minTemp = 1e8;
//! minimum density at which we compute the nuclear network
const double minRho = 1e5;

//! maximum timestep
const double maxDt = 1e-2;
//! maximum timestep evolution
const double maxDtStep = 2;
//! maximum negative timestep evolution
const double minDtStep = 1e-2;
//! timestep jump when a nan is in the solution
const double nanDtStep = 2e-1;

//! relative temperature variation target of the implicit solver
const double dTOverTempTarget = 4e-3;
//! relative temperature variation tolerance of the implicit solver
const double dToverTempTol = 4;

//! the value that is considered null inside a system
const double epsilonSystem = 1e-40;
//! the value that is considered null inside a state
const double epsilonVector = 1e-16;

namespace NR
{
//! maximum timestep
const double maxDt = 1e-2;

//! relative temperature variation target of the implicit solver
const double dTOverTempTarget = 1e-2;
//! relative temperature variation tolerance of the implicit solver
const double dTOverTempTol = 4;

//! minimum number of newton raphson iterations
const int minIt = 1;
//! maximum number of newton raphson iterations
const int maxIt = 11;
//! tolerance for the correction to break out of the newton raphson loop
const double itTol = 1e-7;
} // namespace NR

namespace substep
{
//! timestep tolerance for substepping
const double dtTol = 1e-6;

//! ratio of the nuclear timestep and "super timestep" to jump to NSE
const double dtNseTol = 0; // 1e-8; // !!!! useless for now
} // namespace substep
} // namespace constants

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) reaction class:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*! @brief reaction class */
struct Reaction
{
    /*! @brief class representing a product or reactant */
    struct ReactantProduct
    {
        int speciesId, numConsumed = 1;
    };

    std::vector<ReactantProduct> reactants, products;

    // reaction class print operator
    friend std::ostream& operator<<(std::ostream& os, const Reaction& r);
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) reaction reference class (referenced from a bigger vector):
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*! @brief class referencing a reaction */
struct ReactionReference
{
    HOST_DEVICE_FUN ReactionReference() {}

    /*! @brief class simulating a vector from a pointer */
    template<class T>
    class VectorReference
    {
    private:
        const T* ptr   = nullptr;
        size_t   size_ = 0;

    public:
        HOST_DEVICE_FUN VectorReference() {}
        HOST_DEVICE_FUN VectorReference(const T* ptr_, size_t size)
            : ptr(ptr_)
            , size_(size)
        {
        }
        size_t inline size() const { return size_; }
        HOST_DEVICE_FUN const T* begin() const { return ptr; }
        HOST_DEVICE_FUN const T* end() const { return ptr + size_; }
        HOST_DEVICE_FUN const T& operator[](int i) { return ptr[i]; }
    };

    VectorReference<Reaction::ReactantProduct> reactants, products;

    // reaction class print operator
    friend std::ostream& operator<<(std::ostream& os, const ReactionReference& r);
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) reaction list class:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*! @brief reaction pointer class (pointer to contigous buffers rather then a vector of reaction) */
class ReactionList
{
private:
    // pointer to each reaction
    std::vector<int> reactantBegin = {0};
    std::vector<int> productBegin  = {};

    // actual vectors
    std::vector<Reaction::ReactantProduct> reactantProduct = {};

    friend class PtrReactionList;

public:
    ReactionList() {}
    ReactionList(std::vector<Reaction> const& reactions)
    {
        for (auto& Reaction : reactions)
            pushBack(Reaction);
    }

    /*! @brief push back reaction to list */
    void inline pushBack(Reaction const& Reaction)
    {
        reactantProduct.insert(reactantProduct.end(), Reaction.reactants.begin(), Reaction.reactants.end());
        reactantProduct.insert(reactantProduct.end(), Reaction.products.begin(), Reaction.products.end());

        productBegin.push_back(reactantBegin.back() + Reaction.reactants.size());
        reactantBegin.push_back(productBegin.back() + Reaction.products.size());
    }

    /*! @brief access reaction from reacton list */
    ReactionReference inline operator[](int i) const
    {
        ReactionReference Reaction;

        Reaction.reactants = ReactionReference::VectorReference(reactantProduct.data() + reactantBegin[i],
                                                                productBegin[i] - reactantBegin[i]);
        Reaction.products  = ReactionReference::VectorReference(reactantProduct.data() + productBegin[i],
                                                                reactantBegin[i + 1] - productBegin[i]);

        return Reaction;
    }

    /*! @brief access reaction list size */
    size_t inline size() const { return productBegin.size(); }
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) offloadable reaction list class:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

// forward declaration
class GPUReactionList;

/*! @brief reaction pointer class (pointer to contigous buffers rather then a vector of reaction) */
class PtrReactionList
{
protected:
    // pointer to each reaction
    const int *                      reactantBegin, *productBegin;
    const Reaction::ReactantProduct* reactantProduct;
    int                              numReactions;

    // forward declaration
    friend GPUReactionList;
    friend GPUReactionList moveToGpu(const PtrReactionList& reactions);
    friend void inline free(GPUReactionList& reactions);

public:
    PtrReactionList() {}
    PtrReactionList(ReactionList const& other)
    {
        numReactions = other.size();

        reactantBegin   = other.reactantBegin.data();
        productBegin    = other.productBegin.data();
        reactantProduct = other.reactantProduct.data();
    }

    /*! @brief access reaction from reacton list */
    HOST_DEVICE_FUN ReactionReference inline operator[](int i) const
    {
        ReactionReference Reaction;

        Reaction.reactants =
            ReactionReference::VectorReference(reactantProduct + reactantBegin[i], productBegin[i] - reactantBegin[i]);
        Reaction.products = ReactionReference::VectorReference(reactantProduct + productBegin[i],
                                                               reactantBegin[i + 1] - productBegin[i]);

        return Reaction;
    }

    /*! @brief access reaction list size */
    HOST_DEVICE_FUN size_t inline size() const { return numReactions; }
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print functions:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

// print reaction reference
std::ostream inline& operator<<(std::ostream& os, const ReactionReference& r)
{
    // print reactant
    for (auto [reactant_id, n_reactant_consumed] : r.reactants)
        os << n_reactant_consumed << "*[" << reactant_id << "] ";

    os << " ->  ";

    // print products
    for (auto [product_id, n_product_produced] : r.products)
        os << n_product_produced << "*[" << product_id << "] ";
    return os;
}

// print reaction
std::ostream inline& operator<<(std::ostream& os, const Reaction& r)
{
    // print reactant
    for (auto [reactant_id, n_reactant_consumed] : r.reactants)
        os << n_reactant_consumed << "*[" << reactant_id << "] ";

    os << " ->  ";

    // print products
    for (auto [product_id, n_product_produced] : r.products)
        os << n_product_produced << "*[" << product_id << "] ";
    return os;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
utils functions:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

namespace util
{
/*! @brief clip the values in a Vector
 *
 * @param X          buffer to be cliped
 * @param dimension  size of buffer to be cliped
 * @param epsilon    small tolerence value such that any value x |x|<epsilon is considered equal to zero
 */
template<typename Float>
HOST_DEVICE_FUN void inline clip(Float* X, const int dimension, const Float epsilon)
{
    for (int i = 0; i < dimension; ++i)
        if (X[i] <= epsilon) // if (std::abs(X(i)) <= epsilon)
            X[i] = 0;
}

/*! @brief function to check if there is a nan in both temperature and abundances
 *
 * @param T          temperature to be checked for nan
 * @param Y          buffer to be checked for nan
 * @param dimension  size of buffer to be checked for nan
 */
template<typename Float>
HOST_DEVICE_FUN bool inline containsNan(const Float T, const Float* Y, const int dimension)
{
    if (std::isnan(T)) return true;

    for (int i = 0; i < dimension; ++i)
        if (std::isnan(Y[i])) return true;

    return false;
}

/*! @brief create a first order system from a list of reaction
 *
 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y
 *
 * @param reactions  reaction list
 * @param rates      reaction rates
 * @param rho        density
 * @param Y          abundances
 * @param dY         temporal derivative of abundances to be populated
 * @param dimension  number of nuclear species
 */
template<typename Float>
HOST_DEVICE_FUN void inline derivativesFromReactions(const PtrReactionList& reactions, const Float* rates,
                                                     const Float rho, const Float* Y, Float* dY, const int dimension)
{
    // fill with zero
    for (int i = 0; i < dimension; ++i)
        dY[i] = 0.;

    const int numReactions = reactions.size();
    for (int i = 0; i < numReactions; ++i)
    {
        const auto& Reaction = reactions[i];
        Float       rate     = rates[i];

        // compute rate and order
        for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants)
        {
            // divide by factorial
            if (n_reactant_consumed != 1) rate /= std::tgamma(n_reactant_consumed + 1);

            // multiply by abundance
            rate *= std::pow(Y[reactant_id] * rho, n_reactant_consumed);
        }

        // correct for rho
        rate /= rho;

        // insert consumption rates
        for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants)
            dY[reactant_id] -= rate * n_reactant_consumed;

        // insert production rates
        for (const auto [product_id, n_product_produced] : Reaction.products)
            dY[product_id] += rate * n_product_produced;
    }
}

/*! @brief create a first order system from a list of reaction
 *
 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y
 *
 * @param reactions  reaction list
 * @param rates      reaction rates
 * @param rho        density
 * @param Y          abundances
 * @param M          matrix to be populated
 * @param dimension  number of nuclear species
 */
template<typename Float>
HOST_DEVICE_FUN void inline firstOrderDYFromReactions(const PtrReactionList& reactions, const Float* rates,
                                                      const Float rho, Float const* Y, Float* M, const int dimension)
{
    // fill matrix with zero
    for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension; ++j)
            M[(i + 1) + (dimension + 1) * (j + 1)] = 0.;

    const int numReactions = reactions.size();
    for (int i = 0; i < numReactions; ++i)
    {
        const auto& Reaction = reactions[i];
        Float       rate     = rates[i];

        // compute rate
        for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants)
        {
            // divide by factorial
            if (n_reactant_consumed != 1) rate /= std::tgamma(n_reactant_consumed + 1);

            // multiply by abundance
            rate *= std::pow(Y[reactant_id] * rho, n_reactant_consumed - 1);
        }

        if (rate > (Float)constants::epsilonSystem)
            for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants)
            {
                // compute rate
                Float this_rate = rate;
                for (auto& [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
                    // multiply by abundance
                    if (other_reactant_id != reactant_id) this_rate *= Y[other_reactant_id] * rho;

                // insert consumption rates
                for (const auto [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
                    M[(other_reactant_id + 1) + (dimension + 1) * (reactant_id + 1)] -=
                        this_rate * other_n_reactant_consumed;

                // insert production rates
                for (auto const [product_id, n_product_produced] : Reaction.products)
                    M[(product_id + 1) + (dimension + 1) * (reactant_id + 1)] += this_rate * n_product_produced;
            }
    }
}
} // namespace util

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Interfaces definitions:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

//! @brief EOS output struct.
template<typename Float>
struct eos_struct
{
    HOST_DEVICE_FUN eos_struct() {}
    HOST_DEVICE_FUN ~eos_struct() {}

    Float cv, dpdT, p;
    Float cp, c, u;

    Float dse, dpe, dsp;
    Float cv_gaz, cp_gaz, c_gaz;

    Float dudYe;
};

//! @brief functor interface to compute EOS
template<typename Float>
class EosFunctor
{
public:
    /*! @brief Computes EOS.
     *
     * @param Y    molar proportions
     * @param T    temperature
     * @param rho  density
     *
     * Returns ideal gas EOS output struct.
     */
    HOST_DEVICE_FUN eos_struct<Float> inline virtual operator()(const Float* Y, const Float T, const Float rho) const
    {
        eos_struct<Float> res;
        return res;
    }
};

//! @brief functor interface to compute nuclear reactions rates
template<typename Float>
class ComputeReactionRatesFunctor
{
public:
    ComputeReactionRatesFunctor() {}

    /*! @brief computes rates.
     *
     * @param Y              molar fractions
     * @param T             temperature
     * @param rho           density
     * @param eos_struct    eos struct to populate
     * @param corrected_BE  will be populated by binding energies, corrected by coulombien terms
     * @param rates         will be populated with reaction rates
     * @param drates        will be populated with the temperature derivatives of reaction rates
     */
    HOST_DEVICE_FUN void inline virtual operator()(const Float* Y, const Float T, const Float rho,
                                                   const eos_struct<Float>& eos_struct, Float* corrected_BE,
                                                   Float* rates, Float* drates) const
    {
    }
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
First simple direct solver:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*! @brief generate the system to be solve (with rates computed at a specific "guess")
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 */
template<typename Float>
HOST_DEVICE_FUN void inline prepareSystemFromGuess(const int dimension, Float* Mp, Float* RHS, Float* rates,
                                                   const PtrReactionList&                    reactions,
                                                   const ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                                                   const Float* Y, const Float T, const Float* Y_guess,
                                                   const Float T_guess, const Float rho, const Float drho_dt,
                                                   const eos_struct<Float>& eos_struct, const Float dt)
{
    /* -------------------
    Solves d{Y, T}/dt = M'*Y using eigen:

<=>                               D{T, Y} = Dt*(M + theta*dM/dT*DT)*{T_in,Y_in} + theta*Dt*Myy*D{T,Y})
<=> (I - Dt*theta*(Myy + dM/dT*Y))*D{T, Y} = Dt*M*{T_in,Y_in}

    Energy equation:

    dT/dt*cv = value_1*T + (dY/dt).BE
<=> DT*cv = value_1*(T + theta*DT) + DY.BE
<=> DT*(cv - theta*value_1) - DY.BE = value_1*T
    ------------------- */
#if !DEVICE_CODE
    if (dt == 0)
    {
        std::string error = "Zero timestep in nuclear network\n";
        error += "\tT=" + std::to_string(T) + ",\tTguess=" + std::to_string(T_guess) + "\n";
        error += "\trho=" + std::to_string(rho) + "\tdrho/dt=" + std::to_string(drho_dt) + "\n";
        error += "\tdP/dT=" + std::to_string(eos_struct.dpdT) + ", cv=" + std::to_string(eos_struct.cv) + "\n";
        error += "\tY=";
        for (int i = 0; i < dimension; ++i)
            error += std::to_string(Y[i]) + " ";
        error += "\n\tYguess=";
        for (int i = 0; i < dimension; ++i)
            error += std::to_string(Y_guess[i]) + " ";

        throw std::runtime_error(error);
    }
#endif

    Float* BE        = RHS + 1;
    Float* drates_dT = Mp + dimension + 1;

    // construct BE and rates in plance
    construct_rates_BE(Y_guess, T_guess, rho, eos_struct, BE, rates, drates_dT);
    // include rate derivative
    util::derivativesFromReactions(reactions, drates_dT, rho, Y_guess, &Mp[1], dimension);
    // swap
    for (int i = 0; i < dimension; ++i)
        Mp[0 + (dimension + 1) * (i + 1)] = -BE[i] / eos_struct.cv;

    // compute RHS
    util::derivativesFromReactions(reactions, rates, rho, Y_guess, &RHS[1], dimension);
    for (int i = 0; i < dimension; ++i)
        RHS[i + 1] *= dt;
    // correct RHS based on the derivative of rates
    for (int i = 0; i < dimension; ++i)
    {
        //               __*Dt = __*(next_T - T_guess) = __*(next_T - T + T - T_guess) = __*(next_T - T) - __*(T_guess -
        //               T)
        // <=> -__*theta*dt*Dt = ... - __*theta*dt*(T_guess - T)
        RHS[i + 1] += -constants::theta * dt * Mp[(i + 1) + 0] * (T_guess - T);

        // correct rate derivative
        Mp[(i + 1) + 0] *= -constants::theta * dt;
    }

    // main matrix part
    util::firstOrderDYFromReactions(reactions, rates, rho, Y_guess, Mp, dimension);
    for (int i = 0; i < dimension; ++i)
    {
        //     dY = ... + theta*dt*Mp*(next_Y - Y_guess) = ... + theta*dt*Mp*(next_Y - Y + Y - Y_guess) = ... +
        //     theta*dt*Mp*dY - theta*dt*Mp*(Y_guess - Y)
        // <=> dY*(I - theta*dt*Mp) = ... - theta*Mp*dt*(Y_guess - Y)
        Float RHS_correction = 0;
        for (int j = 0; j < dimension; ++j)
            RHS_correction += Mp[(i + 1) + (dimension + 1) * (j + 1)] * (Y_guess[j] - Y[j]);
        RHS[i + 1] += -constants::theta * dt * RHS_correction;
    }
    for (int i = 0; i < dimension; ++i)
    {
        // diagonal terms
        Mp[(i + 1) + (dimension + 1) * (i + 1)] = 1. - constants::theta * dt * Mp[(i + 1) + (dimension + 1) * (i + 1)];

        // other terms
        for (int j = 0; j < dimension; ++j)
            if (i != j)
                Mp[(i + 1) + (dimension + 1) * (j + 1)] =
                    -constants::theta * dt * Mp[(i + 1) + (dimension + 1) * (j + 1)];
    }

    // compute value1
    const Float drho    = drho_dt * dt;
    const Float value_1 = eos_struct.dpdT * drho / (rho * rho);

    // energy equation
    RHS[0] = T * value_1 / eos_struct.cv;
    Mp[0]  = 1 - constants::theta * value_1 / eos_struct.cv;
}

/*! @brief second part after solving the system (generated in "prepareSystemFromGuess")
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 */
template<typename Float>
HOST_DEVICE_FUN void inline finalizeSystem(const int dimension, const Float* Y, const Float T, Float* next_Y,
                                           Float& next_T, const Float* DY_T)
{
    // increment values
    for (int i = 0; i < dimension; ++i)
        next_Y[i] = Y[i] + DY_T[i + 1];

    // update temperature
    next_T = T + DY_T[0];
}

// actual solver:
/*! @brief solves a system non-iteratively (with rates computed at a specific "guess")
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 * solves non-iteratively and partialy implicitly the system represented by M (computed at a specific "guess")
 */
template<typename Float>
void inline solveSystemFromGuess(const int dimension, Float* Mp, Float* RHS, Float* DY_T, Float* rates,
                                 const PtrReactionList&                    reactions,
                                 const ComputeReactionRatesFunctor<Float>& construct_rates_BE, const Float* Y,
                                 const Float T, const Float* Y_guess, const Float T_guess, Float* next_Y, Float& next_T,
                                 const Float rho, const Float drho_dt, const eos_struct<Float>& eos_struct,
                                 const Float dt)
{
    if (rho < constants::minRho || T < constants::minTemp)
    {
        for (int i = 0; i < dimension; ++i)
            next_Y[i] = Y[i];
        next_T = T;
    }
    else
    {
        // generate system
        prepareSystemFromGuess(dimension, Mp, RHS, DY_T, rates, reactions, construct_rates_BE, Y, T, Y_guess, T_guess,
                               rho, drho_dt, eos_struct, dt);

        // solve M*D{T, Y} = RHS
        eigen::solve(Mp, RHS, DY_T, dimension + 1, (Float)constants::epsilonSystem);

        // finalize
        return finalizeSystem(dimension, Y, T, next_Y, next_T, DY_T);
    }
}

// actual solver:
/*! @brief solves a system non-iteratively.
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 * solves non-iteratively and partialy implicitly the system represented by M
 */
template<typename Float>
void inline solveSystem(const int dimension, const PtrReactionList& reactions,
                        const ComputeReactionRatesFunctor<Float>& construct_rates_BE, const Float* Y, const Float T,
                        Float* next_Y, Float& next_T, const Float cv, const Float rho, const Float value_1,
                        const Float dt)
{

    eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);
    eigen::Vector<Float> RHS(dimension + 1);
    eigen::Vector<Float> DY_T(dimension + 1);
    std::vector<Float>   rates(dimension * dimension);

    solveSystemFromGuess(dimension, Mp.data(), RHS.data(), DY_T.data(), rates.data(), reactions, construct_rates_BE, Y,
                         T, Y, T, next_Y, next_T, cv, rho, value_1, dt);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Iterative solver:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*! @brief generate the system to be solve for the iterative solver
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 */
template<typename Float>
HOST_DEVICE_FUN void inline prepareSystemNR(const int dimension, Float* Mp, Float* RHS, Float* rates,
                                            const PtrReactionList&                    reactions,
                                            const ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                                            const EosFunctor<Float>& eos, const Float* Y, Float T, Float* final_Y,
                                            Float final_T, const Float rho, const Float drho_dt, Float& dt, const int i)
{
    // copy if first iteration
    if (i <= 1)
    {
        for (int j = 0; j < dimension; ++j)
            final_Y[j] = Y[j];
        final_T = T;
    }

    // compute n+theta values
    Float T_theta = (1 - constants::theta) * T + constants::theta * final_T;
    for (int j = 0; j < dimension; ++j)
        final_Y[j] = (1 - constants::theta) * Y[j] + constants::theta * final_Y[j];

    // compute eos
    auto eos_struct = eos(final_Y, T_theta, rho);

    // generate system
    prepareSystemFromGuess(dimension, Mp, RHS, rates, reactions, construct_rates_BE, Y, T, final_Y, T_theta, rho,
                           drho_dt, eos_struct, dt);
}

/*! @brief second part after solving the system (generated in "prepareSystemNR")
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 */
template<typename Float>
HOST_DEVICE_FUN bool inline finalizeSystemNR(const int dimension, const Float* Y, const Float T, Float* final_Y,
                                             Float& final_T, const Float* DY_T, Float& dt, Float& used_dt, int& i)
{

    Float last_T = final_T;
    finalizeSystem(dimension, Y, T, final_Y, final_T, DY_T);

    // check for garbage
    if (util::containsNan(final_T, final_Y, dimension) || final_T < 0)
    {
        // set timestep
        dt *= constants::nanDtStep;

        // jump back
        i       = 0;
        used_dt = 0.;
        return false;
    }

    // break condition
    Float dT_T = std::abs((final_T - T) / final_T);
    if (i >= constants::NR::minIt && dT_T > constants::NR::dTOverTempTarget * constants::NR::dTOverTempTol)
    {
        // set timestep
        dt *= constants::NR::dTOverTempTol / dT_T;

        // jump back
        i       = 0;
        used_dt = 0.;
        for (int j = 0; j < dimension; ++j)
            final_Y[j] = Y[j];
        final_T = T;
        return false;
    }

    // cleanup Vector
    util::clip(final_Y, dimension, (Float)nnet::constants::epsilonVector);

    // return condition
    Float correction = std::abs((final_T - last_T) / final_T);
    if ((i >= constants::NR::minIt && correction < constants::NR::itTol) || i >= constants::NR::maxIt)
    {
        // mass and temperature variation
        Float dT_T = std::abs((final_T - T) / final_T);

        // timestep tweeking
        used_dt = dt;
        dt      = (dT_T == 0 ? constants::maxDtStep : constants::NR::dTOverTempTarget / dT_T) * used_dt;
        dt      = std::min(dt, used_dt * (Float)constants::maxDtStep);
        dt      = std::max(dt, used_dt * (Float)constants::minDtStep);
        dt      = std::min(dt, (Float)constants::NR::maxDt);

        return true;
    }

    // continue the loop
    used_dt = 0.;
    return false;
}

// actual solver:
/*! @brief solve with newton raphson
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 * iterative solver
 */
template<typename Float>
Float inline solveSystemNR(const int dimension, Float* Mp, Float* RHS, Float* DY_T, Float* rates,
                           const PtrReactionList&                    reactions,
                           const ComputeReactionRatesFunctor<Float>& construct_rates_BE, const EosFunctor<Float>& eos,
                           const Float* Y, Float T, Float* final_Y, Float& final_T, const Float rho,
                           const Float drho_dt, Float& dt)
{
    // check for non-burning particles
    if (rho < constants::minRho || T < constants::minTemp)
    {
        dt = constants::maxDt;
        return dt;
    }

    // actual solving
    Float timestep = 0;
    for (int i = 1;; ++i)
    {
        // generate system
        prepareSystemNR(dimension, Mp, RHS, rates, reactions, construct_rates_BE, eos, Y, T, final_Y, final_T, rho,
                        drho_dt, dt, i);

        // solve M*D{T, Y} = RHS
        eigen::solve(Mp, RHS, DY_T, dimension + 1, (Float)constants::epsilonSystem);

        // finalize
        if (finalizeSystemNR(dimension, Y, T, final_Y, final_T, DY_T, dt, timestep, i)) { return timestep; }
    }
}

// actual solver:
/*! @brief solve with newton raphson
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 * iterative solver
 */
template<typename Float>
Float inline solveSystemNR(const int dimension, const PtrReactionList& reactions,
                           const ComputeReactionRatesFunctor<Float>& construct_rates_BE, const EosFunctor<Float>& eos,
                           const Float* Y, Float T, Float* final_Y, Float& final_T, const Float rho,
                           const Float drho_dt, Float& dt)
{
    std::vector<Float>   rates(reactions.size());
    eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);
    eigen::Vector<Float> RHS(dimension + 1);
    eigen::Vector<Float> DY_T(dimension + 1);

    return solveSystemNR(dimension, Mp.data(), RHS.data(), DY_T.data(), rates.data(), reactions, construct_rates_BE,
                         eos, Y, T, final_Y, final_T, rho, drho_dt, dt);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Substeping solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/*! @brief generate the system to be solve for the substepping solver
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 */
template<typename Float, class nseFunction = void*>
HOST_DEVICE_FUN void inline prepareSystemSubstep(const int dimension, Float* Mp, Float* RHS, Float* rates,
                                                 const PtrReactionList&                    reactions,
                                                 const ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                                                 const EosFunctor<Float>& eos, const Float* final_Y, Float final_T,
                                                 Float* next_Y, Float& next_T, const Float final_rho,
                                                 const Float drho_dt, const Float dt_tot, Float& elapsed_time,
                                                 Float& dt, const int i, const nseFunction jumpToNse = NULL)
{
    // compute rho
    Float rho = final_rho - drho_dt * (dt_tot - elapsed_time);

#ifndef COMPILE_DEVICE
    // timejump if needed
    if constexpr (std::is_invocable<std::remove_pointer<nseFunction>>())
        if (dt < dt_tot * constants::substep::dtNseTol)
        {
            dt           = constants::maxDt;
            elapsed_time = dt_tot;

            (*jumpToNse)(reactions, construct_rates_BE, eos, final_Y, final_T, rho, drho_dt);
        }
#endif

    // insure convergence to the right time
    Float used_dt = dt;
    if (dt_tot - elapsed_time < dt) used_dt = dt_tot - elapsed_time;

    // prepare system
    prepareSystemNR(dimension, Mp, RHS, rates, reactions, construct_rates_BE, eos, final_Y, final_T, next_Y, next_T,
                    rho, drho_dt, used_dt, i);
}

/*! @brief second part after solving the system (generated in "prepareSystemFromGuess")
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 */
template<typename Float>
HOST_DEVICE_FUN bool inline finalizeSystemSubstep(const int dimension, Float* final_Y, Float& final_T, Float* next_Y,
                                                  Float& next_T, const Float* DY_T, const Float dt_tot,
                                                  Float& elapsed_time, Float& dt, int& i)
{
    // insure convergence to the right time
    Float timestep, used_dt = dt;
    if (dt_tot - elapsed_time < dt) used_dt = dt_tot - elapsed_time;

    // finalize system
    bool converged = finalizeSystemNR(dimension, final_Y, final_T, next_Y, next_T, DY_T, used_dt, timestep, i);

    // update timestep
    if (dt_tot - elapsed_time < dt)
    {
        if (used_dt < dt_tot - elapsed_time) dt = used_dt;
    }
    else
        dt = used_dt;

    if (converged)
    {
        // update state
        for (int j = 0; j < dimension; ++j)
            final_Y[j] = next_Y[j];
        final_T = next_T;

        // jump back, increment time
        i = 0;
        elapsed_time += timestep;

        // check exit condition
        if ((dt_tot - elapsed_time) / dt_tot < constants::substep::dtTol) return true;
    }

    return false;
}

// actual substepping solver:
/*! @brief function to supperstep (can include jumping to NSE)
 *
 * used in include/nnet/sphexa/nuclear-net.hpp and/or in later function, should not be directly accessed by user
 * Superstepping using solveSystemNR
 */
template<typename Float, class nseFunction = void*>
HOST_DEVICE_FUN void inline solveSystemSubstep(const int dimension, Float* Mp, Float* RHS, Float* DY_T, Float* rates,
                                               const PtrReactionList&                    reactions,
                                               const ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                                               const EosFunctor<Float>& eos, Float* final_Y, Float& final_T,
                                               Float* Y_buffer, const Float final_rho, const Float drho_dt,
                                               Float const dt_tot, Float& dt, const nseFunction jumpToNse = NULL)
{
    // check for non-burning particles
    if (final_rho < constants::minRho || final_T < constants::minTemp) return;

    // actual solving
    Float elapsed_time = 0;
    Float T_buffer;
    for (int i = 1;; ++i)
    {
        // generate system
        prepareSystemSubstep(dimension, Mp, RHS, rates, reactions, construct_rates_BE, eos, final_Y, final_T, Y_buffer,
                             T_buffer, final_rho, drho_dt, dt_tot, elapsed_time, dt, i, jumpToNse);

        // solve M*D{T, Y} = RHS
        eigen::solve(Mp, RHS, DY_T, dimension + 1, (Float)constants::epsilonSystem);

        // finalize
        if (finalizeSystemSubstep(dimension, final_Y, final_T, Y_buffer, T_buffer, DY_T, dt_tot, elapsed_time, dt, i))
        {
            break;
        }
    }
}

// actual substepping solver:
/*! @brief function to supperstep (can include jumping to NSE)
 *
 * used in include/nnet/sphexa/nuclear-net.hpp, should not be directly accessed by user
 * Superstepping using solveSystemNR
 */
template<typename Float, class nseFunction = void*>
void inline solveSystemSubstep(const int dimension, const PtrReactionList& reactions,
                               const ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                               const EosFunctor<Float>& eos, Float* final_Y, Float& final_T, const Float final_rho,
                               const Float drho_dt, Float const dt_tot, Float& dt, const nseFunction jumpToNse = NULL)
{
    std::vector<Float>   rates(reactions.size());
    eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);
    eigen::Vector<Float> RHS(dimension + 1), DY_T(dimension + 1), Y_buffer(dimension);

    solveSystemSubstep(dimension, Mp.data(), RHS.data(), DY_T.data(), rates.data(), reactions, construct_rates_BE, eos,
                       final_Y, final_T, Y_buffer.data(), final_rho, drho_dt, dt_tot, dt, jumpToNse);
}
} // namespace nnet
