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
 * @brief Simple single-particle test for net14.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include <iostream>
#include <chrono>

#include "nnet/parameterization/net14/net14.hpp"
#include "nnet/parameterization/eos/helmholtz.hpp"
#include "nnet/parameterization/eos/ideal_gas.hpp"
#include "nnet/nuclear_net.hpp"

#include "util/arg_parser.hpp"

void printHelp(char* name)
{
    std::cout << "\nUsage:\n\n";
    std::cout << name << " [OPTION]\n";

    std::cout << "\nWhere possible options are:\n\n";

    std::cout << "\t'-n': number of iterations (default = 1000)\n\n";
    std::cout << "\t'--t-lim': limit time (default = 1.5s)\n\n";
    std::cout << "\t'--n-debug': number of debuging prints (default = 30)\n\n";
    std::cout << "\t'--n-save': number of saving prints (to stderr) (default = 0)\n\n";

    std::cout << "\t'--rho': density (default = 1e9)\n\n";
    std::cout << "\t'-T': Temperature (default = 1e9)\n\n";

    std::cout << "\t'--test-case': represent nuclear initial state, can be:\n\n";
    std::cout << "\t\t'C-O-burning: x(12C) = x(16O) = 0.5\n\n";
    std::cout << "\t\t'He-burning: x(4He) = 1\n\n";
    std::cout << "\t\t'Si-burning: x(28Si) = 1\n\n";

    std::cout << "\t'--isotherm', if exists cv=1e20, else use Helmholtz EOS\n\n";
    std::cout << "\t'--skip-coulomb-corr', if exists skip coulombian corrections\n\n";
}

int main(int argc, char* argv[])
{
    const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0]);
        return 0;
    }

    const int    n_max   = parser.get("-n", 1000);
    const int    n_print = parser.get("--n-debug", 30);
    const int    n_save  = parser.get("--n-save", 0);
    const double t_max   = parser.get("--t-lim", 1.5);

    double      rho = parser.get("--rho", 1e9);
    double      T, last_T = parser.get("-T", 1e9);
    std::string test_case = parser.get("--test-case");
    const bool  isotherm  = parser.exists("--isotherm");
    const bool  idealGas  = parser.exists("--ideal-gas") || isotherm;

    nnet::net14::debug = parser.exists("--nnet-debug");

    const bool   expension       = parser.exists("--expansion");
    const int    start_expansion = parser.get("--start-expansion", 600);
    const double rho_half_life   = parser.get("--rho-half-life", 0.02);
    const double rho_lim         = parser.get("--rho-lim", 1e5);

    std::array<double, 14> last_Y, X, Y;
    for (int i = 0; i < 14; ++i)
        X[i] = 0;
    if (test_case == "C-O-burning")
    {
        X[1] = 0.5;
        X[2] = 0.5;
    }
    else if (test_case == "He-burning") { X[0] = 1; }
    else if (test_case == "Si-burning") { X[5] = 1; }
    else
    {
        printHelp(argv[0]);
        throw std::runtime_error("unknown nuclear test case!\n");
    }
    for (int i = 0; i < 14; ++i)
        last_Y[i] = X[i] / nnet::net14::constants::A[i];

    // buffers
    std::vector<double>   rates(nnet::net14::reactionList.size());
    eigen::Matrix<double> Mp(14 + 1, 14 + 1);
    eigen::Vector<double> RHS(14 + 1);
    eigen::Vector<double> DY_T(14 + 1);

    // double E_in = eigen::dot(last_Y.begin(), last_Y.end(), nnet::net14::BE.begin()) + cv*last_T ;
    double m_in = eigen::dot(last_Y.begin(), last_Y.end(), nnet::net14::constants::A.begin());

    if (n_save > 0)
    {
        std::cerr << "\"t\",\"dt\",,\"T\",,";
        for (auto name : nnet::net14::constants::speciesNames)
            std::cerr << "\"x(" << name << ")\",";
        std::cerr << ",\"Dm/m\"\n";
    }

    const nnet::eos::IdealGasFunctor<double>  idea_gas_eos(isotherm ? 1e-20 : 10.0);
    const nnet::eos::HelmholtzFunctor<double> helm_eos(nnet::net14::constants::Z);

    auto start = std::chrono::high_resolution_clock::now();

    double t = 0, dt = nnet::constants::initialDt;
    for (int i = 1; i <= n_max; ++i)
    {
        if (t >= t_max) break;

        // solve the system
        double current_dt = idealGas
                                ? nnet::solveSystemNR(14, Mp.data(), RHS.data(), DY_T.data(), rates.data(),
                                                      nnet::net14::reactionList, nnet::net14::computeReactionRates,
                                                      idea_gas_eos, last_Y.data(), last_T, Y.data(), T, rho, 0., dt)
                                : nnet::solveSystemNR(14, Mp.data(), RHS.data(), DY_T.data(), rates.data(),
                                                      nnet::net14::reactionList, nnet::net14::computeReactionRates,
                                                      helm_eos, last_Y.data(), last_T, Y.data(), T, rho, 0., dt);
        t += current_dt;

        nnet::net14::debug = false;

        // double E_tot = eigen::dot(Y.begin(), Y.end(), nnet::net14::BE.begin()) + cv*T;
        // double dE_E = (E_tot - E_in)/E_in;

        double m_tot = eigen::dot(Y.begin(), Y.end(), nnet::net14::constants::A.begin());
        double dm_m  = (m_tot - m_in) / m_in;

        // formated print (stderr)
        if (n_save > 0)
            if (n_save >= n_max || (n_max - i) % (int)((float)n_max / (float)n_save) == 0 || t >= t_max)
            {
                for (int i = 0; i < 14; ++i)
                    X[i] = Y[i] * nnet::net14::constants::A[i] /
                           eigen::dot(Y.begin(), Y.end(), nnet::net14::constants::A.begin());
                std::cerr << t << "," << dt << ",," << T << ",,";
                for (int i = 0; i < 14; ++i)
                    std::cerr << X[i] << ",";
                std::cerr << "," << dm_m << "\n";
            }

        // debug print
        if (n_print > 0)
            if (n_print >= n_max || (n_max - i) % (int)((float)n_max / (float)n_print) == 0 || t >= t_max)
            {
                for (int i = 0; i < 14; ++i)
                    X[i] = Y[i] * nnet::net14::constants::A[i] /
                           eigen::dot(Y.begin(), Y.end(), nnet::net14::constants::A.begin());
                std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
                for (int i = 0; i < 14; ++i)
                    std::cout << X[i] << ", ";

                auto state = helm_eos(last_Y.begin(), last_T, rho);
                std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tcv=" << state.cv
                          << ",\tdP/dT=" << state.dpdT << ",\t" << T << "\n";
            }

        last_Y = Y;
        last_T = T;
    }

    auto stop     = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(stop - start);
    std::cout << "\nexec time: " << ((float)duration.count()) << "s\n";

    return 0;
}