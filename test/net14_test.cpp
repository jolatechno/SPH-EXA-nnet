#include <iostream>
#include <chrono>

#include "../src/nuclear-net.hpp"
#include "../src/net14/net14.hpp"
#include "../src/eos/helmholtz.hpp"

#include "utils/arg_parser.hpp"

void printHelp() {
	/* TODO */
}

int main(int argc, char* argv[]) {
	const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")) {
        printHelp();
        return 0;
    }

    const int n_max                         = parser.get("-n", 1000);
    const int n_print                       = parser.get("--n-debug", 30);
    const int n_save                        = parser.get("--n-save", 0);
    const double t_max                      = parser.get("--t-lim", 1.5);

    double rho                              = parser.get("--rho", 1e9);
    double last_T                           = parser.get("-T", 1e9);
    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");

    nnet::debug                             = parser.exists("--nnet-debug");

    nnet::net14::skip_coulombian_correction = parser.exists("--skip-coulomb-corr");
    nnet::constants::NR::max_dt             = parser.get("--max_dt",      nnet::constants::NR::max_dt);
    nnet::constants::NR::dT_T_target        = parser.get("--dT_T_target", nnet::constants::NR::dT_T_target);
    nnet::constants::NR::dT_T_tol           = parser.get("--dT_T_tol",    nnet::constants::NR::dT_T_tol);
    nnet::constants::NR::it_tol             = parser.get("--NR_tol",      nnet::constants::NR::it_tol);
    nnet::constants::NR::min_it             = parser.get("--min_NR_it",   nnet::constants::NR::min_it);
    nnet::constants::NR::max_it             = parser.get("--max_NR_it",   nnet::constants::NR::max_it);

    const bool expension                    = parser.exists("--expansion");
    const int start_expansion               = parser.get("--start-expansion", 600);
	const double rho_half_life              = parser.get("--rho-half-life", 0.02);
	const double rho_lim                    = parser.get("--rho-lim", 1e5);

	std::array<double, 14> last_Y, X;
    for (int i = 0; i < 14; ++i) X[i] = 0;
    if  (      test_case == "C-O-burning") {
    	X[1] = 0.5;
		X[2] = 0.5;
    } else if (test_case == "He-burning") {
    	X[0] = 1;
    } else if (test_case == "Si-burning") {
    	X[5] = 1;
    } else
    	throw std::runtime_error("unknown nuclear test case!\n");
    for (int i = 0; i < 14; ++i) last_Y[i] = X[i]/nnet::net14::constants::A[i];


	// double E_in = eigen::dot(last_Y, nnet::net14::BE) + cv*last_T ;
	double m_in = eigen::dot(last_Y, nnet::net14::constants::A);

	if (n_save > 0) {
		std::cerr << "\"t\",\"dt\",,\"T\",,";
		for (auto name : nnet::net14::constants::species_names)
			std::cerr << "\"x(" << name << ")\",";
		std::cerr << ",\"Dm/m\"\n";
	}

	const nnet::eos::helmholtz helm_eos(nnet::net14::constants::Z);
	const auto isotherm_eos = [&](const std::array<double, 14> &Y_, const double T, const double rho_) {
		const double cv = 1e20; //1.5 * /*Rgasid*/8.31e7 * /*mu*/0.72; 		// typical cv from net14 fortran
		struct eos_output {
			double cv, dP_dT;
		} res{cv, 0};
		return res;
	};


	auto start = std::chrono::high_resolution_clock::now();

	double t = 0, dt=1e-12;
	for (int i = 1; i <= n_max; ++i) {
		if (t >= t_max)
			break;

		// solve the system
		auto [Y, T, current_dt] = isotherm ? 
			nnet::solve_system_NR(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, isotherm_eos,
				last_Y, last_T, rho, 0., dt) :
			nnet::solve_system_NR(nnet::net14::reaction_list, nnet::net14::compute_reaction_rates<double>, nnet::net14::compute_BE<double>, helm_eos,
				last_Y, last_T, rho, 0., dt);
		t += current_dt;

		nnet::debug = false;


		// double E_tot = eigen::dot(Y, nnet::net14::BE) + cv*T;
		// double dE_E = (E_tot - E_in)/E_in;

		double m_tot = eigen::dot(Y, nnet::net14::constants::A);
		double dm_m = (m_tot - m_in)/m_in;

		// formated print (stderr)
		if (n_save > 0)
			if (n_save >= n_max || (n_max - i) % (int)((float)n_max/(float)n_save) == 0 || t >= t_max) {
				for (int i = 0; i < 14; ++i) X[i] = Y[i]*nnet::net14::constants::A[i]/eigen::dot(Y, nnet::net14::constants::A);
				std::cerr << t << "," << dt << ",," << T << ",,";
				for (int i = 0; i < 14; ++i) std::cerr << X[i] << ",";
				std::cerr << "," << dm_m << "\n";
			}

		// debug print
		if (n_print > 0)
			if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0 || t >= t_max) {
				for (int i = 0; i < 14; ++i) X[i] = Y[i]*nnet::net14::constants::A[i]/eigen::dot(Y, nnet::net14::constants::A);
				std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
				for (int i = 0; i < 14; ++i) std::cout << X[i] << ", ";
				std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << "),\tcv=" << helm_eos(last_Y, last_T, rho).cv << ",\t" << T << "\n";
			}

		last_Y = Y;
		last_T = T;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nexec time:" << ((float)duration.count())/1e3 << "s\n";

	return 0;
}