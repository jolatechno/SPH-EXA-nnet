#include <iostream>
#include <chrono>

#include "../src/nuclear-net.hpp"
//#include "../src/net86/net86.hpp"
#include "../src/net87/net87.hpp"
#include "../src/eos/helmholtz.hpp"

#include "util/arg_parser.hpp"

void printHelp(char* name) {
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

	std::cout << "\t'--isotherm': if exists cv=1e20, else use Helmholtz EOS\n\n";
	std::cout << "\t'--skip-coulomb-corr': if exists skip coulombian corrections\n\n";

	std::cout << "\t'--output-net14': if exists output results only for net14 species\n\n";
	std::cout << "\t'--debug-net86': if exists output debuging prints for net86 species\n\n";
}

inline static constexpr struct eos_output {
	double cv, dP_dT, dU_dYe;
} isotherm_res{1e20, 0, 0};
struct isotherm_eos_struct {
	CUDA_FUNCTION_DECORATOR isotherm_eos_struct() {}
	
	CUDA_FUNCTION_DECORATOR eos_output inline operator()(const double *Y_, const double T, const double rho_) const {
		return isotherm_res;
	}
} isotherm_eos;

int main(int argc, char* argv[]) {
	const ArgParser parser(argc, argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")) {
        printHelp(argv[0]);
        return 0;
    }

    const int n_max                         = parser.get("-n",        1000);
    const int n_print                       = parser.get("--n-debug", 30);
    const double t_max                      = parser.get("--t-lim",   1.5);

    double rho                              = parser.get("--rho", 1e9);
    double T, last_T                        = parser.get("-T", 1e9);
    std::string test_case                   = parser.get("--test-case");
    const bool isotherm                     = parser.exists("--isotherm");

    nnet::debug                             = parser.exists("--nnet-debug");

    nnet::net86::skip_coulombian_correction = parser.exists("--skip-coulomb-corr");

    const int n_save                        = parser.get("--n-save", 0);
    const bool save_res_net14               = parser.exists("--output-net14");
    const bool debug_net86                  = parser.exists("--debug-net86");

    const bool expension                    = parser.exists("--expansion");
    const int start_expansion               = parser.get("--start-expansion", 600);
	const double rho_half_life              = parser.get("--rho-half-life",   0.02);
	const double rho_lim                    = parser.get("--rho-lim",         1e5);

    std::array<double, 86> last_Y, X, Y;
    for (int i = 0; i < 86; ++i) X[i] = 0;
    if  (      test_case == "C-O-burning") {
    	X[nnet::net86::constants::net14_species_order[1]] = 0.5;
		X[nnet::net86::constants::net14_species_order[2]] = 0.5;
    } else if (test_case == "He-burning") {
    	X[nnet::net86::constants::net14_species_order[0]] = 1;
    } else if (test_case == "Si-burning") {
    	X[nnet::net86::constants::net14_species_order[5]] = 1;
    } else {
    	printHelp(argv[0]);
    	throw std::runtime_error("unknown nuclear test case!\n");
    }
    for (int i = 0; i < 86; ++i) last_Y[i] = X[i]/nnet::net86::constants::A[i];




    // buffers
	std::vector<double> rates(nnet::net86::reaction_list.size()), drates_dT(nnet::net86::reaction_list.size());
	eigen::Matrix<double> Mp(86 + 1, 86 + 1);
	eigen::Vector<double> RHS(86 + 1);
	eigen::Vector<double> DY_T(86 + 1);




	// double E_in = eigen::dot(last_Y.begin(), last_Y.end(), nnet::net86::BE.begin()) + cv*last_T ;
	double m_in = eigen::dot(last_Y.begin(), last_Y.end(), nnet::net86::constants::A.begin());

	if (n_save > 0) {
		std::cerr << "\"t\",\"dt\",,\"T\",";
		if (save_res_net14) {
			for (auto name : nnet::net14::constants::species_names)
				std::cerr << ",\"x(" << name << ")\"";
		} else {
			for (auto name : nnet::net86::constants::species_names)
				std::cerr << ",\"x(" << name << ")\"";
		}
		std::cerr << ",,\"Dm/m\"\n";
	}


	const nnet::eos::helmholtz_functor helm_eos(nnet::net86::constants::Z, 86);


	auto start = std::chrono::high_resolution_clock::now();

	double t = 0, dt = nnet::constants::initial_dt;
	for (int i = 1; i <= n_max; ++i) {
		if (t >= t_max)
			break;

		// solve the system
		double current_dt = isotherm ? 
			nnet::solve_system_NR(86,
				Mp.data(), RHS.data(), DY_T.data(), rates.data(), drates_dT.data(),
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, isotherm_eos,
				last_Y.data(), last_T, Y.data(), T,
				rho, 0., dt) :
			nnet::solve_system_NR(86,
				Mp.data(), RHS.data(), DY_T.data(), rates.data(), drates_dT.data(),
				nnet::net86::reaction_list, nnet::net86::compute_reaction_rates, helm_eos,
				last_Y.data(), last_T, Y.data(), T,
				rho, 0., dt);
		t += current_dt;

		nnet::debug = false;


		// double E_tot = eigen::dot(Y.begin(), Y.end(), nnet::net86::BE.begin()) + cv*T;
		// double dE_E = (E_tot - E_in)/E_in;

		double m_tot = eigen::dot(Y.begin(), Y.end(), nnet::net86::constants::A.begin());
		double dm_m = (m_tot - m_in)/m_in;

		// formated print (stderr)
		if (n_save > 0)
			if (n_save >= n_max || (n_max - i) % (int)((float)n_max/(float)n_save) == 0 || t >= t_max) {
				for (int i = 0; i < 86; ++i) X[i] = Y[i]*nnet::net86::constants::A[i]/eigen::dot(Y.begin(), Y.end(), nnet::net86::constants::A.begin());

				std::cerr << t << "," << dt << ",," << T << ",,";
				if (save_res_net14) {
					for (auto idx : nnet::net86::constants::net14_species_order)
						std::cerr << X[idx] << ", ";
				} else
					for (int i = 0; i < 86; ++i)
						std::cerr << X[nnet::net86::constants::species_order[i]] << ", ";
				std::cerr << "," << dm_m << "\n";
			}

		// debug print
			if (n_print > 0)
			if (n_print >= n_max || (n_max - i) % (int)((float)n_max/(float)n_print) == 0 || t >= t_max) {
				for (int i = 0; i < 86; ++i) X[i] = Y[i]*nnet::net86::constants::A[i]/eigen::dot(Y.begin(), Y.end(), nnet::net86::constants::A.begin());

				std::cout << "\n(t=" << t << ", dt=" << dt << "):\t";
				if (debug_net86) {
					for (int i = 0; i < 86; ++i)
						std::cout << X[nnet::net86::constants::species_order[i]] << ", ";
				} else
					for (auto idx : nnet::net86::constants::net14_species_order)
						std::cout << X[idx] << ", ";
				std::cout << "\t(m=" << m_tot << ",\tdm_m0=" << dm_m << ")\t" << T << "\n";
			}

		last_Y = Y;
		last_T = T;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<double>(stop - start);
	std::cout << "\nexec time: " << ((float)duration.count()) << "s\n";

	return 0;
}