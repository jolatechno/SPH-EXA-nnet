#pragma once


#include <cmath> // factorial
//#include <ranges> // drop

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>


/* !!!!!!!!!!!!
debuging :
!!!!!!!!!!!! */
bool net14_debug = false;


/*
photodesintegration_to_first_order  R_i,j, n_i,j

fusion_to_first_order               F_i,j,k, n_i,j,k, Y_i
	'-> M : dY/dt = M*Y

include_temp : theta, value_1, cv, BE'_i: sum_i(BE'_i*dY_i/dt) + value_1*T = cv*dT/dt
	'-> M' : d{T, Y}/dt = M'*{T, Y}

iterate_system M', theta, Y, T
	'-> D{T, Y} = Dt* M'*(T_in*(1 - theta) + T_out*theta)

	D{T, Y} = Dt*M'*{T_in + theta*DT, Y_in + theta*Y_in}
 	<=> D{T, Y} = Dt*M'*({T_in, Y_in} + theta*{DT, DY})
 	<=> (I - Dt*M'*theta)*D{T, Y} = DT*M'*{T_in, Y_in}
 	<=> D{T, Y} = (I - Dt* M' *theta)^{-1} * DT*M'*{T_in, Y_in}

solve_system : compose_system(Y, T), Y, T:
 	DT = 0, DY = 0

 	while true:
 		M' = compose_system(Y + theta*DY, T + theta*DT)

		DT, DY -> iterate_system(M', Y, T)

		abs((DT - DT_prev)/T) < tol:
		 	T = T + DT, Y = Y + DT

	return T + DT, Y + DY
*/

/* class reaction: std::vector<std::pair<int, int>> reactant -> std::vector<std::pair<int, int>> product */
/* function "reaction_to_first_order": std:vector<std::pair<reaction, double>> -> Eigen::matrix */

namespace nnet {
	struct reaction {
		struct reactant {
			int reactant_id, n_reactant_consumed = 1;
		};
		struct product {
			int product_id, n_product_produced = 1;
		};
		std::vector<reactant> reactants;
		std::vector<product> products;
	};

	namespace utils {
		template<typename Float, int n>
		Eigen::SparseMatrix<Float> sparsify(const Eigen::Matrix<Float, n, n> &Min, const Float epsilon=1e-16) {
			/* -------------------
			put a "sparsified" version of Min into Mout according to epsilon 
			------------------- */
			std::vector<Eigen::Triplet<Float>> coefs;

			for (int i = 0; i < Min.cols(); ++i)
				for (int j = 0; j < Min.rows(); ++j)
					if (std::abs(Min(i, j)) > epsilon)
						coefs.push_back(Eigen::Triplet<Float>(i, j, Min(i, j)));

			Eigen::SparseMatrix<Float> Mout(Min.cols(), Min.rows());
			Mout.setFromTriplets(coefs.begin(), coefs.end());
			return Mout;
		}

		template<typename Float, int n, int m>
		void normalize(Eigen::Matrix<Float, n, n> &M, Eigen::Vector<Float, m> &RHS) {
			const int dimension = RHS.size();

			for (int i = 0; i < dimension; ++i) {

				// find maximum
				Float max_ = std::abs(M(0, i));
				for (int j = 1; j < dimension; ++j) {
					Float val = std::abs(M(j, i));
					if (val > max_) max_ = val;
				}

				// normalize
				RHS[i] /= max_;
				for (int j = 0; j < dimension; ++j)
					M(j, i) /= max_;
			}
		}

		template<typename Float, int n, int m>
		Eigen::Vector<Float, m> solve(Eigen::Matrix<Float, n, n> &M, Eigen::Vector<Float, m> &RHS, const Float epsilon=0) {
			// sparcify M
			auto sparse_M = utils::sparsify(M, epsilon);

			// now solve M*X = RHS
			Eigen::BiCGSTAB<Eigen::SparseMatrix<Float>>  BCGST;
			BCGST.compute(sparse_M);
			return BCGST.solve(RHS);
		}

		template<typename Float, int n>
		void clip(Eigen::Vector<Float, n> &X, const Float epsilon=0) {
			const int dimension = X.size();

			for (int i = 0; i < dimension; ++i)
				if (X(i) < epsilon)
					X(i) = 0;
		}
	}

	template<typename Float>
	Eigen::Matrix<Float, -1, -1> first_order_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const Float rho, Eigen::Vector<Float, -1> const &Y) {
		/* -------------------
		reactes a sparce matrix M such that dY/dt = M*Y*
		from a list of reactions
		------------------- */
		const int dimension = Y.size();

		Eigen::Matrix<Float, -1, -1> M = Eigen::Matrix<Float, -1, -1>::Zero(dimension, dimension);

		for (int i = 0; i < reactions.size() && i < rates.size(); ++i) {
			const reaction &Reaction = reactions[i];
			const Float rate = rates[i];

			// actual reaction speed (including factorials)
			Float corrected_rate = rate;
			for (auto &[reactant_id, n_reactant_consumed] : Reaction.reactants) {
				corrected_rate /= std::tgamma(n_reactant_consumed + 1); // tgamma performas factorial with n - 1 -> hence we use n + 1
				if (i > n_reactant_consumed)
					corrected_rate *= std::pow(Y(reactant_id), n_reactant_consumed - 1);
			}

			// correction with rho
			int order = 0;
			for (auto &[_, n_reactant_consumed] : Reaction.reactants)
				order += n_reactant_consumed;
			corrected_rate *= std::pow(rho, (Float)(order - 1));





			/* !!!!!!!!!!!!
			debuging :
			!!!!!!!!!!!! */
			if (net14_debug) {
				std::cerr << "\t\t";
				for (auto &[reactant_id, n_reactant_consumed] : Reaction.reactants)
					std::cerr << n_reactant_consumed << "*[" << reactant_id << "] ";
				std::cerr << " -> ";
				for (auto &[product_id, n_product_produced] : Reaction.products)
					std::cerr << n_product_produced << "*[" << product_id << "] ";
				std::cerr << ", " << order << ", " << rate << " -> " << (corrected_rate/std::pow(rho, (Float)(order - 1))) << " ->\t" << corrected_rate << "\n";
			}





			// stop if the rate is 0
			if (std::abs(corrected_rate) > 0) {
				// compute diagonal terms (consumption)
				for (auto &[reactant_id, n_reactant_consumed] : Reaction.reactants) {
					Float consumption_rate = n_reactant_consumed*corrected_rate;
					for (auto &[other_reactant_id, _] : Reaction.reactants)
						if (other_reactant_id != reactant_id)
							consumption_rate *= Y(other_reactant_id);
					M(reactant_id, reactant_id) -= consumption_rate;
				}

#ifdef DISTRIBUTE_ELEMENT
				// compute non-diagonal terms (production)
				const int n_reactants = Reaction.reactants.size();
				for (auto &[product_id, n_product_produced] : Reaction.products) {
					for (auto &[reactant_id, _] : Reaction.products) {
						Float production_rate = n_product_produced*corrected_rate/n_reactants;
						for (auto &[other_reactant_id, _] : Reaction.reactants)
							if (other_reactant_id != reactant_id)
								production_rate *= Y(other_reactant_id);
						M(product_id, reactant_id) += production_rate;
					}
				}
#else
				// compute non-diagonal terms (production)
				for (auto &[product_id, n_product_produced] : Reaction.products) {
					// find the optimum place to put the coeficient
					int best_reactant_id = Reaction.reactants[0].reactant_id;
					bool is_non_zero = M(product_id, best_reactant_id) != 0.;
					for (int j = 1; j < Reaction.reactants.size(); ++j) { // for (auto &[reactant_id, _] : Reaction.reactants | std::ranges::views::drop(1))
						int reactant_id = Reaction.reactants[j].reactant_id;

						// prioritize the minimization of the number of non-zero terms
						if (M(product_id, reactant_id) != 0.) {
							// overwise if it is the first non-zero term
							if (!is_non_zero) {
								is_non_zero = true;
								best_reactant_id = reactant_id;
							} else
								// otherwise keep the "relativly closest" element
								if (std::abs((reactant_id - product_id)*M(product_id, reactant_id)/M(product_id, product_id)) <
									std::abs((best_reactant_id - product_id)*M(product_id, best_reactant_id)/M(product_id, product_id)))
									best_reactant_id = reactant_id;
						} else if (!is_non_zero)
							// if still populating a zero element, then keep the closest to the diagonal
							if (std::abs(reactant_id - product_id) < std::abs(best_reactant_id - product_id))
								best_reactant_id = reactant_id;
					}


					// insert into the matrix
					Float production_rate = n_product_produced*corrected_rate;
					for (auto &[other_reactant_id, _] : Reaction.reactants)
						if (other_reactant_id != best_reactant_id)
							production_rate *= Y(other_reactant_id);
					M(product_id, best_reactant_id) += production_rate;
				}
#endif
			}
		}
			

		return M;
	}

	template<class matrix, class vector, typename Float>
	matrix include_temp(const matrix &M, const Float value_1, const Float cv, const vector &BE, const vector &Y) {
		/* -------------------
		add a row to M based on BE so that, d{T, Y}/dt = M'*{T, Y}

		value_1, cv : BE.dY/dt + value_1*T = cv*dT/dt
					   <=> (M * Y).BE/cv + value_1/cv*T = dT/dt
					   <=> (M.t * BE).Y/cv + value_1/cv*T = dT/dt
		------------------- */

		const int dimension = Y.size();
		matrix Mp = matrix::Zero(dimension + 1, dimension + 1);

		// insert M
		Mp(Eigen::seq(1, dimension), Eigen::seq(1, dimension)) = M;

		// insert (T, Y) -> dT terms
		Mp(0, Eigen::seq(1, dimension)) = M.transpose()*BE/cv;
		Mp(0, 0) = value_1/cv; // T -> dT term

		return Mp;
	}

	template<class matrix, class vector, typename Float>
	vector solve_first_order(const vector &Y, const Float T, const matrix &Mp, const Float dt, const Float theta=1, const Float epsilon=1e-100) {
		const int dimension = Y.size();

		// construct vector
		vector Y_T(dimension + 1);
		Y_T << T, Y;

#ifndef DIFFERENT_SOLVER
		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

		{T_out, Y_out} = {T_in, Y_in} + Dt* M'*{T_in*(1 - theta) + theta*T_out, Y_in*(1 - theta) + theta*Y_out}
 	<=> {T_out, Y_out} = {T_in, Y_in} + Dt* M'*([1 - theta]*{T_in, Y_in} + theta*{T_out, Y_out})
 	<=> (I - Dt*M'*theta)*{T_out, Y_out} = (I + Dt*M'*[1- theta])*{T_in, Y_in}
		------------------- */

		// right hand side
		vector RHS = Y_T + dt*(1 - theta)*Mp*Y_T;

	/* !!!!!!!!!!!!
	no idea why
	!!!!!!!!!!!! */
	#ifdef ZERO_T_RHS
		RHS(0) = Y_T(0);
	#endif

		// construct M
		matrix M = matrix::Identity(dimension + 1, dimension + 1) - dt*theta*Mp;

		// normalize
		utils::normalize(M, RHS);

		// now solve M*{T_out, Y_out} = RHS
		return utils::solve(M, RHS, epsilon);
#else
		/* different solver : */
		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

		D{T, Y} = Dt* M'*{T_in + theta*DT, Y_in + theta*DY}
 	<=> D{T, Y} = Dt* M'*({T_in, Y_in} + theta*D{T, Y})
 	<=> (I - Dt*M'*theta)*D{T, Y} = Dt*M'*{T_in, Y_in}
		------------------- */

		// right hand side
		vector RHS = Mp*Y_T*dt;

		/* !!!!!!!!!!!!
		no idea why
		!!!!!!!!!!!! */
	#ifdef ZERO_T_RHS
		RHS(0) = 0;
	#endif

		// construct M
		matrix M = matrix::Identity(dimension + 1, dimension + 1) - theta*dt*Mp;

		// normalize
		utils::normalize(M, RHS);

		// now solve M*D{T, Y} = RHS
		vector DY_T = utils::solve(M, RHS, epsilon);
		return Y_T + DY_T;
#endif
	}

	template<class problem, class vector, typename Float>
	std::tuple<vector, Float> solve_system(const problem construct_system, const vector &Y, const Float T, const Float dt, const Float theta=1, const Float tol=1e-5, const Float epsilon=1e-16) {
		const int dimension = Y.size();

		// construct vector
		vector prev_Y_T_out(dimension + 1);
		prev_Y_T_out << T, Y;

		int max_iter = (int)std::max(1., -std::log2(tol));
		for (int i = 0;; ++i) {
			// intermediary vecor
			Float scaled_T_out = T*(1 - theta) + theta*prev_Y_T_out(0); 
			vector scaled_Y_out = Y*(1 - theta) + theta*prev_Y_T_out(Eigen::seq(1, dimension));

			// construct system
			auto M = construct_system(scaled_Y_out, scaled_T_out);

			// solve system
			auto Y_T_out = solve_first_order(Y, T, M, dt, theta, epsilon);

			// clip system
			utils::clip(Y_T_out, epsilon);

			// exit on condition
			if (i >= max_iter || std::abs((prev_Y_T_out(0) - Y_T_out(0))/scaled_T_out) < tol)
				return {Y_T_out(Eigen::seq(1, dimension)), Y_T_out(0)};

			prev_Y_T_out = Y_T_out;
		}
	}
}
