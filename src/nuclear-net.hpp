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

namespace nnet {
	namespace constants {
		/// safety factor when determining the maximum timestep.
		double safety_margin = 0.3;



		/// theta for the implicit method
		double theta = 0.8;
		/// tolerance of the implicit solver
		double implicit_tol = 1e-9;

		/// minimum timestep
		double min_dt = 1e-20;
		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 1.5;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 5e-5;
		/// relative mass conservation target of the implicit solver
		double dm_m_target = 1e-6;
		/// relative mass conservation tolerance of the implicit solver
		double dm_m_tol = 1e-5;



		/// the value that is considered null inside a system
		double epsilon_system = 1e-100;
		/// the value that is considered null inside a state
		double epsilon_vector = 1e-100;
	}




	/// reaction class
	/**
	 * ...TODO
	 */
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
		/// sparcify a matrix.
		/**
		 * retursn a "sparsified" version of Min, with non-zero elements having a higher absolute value thab epsilon.
		 * ...TODO
		 */
		template<typename Float, int n>
		Eigen::SparseMatrix<Float> sparsify(const Eigen::Matrix<Float, n, n> &Min, const Float epsilon) {
			std::vector<Eigen::Triplet<Float>> coefs;

			for (int i = 0; i < Min.cols(); ++i)
				for (int j = 0; j < Min.rows(); ++j)
					if (std::abs(Min(i, j)) > epsilon)
						coefs.push_back(Eigen::Triplet<Float>(i, j, Min(i, j)));

			Eigen::SparseMatrix<Float> Mout(Min.cols(), Min.rows());
			Mout.setFromTriplets(coefs.begin(), coefs.end());
			return Mout;
		}




		/// normalizes a system.
		/**
		 * normalizes each equation of a system represented by a matrix (M) and a Right Hand Side (RHS).
		 * ...TODO
		 */
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




		/// solves a system
		/**
		 * solves a system represented by a matrix (M) and a Right Hand Side (RHS).
		 * ...TODO
		 */
		template<typename Float, int n, int m>
		Eigen::Vector<Float, m> solve(Eigen::Matrix<Float, n, n> &M, Eigen::Vector<Float, m> &RHS, const Float epsilon) {
			// sparcify M
			auto sparse_M = utils::sparsify(M, epsilon);

			// now solve M*X = RHS
			Eigen::BiCGSTAB<Eigen::SparseMatrix<Float>>  BCGST;
			BCGST.compute(sparse_M);
			return BCGST.solve(RHS);
		}

		/// clip the values in a vector
		/**
		 * clip the values in a vector, to make 0 any negative value, or values smaller than a tolerance epsilon
		 * ...TODO
		 */
		template<typename Float, int n>
		void clip(Eigen::Vector<Float, n> &X, const Float epsilon) {
			const int dimension = X.size();

			for (int i = 0; i < dimension; ++i)
				if (X(i) < epsilon)
					X(i) = 0;
		}
	}






	/// create a first order system from a list of reaction.
	/**
	 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
	 * ...TODO
	 */
	template<typename Float>
	Eigen::Matrix<Float, -1, -1> first_order_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const Float rho, Eigen::Vector<Float, -1> const &Y) {
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
				for (auto &[reactant_id, n_reactant_consumed] : Reaction.reactants)
					std::cout << n_reactant_consumed << "*[" << reactant_id << "] ";
				std::cout << "\t->\t";
				for (auto &[product_id, n_product_produced] : Reaction.products)
					std::cout << n_product_produced << "*[" << product_id << "] ";
				std::cout << ", " << order << ", " << rate << "\t->\t" << (corrected_rate/std::pow(rho, (Float)(order - 1))) << "\t->\t" << corrected_rate << "\n";
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
			}
		}
			

		return M;
	}





	/// includes temperature in the system represented by M.
	/**
	 * add a row to M based on BE to obtain M' such that, d{T,Y}/dt = M'*{T,Y}.
	 * ...TODO
	 */
	template<class matrix, class vector, typename Float>
	matrix include_temp(const matrix &M, const Float cv, const Float value_1, const vector &BE, const vector &Y) {
		/* -------------------
		Add a row to M (dY/dt = M*Y) such that d{T,Y}/dt = M'*{T,Y}:

		  (dY/dt).BE    + value_1*T    = dT/dt*cv
	<=>     (M*Y).BE/cv + value_1*T/cv = dT/dt
	<=>  (M.T*BE). Y/cv + value_1*T/cv = dT/dt
		------------------- */

		const int dimension = Y.size();
		matrix Mp = matrix::Zero(dimension + 1, dimension + 1);

		// insert M
		Mp(Eigen::seq(1, dimension), Eigen::seq(1, dimension)) = M;

		// insert Y -> dT terms (first row)
		Mp(0, Eigen::seq(1, dimension)) = M.transpose()*BE/cv;

		// insert T -> dT term  (first row diagonal value)
		Mp(0, 0) = value_1/cv;

		return Mp;
	}




	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class matrix, class vector, typename Float>
	vector solve_first_order(const vector &Y, const Float T, const matrix &Mp, const Float dt) {
		const int dimension = Y.size();

		// construct vector
		vector Y_T(dimension + 1);
		Y_T << T, Y;

		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

		                  D{T, Y} = Dt* M'*{T_in+theta*DT, Y_in+theta*DY}
 	<=>                   D{T, Y} = Dt* M'*({T_in,Y_in}  + theta*D{T,Y})
 	<=> (I - Dt*M'*theta)*D{T, Y} = Dt* M'* {T_in,Y_in}
		------------------- */

		// right hand side
		vector RHS = Mp*Y_T*dt;

		// construct M
		matrix M = matrix::Identity(dimension + 1, dimension + 1) - constants::theta*dt*Mp;

		// normalize
		utils::normalize(M, RHS);

		// now solve M*D{T, Y} = RHS
		vector DY_T = utils::solve(M, RHS, constants::epsilon_system);
		return Y_T + DY_T;
	}






	/// fully solves a system.
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system.
	 * ...TODO
	 */
	template<class problem, class vector, typename Float>
	std::tuple<vector, Float> solve_system(const problem construct_system, const vector &Y, const Float T, const Float dt) {
		const int dimension = Y.size();

		// construct vector
		vector prev_Y_T_out(dimension + 1);
		prev_Y_T_out << T, Y;

		// actual solving
		int max_iter = std::max(1., -std::log2(constants::implicit_tol));
		for (int i = 0;; ++i) {

			// intermediary vecor
			Float scaled_T_out  = (1 - constants::theta)*T + constants::theta*prev_Y_T_out(0); 
			vector scaled_Y_out = (1 - constants::theta)*Y + constants::theta*prev_Y_T_out(Eigen::seq(1, dimension));

			// construct system
			auto M = construct_system(scaled_Y_out, scaled_T_out);

			// solve system
			auto Y_T_out = solve_first_order(Y, T, M, dt);

			// clip system
			utils::clip(Y_T_out, constants::epsilon_vector);

			// exit on condition
			if (i >= max_iter || std::abs((prev_Y_T_out(0) - Y_T_out(0))/scaled_T_out) <= constants::implicit_tol)
				return {Y_T_out(Eigen::seq(1, dimension)), Y_T_out(0)};

			prev_Y_T_out = Y_T_out;
		}
	}




	/// fully solves a system, with timestep tweeking
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system, with added timestep tweeking
	 * ...TODO
	 */
	template<class problem, class vector, typename Float>
	std::tuple<vector, Float, Float> solve_system_var_timestep(const vector &A, const problem construct_system, const vector &Y, const Float T, Float dt) {
		const Float m_in = Y.dot(A);

		// actual solving
		int max_iter = std::max(1., -std::log2(constants::dm_m_target));
		for (int i = 0;; ++i) {
			// solve the system
			auto [next_Y, next_T] = solve_system(construct_system, Y, T, dt);

			// mass temperature variation
			Float dm_m = std::abs((next_Y.dot(A) - m_in)/m_in);
			Float dT_T = std::abs((next_T - T)/((1 - constants::theta)*T + constants::theta*next_T));

			// timestep tweeking
			Float dt_multiplier = std::min((Float)constants::max_dt_step, std::min(constants::dT_T_target/dT_T, constants::dm_m_target/dm_m));
			dt = std::max((Float)constants::min_dt, dt*dt_multiplier);
			dt = std::min((Float)constants::max_dt, dt);

			// exit on condition
			if (i >= max_iter || dm_m <= constants::dm_m_tol)
				return {next_Y, next_T, dt};
		}
	}
}