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
		double theta = 0.6;
		/// tolerance of the implicit solver
		double implicit_tol = 1e-9;

		/// minimum timestep
		double min_dt = 1e-20;
		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 1.5;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 1e-2; //1e-4;
		/// relative mass conservation target of the implicit solver
		double dm_m_target = 1e-3;
		/// relative temperature variation tolerance of the implicit solver
		double dT_T_tol = 1e-1;
		/// relative mass conservation tolerance of the implicit solver
		double dm_m_tol = 1e-2;



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
	Eigen::Matrix<Float, -1, -1> first_order_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, Eigen::Vector<Float, -1> const &Y, Eigen::Vector<Float, -1> const &A, const Float rho) {
		const int dimension = Y.size();

		Eigen::Matrix<Float, -1, -1> M = Eigen::Matrix<Float, -1, -1>::Zero(dimension, dimension);

		for (int i = 0; i < reactions.size() && i < rates.size(); ++i) {
			const reaction &Reaction = reactions[i];
			const Float rate = rates[i];

			// actual reaction speed (including factorials)
			int order = 0;
			Float corrected_rate = rate;
			for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants) {
				corrected_rate /= std::tgamma(n_reactant_consumed + 1); // tgamma performas factorial with n - 1 -> hence we use n + 1
				if (i > n_reactant_consumed)
					corrected_rate *= std::pow(Y(reactant_id), n_reactant_consumed - 1);

				order += n_reactant_consumed;
			}

			// correction with rho
			corrected_rate *= std::pow(rho, (Float)(order - 1));



			/* !!!!!!!!!!!!
			debuging :
			!!!!!!!!!!!! */
			if (net14_debug) {
				for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants)
					std::cout << n_reactant_consumed << "*[" << A(reactant_id) << "] ";
				std::cout << "\t->\t";
				for (auto const [product_id, n_product_produced] : Reaction.products)
					std::cout << n_product_produced << "*[" << A(product_id) << "] ";
				std::cout << ", " << order << ", " << rate << "\t->\t" << (corrected_rate/std::pow(rho, (Float)(order - 1))) << "\t->\t" << corrected_rate << "\n";
			}



			// stop if the rate is 0
			if (std::abs(corrected_rate) > 0) {
				// total reactant mass
				Float total_reactant_mass = 0;
				for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants)
					total_reactant_mass += n_reactant_consumed*A(reactant_id);



				for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants) {
					// compute reaction rate including other species
					Float this_rate = corrected_rate;
					for (auto &[other_reactant_id, _] : Reaction.reactants)
						if (other_reactant_id != reactant_id)
							this_rate *= Y(other_reactant_id);

					// insert diagonal terms (consumption)
					M(reactant_id, reactant_id) -= n_reactant_consumed*this_rate;

					

					// reaction mass proportion
					Float this_reactant_mass_proportion = n_reactant_consumed*A(reactant_id)/total_reactant_mass;

					// compute non-diagonal terms (production)
					Float production_rate = this_rate*this_reactant_mass_proportion;

					// insert production rates
					for (auto const [product_id, n_product_produced] : Reaction.products)
						M(product_id, reactant_id) += n_product_produced*production_rate;
				}
			}
		}

		auto const dM_dt = M.transpose()*A;
		for (int i = 0; i < dimension; ++i)
			M(i, i) -= dM_dt(i)/A(i);

		return M;
	}




	/// includes temperature in the system represented by M.
	/**
	 * add a row to M based on BE to obtain M' such that, d{T,Y}/dt = M'*{T,Y}.
	 * ...TODO
	 */
	template<class matrix, class vector, typename Float>
	matrix include_temp(const matrix &M, const vector &Y, const vector &BE, const Float cv, const Float value_1) {
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




	/// includes the derivative of rates in the system represented by M.
	/**
	 * fills a column to M' based on dM/dT.
	 * ...TODO
	 */
	template<class matrix, class vector>
	matrix include_rate_derivative(const matrix &Mp, const matrix &dM_dT, const vector &Y) {
		/* -------------------
		D{T, Y} = Dt*(M'  + theta*dM/dT*DT)*{T_in+theta*DT, Y_in+theta*DY}
 	<=>                               D{T, Y} = Dt*((M' + theta*dM/dT*DT)*{T_in,Y_in}  + theta*M'*D{T,Y})
 	<=> (I - Dt*theta*(M' + dM/dT*Y))*D{T, Y} = Dt*M'*{T_in,Y_in}

	<=> Mp[1:dimension, 0] = dM/dT*Y
		------------------- */

		const int dimension = Y.size();

		matrix MpT = Mp;

		// derivative of Y compared to T
		MpT(Eigen::seq(1, dimension), 0) = dM_dT*Y;

		return MpT;
	}




	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class matrix, class vector, typename Float>
	std::tuple<vector, Float> solve_system(const matrix &Mp, const matrix &dM_dT, const vector &Y, const Float T, const Float dt) {
		const int dimension = Y.size();

		// construct vector
		vector Y_T(dimension + 1);
		Y_T << T, Y;

		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

		                              D{T, Y} = Dt*(M'  + theta*dM/dT*DT)*{T_in+theta*DT, Y_in+theta*DY}
 	<=>                               D{T, Y} = Dt*((M' + theta*dM/dT*DT)*{T_in,Y_in}  + theta*M'*D{T,Y})
 	<=> (I - Dt*theta*(M' + dM/dT*Y))*D{T, Y} = Dt*M'*{T_in,Y_in}
		------------------- */

		// right hand side
		vector RHS = Mp*Y_T;

		// include rate derivative
		matrix MpT = include_rate_derivative(Mp, dM_dT, Y);

		// construct M
		matrix M = matrix::Identity(dimension + 1, dimension + 1)/dt - constants::theta*MpT;

		// normalize
		utils::normalize(M, RHS);

		// now solve M*D{T, Y} = RHS
		vector DY_T = utils::solve(M, RHS, constants::epsilon_system);

		// add values
		vector next_Y = Y + DY_T(Eigen::seq(1, dimension));
		Float next_T = T + DY_T(0);

		// cleanup
		utils::clip(next_Y, nnet::constants::epsilon_vector);
		return {next_Y, next_T};
	}




	/// fully solves a system, with timestep tweeking
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system, with added timestep tweeking
	 * ...TODO
	 */
	template<class matrix, class vector, typename Float>
	std::tuple<vector, Float, Float> solve_system_var_timestep(const matrix &Mp, const matrix &dM_dT, const vector &Y, const Float T, const vector &A, Float &dt) {
		const Float m_in = Y.dot(A);

		// actual solving
		while (true) {
			// solve the system
			auto [next_Y, next_T] = solve_system(Mp, dM_dT, Y, T, dt);

			// mass temperature variation
			Float dm_m = std::abs((next_Y.dot(A) - m_in)/m_in);
			Float dT_T = std::abs((next_T - T)/((1 - constants::theta)*T + constants::theta*next_T));

			// timestep tweeking
			Float actual_dt = dt;
			Float dt_multiplier = std::min(
				(Float)constants::max_dt_step,
				std::min(
					dT_T == 0 ? (Float)constants::max_dt_step : constants::dT_T_target/dT_T,
					dm_m == 0 ? (Float)constants::max_dt_step : constants::dm_m_target/dm_m
				));
			dt = std::min(
				(Float)constants::max_dt,
				std::max(
					(Float)constants::min_dt,
					dt*dt_multiplier
				));

			// exit on condition
			if (dm_m <= constants::dm_m_tol && dT_T <= constants::dT_T_tol)
				return {next_Y, next_T, actual_dt};
		}
	}
}
