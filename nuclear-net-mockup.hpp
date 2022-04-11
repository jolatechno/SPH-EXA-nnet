#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

namespace nnet {
	/* -------------------
	Iteration protocol (r : desintegration rates, dr/dT, fi : fission rates, dfi="dfi/dT", fu : fusion rates, dfu="dfu/dT", Q : mass excedent, rho : specific heat, M : molar masses):
		
		// if needed, convert to mols
		masses_to_mols(Y, M);

		// if needed, rework desintegration rates 
		//to insure that the equation is dY/dt = r*Y
		auto r_included = desintegration_rate_to_first_order(r);
		auto dr_included = desintegration_rate_to_first_order(dr);

		// add fission rates to desintegration rates
		r_included += nnet::fission_to_desintegration_rates(fi, Y);
		dr_included += nnet::fission_to_desintegration_rates(dfi, Y);

		// add fusion rates to desintegration rates
		r_included += nnet::fusion_to_desintegration_rates(fu, Y);
		dr_included += nnet::fusion_to_desintegration_rates(dfu, Y);

		// add temperature to the problem
		auto RQ = include_temp(r_included, Q, rho);

		// solve the system
		auto [Y_next, T_next] = solve_first_order(Y, T, RQ, dr_included, Q, dt)

		// if needed, convert back to masses
		mols_to_masses(Y_next, M);

	------------------- */

	namespace utils {
		template<typename Float>
		Eigen::SparseMatrix<Float> sparsify(const Eigen::MatrixXd &Min, const Float epsilon=1e-16) {
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
	}

	template<class vector>
	void masses_to_mols(vector &X, const vector &M) {
		/* -------------------
		goes from a vector of masses to a vector of mol

		inverse of mols_to_masses
		------------------- */

		const int dimension = X.size();
		for (int i = 0; i < dimension; ++i)
			X(i) /= M(i);
	}

	template<class vector>
	void mols_to_masses(vector &Y, const vector &M) {
		/* -------------------
		goes from a vector of masses to a vector of mol

		inverse of masses_to_mols
		------------------- */

		const int dimension = Y.size();
		for (int i = 0; i < dimension; ++i)
			Y(i) *= M(i);
	}

	template<class matrix>
	Eigen::MatrixXd desintegration_rate_to_first_order(const matrix &r) {
		/* -------------------
		simply add the diagonal desintegration terms to the desintegration rates if not included

		makes sure that the equation ends up being : dY/dt = r*Y
		------------------- */

		int dimension = r.cols();
		Eigen::MatrixXd r_out = r;

		for (int i = 0; i < dimension; ++i) {
			r_out(i, i) = 0;
			for (int j = 0; j < dimension; ++j) 
				if (j != i)
					r_out(i, i) -= r(j, i);
		}

		return r_out;
	}

	template<class tensor, class vector>
	Eigen::MatrixXd fusion_to_desintegration_rates(const tensor &fu, const vector &Y) {
		/* -------------------
		include fusion rate into desintegration rates for a given state Y
		------------------- */

		const int dimension = Y.size();
		Eigen::MatrixXd r_out = Eigen::MatrixXd::Zero(dimension, dimension);

		// add fusion rates
		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				if (i != j) {
					// add i + i -> j
					r_out(j, i) += fu(j, i, i)*Y(i);
					r_out(i, i) -= fu(j, i, i)*Y(i)*2;

					// add i + k -> j
					for (int k = 0; k < dimension; ++k)
						if (i != k) {
							r_out(j, i) += (fu(j, i, k) + fu(j, k, i))*Y(k)/2;
							r_out(i, i) -= (fu(j, i, k) + fu(j, k, i))*Y(k);
						}
				}

		return r_out;
	}

	template<class tensor>
	Eigen::MatrixXd fission_to_desintegration_rates(const tensor &fi) {
		/* -------------------
		include fission rate into desintegration rates
		------------------- */

		const int dimension = fi.dimension(0);
		Eigen::MatrixXd r_out = Eigen::MatrixXd::Zero(dimension, dimension);

		// add fission rates
		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				if (i != j) {
					// add i -> j + j
					r_out(j, i) += fi(j, j, i)*2;
					r_out(i, i) -= fi(j, j, i);

					// add i -> j + k
					for (int k = 0; k < dimension; ++k)
						if (j != k) {
							r_out(j, i) += fi(j, k, i) + fi(k, j, i);
							r_out(i, i) -= (fi(j, k, i) + fi(k, j, i))/2;
						}
				}

		return r_out;
	}

	template<class matrix, class tensor, class vector>
	Eigen::MatrixXd fusion_to_desintegration_rates(const matrix &r, const tensor &f, const vector &Y) {
		/* -------------------
		include fusion rate into desintegration rates for a given state Y
		------------------- */

		const int dimension = Y.size();
		Eigen::MatrixXd r_out = Eigen::MatrixXd::Zero(dimension, dimension);

		// add fusion rates
		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				if (i != j) {
					// add i + i -> j
					r_out(j, i) += f(j, i, i)*Y(i);

					// add i + k -> j
					for (int k = 0; k < dimension; ++k)
						if (i != k) {
							r_out(j, i) += (f(j, i, k) + f(j, k, i))*Y(k)/2;
							r_out(i, i) -= (f(j, i, k) + f(j, k, i))*Y(k);
						}
				}

		return r_out;
	}

	template<class matrix, class vector, typename Float>
	Eigen::MatrixXd include_temp(const matrix &r, const vector &Q, const Float rho) {
		/* -------------------
		add a row to r based on Q so that, d{Y, T}/dt = r*Y
		------------------- */

		const int dimension = Q.size();
		Eigen::MatrixXd RQ(dimension + 1, dimension);

		// insert r
		RQ(Eigen::seq(1, dimension), Eigen::seq(0, dimension - 1)) = r;

		// temperature terms
		for (int i = 0; i < dimension; ++i) {
			RQ(0, i) = -r(i, i)*Q(i)/rho;

			// add i -> j
			for (int j = 0; j < dimension; ++j) 
				if (i != j)
					RQ(0, i) += -r(j, i)*Q(j)/rho;
		}

		return RQ;
	}

	template<class matrix, class vector, typename Float>
	std::tuple<vector, Float> solve_first_order(const vector &Y, const Float T, const matrix &RQ, const matrix &dr, const Float dt, const Float theta=1, const Float epsilon=1e-16) {
		/* -------------------
		Solves d{Y, T}/dt = RQ*Y using eigen:
		{DY, DT} = RQ*(Y + teta*DY)*dt + (dr*Y*DT, 0)*dt
		=> {DY, DT} * (I - dt*teta*{RQ, dr*Y)} = dt*RQ*Y
		------------------- */

		const int dimension = Y.size();

		// right hand side
		const vector RHS = RQ*Y*dt;

		// left hand side matrix (M)
		matrix M(dimension + 1, dimension + 1);
		M(Eigen::seq(0, dimension), Eigen::seq(1, dimension)) = RQ;
		M(Eigen::seq(1, dimension), 0) = dr*Y;

		M *= -theta*dt;
		M += Eigen::MatrixXd::Identity(dimension + 1, dimension + 1);

		// sparcify M
		auto sparse_M = utils::sparsify(M, epsilon);

		/* -------------------
		TODO (if needed): Add loop
		------------------- */
		// now solve {Dy, DT}*M = RHS
		Eigen::BiCGSTAB<Eigen::SparseMatrix<Float>>  BCGST;
		BCGST.compute(sparse_M);
		auto const DY_T = BCGST.solve(RHS);

		// add to solution
		Float T_next = T + DY_T(0);
		auto Y_next = Y + DY_T(Eigen::seq(1, dimension));

		return {Y_next, T_next};
	}


	namespace net14 {
		template<class matrix, typename Float>
		void get_net14_desintegration_rates(matrix &r, const Float T) {
			/* -------------------
			simply copute desintegration rates within net-14
			------------------- */

			// TODO
		}

		template<class matrix, typename Float>
		void get_net14_desintegration_rates_derivatives(matrix &dr, const Float T) {
			/* -------------------
			simply copute desintegration rates derivatives within net-14
			------------------- */

			// TODO
		}

		template<class tensor, typename Float>
		void get_net14_fusion_rates(tensor &f, const Float T) {
			/* -------------------
			simply copute fusion rates within net-14
			------------------- */

			// TODO
		}

		template<class tensor, typename Float>
		void get_net14_fusion_rates_derivatives(tensor &df, const Float T) {
			/* -------------------
			simply copute fusion rates derivatives within net-14
			------------------- */

			// TODO
		}
	}

	namespace net87 {
		template<class matrix, typename Float>
		void get_net87_desintegration_rates(matrix &r, const Float T) {
			/* -------------------
			simply copute desintegration rates within net-14
			------------------- */

			// TODO
		}

		template<class matrix, typename Float>
		void get_net87_desintegration_rates_derivatives(matrix &dr, const Float T) {
			/* -------------------
			simply copute desintegration rates derivatives within net-14
			------------------- */

			// TODO
		}

		template<class tensor, typename Float>
		void get_net87_fusion_rates(tensor &f, const Float T) {
			/* -------------------
			simply copute fusion rates within net-14
			------------------- */

			// TODO
		}

		template<class tensor, typename Float>
		void get_net87_fusion_rates_derivatives(tensor &df, const Float T) {
			/* -------------------
			simply copute fusion rates derivatives within net-14
			------------------- */

			// TODO
		}
	}
}