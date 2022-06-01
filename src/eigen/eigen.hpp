#pragma once

#include <vector>
#include <cmath>
#include <tuple>

// base implementations
namespace eigen {
	/// custom fixed-size matrix type
	template<typename Type, int n, int m>
	class fixed_size_matrix {
	private:
		std::vector<Type> weights;

	public:
		fixed_size_matrix() {
			weights.resize(n*m, 0);
		}

		Type inline &operator()(int i, int j) {
			return weights[i + j*n];
		}

		Type inline operator()(int i, int j) const {
			return weights[i + j*n];
		}
	};



	/// dot product function
	template<class Vector1, class Vector2>
	double dot(Vector1 const &X, Vector2 const &Y) {
		double res = 0;
		const int dimension = std::min(X.size(), Y.size());

		for (int i = 0; i < dimension; ++i)
			res += X[i]*Y[i];

		return res;
	}
}


// implementation using eigen
#ifdef USE_EIGEN
	// to evoid parallelization
	#define EIGEN_DONT_PARALLELIZE

	#include <Eigen/Core>
#ifdef USE_SPARSE
	#include <Eigen/SparseCore>
	#include <Eigen/SparseLU>
#else
	#include <Eigen/LU>
#endif

	namespace eigen {
		template<typename Type>
		class Vector : public Eigen::VectorX<Type> {
		public:
			Vector(int n) : Eigen::VectorX<Type>(n) {}

			Type inline &operator[](int i) {
				return Eigen::VectorX<Type>::operator()(i);
			}
			const Type inline &operator[](int i) const {
				return Eigen::VectorX<Type>::operator()(i);
			}

			template<typename otherVector>
			Vector<Type> &operator=(const otherVector &other) {
				const int size = other.size();
				Eigen::VectorX<Type>::resize(size);

				for (int i = 0; i < size; ++i)
					operator[](i) = other[i];

				return *this;
			}
		};

		template<typename Type>
		class Matrix : public Eigen::Matrix<Type, -1, -1> {
		public:
			Matrix(int n, int m) : Eigen::Matrix<Type, -1, -1>(n, m) {
				Eigen::Matrix<Type, -1, -1>::setZero();
			}
		};

		/// custom analytical solver
		template<typename Float>
		Vector<Float> solve(Matrix<Float> M, Vector<Float> RHS, Float epsilon=0) {
#ifdef USE_SPARSE
			// sparsify
			Eigen::SparseMatrix<Float> sparceM = M.sparseView(1., epsilon);

			// solve
			Eigen::SparseLU<Eigen::SparseMatrix<Float>> solver;
			solver.compute(sparceM);
#else
			// solve
			Eigen::PartialPivLU<Eigen::Matrix<Float, -1, -1>> solver;
			solver.compute(M);
#endif
			Eigen::VectorX<Float> res = solver.solve(RHS);
			return static_cast<Vector<Float>&>(res);
		}
	}

// implementation from scratch
#else
	namespace eigen {
		template<typename Type>
		using Vector = std::vector<Type>;

		/// custom matrix type
		template<typename Type>
		class Matrix {
		private:
			std::vector<Type> weights;

		public:
			int n, m;

			Matrix(int n_, int m_) : n(n_), m(m_) {
				weights.resize(n*m, 0);
			}

			Type inline &operator()(int i, int j) {
				return weights[i + j*n];
			}

			Type inline operator()(int i, int j) const {
				return weights[i + j*n];
			}
		};



		/// custom analytical solver
		template<typename Float, class Vector, class Matrix>
		Vector solve(Matrix M, Vector RHS, Float epsilon=0) {
			if (M.n != M.m)
				throw std::runtime_error("can't use gaussian elimination on non-square matrices !");

			const int n = RHS.size();
			Vector X(n);

			// reduce into upper triangular
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < i; ++j) {
					// include the jth line
					Float weight = M(i, j);
					M(i, j) = 0;

					// eliminate
					if (std::abs(weight) > epsilon) {
						RHS[i] -= weight*RHS[j];
						for (int k = j + 1; k < n; ++k)
							M(i, k) -= weight*M(j, k);
					}
				}

				Float diagonal = M(i, i);
				M(i, i) = 1;

				// normalize
				RHS[i] /= diagonal;
				for (int j = i + 1; j < n; ++j)
					M(i, j) /= diagonal;
			}

			// "back propagate" to solve
			for (int i = n - 1; i >= 0; --i) {
				Float res = RHS[i];

				for (int j = i + 1; j < n; ++j) {
					Float weight = M(i, j);

					// if (std::abs(weight) > epsilon) // probably slower
					res -= weight*X[j];
				}

				X[i] = res;
			}

			return X;
		}
	}
#endif