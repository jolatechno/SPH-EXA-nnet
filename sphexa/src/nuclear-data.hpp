#pragma once

#include <vector>
#include <array>

namespace sphexa::sphnnet {
	/// nuclear abundances type, that is integrated into NuclearData, or should be integrated into ParticlesData
	template <int n_species, typename Float=double>
	class NuclearAbundances {
	private:
		Float data[n_species];
	public:
		NuclearAbundances(Float x=0) {
			for (int i = 0; i < n_species; ++i)
				data[i] = x;
		}

		~NuclearAbundances() {}

		size_t size() const {
			return n_species;
		}

		Float &operator[](const int i) {
			return data[i];
		}

		const Float &operator[](const int i) const {
			return data[i];
		}

		template<class Vector>
		NuclearAbundances &operator=(const Vector &other) {
			for (int i = 0; i < n_species; ++i)
				data[i] = other[i];

			return *this;
		}
	};

	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double>
	struct NuclearDataType {
		/// hydro data
		std::vector<Float> rho, drho_dt, T;

		/// nuclear abundances (vector of vector)
		std::vector<NuclearAbundances<n_species, Float>> Y;

		/// timesteps
		std::vector<Float> dt;

		/// resize the number of particules
		void resize(const size_t N) {
			rho.resize(N);
			drho_dt.resize(N);
			T.resize(N);

			Y.resize(N);

			dt.resize(N, 1e-12);
		}
	};
}