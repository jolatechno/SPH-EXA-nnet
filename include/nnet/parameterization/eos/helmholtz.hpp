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
 * @brief Helmholtz EOS definition.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <math.h>

#include "nnet_util/CUDA/cuda.inl"
#if COMPILE_DEVICE
#include "nnet_util/CUDA/cuda-util.hpp"
#endif

#include "nnet_util/eigen.hpp"

#include "nnet_util/algorithm.hpp"

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

#ifndef IMAX
#define IMAX 541
#endif
#ifndef JMAX
#define JMAX 201
#endif
#ifndef HELM_TABLE_PATH
#define HELM_TABLE_PATH "./helm_table.dat"
#endif

namespace nnet::eos
{

namespace helmholtz
{

/*! @brief if true print debuging prints */
extern bool debug;

namespace constants
{
/*! @brief table initialize sucess */
extern bool table_read_success;

// table size
const int imax = IMAX, jmax = JMAX;

// table limits
const double tlo   = 3.;
const double thi   = 13.;
const double tstp  = (thi - tlo) / (double)(jmax - 1);
const double tstpi = 1. / tstp;
const double dlo   = -12.;
const double dhi   = 15.;
const double dstp  = (dhi - dlo) / (double)(imax - 1);
const double dstpi = 1. / dstp;

// physical constants
const double pi =
    3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198938095257201065485863278865936153381827968230301952035301852968995773622599413891249721775283479131515574857242454150695950829533116861727855889075098381754637464939319;
const double g       = 6.6742867e-8;
const double h       = 6.6260689633e-27;
const double hbar    = 0.5 * h / pi;
const double qe      = 4.8032042712e-10;
const double avo     = 6.0221417930e23;
const double clight  = 2.99792458e10;
const double kerg    = 1.380650424e-16;
const double ev2erg  = 1.60217648740e-12;
const double kev     = kerg / ev2erg;
const double amu     = 1.66053878283e-24;
const double mn      = 1.67492721184e-24;
const double mp      = 1.67262163783e-24;
const double me      = 9.1093821545e-28;
const double rbohr   = hbar * hbar / (me * qe * qe);
const double fine    = qe * qe / (hbar * clight);
const double hion    = 13.605698140;
const double ssol    = 5.6704e-5;
const double asol    = 4.0 * ssol / clight;
const double weinlam = h * clight / (kerg * 4.965114232);
const double weinfre = 2.821439372 * kerg / h;
const double rhonuc  = 2.342e14;
const double kergavo = kerg * avo;
const double sioncon = (2.0 * pi * amu * kerg) / (h * h);

// parameters
const double a1   = -0.898004;
const double b1   = 0.96786;
const double c1   = 0.220703;
const double d1   = -0.86097;
const double e1   = 2.5269;
const double a2   = 0.29561;
const double b2   = 1.9885;
const double c2   = 0.288675;
const double esqu = qe * qe;

DEVICE_DEFINE(extern double, d[IMAX], ;)
DEVICE_DEFINE(extern double, dd_sav[IMAX - 1], ;)
DEVICE_DEFINE(extern double, dd2_sav[IMAX - 1], ;)
DEVICE_DEFINE(extern double, ddi_sav[IMAX - 1], ;)
DEVICE_DEFINE(extern double, dd2i_sav[IMAX - 1], ;)
DEVICE_DEFINE(extern double, dd3i_sav[IMAX - 1], ;)

DEVICE_DEFINE(extern double, t_[JMAX], ;)
DEVICE_DEFINE(extern double, dt_sav[JMAX - 1], ;)
DEVICE_DEFINE(extern double, dt2_sav[JMAX - 1], ;)
DEVICE_DEFINE(extern double, dti_sav[JMAX - 1], ;)
DEVICE_DEFINE(extern double, dt2i_sav[JMAX - 1], ;)
DEVICE_DEFINE(extern double, dt3i_sav[JMAX - 1], ;)

DEVICE_DEFINE(extern double, f[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, fd[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, ft[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, fdd[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, ftt[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, fdt[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, fddt[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, fdtt[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, fddtt[IMAX][JMAX], ;)

DEVICE_DEFINE(extern double, dpdf[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, dpdfd[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, dpdft[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, dpdfdt[IMAX][JMAX], ;)

DEVICE_DEFINE(extern double, ef[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, efd[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, eft[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, efdt[IMAX][JMAX], ;)

DEVICE_DEFINE(extern double, xf[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, xfd[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, xft[IMAX][JMAX], ;)
DEVICE_DEFINE(extern double, xfdt[IMAX][JMAX], ;)

/*! @brief Read helmholtz constant table. */
bool inline readCPUTable()
{
    // read table
    const std::string helmoltz_table = {
#include HELM_TABLE_PATH
    };

    // read file
    std::stringstream helm_table;
    helm_table << helmoltz_table;

    // read the helmholtz free energy and its derivatives
    for (int i = 0; i < imax; ++i)
    {
        double dsav = dlo + i * dstp;
        d[i]        = std::pow((double)10., dsav);
    }
    for (int j = 0; j < jmax; ++j)
    {
        double tsav = tlo + j * tstp;
        t_[j]       = std::pow((double)10., tsav);

        for (int i = 0; i < imax; ++i)
        {
            helm_table >> f[i][j] >> fd[i][j] >> ft[i][j] >> fdd[i][j] >> ftt[i][j] >> fdt[i][j] >> fddt[i][j] >>
                fdtt[i][j] >> fddtt[i][j];
        }
    }

    // read the pressure derivative with rhosity table
    for (int j = 0; j < jmax; ++j)
        for (int i = 0; i < imax; ++i)
        {
            helm_table >> dpdf[i][j] >> dpdfd[i][j] >> dpdft[i][j] >> dpdfdt[i][j];
        }

    // read the electron chemical potential table
    for (int j = 0; j < jmax; ++j)
        for (int i = 0; i < imax; ++i)
        {
            helm_table >> ef[i][j] >> efd[i][j] >> eft[i][j] >> efdt[i][j];
        }

    // read the number rhosity table
    for (int j = 0; j < jmax; ++j)
        for (int i = 0; i < imax; ++i)
        {
            helm_table >> xf[i][j] >> xfd[i][j] >> xft[i][j] >> xfdt[i][j];
        }

    // construct the temperature and rhosity deltas and their inverses
    for (int j = 0; j < jmax - 1; ++j)
    {
        const double dth  = t_[j + 1] - t_[j];
        const double dt2  = dth * dth;
        const double dti  = 1. / dth;
        const double dt2i = 1. / dt2;
        const double dt3i = dt2i * dti;

        dt_sav[j]   = dth;
        dt2_sav[j]  = dt2;
        dti_sav[j]  = dti;
        dt2i_sav[j] = dt2i;
        dt3i_sav[j] = dt3i;
    }

    // construct the temperature and rhosity deltas and their inverses
    for (int i = 0; i < imax - 1; ++i)
    {
        const double dd   = d[i + 1] - d[i];
        const double dd2  = dd * dd;
        const double ddi  = 1. / dd;
        const double dd2i = 1. / dd2;
        const double dd3i = dd2i * ddi;

        dd_sav[i]   = dd;
        dd2_sav[i]  = dd2;
        ddi_sav[i]  = ddi;
        dd2i_sav[i] = dd2i;
        dd3i_sav[i] = dd3i;
    }

    return true;
}

bool inline copyTableToGPU()
{
#if COMPILE_DEVICE
    // copy to device
    gpuErrchk(cudaMemcpyToSymbol(dev_d, d, imax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dd_sav, dd_sav, (imax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dd2_sav, dd2_sav, (imax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_ddi_sav, ddi_sav, (imax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dd2i_sav, dd2i_sav, (imax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dd3i_sav, dd3i_sav, (imax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(dev_t_, t_, jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dt_sav, dt_sav, (jmax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dt2_sav, dt2_sav, (jmax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dti_sav, dti_sav, (jmax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dt2i_sav, dt2i_sav, (jmax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dt3i_sav, dt3i_sav, (jmax - 1) * sizeof(double), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(dev_f, f, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_fd, fd, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_ft, ft, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_fdd, fdd, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_ftt, ftt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_fdt, fdt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_fddt, fddt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_fdtt, fdtt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_fddtt, fddtt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(dev_dpdf, dpdf, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dpdfd, dpdfd, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dpdft, dpdft, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_dpdfdt, dpdfdt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(dev_ef, ef, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_efd, efd, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_eft, eft, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_efdt, efdt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(dev_xf, xf, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_xfd, xfd, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_xft, xft, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_xfdt, xfdt, imax * jmax * sizeof(double), 0, cudaMemcpyHostToDevice));
#endif
    return true;
};

// quintic hermite polynomial statement functions
// psi0 and its derivatives
template<typename Float>
HOST_DEVICE_FUN Float inline psi0(const Float z)
{
    return z * z * z * (z * (-6. * z + 15.) - 10.) + 1.;
}
template<typename Float>
HOST_DEVICE_FUN Float inline dpsi0(const Float z)
{
    return z * z * (z * (-30. * z + 60.) - 30.);
};
template<typename Float>
HOST_DEVICE_FUN Float inline ddpsi0(const Float z)
{
    return z * (z * (-120. * z + 180.) - 60.);
};

// psi1 and its derivatives
template<typename Float>
HOST_DEVICE_FUN Float inline psi1(const Float z)
{
    return z * (z * z * (z * (-3. * z + 8.) - 6.) + 1.);
};
template<typename Float>
HOST_DEVICE_FUN Float inline dpsi1(const Float z)
{
    return z * z * (z * (-15. * z + 32.) - 18.) + 1.;
};
template<typename Float>
HOST_DEVICE_FUN Float inline ddpsi1(const Float z)
{
    return z * (z * (-60. * z + 96.) - 36.);
};

// psi2  and its derivatives
template<typename Float>
HOST_DEVICE_FUN Float inline psi2(const Float z)
{
    return 0.5 * z * z * (z * (z * (-z + 3.) - 3.) + 1.);
};
template<typename Float>
HOST_DEVICE_FUN Float inline dpsi2(const Float z)
{
    return 0.5 * z * (z * (z * (-5. * z + 12.) - 9.) + 2.);
};
template<typename Float>
HOST_DEVICE_FUN Float inline ddpsi2(const Float z)
{
    return 0.5 * (z * (z * (-20. * z + 36.) - 18.) + 2.);
};

// biquintic hermite polynomial statement function
template<typename Float>
HOST_DEVICE_FUN Float inline h5(const Float* fi, const Float w0t, const Float w1t, const Float w2t, const Float w0mt,
                                const Float w1mt, const Float w2mt, const Float w0d, const Float w1d, const Float w2d,
                                const Float w0md, const Float w1md, const Float w2md)
{
    return fi[0] * w0d * w0t + fi[1] * w0md * w0t + fi[2] * w0d * w0mt + fi[3] * w0md * w0mt + fi[4] * w0d * w1t +
           fi[5] * w0md * w1t + fi[6] * w0d * w1mt + fi[7] * w0md * w1mt + fi[8] * w0d * w2t + fi[9] * w0md * w2t +
           fi[10] * w0d * w2mt + fi[11] * w0md * w2mt + fi[12] * w1d * w0t + fi[13] * w1md * w0t + fi[14] * w1d * w0mt +
           fi[15] * w1md * w0mt + fi[16] * w2d * w0t + fi[17] * w2md * w0t + fi[18] * w2d * w0mt +
           fi[19] * w2md * w0mt + fi[20] * w1d * w1t + fi[21] * w1md * w1t + fi[22] * w1d * w1mt +
           fi[23] * w1md * w1mt + fi[24] * w2d * w1t + fi[25] * w2md * w1t + fi[26] * w2d * w1mt +
           fi[27] * w2md * w1mt + fi[28] * w1d * w2t + fi[29] * w1md * w2t + fi[30] * w1d * w2mt +
           fi[31] * w1md * w2mt + fi[32] * w2d * w2t + fi[33] * w2md * w2t + fi[34] * w2d * w2mt + fi[35] * w2md * w2mt;
};

// cubic hermite polynomial statement functions
// psi0 and its derivatives
template<typename Float>
HOST_DEVICE_FUN Float inline xpsi0(const Float z)
{
    return z * z * (2. * z - 3.) + 1.;
};
template<typename Float>
HOST_DEVICE_FUN Float inline xdpsi0(const Float z)
{
    return z * (6. * z - 6.);
};

// psi1 & derivatives
template<typename Float>
HOST_DEVICE_FUN Float inline xpsi1(const Float z)
{
    return z * (z * (z - 2.) + 1.);
};
template<typename Float>
HOST_DEVICE_FUN Float inline xdpsi1(const Float z)
{
    return z * (3. * z - 4.) + 1.;
};

// bicubic hermite polynomial statement function
template<typename Float>
HOST_DEVICE_FUN Float inline h3(const Float* fi, const Float w0t, const Float w1t, const Float w0mt, const Float w1mt,
                                const Float w0d, const Float w1d, const Float w0md, const Float w1md)
{
    return fi[0] * w0d * w0t + fi[1] * w0md * w0t + fi[2] * w0d * w0mt + fi[3] * w0md * w0mt + fi[4] * w0d * w1t +
           fi[5] * w0md * w1t + fi[6] * w0d * w1mt + fi[7] * w0md * w1mt + fi[8] * w1d * w0t + fi[9] * w1md * w0t +
           fi[10] * w1d * w0mt + fi[11] * w1md * w0mt + fi[12] * w1d * w1t + fi[13] * w1md * w1t + fi[14] * w1d * w1mt +
           fi[15] * w1md * w1mt;
};

// get correspong table indices
template<typename Float>
HOST_DEVICE_FUN void inline getTableIndices(int& iat, int& jat, const Float temp, const Float rho, const Float abar,
                                            const Float zbar)
{
    const Float ye  = std::max((Float)1e-16, zbar / abar);
    const Float din = ye * rho;

    jat = int((std::log10(temp) - tlo) * tstpi);
    jat = std::max<int>(1, std::min<int>(jat, jmax - 2));

    iat = int((std::log10(din) - dlo) * dstpi);
    iat = std::max<int>(1, std::min<int>(iat, imax - 2));
}
} // namespace constants

/*! @brief Helmholtz EOS
 *
 * @param abar_ average A (number of mass)
 * @param zbar_ average Z (number of charge)
 * @param temp  temperature
 * @param rho   density
 *
 * Returns Helmholtz EOS output struct.
 */
template<typename Float>
HOST_DEVICE_FUN nnet::eos_struct<Float> inline helmholtzEos(double abar_, double zbar_, const Float temp,
                                                            const Float rho)
{
    // coefs
    // Float fi[36];
    Float* fi = new Float[36];

    Float abar = 1 / abar_;
    Float zbar = zbar_ / abar_;

    /* debug: */
#if !DEVICE_CODE
    if (debug) std::cout << "temp=" << temp << ", rho=" << rho << ", abar=" << abar << ", zbar=" << zbar << "\n";
#endif

    // compute polynoms rates
    int iat, jat;
    constants::getTableIndices(iat, jat, temp, rho, abar, zbar);

    Float ytot1 = 1 / abar;
    Float ye    = std::max<Float>((Float)1e-16, zbar / abar);
    Float din   = ye * rho;

    // initialize
    Float rhoi  = 1. / rho;
    Float tempi = 1. / temp;
    Float kt    = constants::kerg * temp;
    Float ktinv = 1. / kt;

    // adiation section:
    Float prad    = constants::asol * temp * temp * temp * temp / 3;
    Float dpraddd = 0.;
    Float dpraddt = 4. * prad * tempi;
    Float dpradda = 0.;
    Float dpraddz = 0.;

    Float erad    = 3. * prad * rhoi;
    Float deraddd = -erad * rhoi;
    Float deraddt = 3. * dpraddt * rhoi;
    Float deradda = 0.;
    Float deraddz = 0.;

    Float srad    = (prad * rhoi + erad) * tempi;
    Float dsraddd = (dpraddd * rhoi - prad * rhoi * rhoi + deraddd) * tempi;
    Float dsraddt = (dpraddt * rhoi + deraddt - srad) * tempi;
    Float dsradda = 0.;
    Float dsraddz = 0.;

    // ion section:
    Float xni    = constants::avo * ytot1 * rho;
    Float dxnidd = constants::avo * ytot1;
    Float dxnida = -xni * ytot1;

    Float pion    = xni * kt;
    Float dpiondd = dxnidd * kt;
    Float dpiondt = xni * constants::kerg;
    Float dpionda = dxnida * kt;
    Float dpiondz = 0.;

    Float eion    = 1.5 * pion * rhoi;
    Float deiondd = (1.5 * dpiondd - eion) * rhoi;
    Float deiondt = 1.5 * dpiondt * rhoi;
    Float deionda = 1.5 * dpionda * rhoi;
    Float deiondz = 0.;

    // sackur-tetrode equation for the ion entropy of
    // a single ideal gas characterized by abar
    Float x = abar * abar * std::sqrt(abar) * rhoi / constants::avo;
    Float s = constants::sioncon * temp;
    Float z = x * s * std::sqrt(s);
    Float y = std::log(z);

    // y       = 1./(abar*kt)
    // yy      = y*sqrt(y)
    // z       = xni*sifac*yy
    // etaion  = log(z)

    Float sion    = (pion * rhoi + eion) * tempi + constants::kergavo * ytot1 * y;
    Float dsiondd = (dpiondd * rhoi - pion * rhoi * rhoi + deiondd) * tempi - constants::kergavo * rhoi * ytot1;
    Float dsiondt = (dpiondt * rhoi + deiondt) * tempi - (pion * rhoi + eion) * tempi * tempi +
                    1.5 * constants::kergavo * tempi * ytot1;
    x             = constants::avo * constants::kerg / abar;
    Float dsionda = (dpionda * rhoi + deionda) * tempi + constants::kergavo * ytot1 * ytot1 * (2.5 - y);
    Float dsiondz = 0.;

    // electron-positron section:

    // assume complete ionization
    Float xnem = xni * zbar;

    // move table values into coefficient table
    fi[0]  = constants::DEVICE_ACCESS(f)[iat + 0][jat + 0];
    fi[1]  = constants::DEVICE_ACCESS(f)[iat + 1][jat + 0];
    fi[2]  = constants::DEVICE_ACCESS(f)[iat + 0][jat + 1];
    fi[3]  = constants::DEVICE_ACCESS(f)[iat + 1][jat + 1];
    fi[4]  = constants::DEVICE_ACCESS(ft)[iat + 0][jat + 0];
    fi[5]  = constants::DEVICE_ACCESS(ft)[iat + 1][jat + 0];
    fi[6]  = constants::DEVICE_ACCESS(ft)[iat + 0][jat + 1];
    fi[7]  = constants::DEVICE_ACCESS(ft)[iat + 1][jat + 1];
    fi[8]  = constants::DEVICE_ACCESS(ftt)[iat + 0][jat + 0];
    fi[9]  = constants::DEVICE_ACCESS(ftt)[iat + 1][jat + 0];
    fi[10] = constants::DEVICE_ACCESS(ftt)[iat + 0][jat + 1];
    fi[11] = constants::DEVICE_ACCESS(ftt)[iat + 1][jat + 1];
    fi[12] = constants::DEVICE_ACCESS(fd)[iat + 0][jat + 0];
    fi[13] = constants::DEVICE_ACCESS(fd)[iat + 1][jat + 0];
    fi[14] = constants::DEVICE_ACCESS(fd)[iat + 0][jat + 1];
    fi[15] = constants::DEVICE_ACCESS(fd)[iat + 1][jat + 1];
    fi[16] = constants::DEVICE_ACCESS(fdd)[iat + 0][jat + 0];
    fi[17] = constants::DEVICE_ACCESS(fdd)[iat + 1][jat + 0];
    fi[18] = constants::DEVICE_ACCESS(fdd)[iat + 0][jat + 1];
    fi[19] = constants::DEVICE_ACCESS(fdd)[iat + 1][jat + 1];
    fi[20] = constants::DEVICE_ACCESS(fdt)[iat + 0][jat + 0];
    fi[21] = constants::DEVICE_ACCESS(fdt)[iat + 1][jat + 0];
    fi[22] = constants::DEVICE_ACCESS(fdt)[iat + 0][jat + 1];
    fi[23] = constants::DEVICE_ACCESS(fdt)[iat + 1][jat + 1];
    fi[24] = constants::DEVICE_ACCESS(fddt)[iat + 0][jat + 0];
    fi[25] = constants::DEVICE_ACCESS(fddt)[iat + 1][jat + 0];
    fi[26] = constants::DEVICE_ACCESS(fddt)[iat + 0][jat + 1];
    fi[27] = constants::DEVICE_ACCESS(fddt)[iat + 1][jat + 1];
    fi[28] = constants::DEVICE_ACCESS(fdtt)[iat + 0][jat + 0];
    fi[29] = constants::DEVICE_ACCESS(fdtt)[iat + 1][jat + 0];
    fi[30] = constants::DEVICE_ACCESS(fdtt)[iat + 0][jat + 1];
    fi[31] = constants::DEVICE_ACCESS(fdtt)[iat + 1][jat + 1];
    fi[32] = constants::DEVICE_ACCESS(fddtt)[iat + 0][jat + 0];
    fi[33] = constants::DEVICE_ACCESS(fddtt)[iat + 1][jat + 0];
    fi[34] = constants::DEVICE_ACCESS(fddtt)[iat + 0][jat + 1];
    fi[35] = constants::DEVICE_ACCESS(fddtt)[iat + 1][jat + 1];

    // various differences
    Float xt = std::max<Float>((temp - constants::DEVICE_ACCESS(t_)[jat]) * constants::DEVICE_ACCESS(dti_sav)[jat], 0.);
    Float xd = std::max<Float>((din - constants::DEVICE_ACCESS(d)[iat]) * constants::DEVICE_ACCESS(ddi_sav)[iat], 0.);
    Float mxt = 1. - xt;
    Float mxd = 1. - xd;

    /* debug: */
#if !DEVICE_CODE
    if (debug)
        std::cout << "xt=" << xt << " = (temp - t[" << jat << "]=" << constants::t_[jat] << ")* dti_sav[" << jat
                  << "]=" << constants::dti_sav[jat] << "\n";
#endif

    // the six rhosity and six temperature basis functions;
    Float si0t = constants::psi0(xt);
    Float si1t = constants::psi1(xt) * constants::DEVICE_ACCESS(dt_sav)[jat];
    Float si2t = constants::psi2(xt) * constants::DEVICE_ACCESS(dt2_sav)[jat];

    /* debug: */
#if !DEVICE_CODE
    if (debug) std::cout << "si0t=" << si0t << " = psi0(xt=" << xt << ")\n";
#endif

    Float si0mt = constants::psi0(mxt);
    Float si1mt = -constants::psi1(mxt) * constants::DEVICE_ACCESS(dt_sav)[jat];
    Float si2mt = constants::psi2(mxt) * constants::DEVICE_ACCESS(dt2_sav)[jat];

    Float si0d = constants::psi0(xd);
    Float si1d = constants::psi1(xd) * constants::DEVICE_ACCESS(dd_sav)[iat];
    Float si2d = constants::psi2(xd) * constants::DEVICE_ACCESS(dd2_sav)[iat];

    Float si0md = constants::psi0(mxd);
    Float si1md = -constants::psi1(mxd) * constants::DEVICE_ACCESS(dd_sav)[iat];
    Float si2md = constants::psi2(mxd) * constants::DEVICE_ACCESS(dd2_sav)[iat];

    // derivatives of the weight functions
    Float dsi0t = constants::dpsi0(xt) * constants::DEVICE_ACCESS(dti_sav)[jat];
    Float dsi1t = constants::dpsi1(xt);
    Float dsi2t = constants::dpsi2(xt) * constants::DEVICE_ACCESS(dt_sav)[jat];

    Float dsi0mt = -constants::dpsi0(mxt) * constants::DEVICE_ACCESS(dti_sav)[jat];
    Float dsi1mt = constants::dpsi1(mxt);
    Float dsi2mt = -constants::dpsi2(mxt) * constants::DEVICE_ACCESS(dt_sav)[jat];

    Float dsi0d = constants::dpsi0(xd) * constants::DEVICE_ACCESS(ddi_sav)[iat];
    Float dsi1d = constants::dpsi1(xd);
    Float dsi2d = constants::dpsi2(xd) * constants::DEVICE_ACCESS(dd_sav)[iat];

    Float dsi0md = -constants::dpsi0(mxd) * constants::DEVICE_ACCESS(ddi_sav)[iat];
    Float dsi1md = constants::dpsi1(mxd);
    Float dsi2md = -constants::dpsi2(mxd) * constants::DEVICE_ACCESS(dd_sav)[iat];

    // second derivatives of the weight functions
    Float ddsi0t = constants::ddpsi0(xt) * constants::DEVICE_ACCESS(dt2i_sav)[jat];
    Float ddsi1t = constants::ddpsi1(xt) * constants::DEVICE_ACCESS(dti_sav)[jat];
    Float ddsi2t = constants::ddpsi2(xt);

    Float ddsi0mt = constants::ddpsi0(mxt) * constants::DEVICE_ACCESS(dt2i_sav)[jat];
    Float ddsi1mt = -constants::ddpsi1(mxt) * constants::DEVICE_ACCESS(dti_sav)[jat];
    Float ddsi2mt = constants::ddpsi2(mxt);

    // ddsi0d =   ddpsi0(xd)*dd2i_sav[iat];
    // ddsi1d =   ddpsi1(xd)*ddi_sav[iat];
    // ddsi2d =   ddpsi2(xd);

    // ddsi0md =  ddpsi0(mxd)*dd2i_sav[iat];
    // ddsi1md = -ddpsi1(mxd)*ddi_sav[iat];
    // ddsi2md =  ddpsi2(mxd);

    // the free energy
    Float free = constants::h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md);

    // derivative with respect to rhosity
    Float df_d = constants::h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    // derivative with respect to temperature
    Float df_t = constants::h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);

    // derivative with respect to rhosity**2
    // df_dd = h5(fi,
    //		si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
    //		ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md)

    // derivative with respect to temperature**2
    Float df_tt =
        constants::h5(fi, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);

    // derivative with respect to temperature and rhosity
    Float df_dt =
        constants::h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    // now get the pressure derivative with rhosity, chemical potential, and
    // electron positron number rhosities
    // get the interpolation weight functions
    si0t = constants::xpsi0(xt);
    si1t = constants::xpsi1(xt) * constants::DEVICE_ACCESS(dt_sav)[jat];

    si0mt = constants::xpsi0(mxt);
    si1mt = -constants::xpsi1(mxt) * constants::DEVICE_ACCESS(dt_sav)[jat];

    si0d = constants::xpsi0(xd);
    si1d = constants::xpsi1(xd) * constants::DEVICE_ACCESS(dd_sav)[iat];

    si0md = constants::xpsi0(mxd);
    si1md = -constants::xpsi1(mxd) * constants::DEVICE_ACCESS(dd_sav)[iat];

    // derivatives of weight functions
    dsi0t = constants::xdpsi0(xt) * constants::DEVICE_ACCESS(dti_sav)[jat];
    dsi1t = constants::xdpsi1(xt);

    dsi0mt = -constants::xdpsi0(mxt) * constants::DEVICE_ACCESS(dti_sav)[jat];
    dsi1mt = constants::xdpsi1(mxt);

    dsi0d = constants::xdpsi0(xd) * constants::DEVICE_ACCESS(ddi_sav)[iat];
    dsi1d = constants::xdpsi1(xd);

    dsi0md = -constants::xdpsi0(mxd) * constants::DEVICE_ACCESS(ddi_sav)[iat];
    dsi1md = constants::xdpsi1(mxd);

    // move table values into coefficient table
    fi[0]  = constants::DEVICE_ACCESS(dpdf)[iat + 0][jat + 0];
    fi[1]  = constants::DEVICE_ACCESS(dpdf)[iat + 1][jat + 0];
    fi[2]  = constants::DEVICE_ACCESS(dpdf)[iat + 0][jat + 1];
    fi[3]  = constants::DEVICE_ACCESS(dpdf)[iat + 1][jat + 1];
    fi[4]  = constants::DEVICE_ACCESS(dpdft)[iat + 0][jat + 0];
    fi[5]  = constants::DEVICE_ACCESS(dpdft)[iat + 1][jat + 0];
    fi[6]  = constants::DEVICE_ACCESS(dpdft)[iat + 0][jat + 1];
    fi[7]  = constants::DEVICE_ACCESS(dpdft)[iat + 1][jat + 1];
    fi[8]  = constants::DEVICE_ACCESS(dpdfd)[iat + 0][jat + 0];
    fi[9]  = constants::DEVICE_ACCESS(dpdfd)[iat + 1][jat + 0];
    fi[10] = constants::DEVICE_ACCESS(dpdfd)[iat + 0][jat + 1];
    fi[11] = constants::DEVICE_ACCESS(dpdfd)[iat + 1][jat + 1];
    fi[12] = constants::DEVICE_ACCESS(dpdfdt)[iat + 0][jat + 0];
    fi[13] = constants::DEVICE_ACCESS(dpdfdt)[iat + 1][jat + 0];
    fi[14] = constants::DEVICE_ACCESS(dpdfdt)[iat + 0][jat + 1];
    fi[15] = constants::DEVICE_ACCESS(dpdfdt)[iat + 1][jat + 1];

    Float dpepdd = constants::h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);
    dpepdd       = std::max<Float>(ye * dpepdd, (Float)1.e-30);

    // move table values into coefficient table
    fi[0]  = constants::DEVICE_ACCESS(ef)[iat + 0][jat + 0];
    fi[1]  = constants::DEVICE_ACCESS(ef)[iat + 1][jat + 0];
    fi[2]  = constants::DEVICE_ACCESS(ef)[iat + 0][jat + 1];
    fi[3]  = constants::DEVICE_ACCESS(ef)[iat + 1][jat + 1];
    fi[4]  = constants::DEVICE_ACCESS(eft)[iat + 0][jat + 0];
    fi[5]  = constants::DEVICE_ACCESS(eft)[iat + 1][jat + 0];
    fi[6]  = constants::DEVICE_ACCESS(eft)[iat + 0][jat + 1];
    fi[7]  = constants::DEVICE_ACCESS(eft)[iat + 1][jat + 1];
    fi[8]  = constants::DEVICE_ACCESS(efd)[iat + 0][jat + 0];
    fi[9]  = constants::DEVICE_ACCESS(efd)[iat + 1][jat + 0];
    fi[10] = constants::DEVICE_ACCESS(efd)[iat + 0][jat + 1];
    fi[11] = constants::DEVICE_ACCESS(efd)[iat + 1][jat + 1];
    fi[12] = constants::DEVICE_ACCESS(efdt)[iat + 0][jat + 0];
    fi[13] = constants::DEVICE_ACCESS(efdt)[iat + 1][jat + 0];
    fi[14] = constants::DEVICE_ACCESS(efdt)[iat + 0][jat + 1];
    fi[15] = constants::DEVICE_ACCESS(efdt)[iat + 1][jat + 1];

    // electron chemical potential etaele
    Float etaele = constants::h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to rhosity
    x            = constants::h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md);
    Float detadd = ye * x;

    // derivative with respect to temperature
    Float detadt = constants::h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to abar and zbar
    Float detada = -x * din * ytot1;
    Float detadz = x * rho * ytot1;

    // move table values into coefficient table
    fi[0]  = constants::DEVICE_ACCESS(xf)[iat + 0][jat + 0];
    fi[1]  = constants::DEVICE_ACCESS(xf)[iat + 1][jat + 0];
    fi[2]  = constants::DEVICE_ACCESS(xf)[iat + 0][jat + 1];
    fi[3]  = constants::DEVICE_ACCESS(xf)[iat + 1][jat + 1];
    fi[4]  = constants::DEVICE_ACCESS(xft)[iat + 0][jat + 0];
    fi[5]  = constants::DEVICE_ACCESS(xft)[iat + 1][jat + 0];
    fi[6]  = constants::DEVICE_ACCESS(xft)[iat + 0][jat + 1];
    fi[7]  = constants::DEVICE_ACCESS(xft)[iat + 1][jat + 1];
    fi[8]  = constants::DEVICE_ACCESS(xfd)[iat + 0][jat + 0];
    fi[9]  = constants::DEVICE_ACCESS(xfd)[iat + 1][jat + 0];
    fi[10] = constants::DEVICE_ACCESS(xfd)[iat + 0][jat + 1];
    fi[11] = constants::DEVICE_ACCESS(xfd)[iat + 1][jat + 1];
    fi[12] = constants::DEVICE_ACCESS(xfdt)[iat + 0][jat + 0];
    fi[13] = constants::DEVICE_ACCESS(xfdt)[iat + 1][jat + 0];
    fi[14] = constants::DEVICE_ACCESS(xfdt)[iat + 0][jat + 1];
    fi[15] = constants::DEVICE_ACCESS(xfdt)[iat + 1][jat + 1];

    // electron + positron number rhosities
    Float xnefer = constants::h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to rhosity
    x            = constants::h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md);
    x            = std::max<Float>(x, (Float)1e-30);
    Float dxnedd = ye * x;

    // derivative with respect to temperature
    Float dxnedt = constants::h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to abar and zbar
    Float dxneda = -x * din * ytot1;
    Float dxnedz = x * rho * ytot1;

    // the desired electron-positron thermodynamic quantities

    // dpepdd at high temperatures and low rhosities is below the
    // floating point limit of the subtraction of two large terms.
    // since dpresdd doesn't enter the maxwell relations at all, use the
    // bicubic interpolation done above instead of the formally correct expression
    x            = din * din;
    Float pele   = x * df_d;
    Float dpepdt = x * df_dt;
    // dpepdd  = ye*(x*df_dd + 2.0*din*df_d)
    s            = dpepdd / ye - 2.0 * din * df_d;
    Float dpepda = -ytot1 * (2.0 * pele + s * din);
    Float dpepdz = rho * ytot1 * (2.0 * din * df_d + s);

    x            = ye * ye;
    Float sele   = -df_t * ye;
    Float dsepdt = -df_tt * ye;
    Float dsepdd = -df_dt * x;
    Float dsepda = ytot1 * (ye * df_dt * din - sele);
    Float dsepdz = -ytot1 * (ye * df_dt * rho + df_t);

    /* debug: */
#if !DEVICE_CODE
    if (debug) std::cout << "dsepdt=" << dsepdt << " = -df_tt=" << df_tt << " * ye=" << ye << "\n";
#endif

    Float eele   = ye * free + temp * sele;
    Float deepdt = temp * dsepdt;
    Float deepdd = x * df_d + temp * dsepdd;
    Float deepda = -ye * ytot1 * (free + df_d * din) + temp * dsepda;
    Float deepdz = ytot1 * (free + ye * df_d * rho) + temp * dsepdz;

    /* debug: */
#if !DEVICE_CODE
    if (debug)
        std::cout << "deepdt=" << deepdt << " = dsepdt=" << dsepdt << " * temp"
                  << "\n";
#endif

    // coulomb section:

    // uniform background corrections only
    // from yakovlev & shalybkov 1989
    // lami is the average ion seperation
    // plasg is the plasma coupling parameter

    z          = constants::pi * 4. / 3.;
    s          = z * xni;
    Float dsdd = z * dxnidd;
    Float dsda = z * dxnida;

    /* debug: */
#if !DEVICE_CODE
    if (debug) std::cout << "s=" << s << " = z=" << z << " * xni=" << xni << "\n";
#endif

    Float lami     = std::pow((Float)1. / s, (Float)1. / 3.);
    Float inv_lami = 1. / lami;
    z              = -lami / 3;
    Float lamidd   = z * dsdd / s;
    Float lamida   = z * dsda / s;

    Float plasg   = zbar * zbar * constants::esqu * ktinv * inv_lami;
    z             = -plasg * inv_lami;
    Float plasgdd = z * lamidd;
    Float plasgda = z * lamida;
    Float plasgdt = -plasg * ktinv * constants::kerg;
    Float plasgdz = 2.0 * plasg / zbar;

    /* debug: */
#if !DEVICE_CODE
    if (debug)
        std::cout << "plasg=" << plasg << " = zbar=" << zbar << "^2 * esqu=" << constants::esqu << " * ktinv=" << ktinv
                  << " * inv_lami=" << inv_lami << "\n";
#endif

    Float ecoul, pcoul, scoul, decouldd, decouldt, decoulda, decouldz, dpcouldd, dpcouldt, dpcoulda, dpcouldz, dscouldd,
        dscouldt, dscoulda, dscouldz;

    // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.)
    {
        x     = std::pow(plasg, (Float)0.25);
        y     = constants::avo * ytot1 * constants::kerg;
        ecoul = y * temp * (constants::a1 * plasg + constants::b1 * x + constants::c1 / x + constants::d1);
        pcoul = rho * ecoul / 3.;
        scoul = -y * (3.0 * constants::b1 * x - 5.0 * constants::c1 / x + constants::d1 * (std::log(plasg) - 1.) -
                      constants::e1);

        y = constants::avo * ytot1 * kt * (constants::a1 + 0.25 / plasg * (constants::b1 * x - constants::c1 / x));
        decouldd = y * plasgdd;
        decouldt = y * plasgdt + ecoul / temp;
        decoulda = y * plasgda - ecoul / abar;
        decouldz = y * plasgdz;

        /* debug: */
#if !DEVICE_CODE
        if (debug)
            std::cout << "decouldt=" << decouldt << " = y=" << y << " * plasgdt=" << decouldt << " + ecoul=" << ecoul
                      << " / temp"
                      << "\n";
#endif

        y        = rho / 3.;
        dpcouldd = ecoul + y * decouldd / 3.;
        dpcouldt = y * decouldt;
        dpcoulda = y * decoulda;
        dpcouldz = y * decouldz;

        y = -constants::avo * constants::kerg / (abar * plasg) *
            (0.75 * constants::b1 * x + 1.25 * constants::c1 / x + constants::d1);
        dscouldd = y * plasgdd;
        dscouldt = y * plasgdt;
        dscoulda = y * plasgda - scoul / abar;
        dscouldz = y * plasgdz;

        // yakovlev & shalybkov 1989 equations 102, 103, 104
    }
    else if (plasg < 1.)
    {
        x     = plasg * std::sqrt(plasg);
        y     = std::pow(plasg, (Float)constants::b2);
        z     = constants::c2 * x - constants::a2 * y / 3.;
        pcoul = -pion * z;
        ecoul = 3.0 * pcoul / rho;
        scoul = -constants::avo / abar * constants::kerg *
                (constants::c2 * x - constants::a2 * (constants::b2 - 1.) / constants::b2 * y);

        s        = 1.5 * constants::c2 * x / plasg - constants::a2 * constants::b2 * y / plasg / 3.;
        dpcouldd = -dpiondd * z - pion * s * plasgdd;
        dpcouldt = -dpiondt * z - pion * s * plasgdt;
        dpcoulda = -dpionda * z - pion * s * plasgda;
        dpcouldz = -dpiondz * z - pion * s * plasgdz;

        s        = 3.0 / rho;
        decouldd = s * dpcouldd - ecoul / rho;
        decouldt = s * dpcouldt;
        decoulda = s * dpcoulda;
        decouldz = s * dpcouldz;

        /* debug: */
#if !DEVICE_CODE
        if (debug) std::cout << "decouldt=" << decouldt << " = s=" << s << " * dpcouldt=" << dpcouldt << "\n";
#endif

        s = -constants::avo * constants::kerg / (abar * plasg) *
            (1.5 * constants::c2 * x - constants::a2 * (constants::b2 - 1.) * y);
        dscouldd = s * plasgdd;
        dscouldt = s * plasgdt;
        dscoulda = s * plasgda - scoul / abar;
        dscouldz = s * plasgdz;
    }

    // bomb proof
    x = prad + pion + pele + pcoul;
    y = erad + eion + eele + ecoul;
    z = srad + sion + sele + scoul;

    // if (x .le. 0.0 .or. y .le. 0.0 .or. z .le. 0.0) then
    // if (x .le. 0.0) then
    if (x <= 0. || y <= 0.)
    {
        pcoul    = 0.;
        dpcouldd = 0.;
        dpcouldt = 0.;
        dpcoulda = 0.;
        dpcouldz = 0.;
        ecoul    = 0.;
        decouldd = 0.;
        decouldt = 0.;
        decoulda = 0.;
        decouldz = 0.;
        scoul    = 0.;
        dscouldd = 0.;
        dscouldt = 0.;
        dscoulda = 0.;
        dscouldz = 0.;
    }

    // sum all the gas components
    Float pgas = pion + pele + pcoul;
    Float egas = eion + eele + ecoul;
    Float sgas = sion + sele + scoul;

    Float dpgasdd = dpiondd + dpepdd + dpcouldd;
    Float dpgasdt = dpiondt + dpepdt + dpcouldt;
    Float dpgasda = dpionda + dpepda + dpcoulda;
    Float dpgasdz = dpiondz + dpepdz + dpcouldz;

    Float degasdd = deiondd + deepdd + decouldd;
    Float degasdt = deiondt + deepdt + decouldt;
    Float degasda = deionda + deepda + decoulda;
    Float degasdz = deiondz + deepdz + decouldz;

    Float dsgasdd = dsiondd + dsepdd + dscouldd;
    Float dsgasdt = dsiondt + dsepdt + dscouldt;
    Float dsgasda = dsionda + dsepda + dscoulda;
    Float dsgasdz = dsiondz + dsepdz + dscouldz;

    /* debug: */
#if !DEVICE_CODE
    if (debug)
        std::cout << "degasdt=" << degasdt << " = deiondt=" << deiondt << " + deepdt=" << deepdt
                  << " + decouldt=" << decouldt << "\n";
#endif

    // add in radiation to get the total
    Float pres = prad + pgas;
    Float ener = erad + egas;
    Float entr = srad + sgas;

    Float dpresdd = dpraddd + dpgasdd;
    Float dpresdt = dpraddt + dpgasdt;
    Float dpresda = dpradda + dpgasda;
    Float dpresdz = dpraddz + dpgasdz;

    Float rhoerdd = deraddd + degasdd;
    Float rhoerdt = deraddt + degasdt;
    Float rhoerda = deradda + degasda;
    Float rhoerdz = deraddz + degasdz;

    Float rhotrdd = dsraddd + dsgasdd;
    Float rhotrdt = dsraddt + dsgasdt;
    Float rhotrda = dsradda + dsgasda;
    Float rhotrdz = dsraddz + dsgasdz;

    /* debug: */
#if !DEVICE_CODE
    if (debug) std::cout << "rhoerdt(cv)=" << rhoerdt << " = deraddt=" << deraddt << " + degasdt=" << degasdt << "\n\n";
#endif

    // for the gas
    // the temperature and rhosity exponents (c&g 9.81 9.82)
    // the specific heat at constant volume (c&g 9.92)
    // the third adiabatic exponent (c&g 9.93)
    // the first adiabatic exponent (c&g 9.97)
    // the second adiabatic exponent (c&g 9.105)
    // the specific heat at constant pressure (c&g 9.98)
    // and relativistic formula for the sound speed (c&g 14.29)

    nnet::eos_struct<Float> res;

    Float zz        = pgas * rhoi;
    Float zzi       = rho / pgas;
    Float chit_gas  = temp / pgas * dpgasdt;
    Float chid_gas  = dpgasdd * zzi;
    res.cv_gaz      = degasdt;
    x               = zz * chit_gas / (temp * res.cv_gaz);
    Float gam3_gas  = x + 1.;
    Float gam1_gas  = chit_gas * x + chid_gas;
    Float nabad_gas = x / gam1_gas;
    Float gam2_gas  = 1. / (1. - nabad_gas);
    res.cp_gaz      = res.cv_gaz * gam1_gas / chid_gas;
    z               = 1. + (egas + constants::clight * constants::clight) * zzi;
    res.c_gaz       = constants::clight * std::sqrt(gam1_gas / z);

    // for the totals
    zz          = pres * rhoi;
    zzi         = rho / pres;
    Float chit  = temp / pres * dpresdt;
    Float chid  = dpresdd * zzi;
    res.cv      = rhoerdt;
    x           = zz * chit / (temp * res.cv);
    Float gam3  = x + 1.;
    Float gam1  = chit * x + chid;
    Float nabad = x / gam1;
    Float gam2  = 1. / (1. - nabad);
    res.cp      = res.cv * gam1 / chid;
    z           = 1. + (ener + constants::clight * constants::clight) * zzi;
    res.c       = constants::clight * std::sqrt(gam1 / z);

    // maxwell relations; each is zero if the consistency is perfect
    x       = rho * rho;
    res.dse = temp * rhotrdt / rhoerdt - 1.;
    res.dpe = (rhoerdd * x + temp * dpresdt) / pres - 1.;
    res.dsp = -rhotrdd * x / dpresdt - 1.;

    // Needed output
    res.dpdT  = dpresdt;
    res.dudYe = degasdz * abar;
    res.p     = pres;
    res.u     = ener;

    delete[] fi;

#ifdef DEBUG_HELM
    res.cv   = constants::DEVICE_ACCESS(xf)[iat][jat]; // rho;  // (Float)iat
    res.u    = constants::DEVICE_ACCESS(d)[iat];       // temp; // (Float)jat
    res.dpdT = constants::DEVICE_ACCESS(t_)[jat];      // zbar_;
#endif

    return res;
}

} // namespace helmholtz

/*! @brief Helmholtz functor class */
template<typename Float>
class HelmholtzFunctor : public nnet::EosFunctor<Float>
{
private:
    const Float* Z;
    int          dimension;
#if COMPILE_DEVICE
    Float* devZ;
#endif

public:
    HelmholtzFunctor(const Float* Z_, int dimension_)
        : Z(Z_)
        , dimension(dimension_)
    {
#if COMPILE_DEVICE
        gpuErrchk(cudaMalloc(&devZ, dimension * sizeof(Float)));
        gpuErrchk(cudaMemcpy(devZ, Z, dimension * sizeof(Float), cudaMemcpyHostToDevice));
#endif
    }
    template<class Vector>
    HelmholtzFunctor(const Vector& Z_, int dimension_)
        : HelmholtzFunctor(Z_.data(), dimension_)
    {
    }
    template<class Vector>
    HelmholtzFunctor(const Vector& Z_)
        : HelmholtzFunctor(Z_.data(), Z_.size())
    {
    }

    HelmholtzFunctor(const std::vector<Float>& Z_, int dimension_)
        : HelmholtzFunctor(Z_.data(), dimension_)
    {
    }
    HelmholtzFunctor(const std::vector<Float>& Z_)
        : HelmholtzFunctor(Z_.data(), Z_.size())
    {
    }

    template<size_t n>
    HelmholtzFunctor(const std::array<Float, n>& Z_, int dimension_)
        : HelmholtzFunctor(Z_.data(), dimension_)
    {
    }
    template<size_t n>
    HelmholtzFunctor(const std::array<Float, n>& Z_)
        : HelmholtzFunctor(Z_.data(), Z_.size())
    {
    }

    HOST_DEVICE_FUN HelmholtzFunctor() {}
    HOST_DEVICE_FUN ~HelmholtzFunctor()
    {
#if COMPILE_DEVICE && !DEVICE_CODE
        cudaFree(devZ);
#endif
    }

    /*! @brief Ideal gas EOS for nuclear networks.
     *
     * @param Y    molar proportions
     * @param temp temperature
     * @param rho  density
     *
     * Returns Helmholtz EOS output struct.
     */
    HOST_DEVICE_FUN nnet::eos_struct<Float> inline operator()(const Float* Y, const Float temp,
                                                              const Float rho) const override
    {
        // compute abar and zbar
        double abar = algorithm::accumulate(Y, Y + dimension, (double)0.);
        double zbar = eigen::dot(Y, Y + dimension, DEVICE_ACCESS(Z));

        return helmholtz::helmholtzEos(abar, zbar, temp, rho);
    }
};
} // namespace nnet::eos