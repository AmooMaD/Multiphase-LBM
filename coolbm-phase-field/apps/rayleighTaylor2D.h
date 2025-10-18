#pragma once
// ─────────────────────────   standard headers   ────────────────────────────
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <execution>
#include <functional>
#include <chrono>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <cstdlib>
#include <numeric>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

using namespace std;
using namespace std::chrono;

// ─────────────────────────   cell-level helpers   ───────────────────────────
enum class CellType_rayleighTaylor2D : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
inline auto d2q9_constants_rayleighTaylor2D()
{
    vector<array<int,2>> c_vect = {
        {-1,  0}, { 0, -1}, {-1, -1}, {-1,  1}, { 0,  0},
        { 1,  0}, { 0,  1}, { 1,  1}, { 1, -1}
    };
    vector<int>    opp_vect = { 5, 6, 7, 8, 4, 0, 1, 2, 3 };
    vector<double> t_vect   = {
        1./9., 1./9., 1./36., 1./36., 4./9.,
        1./9., 1./9., 1./36., 1./36.
    };
    return make_tuple(c_vect, opp_vect, t_vect);
}

// ─────────────────────────   geometry helpers   ────────────────────────────
struct Dim_rayleighTaylor2D {

    Dim_rayleighTaylor2D(int nx_, int ny_)
        : nx(nx_), ny(ny_),
          nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny)),
          npop (9 * nelem) {}
          int    nx, ny;
          size_t nelem, npop;
};

// ─────────────────────────   lattice-unit parameters   ─────────────────────
inline auto lbParameters_rayleighTaylor2D(double ulb, int lref, double Re)
{
    double nu    = ulb * static_cast<double>(lref) / Re;
    double omega = 1. / (3.*nu + 0.5);
    double dx    = 1. / static_cast<double>(lref);
    double dt    = dx * ulb;
    return make_tuple(nu, omega, dx, dt);
}

inline void printParameters_rayleighTaylor2D(const Dim_rayleighTaylor2D &dim, double Re, double omega,
                                             double ulb, int N, double max_t , double nu)
{
    cout << "Rayleigh–Taylor 2-D problem\n"
         << "N      = " << N      << '\n'
         << "nx     = " << dim.nx      << '\n'
         << "ny     = " << dim.ny      << '\n'
         << "Re     = " << Re     << '\n'
         << "omega  = " << omega  << '\n'
         << "tau    = " << 1. / omega  << '\n'
         << "nu     = " << nu  << '\n'
         << "ulb    = " << ulb    << '\n'
         << "max_t  = " << max_t  << '\n';
}

// ─────────────────────────   timing helpers   ──────────────────────────────
inline auto restartClock_rayleighTaylor2D()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_rayleighTaylor2D(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = us.count();
    cout << "Runtime: " << dure/1000000. << " seconds " << endl;
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_rayleighTaylor2D
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem
                                                     + 2 * 9 * nelem
                                                     //+ 2 * nelem
                                                     ; }

    // raw buffers
    CellData*                 lattice;
    CellType_rayleighTaylor2D* flag;
    int*                      parity;
    std::array<int,2>*        c;
    int*                      opp;
    double*                   t;
    double                    omega;
    
    double phi_l;
    double phi_g;
    double rho_l;
    double rho_g;
    double a;
    double b ;
    double kappa;
    double gravity;
    Dim_rayleighTaylor2D      dim;

    // ───────── index helpers ─────────
    size_t xyz_to_i (int x, int y) {
        return (y + dim.ny * x);
    };

    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);
        int remainder = i % (dim.ny);
        int iY = remainder;
        return std::make_tuple(iX, iY);
    };

    // ───────── array accessors ───────
    double& f   (int i,int k){ return lattice[              k*dim.nelem + i]; }
    double& fin (int i,int k){ return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    double& fout(int i,int k){ return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    double& g   (int i,int k){ return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    double& gin (int i,int k){ return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    double& gout(int i,int k){ return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    //double& Pin (int i)      { return lattice[*parity  *dim.nelem + 4 * dim.npop + i];}
    //double& Pout(int i)      { return lattice[(1-*parity)*dim.nelem + 4 * dim.npop + i];}

    // ─────────────────────────────────── physics kernels ────────────────────

    /* =====================================================================
    Step + cosinus interface initialization (like LB::Init)
    ===================================================================== */

    void iniLattice(double& f0)
    {
        // 1) compute linear index i and (iX, iY)
        size_t i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(int(i));

        // 2) compute the “mean” interface height as before
        double x = static_cast<double>(iX);
        double interface = (static_cast<double>(dim.ny) / 2.0)
                        + (static_cast<double>(dim.nx)) * 0.1 * std::cos(2.0 * M_PI * x / (static_cast<double>(dim.nx - 1)));

        // width parameter for the tanh transition
        double w = 1.25;

        // 3) smooth φ with a tanh profile across the interface
        //    φ = ½(φ_l + φ_g) + ½(φ_l – φ_g)·tanh[(y – interface)/(2·w)]
        double y = static_cast<double>(iY);
        double phi = 0.5 * (phi_l + phi_g)
                + 0.5 * (phi_l - phi_g) * std::tanh((y - interface) / (2.0 * w));

        double rho = rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);


        // 4) compute pth = p_th(phi)
        double rt  = b * phi / 4.0;
        double pth = (phi / 3.0) * (1.0 + rt + rt * rt - rt * rt * rt) / std::pow(1.0 - rt, 3)
                - a * phi * phi;

        double rt_rho  = b * rho / 4.0;
        double p_rho = (rho / 3.0) * (1.0 + rt_rho + rt_rho * rt_rho - rt_rho * rt_rho * rt_rho) / std::pow(1.0 - rt_rho, 3) - a * rho * rho;
        // 5) initialize populations to equilibrium (Ux=Uy=0)
        for (int k = 0; k < 9; ++k)
        {
            fin(i, k) = phi * t[k];
            gin(i, k) = p_rho * t[k];

        }
        //Pin(i) = p_rho;
    }


    // macro-vars ϕ, P, u
    auto macro_phi_P(double& f0)
    {
        int i = &f0 - lattice;
        double phi , P;

        double Xf_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double Xf_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);
        double Xf_0  = fin(i, 6) + fin(i, 1) + fin(i, 4);

        double Xg_M1 = gin(i, 0) + gin(i, 2) + gin(i, 3);
        double Xg_P1 = gin(i, 5) + gin(i, 7) + gin(i, 8);
        double Xg_0  = gin(i, 6) + gin(i, 1) + gin(i, 4);

        phi  = Xf_M1 + Xf_P1 + Xf_0;
        P    = Xg_M1 + Xg_P1 + Xg_0;

        return make_tuple(phi, P);
    }

    auto macro_u(double& f0)
    {
        int i = &f0 - lattice;
        array<double,2> u;

        double Xg_M1 = gin(i, 0) + gin(i, 2) + gin(i, 3);
        double Xg_P1 = gin(i, 5) + gin(i, 7) + gin(i, 8);

        double Yg_P1 = gin(i, 3) + gin(i, 7) + gin(i, 6);
        double Yg_M1 = gin(i, 2) + gin(i, 1) + gin(i, 8);

        u[0] = Xg_P1 - Xg_M1;
        u[1] = Yg_P1 - Yg_M1;
        return u;
    }

    double total_rho(double& f0){
        auto [phi, P] = macro_phi_P(f0);
        return rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);
    }

    double psi_phi(double& f0){
        auto [phi,P] = macro_phi_P(f0);
        double rt = b*phi/4.0;
        double pth = (phi/3.0)*(1 + rt + rt*rt - rt*rt*rt)/pow(1 - rt, 3) - a*phi*phi;
        return pth - phi/3.0;
    }

    // ------------------------------------------------------------------
    // ∇²ρ   (lattice Laplacian of the local density ρ)
    // ------------------------------------------------------------------

    double laplacian_rho(double& f0)
    {
        size_t i   = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));
        double rho_c = total_rho(lattice[i]);

        double sum   = 0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);

                double rho_nb = total_rho(lattice[nbb]);
                sum += t[k] * (rho_nb - rho_c);
            }
            else{
                double rho_nb = total_rho(lattice[nb]);
                sum += t[k] * (rho_nb - rho_c);
            }
        }
        return 6.0 * sum;  // 2 / c_s^2  with c_s^2 = 1/3
    }
    // ------------------------------------------------------------------
    // ∇(∇²ρ)   : gradient of the Laplacian of ρ
    // ------------------------------------------------------------------


     std::array<double,2> grad_lap_rho(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);

                double lap_nb = laplacian_rho(lattice[nbb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = laplacian_rho(lattice[nb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gx, 3.0*gy };  // 1 / c_s^2 factor
    }


    //───────────────────────────────────────────────────────────────//
    //  helper: full velocity  u  (adds interfacial / gravity force)
    //───────────────────────────────────────────────────────────────//
    std::array<double,2> velocity(double& f0)
    {
        auto u = macro_u(f0);
        double rho = total_rho(f0);
        auto [phi, p] = macro_phi_P(f0);

        auto glap_rho = grad_lap_rho(f0);
        auto glap_phi = grad_lap_phi(f0);

        double forcex = kappa * rho * glap_phi[0];	///////////////
        double forcey = kappa * rho * glap_phi[1];	///////////////

        forcey += gravity * rho;

        u[0] += forcex / 6.0;
        u[1] += forcey / 6.0;

        u[0] /= (rho/3.0);
        u[1] /= (rho/3.0);
        
        return u;
    }
    // ------------------------------------------------------------------
    // ∇ψ(φ)
    // ------------------------------------------------------------------
    std::array<double,2> grad_psi_phi(double& f0)  // changed psi_phi to psi_rho!!!!
        {
            size_t i     = &f0 - lattice;
            auto [iX,iY] = i_to_xyz(int(i));

            double gx=0.0, gy=0.0;
            for (int k=0;k<9;++k) {
                int ix = iX + c[k][0];
                int iy = iY + c[k][1];
                ix = (ix + dim.nx) % dim.nx;
                int nb = xyz_to_i(ix,iy);

                double psi_nb;
                if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                    int ixbb = iX - c[k][0];
                    int iybb = iY - c[k][1];
                    ixbb = (ixbb + dim.nx) % dim.nx;
                    int nbb = xyz_to_i(ixbb,iybb);

                    psi_nb = psi_phi(lattice[nbb]);
                    gx += t[k] * c[k][0] * psi_nb;
                    gy += t[k] * c[k][1] * psi_nb;
                }
                else {
                    psi_nb = psi_phi(lattice[nb]);
                    gx += t[k] * c[k][0] * psi_nb;
                    gy += t[k] * c[k][1] * psi_nb;
                }
            }
            return { 3.0*gx, 3.0*gy };  // 1 / c_s^2 factor
        }


    double psi_rho(double& f0) {
        double rho = total_rho(f0);
        double rt = b*rho/4.0;
        double p_rho = (rho/3.0)*(1 + rt + rt*rt - rt*rt*rt)/pow(1 - rt, 3) - a*rho*rho;
        return p_rho - rho/3.0;
    }

    // ------------------------------------------------------------------
    // ∇ψ(ρ)
    // ------------------------------------------------------------------

     std::array<double,2> grad_psi_rho(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);
                double lap_nb = psi_rho(lattice[nbb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = psi_rho(lattice[nb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gx, 3.0*gy };  // 1 / c_s^2 factor
    }


    // ------------------------------------------------------------------
    // ∇ρ
    // ------------------------------------------------------------------

     std::array<double,2> grad_rho(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);
                double lap_nb = total_rho(lattice[nbb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = total_rho(lattice[nb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gx, 3.0*gy };  // 1 / c_s^2 factor
    }


    //───────────────────────────────────────────────────────────────//
    //  helper: full pressure  P  (bulk + kinetic correction)
    //───────────────────────────────────────────────────────────────//
    double total_P(double& f0)					// if you wanna use pth like pressure, check the equation 2.30 in the multiphase lattice boltzmann methods
    {
        auto [phi, P_term] = macro_phi_P(f0);
        auto u = velocity(f0);
        //auto gpsi = grad_psi_rho(f0);
        auto gpsi = grad_rho(f0);

        return P_term - 0.5 * (u[0] * -gpsi[0] / 3. + u[1] * -gpsi[1] / 3.);
    }



    // ------------------------------------------------------------------
    // ∇²φ   (lattice Laplacian of the local order parameter φ)
    // ------------------------------------------------------------------
    double laplacian_phi(double& f0)
    {
        size_t i   = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));
        auto [phi_c, p] = macro_phi_P(lattice[i]);

        double sum   = 0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);

                auto [phi_nbb, p_nbb] = macro_phi_P(lattice[nbb]);
                sum += t[k] * (phi_nbb - phi_c);
            }
            else{
                auto [phi_nb, p_nb] = macro_phi_P(lattice[nb]);
                sum += t[k] * (phi_nb - phi_c);
            }
        }
        return 6.0 * sum;  // 2 / c_s^2  with c_s^2 = 1/3
    }


    // ------------------------------------------------------------------
    // ∇(∇²φ)   : gradient of the Laplacian of φ
    // ------------------------------------------------------------------
    std::array<double,2> grad_lap_phi(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0;
        for (int k=0;k<9;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            ix = (ix + dim.nx) % dim.nx;
            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
                int ixbb = iX - c[k][0];
                int iybb = iY - c[k][1];
                ixbb = (ixbb + dim.nx) % dim.nx;
                int nbb = xyz_to_i(ixbb,iybb);

                double lap_nb = laplacian_phi(lattice[nbb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = laplacian_phi(lattice[nb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gx, 3.0*gy };  // 1 / c_s^2 factor
    }


    // single-population stream
    void stream(int i,int k,int iX,int iY,double pf,double pg)
    {
        int XX = iX + c[k][0];
        int YY = iY + c[k][1];
        XX = (XX + dim.nx) % dim.nx;

        size_t nb = xyz_to_i(XX,YY);

        if (flag[nb] == CellType_rayleighTaylor2D::bounce_back) {
            fout(i,opp[k]) = pf;
            gout(i,opp[k]) = pg;
        }
        else {
            fout(nb,k) = pf;
            gout(nb,k) = pg;
        }
    }

    // BGK collision for the pair (f,g)
    std::tuple<double,double,double,double>
    collideBgk(int i,int k,
               const std::array<double,2>& u,
               double rho, double phi, double P, double usqr,
               const std::array<double,2>& grad_psi_rho,
               const std::array<double,2>& grad_psi_phi,
               const std::array<double,2>& grad_lap_rho,
               const std::array<double,2>& grad_lap_phi)
    {
        double ck_u = c[k][0]*u[0] + c[k][1]*u[1];
        double ck_u_op = c[opp[k]][0]*u[0] + c[opp[k]][1]*u[1];

        double eqf    = phi * t[k] * (1 + 3*ck_u + 4.5*ck_u*ck_u - usqr);
        double eqf_op = phi * t[opp[k]] * (1 + 3*ck_u_op + 4.5*ck_u_op*ck_u_op - usqr);

        double eqg    = t[k] * (P + (rho/3.0)*(3*ck_u + 4.5*ck_u*ck_u - usqr));
        double eqg_op = t[opp[k]] * (P + (rho/3.0)*(3*ck_u_op + 4.5*ck_u_op*ck_u_op - usqr));

        double e_u_x = c[k][0] - u[0];
        double e_u_x_op = c[opp[k]][0] - u[0];

        double e_u_y = c[k][1] - u[1];
        double e_u_y_op = c[opp[k]][1] - u[1];

        double forcex = kappa * rho * grad_lap_phi[0];  ////////////
        double forcey = kappa * rho * grad_lap_phi[1];	////////////
        forcey += gravity * rho;

        double Ex = grad_psi_rho[0];
        double Ey = grad_psi_rho[1];

        double fg = (1. - 0.5 * omega ) *
                    ((e_u_x * forcex + e_u_y * forcey) * eqf / phi)
                  + (1. - 0.5 * omega ) *
                    ((e_u_x * -Ex) + (e_u_y * -Ey)) * (eqf/phi - t[k]);

        double fg_op = (1. - 0.5 * omega ) *
                       ((e_u_x_op * forcex + e_u_y_op * forcey) * eqf_op / phi)
                     + (1. - 0.5 * omega ) *
                       ((e_u_x_op * -Ex) + (e_u_y_op * -Ey)) * (eqf_op/phi - t[k]);

        double ff = (1. - 0.5 * omega ) *
                    ((e_u_x * -grad_psi_phi[0]) + (e_u_y * -grad_psi_phi[1])) * 3.0 * eqf / phi;                   

        double ff_op = (1. - 0.5 * omega ) *
                       ((e_u_x_op * -grad_psi_phi[0]) + (e_u_y_op * -grad_psi_phi[1])) * 3.0 * eqf_op / phi;	
                       	
        double pf    = (1. - omega ) * fin(i,k)      + omega * eqf    + ff;
        double pg    = (1. - omega ) * gin(i,k)      + omega * eqg    + fg;

        double pf_op = (1. - omega ) * fin(i,opp[k]) + omega * eqf_op + ff_op;
        double pg_op = (1. - omega ) * gin(i,opp[k]) + omega * eqg_op + fg_op;

        return {pf, pg, pf_op, pg_op};
    }

    // main per-cell operator
    void operator()(double& f0)
    {
        int i = &f0 - lattice;
        if (flag[i] == CellType_rayleighTaylor2D::bulk) {
            auto  u   = velocity(f0);

            auto [phi, /*P_term*/ __] = macro_phi_P(f0);
            double rho = total_rho(f0);

            auto gpsi_rho = grad_psi_rho(f0);
            auto glap_rho = grad_lap_rho(f0);
            auto glap_phi = grad_lap_phi(f0);
            auto gpsi_phi = grad_psi_phi(f0);

            double P  = total_P(f0);
            //Pout(i) = P;

            double usqr = 1.5*(u[0]*u[0] + u[1]*u[1]);

            auto [iX,iY] = i_to_xyz(i);

            for (int k = 0; k < 4; ++k) {
                auto [pf, pg, pf_op, pg_op] = collideBgk(
                    i, k,
                    u, rho, phi, P, usqr,
                    gpsi_rho, gpsi_phi,
                    glap_rho, glap_phi
                );
                stream(i,     k   , iX, iY, pf   , pg);
                stream(i, opp[k], iX, iY, pf_op, pg_op);
            }

            // rest population (k = 4)
            {
                int k = 4;
                double eqf0 = phi * t[k] * (1. - usqr);
                double eqg0 = t[k] * (P  - (rho/3.0)*usqr);

                double forcex = kappa * rho * glap_phi[0];
                double forcey = kappa * rho * glap_phi[1];
                forcey += gravity * rho;

                double Ex = gpsi_rho[0];
                double Ey = gpsi_rho[1];

                double fg0 = (1. - 0.5 * omega) *
                             (- (u[0] * forcex + u[1] * forcey) * eqf0/phi
                              + ((u[0] * -Ex + u[1] * -Ey) * (eqf0/phi - t[k])));

                double ff0 = (1. - 0.5 * omega) *
                             (-3.0 * ( u[0] * -gpsi_phi[0] + u[1] * -gpsi_phi[1]) * eqf0 / phi);

                fout(i,k) = (1 - omega) * fin(i,k) + omega * eqf0 + ff0;
                gout(i,k) = (1 - omega) * gin(i,k) + omega * eqg0 + fg0;
            }
        }
    }
};

void findInterfaceHeights(
    LBM_rayleighTaylor2D& lbm,
    double phi_l,
    double phi_g,
    int&    spike_y,     // output: y‐index of interface at x=0 (or −1 if none)
    int&    bubble_y     // output: y‐index of interface at x=nx/2 (or −1 if none)
) {
    // 1) compute mid‐φ
    double phi_mid = 0.5 * (phi_l + phi_g);

    // 2) fixed x‐locations
    int x_spike  = 0;
    int x_bubble = lbm.dim.nx / 2;

    // 3) initialize outputs
    spike_y  = 0.05;
    bubble_y = -0.05;

    // 4) scan from y=dim.ny-2 down to y=1 at x=0
    for(int y = lbm.dim.ny - 2; y >= 1; --y) {
        int i = lbm.xyz_to_i(x_spike, y);
        // macro_phi_P returns std::tuple<double phi, double P>
        auto [phi_val, P_val] = lbm.macro_phi_P(lbm.lattice[i]);

        if(phi_val <= phi_mid) {
            bubble_y = y;
            break;
        }
    }

    // 5) scan from y=dim.ny-2 down to y=1 at x=nx/2
    for(int y = lbm.dim.ny - 2; y >= 1; --y) {
        int i = lbm.xyz_to_i(x_bubble, y);
        auto [phi_val, P_val] = lbm.macro_phi_P(lbm.lattice[i]);

        if(phi_val <= phi_mid) {
            spike_y = y;
            break;
        }
    }
}



// ───────────────────────── VTK output ─────────────────────────────────
void saveVtkFields_rayleighTaylor2D(LBM_rayleighTaylor2D& lbm, int time_iter, double dx = 0.) {
    using namespace std;
    Dim_rayleighTaylor2D const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / dim.nx;
    int dimZ = 1;

    stringstream ss;
    ss << "sol_" << setw(7) << setfill('0') << time_iter << ".vtk";
    ofstream os(ss.str());
    os << "# vtk DataFile Version 2.0" << endl;
    os << "iteration " << time_iter << endl;
    os << "ASCII" << endl << endl;
    os << "DATASET STRUCTURED_POINTS" << endl;
    os << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dimZ << endl;
    os << "ORIGIN 0 0 0" << endl;
    os << "SPACING " << dx << " " << dx << " " << dx << endl << endl;
    os << "POINT_DATA " << dim.nx * dim.ny * dimZ << endl;

    // φ
    os << "SCALARS phi float 1" << endl;
    os << "LOOKUP_TABLE default" << endl;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            auto [phi,P] = lbm.macro_phi_P(lbm.lattice[i]);
            if (lbm.flag[i] == CellType_rayleighTaylor2D::bounce_back) {
                double phim = 0.;
            }
            os << phi << " ";
        }
        os << endl;
    }
    os << endl;

    os << "SCALARS density float 1" << endl;
    os << "LOOKUP_TABLE default" << endl;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            auto rho = lbm.total_rho(lbm.lattice[i]);
            if (lbm.flag[i] == CellType_rayleighTaylor2D::bounce_back) {
                double rhom = 0.;
            }
            os << rho << " ";
        }
        os << endl;
    }
    os << endl;

    // Flag
    os << "SCALARS Flag int 1" << endl ;
    os << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >= 0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                if (lbm.flag[i] == CellType_rayleighTaylor2D::bounce_back) {
                    os << "1" << " ";
                }
                else {
                    os << "0" << " ";
                }
            }
            os << endl ;
        }
        os << endl ;
    }
    os << endl ;
}

// ───────────────────────── Energy computation ──────────────────────────────
double computeEnergy_rayleighTaylor2D(LBM_rayleighTaylor2D& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_rayleighTaylor2D::bulk) {
                auto u = lbm.velocity(lbm.lattice[i]);
                energy += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * energy / (dim.nx * dim.ny);
}

// ───────────────────────── Geometry initialization ─────────────────────────


void inigeom_rayleighTaylor2D(LBM_rayleighTaylor2D& lbm) {
    Dim_rayleighTaylor2D const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == 0 || y == dim.ny-1) {
                lbm.flag[i] = CellType_rayleighTaylor2D::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i,k) = 0.0;
                    lbm.gin(i,k) = 0.0;
                    //lbm.Pin(i)   = 0.0;
                }
            }
            else {
                lbm.flag[i] = CellType_rayleighTaylor2D::bulk;
            }
        }
    }
}


/*
#include <random>

void inigeom_rayleighTaylor2D(LBM_rayleighTaylor2D& lbm, double porosity = 0.85) {
    Dim_rayleighTaylor2D const& dim = lbm.dim;

    // Random number generator with fixed seed for reproducibility
    std::mt19937 gen(42);
    std::bernoulli_distribution dist(porosity);

    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);

            // Always solid boundaries at top and bottom
            if (y == 0 || y == dim.ny - 1) {
                lbm.flag[i] = CellType_rayleighTaylor2D::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.f(i, k) = 0.;
                }
            }
            else {
                // Randomly assign interior cell as fluid or solid
                if (dist(gen)) {
                    lbm.flag[i] = CellType_rayleighTaylor2D::bulk;
                } else {
                    lbm.flag[i] = CellType_rayleighTaylor2D::bounce_back;
                    for (int k = 0; k < 9; ++k) {
                        lbm.fin(i, k) = 0.;
                        lbm.gin(i,k) = 0.0;
                    }
                }
            }
        }
    }
}
*/

// ─────────────────────────   main driver   ──────────────────────────────────
void rayleighTaylor2D()
{
    ifstream contfile("../apps/Config_Files/config_rayleighTaylor2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. It should be named "
          "\"config_rayleighTaylor2D.txt\" in Files_Config.");
    }

    double Re = 0, ulb = 0, max_t = 0, phi_l = 0, phi_g = 0, rho_l = 0, rho_g = 0, a = 0, b = 0, kappa = 0, gravity = 0;
    int N = 0, out_freq = 0, vtk_freq = 0;
    string line;
    string value;
    string param;

    while (getline(contfile, line)) {
        // 1) strip comments
        if (auto pos = line.find('#'); pos != string::npos)
            line.erase(pos);

        contfile >> param;
        contfile >> value;

        if      (param == "Re")       { Re      = stod(value); }
        else if (param == "ulb")      { ulb     = stod(value); }
        else if (param == "N")        { N       = stoi(value); }
        else if (param == "max_t")    { max_t   = stod(value); }
        else if (param == "out_freq"){ out_freq = stoi(value); }
        else if (param == "vtk_freq"){ vtk_freq = stoi(value); }
        else if (param == "phi_l")    { phi_l   = stod(value); }
        else if (param == "phi_g")    { phi_g   = stod(value); }
        else if (param == "rho_l")    { rho_l   = stod(value); }
        else if (param == "rho_g")    { rho_g   = stod(value); }
        else if (param == "a")        { a       = stod(value); }
        else if (param == "b")        { b       = stod(value); }
        else if (param == "kappa")    { kappa   = stod(value); }
        else if (param == "gravity")  { gravity = stod(value); }
        else {
            cerr << "Warning: unknown parameter \"" << param << "\"\n";
        }
    }


    // lattice parameters
    Dim_rayleighTaylor2D dim {N, 4 * N+2};
    auto [nu,omega,dx,dt] = lbParameters_rayleighTaylor2D(ulb, N, Re);
    printParameters_rayleighTaylor2D(dim, Re, omega, ulb, N, max_t, nu);

    // allocate
    vector<CellData> lattice_vect(LBM_rayleighTaylor2D::sizeOfLattice(dim.nelem));
    CellData *lattice = &lattice_vect[0];

    vector<CellType_rayleighTaylor2D> flag_vect(dim.nelem);
    CellType_rayleighTaylor2D* flag = &flag_vect[0];

    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_rayleighTaylor2D();

    LBM_rayleighTaylor2D lbm{
        lattice, flag, parity,
        &c_vect[0], &opp_vect[0], &t_vect[0],
        omega, phi_l, phi_g, rho_l, rho_g, a, b, kappa, gravity, dim
    };

    // 1) initialization (step+cosinus) in parallel
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });

    // 2) mark walls and zero populations at y=0, y=ny-1
    inigeom_rayleighTaylor2D(lbm);

    auto [start, clock_iter] = restartClock_rayleighTaylor2D();
    ofstream efile("energy.dat");

    ofstream interfacepositionFile("spike_bubble_position.dat");
    
    ofstream interfacevelocityFile("spike_bubble_velocity.dat");

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_rayleighTaylor2D(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]" << endl;
            double energy = computeEnergy_rayleighTaylor2D(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
        

            // --- find the interface heights in y at x=0 and x=nx/2 ---
            int spike_y, bubble_y;
            findInterfaceHeights(lbm,
                                 lbm.phi_l,  // pass in the same phi_l
                                 lbm.phi_g,  // and phi_g you used in initialization
                                 spike_y,
                                 bubble_y);

            // If spike_y == -1 (never crossed mid_phi), you can still write -1
            // or choose to write '0' or skip. Here we'll write -1 as a sentinel.
            double y_spike_phys  = (spike_y  < 0 ? -1.0 : spike_y  * dx);
            double y_bubble_phys = (bubble_y < 0 ? -1.0 : bubble_y * dx);

            interfacepositionFile
                << setw(10) << time_iter * dt
                << setw(16) << y_spike_phys
                << setw(16) << y_bubble_phys
                << "\n";
                
                
               
                
        }

        // 3) collision + streaming in parallel
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);

        // 4) swap buffers (parity)
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_rayleighTaylor2D(start, clock_iter, dim.nelem);
}
