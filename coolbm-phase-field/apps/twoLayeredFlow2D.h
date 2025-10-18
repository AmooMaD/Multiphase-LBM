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
#include <chrono>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <cstdlib>
#include <numeric>

using namespace std;
using namespace std::chrono;

// ─────────────────────────   cell-level helpers   ───────────────────────────
enum class CellType_twoLayeredPF2D : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants   ───────────────────────────────
inline auto d2q9_constants_twoLayeredPF2D()
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
struct Dim_twoLayeredPF2D {
    Dim_twoLayeredPF2D(int nx_, int ny_)
        : nx(nx_), ny(ny_),
          nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
          npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   lattice-unit parameters   ─────────────────────
inline auto lbParameters_twoLayeredPF2D(double ulb, int lref, double Re)
{
    double nu    = ulb * static_cast<double>(lref) / Re;
    double omega = 1. / (3.*nu + 0.5);
    double dx    = 1. / static_cast<double>(lref);
    double dt    = dx * ulb;
    return make_tuple(nu, omega, dx, dt);
}

inline void printParameters_twoLayeredPF2D(const Dim_twoLayeredPF2D &dim, double Re, double omega,
                                           double ulb, int N, double max_t , double nu,
                                           double h_lower, int w_int, double gx, double Gx_const)
{
    cout << "Two-Layered 2-D (Phase-Field) problem\n"
         << "N      = " << N      << '\n'
         << "nx     = " << dim.nx      << '\n'
         << "ny     = " << dim.ny      << '\n'
         << "Re     = " << Re     << '\n'
         << "omega  = " << omega  << '\n'
         << "tau    = " << 1. / omega  << '\n'
         << "nu     = " << nu  << '\n'
         << "ulb    = " << ulb    << '\n'
         << "max_t  = " << max_t  << '\n'
         << "h_lower (frac of H) = " << h_lower << '\n'
         << "w_int (nodes)       = " << w_int << '\n'
         << "Drive: gx="<<gx<<"  Gx_const="<<Gx_const << '\n';
}

// ─────────────────────────   timing helpers   ──────────────────────────────
inline auto restartClock_twoLayeredPF2D()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_twoLayeredPF2D(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    cout << "Runtime: " << us.count()/1e6 << " seconds\n";
    cout << "Runtime: " << setprecision(4) << mlups << " MLUPS\n";
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_twoLayeredPF2D
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem   // f
                                                     + 2 * 9 * nelem; // g
    }

    // raw buffers
    CellData*                  lattice;
    CellType_twoLayeredPF2D*   flag;
    int*                       parity;
    std::array<int,2>*         c;
    int*                       opp;
    double*                    t;
    double                     omega;

    // PF / EOS / forcing params
    double phi_l, phi_g;   // order parameter bulk values
    double rho_l, rho_g;   // density bulk values
    double a, b;           // EOS params for pressure(ρ)
    double kappa;          // interfacial stiffness
    double gx;             // x-acceleration-like: adds ρ*gx to Fx
    double Gx_const;       // constant x-force per node

    Dim_twoLayeredPF2D      dim;

    // ───────── index helpers ─────────
    size_t xyz_to_i (int x, int y) { return (y + dim.ny * x); };

    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
        return std::make_tuple(iX, iY);
    };

    // ───────── array accessors ───────
    double& f   (int i,int k){ return lattice[              k*dim.nelem + i]; }
    double& fin (int i,int k){ return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    double& fout(int i,int k){ return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    double& g   (int i,int k){ return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    double& gin (int i,int k){ return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    double& gout(int i,int k){ return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }

    // =====================================================================
    // Two-layer (wall-liquid / middle-gas) tanh-smoothed initialization
    // =====================================================================
    void iniLattice_layers(double& f0, double h_lower, int w_int)
    {
        size_t i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz((int)i);

        const double H      = double(dim.ny - 1);
        const double y_low  = std::clamp(h_lower, 0.0, 0.5) * H; // bottom layer height
        const double y_high = H - y_low;                         // top layer start
        const double w      = std::max(1, w_int);

        const double yy = double(iY);
        // smooth indicators (~1 in liquid layer, ~0 outside)
        const double s_bottom = 0.5 * (1.0 - std::tanh((yy - y_low ) / w));
        const double s_top    = 0.5 * (1.0 + std::tanh((yy - y_high) / w));

        double s_liq = s_bottom + s_top;               // liquid near both walls
        s_liq = std::clamp(s_liq, 0.0, 1.0);
        const double s_gas = 1.0 - s_liq;

        // order parameter & density profiles
        //const double phi = s_liq * phi_l + s_gas * phi_g;
        //const double rho = s_liq * rho_l + s_gas * rho_g;




        // AFTER  (gas near both walls, liquid in middle)  ⟵ swap places
        const double phi = s_liq * phi_g + s_gas * phi_l;
        const double rho = s_liq * rho_g + s_gas * rho_l;

        // EOS p(ρ) (Carnahan–Starling-like, as in your RT code)
        const double rt     = b * rho / 4.0;
        const double denom  = std::pow(1.0 - rt, 3);
        const double p_rho  = (rho / 3.0) * (1.0 + rt + rt*rt - rt*rt*rt) / denom - a * rho * rho;

        // initialize to equilibrium at rest
        for (int k = 0; k < 9; ++k) {
            fin(i, k) = phi   * t[k];
            gin(i, k) = p_rho * t[k];



            fout(i,k) = fin(i,k);
            gout(i,k) = gin(i,k);

        }
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

    // reconstruct ρ from φ via linear map between bulk states
    double total_rho(double& f0){
        auto [phi, P] = macro_phi_P(f0);
        if (std::abs(phi_l - phi_g) > 1e-14)
            return rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);
        else
            return 0.5*(rho_l + rho_g);
    }

    double psi_phi(double& f0){
        auto [phi,P] = macro_phi_P(f0);
        double rt = b*phi/4.0;
        double pth = (phi/3.0)*(1 + rt + rt*rt - rt*rt*rt)/pow(1 - rt, 3) - a*phi*phi;
        return pth - phi/3.0;
    }

    // ∇²ρ
    double laplacian_rho(double& f0)
    {
        size_t i   = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);
        double rho_c = total_rho(lattice[i]);

        double sum   = 0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);
            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
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

    // ∇(∇²ρ)
    std::array<double,2> grad_lap_rho(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);

        double gxloc=0.0, gyloc=0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
                int nbb = xyz_to_i(ixbb,iybb);

                double lap_nb = laplacian_rho(lattice[nbb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = laplacian_rho(lattice[nb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gxloc, 3.0*gyloc };  // 1 / c_s^2
    }

    // velocity with interfacial + x-driving forces
    std::array<double,2> velocity(double& f0)
    {
        auto u = macro_u(f0);
        double rho = total_rho(f0);

        auto glap_phi = grad_lap_phi(f0);
        auto glap_rho = grad_lap_rho(f0);
        double forcex = kappa * rho * glap_phi[0] + rho * gx + Gx_const; // x-drive here
        double forcey = kappa * rho * glap_phi[1];                       // no y-drive

        u[0] += forcex / 6.0;
        u[1] += forcey / 6.0;

        u[0] /= (rho/3.0);
        u[1] /= (rho/3.0);

        return u;
    }

    // ∇ψ(φ)
    std::array<double,2> grad_psi_phi(double& f0)
    {
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);

        double gxloc=0.0, gyloc=0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);

            double psi_nb;
            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
                int nbb = xyz_to_i(ixbb,iybb);

                psi_nb = psi_phi(lattice[nbb]);
                gxloc += t[k] * c[k][0] * psi_nb;
                gyloc += t[k] * c[k][1] * psi_nb;
            }
            else {
                psi_nb = psi_phi(lattice[nb]);
                gxloc += t[k] * c[k][0] * psi_nb;
                gyloc += t[k] * c[k][1] * psi_nb;
            }
        }
        return { 3.0*gxloc, 3.0*gyloc };  // 1 / c_s^2
    }

    double psi_rho(double& f0) {
        double rho = total_rho(f0);
        double rt = b*rho/4.0;
        double p_rho = (rho/3.0)*(1 + rt + rt*rt - rt*rt*rt)/pow(1 - rt, 3) - a*rho*rho;
        return p_rho - rho/3.0;
    }

    // ∇ψ(ρ)
    std::array<double,2> grad_psi_rho(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);

        double gxloc=0.0, gyloc=0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
                int nbb = xyz_to_i(ixbb,iybb);
                double lap_nb = psi_rho(lattice[nbb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = psi_rho(lattice[nb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gxloc, 3.0*gyloc };  // 1 / c_s^2
    }

    // ∇ρ
    std::array<double,2> grad_rho(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);

        double gxloc=0.0, gyloc=0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
                int nbb = xyz_to_i(ixbb,iybb);
                double lap_nb = total_rho(lattice[nbb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = total_rho(lattice[nb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gxloc, 3.0*gyloc };  // 1 / c_s^2
    }

    // total pressure with kinetic correction
    double total_P(double& f0)
    {
        auto [phi, P_term] = macro_phi_P(f0);
        auto u = velocity(f0);
        auto gpsi = grad_rho(f0);
        return P_term - 0.5 * (u[0] * -gpsi[0] / 3. + u[1] * -gpsi[1] / 3.);
    }

    // ∇²φ
    double laplacian_phi(double& f0)
    {
        size_t i   = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);
        auto [phi_c, p] = macro_phi_P(lattice[i]);

        double sum   = 0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
                int nbb = xyz_to_i(ixbb,iybb);

                auto [phi_nbb, p_nbb] = macro_phi_P(lattice[nbb]);
                sum += t[k] * (phi_nbb - phi_c);
            }
            else{
                auto [phi_nb, p_nb] = macro_phi_P(lattice[nb]);
                sum += t[k] * (phi_nb - phi_c);
            }
        }
        return 6.0 * sum;  // 2 / c_s^2
    }

    // ∇(∇²φ)
    std::array<double,2> grad_lap_phi(double& f0){
        size_t i     = &f0 - lattice;
        auto [iX,iY] = i_to_xyz((int)i);

        double gxloc=0.0, gyloc=0.0;
        for (int k=0;k<9;++k) {
            int ix = (iX + c[k][0] + dim.nx) % dim.nx;
            int iy = iY + c[k][1];
            if (iy < 0 || iy >= dim.ny) continue;

            int nb = xyz_to_i(ix,iy);

            if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
                int ixbb = (iX - c[k][0] + dim.nx) % dim.nx;
                int iybb = iY - c[k][1];
                if (iybb < 0 || iybb >= dim.ny) continue;
                int nbb = xyz_to_i(ixbb,iybb);

                double lap_nb = laplacian_phi(lattice[nbb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
            else {
                double lap_nb = laplacian_phi(lattice[nb]);
                gxloc += t[k] * c[k][0] * lap_nb;
                gyloc += t[k] * c[k][1] * lap_nb;
            }
        }
        return { 3.0*gxloc, 3.0*gyloc };  // 1 / c_s^2 factor
    }


    // streaming (periodic x, on-site BB in y)
    void stream(int i,int k,int iX,int iY,double pf,double pg)
    {
        int XX = (iX + c[k][0] + dim.nx) % dim.nx;
        int YY = iY + c[k][1];

        if (YY < 0 || YY >= dim.ny) {
            // reflect at domain boundary (top/bottom walls)
            fout(i,opp[k]) = pf;
            gout(i,opp[k]) = pg;
            return;
        }
        size_t nb = xyz_to_i(XX,YY);

        if (flag[nb] == CellType_twoLayeredPF2D::bounce_back) {
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
        double ck_u    = c[k][0]*u[0] + c[k][1]*u[1];
        double ck_u_op = c[opp[k]][0]*u[0] + c[opp[k]][1]*u[1];

        double eqf    = phi * t[k] * (1 + 3*ck_u + 4.5*ck_u*ck_u - usqr);
        double eqf_op = phi * t[opp[k]] * (1 + 3*ck_u_op + 4.5*ck_u_op*ck_u_op - usqr);

        double eqg    = t[k] * (P + (rho/3.0)*(3*ck_u + 4.5*ck_u*ck_u - usqr));
        double eqg_op = t[opp[k]] * (P + (rho/3.0)*(3*ck_u_op + 4.5*ck_u_op*ck_u_op - usqr));

        double e_u_x    = c[k][0]        - u[0];
        double e_u_x_op = c[opp[k]][0]   - u[0];

        double e_u_y    = c[k][1]        - u[1];
        double e_u_y_op = c[opp[k]][1]   - u[1];

        double forcex = kappa * rho * grad_lap_phi[0] + rho * gx + Gx_const; // x-drive
        double forcey = kappa * rho * grad_lap_phi[1];                       // no y-drive

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
        if (flag[i] == CellType_twoLayeredPF2D::bulk) {
            auto  u   = velocity(f0);

            auto [phi, __] = macro_phi_P(f0);
            double rho = total_rho(f0);

            auto gpsi_rho = grad_psi_rho(f0);
            auto glap_rho = grad_lap_rho(f0);
            auto glap_phi = grad_lap_phi(f0);
            auto gpsi_phi = grad_psi_phi(f0);

            double P  = total_P(f0);
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

                double forcex = kappa * rho * glap_rho[0] + rho * gx + Gx_const; // x-drive
                double forcey = kappa * rho * glap_rho[1];                       // no y-drive

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

// ───────────────────────── VTK output ─────────────────────────────────
void saveVtkFields_twoLayeredPF2D(LBM_twoLayeredPF2D& lbm, int time_iter, double dx = 0.) {
    using namespace std;
    Dim_twoLayeredPF2D const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / dim.nx;
    int dimZ = 1;

    stringstream ss;
    ss << "sol_" << setw(7) << setfill('0') << time_iter << ".vtk";
    ofstream os(ss.str());
    os << "# vtk DataFile Version 2.0\n";
    os << "iteration " << time_iter << "\n";
    os << "ASCII\n\n";
    os << "DATASET STRUCTURED_POINTS\n";
    os << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dimZ << "\n";
    os << "ORIGIN 0 0 0\n";
    os << "SPACING " << dx << " " << dx << " " << dx << "\n\n";
    os << "POINT_DATA " << dim.nx * dim.ny * dimZ << "\n";

    // φ
    os << "SCALARS phi float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            auto [phi,P] = lbm.macro_phi_P(lbm.lattice[i]);
            os << phi << " ";
        }
        os << "\n";
    }
    os << "\n";

    // ρ
    os << "SCALARS density float 1\nLOOKUP_TABLE default\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            auto rho = lbm.total_rho(lbm.lattice[i]);
            os << rho << " ";
        }
        os << "\n";
    }
    os << "\n";

    // u
    os << "VECTORS Velocity float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_twoLayeredPF2D::bounce_back) {
                os << 0.0 << " " << 0.0 << " " << 0.0 << "\n";
            } else {
                auto u = lbm.velocity(lbm.lattice[i]);
                os << u[0] << " " << u[1] << " " << 0.0 << "\n";
            }
        }
    }
    os << "\n";

    // Flag
    os << "SCALARS Flag int 1\nLOOKUP_TABLE default\n";
    for (int iY = 0; iY < dim.ny; ++iY) {
        for (int iX = 0; iX < dim.nx; ++iX) {
            size_t i = lbm.xyz_to_i(iX, iY);
            os << ((lbm.flag[i] == CellType_twoLayeredPF2D::bounce_back) ? 1 : 0) << " ";
        }
        os << "\n";
    }
    os << "\n";
}

// ───────────────────────── Energy & Mass ──────────────────────────────
double computeEnergy_twoLayeredPF2D(LBM_twoLayeredPF2D& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_twoLayeredPF2D::bulk) {
                auto u = lbm.velocity(lbm.lattice[i]);
                energy += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * energy / (dim.nx * dim.ny);
}

double totalMass_twoLayeredPF2D(LBM_twoLayeredPF2D& lbm, bool ignore_solids = true) {
    const auto& dim = lbm.dim;
    double M = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            if (ignore_solids && lbm.flag[i] == CellType_twoLayeredPF2D::bounce_back) continue;
            M += lbm.total_rho(lbm.lattice[i]);
        }
    }
    return M;
}

// ───────────────────────── Geometry initialization ─────────────────────────
void inigeom_twoLayeredPF2D(LBM_twoLayeredPF2D& lbm) {
    Dim_twoLayeredPF2D const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == 0 || y == dim.ny-1) {
                lbm.flag[i] = CellType_twoLayeredPF2D::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i,k) = 0.0;
                    lbm.gin(i,k) = 0.0;
                }
            }
            else {
                lbm.flag[i] = CellType_twoLayeredPF2D::bulk;
            }
        }
    }
}

// ─────────────────────────   main driver   ─────────────────────────────
// reads: ../apps/Config_Files/config_twoLayeredFlow2D.txt
void twoLayered2D()
{
    ifstream contfile("../apps/Config_Files/config_twoLayeredFlow2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. It should be named "
          "\"config_twoLayeredFlow2D.txt\" in Files_Config.");
    }

    // Defaults
    double Re=60, ulb=0.1, max_t=10.0;
    int    N=100, out_freq=400, vtk_freq=400, data_freq=0;

    double phi_l=0.25, phi_g=0.02;
    double rho_l=1.0,  rho_g=0.1;
    double a=4.0, b=4.0, kappa=0.01;
    double h_lower=0.5; int w_int=4;
    double gx = 0.0, Gx_const = 0.0;
    double tau_in = -1.0;

    // robust parse: line-by-line, strip comments
    string line, param, value;
    while (std::getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        std::istringstream iss(line);
        if (!(iss >> param >> value)) continue;

        if      (param=="Re") Re=stod(value);
        else if (param=="ulb") ulb=stod(value);
        else if (param=="N") N=stoi(value);
        else if (param=="max_t") max_t=stod(value);
        else if (param=="out_freq") out_freq=stoi(value);
        else if (param=="vtk_freq") vtk_freq=stoi(value);
        else if (param=="data_freq") data_freq=stoi(value);

        else if (param=="phi_l") phi_l=stod(value);
        else if (param=="phi_g") phi_g=stod(value);
        else if (param=="rho_l") rho_l=stod(value);
        else if (param=="rho_g") rho_g=stod(value);

        else if (param=="a") a=stod(value);
        else if (param=="b") b=stod(value);
        else if (param=="kappa") kappa=stod(value);

        else if (param=="h_lower") h_lower=stod(value);
        else if (param=="w_int")   w_int=stoi(value);

        else if (param=="gx") gx = stod(value);
        else if (param=="Gx_const") Gx_const = stod(value);

        else if (param=="tau") tau_in=stod(value);
    }

    // Domain like SC: wider in x, bounded in y
    Dim_twoLayeredPF2D dim {10, N+1};

    // tau/omega/nu/dx/dt
    double nu=0.0, omega=1.0, dx=1.0/N, dt=dx*ulb;
    if (tau_in > 0.0) {
        const double tau = tau_in;
        omega = 1.0 / tau;
        nu    = (tau - 0.5) / 3.0;
    } else {
        tie(nu, omega, dx, dt) = lbParameters_twoLayeredPF2D(ulb, N, Re);
    }
    printParameters_twoLayeredPF2D(dim, Re, omega, ulb, N, max_t, nu, h_lower, w_int, gx, Gx_const);

    // allocate
    vector<CellData> lattice_vect(LBM_twoLayeredPF2D::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    vector<CellType_twoLayeredPF2D> flag_vect(dim.nelem);
    auto* flag = flag_vect.data();

    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_twoLayeredPF2D();

    LBM_twoLayeredPF2D lbm{
        lattice, flag, parity,
        &c_vect[0], &opp_vect[0], &t_vect[0],
        omega,
        phi_l, phi_g,
        rho_l, rho_g,
        a, b, kappa,
        gx, Gx_const,
        dim
    };

    // init populations and geometry
    std::for_each(lattice, lattice + dim.nelem,
                  [&lbm,h_lower,w_int](double& f0){ lbm.iniLattice_layers(f0, h_lower, w_int); });
    inigeom_twoLayeredPF2D(lbm);

    // time loop
    auto [start, clock_iter] = restartClock_twoLayeredPF2D();
    ofstream efile("energy.dat");
    ofstream mass_log("mass.dat");

    // Density probe(s) on the midline
    ofstream dprobe("density_probe.dat");
    dprobe << "# t  rho_center  rho_qbot  rho_qtop\n";
    auto sample_rho = [&lbm](int x, int y) -> double {
        size_t i = lbm.xyz_to_i(x,y);
        return lbm.total_rho(lbm.lattice[i]);
    };


    double M0 = -1.0;

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_twoLayeredPF2D(lbm, time_iter, dx);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]\n";

            double energy = computeEnergy_twoLayeredPF2D(lbm) * dx * dx / (dt * dt);
            cout << "Average energy: " << setprecision(10) << energy << "\n";
            efile << setw(10) << time_iter * dt << setw(16) << setprecision(10) << energy << "\n";

            const double M  = totalMass_twoLayeredPF2D(lbm);
            if (M0 < 0.0) M0 = M;
            const double rel = (M - M0) / M0 * 100.0;
            std::cout << std::setprecision(12)
                      << "[Mass] M="<< M << "   ΔM/M0=" << std::setprecision(6) << rel << "%\n";
            if (mass_log) mass_log << std::setprecision(16) << time_iter*dt << " " << M << "\n";

        }

        // collision + streaming
        std::for_each(std::execution::par_unseq, lattice, lattice + dim.nelem, lbm);

        // parity swap
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_twoLayeredPF2D(start, clock_iter, dim.nelem);
}
