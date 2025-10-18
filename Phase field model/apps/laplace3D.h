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
enum class CellType_laplace3D : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D3Q19 constants   ───────────────────────────────
inline auto d3q19_constants_laplace3D()
{
    vector<array<int, 3>> c_vect = {
        {-1, 0, 0}, { 0,-1, 0}, { 0, 0,-1},
        {-1,-1, 0}, {-1, 1, 0}, {-1, 0,-1},
        {-1, 0, 1}, { 0,-1,-1}, { 0,-1, 1},
        { 0, 0, 0},
        { 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1},
        { 1, 1, 0}, { 1,-1, 0}, { 1, 0, 1},
        { 1, 0,-1}, { 0, 1, 1}, { 0, 1,-1}
    };

    // The opposite of a given direction.
    vector<int> opp_vect =
        { 10, 11, 12, 13, 14, 15, 16, 17, 18, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    // The lattice weights.
    vector<double> t_vect =
        {
            1./18., 1./18., 1./18., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.,
            1./3.,
            1./18., 1./18., 1./18., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.
        };
    return make_tuple(c_vect, opp_vect, t_vect);
}

// ─────────────────────────   geometry helpers   ────────────────────────────
struct Dim_laplace3D {

    Dim_laplace3D(int nx_, int ny_, int nz_)
        : nx(nx_), ny(ny_), nz(nz_),
          nelem(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz)),
          npop(19 * nelem)
    { }
    int nx, ny, nz;
    size_t nelem, npop;
};

// ─────────────────────────   lattice-unit parameters   ─────────────────────
inline auto lbParameters_laplace3D(double ulb, int lref, double Re)
{
    double nu    = ulb * static_cast<double>(lref) / Re;
    double omega = 1. / (3.*nu + 0.5);
    //double tau = 1. / omega;
    double dx    = 1. / static_cast<double>(lref);
    double dt    = dx * ulb;
    return make_tuple(nu, omega, dx, dt);
}

inline void printParameters_laplace3D(const Dim_laplace3D &dim, double Re, double omega,
                                             double ulb, int N, double max_t , double nu)
{

    cout << "laplace 3-D problem\n"
         << "N      = " << N      << '\n'
         << "nx     = " << dim.nx      << '\n'
         << "ny     = " << dim.ny      << '\n'
         << "nz     = " << dim.nz      << '\n'
         << "Re     = " << Re     << '\n'
         << "omega  = " << omega  << '\n'
         << "tau    = " << 1. / omega  << '\n'
         << "nu     = " << nu  << '\n'
         << "ulb    = " << ulb    << '\n'
         << "max_t  = " << max_t  << '\n';
}

// ─────────────────────────   timing helpers   ──────────────────────────────
inline auto restartClock_laplace3D()
{
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TP>
inline void printMlups_laplace3D(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    int dure = us.count();
        cout << "Runtime: " << dure/1000000. << " seconds " << endl;
        cout << "Runtime: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_laplace3D
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 19 * nelem
                                                     + 2 * 19 * nelem ; }

    // raw buffers
    CellData*                 lattice;
    CellType_laplace3D*       flag;
    int*                      parity;
    std::array<int,3>*        c;
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
    Dim_laplace3D      dim;

    // ───────── index helpers ─────────
    auto i_to_xyz (int i) {
        int iX = i / (dim.ny * dim.nz);
        int remainder = i % (dim.ny * dim.nz);
        int iY = remainder / dim.nz;
        int iZ = remainder % dim.nz;
        return std::make_tuple(iX, iY, iZ);
    };

    // Convert Cartesian indices to linear index.
    size_t xyz_to_i (int x, int y, int z) {
        return z + dim.nz * (y + dim.ny * x);
    };

    // ───────── array accessors ───────
    double& f   (int i,int k){ return lattice[              k*dim.nelem + i]; }
    double& fin (int i,int k){ return lattice[*parity  *dim.npop + k*dim.nelem + i]; }
    double& fout(int i,int k){ return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }
    //double& feq (int i,int k){ return lattice[ 2*dim.npop +             k*dim.nelem + i]; }

    double& g   (int i,int k){ return lattice[2*dim.npop                  + k*dim.nelem + i]; }
    double& gin (int i,int k){ return lattice[*parity  *dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    double& gout(int i,int k){ return lattice[(1-*parity)*dim.npop + 2*dim.npop + k*dim.nelem + i]; }
    //double& geq (int i,int k){ return lattice[5*dim.npop + k*dim.nelem + i]; }

    // ─────────────────────────────────── physics kernels ────────────────────

    /* =====================================================================
    Smooth interface initialisation — tanh profile
    ===================================================================== */
    void iniLattice(double& f0)
    {
        // compute global cell index and (iX,iY)
        auto i = &f0 - lattice;
        auto [iX, iY, iZ] = i_to_xyz(i);

        // --- droplet geometry ---
        double xc = double(dim.nx)/2.0;      // droplet center x
        double yc = double(dim.ny)/2.0;      // droplet center y
        double zc = double(dim.nz)/2.0;      // droplet center z
        double R  = 0.25 * dim.nx; // choose radius

        // interface thickness for tanh profile
        const double xi = 1.0;

        // signed‐distance from circle
        double dx = double(iX) - xc;
        double dy = double(iY) - yc;
        double dz = double(iZ) - zc;
        double delta = sqrt(dx*dx + dy*dy + dz*dz) - R;

        // --- bulk binodal values (your a, b, phi_l/g are assumed in scope) ---
        double rt_l  =  b * phi_l / 4.0;
        double pth_l = (phi_l/3.0)*(1 + rt_l + rt_l*rt_l - rt_l*rt_l*rt_l)
                       / pow(1 - rt_l,3) - a * phi_l*phi_l;

        double rt_g  =  b * phi_g / 4.0;
        double pth_g = (phi_g/3.0)*(1 + rt_g + rt_g*rt_g - rt_g*rt_g*rt_g)
                       / pow(1 - rt_g,3) - a * phi_g*phi_g;

        // smooth Heaviside across the interface
        double w   = 0.5 * (1.0 - tanh(delta / xi));

        // local order‐parameter and pressure
        double phi = phi_g + w * (phi_l - phi_g);
        double pth = pth_g + w * (pth_l - pth_g);

        // populate both distribution sets
        for (int k = 0; k < 19; ++k)
        {
            fin(i,k) = phi * t[k];
            gin(i,k) = pth * t[k];
        }
    };

    // macro-vars ϕ, P, u
    auto macro_phi_P(double& f0)
    {
        int i = &f0 - lattice;
        double phi , P ;


        double Xg_M1 = gin(i, 0) + gin(i, 3) + gin(i, 4) + gin(i, 5) + gin(i, 6);
        double Xg_P1 = gin(i, 10) + gin(i, 13) + gin(i, 14) + gin(i, 15) + gin(i, 16);
        double Xg_0  = gin(i, 9) + gin(i, 1) + gin(i, 2) + gin(i, 7) + gin(i, 8) + gin(i, 11) + gin(i, 12) + gin(i, 17) + gin(i, 18);


        double Xf_M1 = fin(i, 0) + fin(i, 3) + fin(i, 4) + fin(i, 5) + fin(i, 6);
        double Xf_P1 = fin(i, 10) + fin(i, 13) + fin(i, 14) + fin(i, 15) + fin(i, 16);
        double Xf_0  = fin(i, 9) + fin(i, 1) + fin(i, 2) + fin(i, 7) + fin(i, 8) + fin(i, 11) + fin(i, 12) + fin(i, 17) + fin(i, 18);


        phi  = Xf_M1 + Xf_P1 + Xf_0;
        P    = Xg_M1 + Xg_P1 + Xg_0;

        //if (phi < 1e-10) phi = 1e-10;
        return make_tuple(phi, P);
    }

    auto macro_u(double& f0)
    {
        int i = &f0 - lattice;
        array<double,3> u;

        double Xg_M1 = gin(i, 0) + gin(i, 3) + gin(i, 4) + gin(i, 5) + gin(i, 6);
        double Xg_P1 = gin(i, 10) + gin(i, 13) + gin(i, 14) + gin(i, 15) + gin(i, 16);

        double Yg_M1 = gin(i, 1) + gin(i, 3) + gin(i, 7) + gin(i, 8) + gin(i, 14);
        double Yg_P1 = gin(i, 4) + gin(i, 11) + gin(i, 13) + gin(i, 17) + gin(i, 18);

        double Zg_M1 = gin(i, 2) + gin(i, 5) + gin(i, 7) + gin(i, 16) + gin(i, 18);
        double Zg_P1 = gin(i, 6) + gin(i, 8) + gin(i, 12) + gin(i, 15) + gin(i, 17);


        u[0] = Xg_P1 - Xg_M1;
        u[1] = Yg_P1 - Yg_M1;
        u[2] = Zg_P1 - Zg_M1;
        return u;
    }


    double total_rho(double& f0){
        int i = &f0 - lattice;

        auto [phi, P] = macro_phi_P(f0);
        return rho_g + ((phi - phi_g)/(phi_l - phi_g)) * (rho_l - rho_g);
    }

    double psi_phi(double& f0){
        int i = &f0 - lattice;

        auto [phi,P]=macro_phi_P(f0);
        double rt = b*phi/4.0;
        double pth = (phi/3.0)*(1+rt+rt*rt-rt*rt*rt)/pow(1-rt,3) - a*phi*phi;
        return pth - phi/3.0;
    }
    
    //───────────────────────────────────────────────────────────────//
    //  helper: full velocity  u  (adds interfacial / gravity force)
    //───────────────────────────────────────────────────────────────//
    std::array<double,3> velocity(double& f0)
    {
        int i = &f0 - lattice;

        auto u = macro_u(f0);

        double rho = total_rho(f0);
        auto [phi, p] = macro_phi_P(f0);

        //auto glap_rho = grad_lap_rho(f0);
        auto glap_phi = grad_lap_phi(f0);

        //double forcex = kappa * rho * glap_rho[0];
        //double forcey = kappa * rho * glap_rho[1]; //check eq 7.49

        double forcex = kappa * phi * glap_phi[0];
        double forcey = kappa * phi * glap_phi[1];
        double forcez = kappa * phi * glap_phi[2];


        forcey += gravity * rho;

        u[0] += forcex / 6.;
        u[1] += forcey / 6.;
        u[2] += forcey / 6.;


        u[0] /= (rho/3.);
        u[1] /= (rho/3.);
        u[2] /= (rho/3.);
        
        return u;
    }


    //───────────────────────────────────────────────────────────────//
    //  helper: full pressure  P  (bulk + kinetic correction)
    //───────────────────────────────────────────────────────────────//
    double total_P(double& f0)
    {
        int i = &f0 - lattice;

        auto [phi, P_term] = macro_phi_P(f0);
        auto u = velocity(f0);
        auto gpsi = grad_psi_phi(f0);
        //gpsi[0] /= ((0.12 - 0.04) / ( 0.251 - 0.024));           // d psi / d rho = ( d spi / d phi) * ( d phi / d rho )
        //gpsi[1] /= ((0.12 - 0.04) / ( 0.251 - 0.024));
        return P_term - 0.5 * (u[0] * gpsi[0] + u[1] * gpsi[1] + u[2] * gpsi[2] );
    }

    double psi_rho(double& f0) { 
        int i = &f0 - lattice;

        double P = total_P(f0);
        double rho = total_rho(f0);
        return P - rho/3.0; 
    }

    // ------------------------------------------------------------------
    // ∇²ρ   (lattice Laplacian of the local density ρ)
    // ------------------------------------------------------------------
    double laplacian_rho(double& f0)
    {
        size_t i   = &f0 - lattice;
        auto [iX,iY, iZ] = i_to_xyz(i);
        double rho_c = total_rho(lattice[i]);

        double sum   = 0.0;

        for (int k=0;k<19;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            int iz = iZ + c[k][2];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            iz = (iz + dim.nz) % dim.nz;
            int nb = xyz_to_i(ix,iy,iz);

            if (flag[nb] != CellType_laplace3D::bounce_back) {
                
                double rho_nb = total_rho(lattice[nb]);
                sum += t[k] * (rho_nb - rho_c);
            }
        }
        return 6.0 * sum;            // 2 / c_s^2  with  c_s^2 = 1/3
    }

    // ------------------------------------------------------------------
    // ∇²phi   (lattice Laplacian of the local density phi)
    // ------------------------------------------------------------------
    double laplacian_phi(double& f0)
    {
        size_t i   = &f0 - lattice;
        auto [iX,iY, iZ] = i_to_xyz(i);
        auto [phi_c, p] = macro_phi_P(lattice[i]);

        double sum   = 0.0;

        for (int k=0;k<19;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            int iz = iZ + c[k][2];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            iz = (iz + dim.nz) % dim.nz;
            int nb = xyz_to_i(ix,iy,iz);

            if (flag[nb] != CellType_laplace3D::bounce_back) {
                auto [phi , p] = macro_phi_P(lattice[nb]);
                sum += t[k] * (phi - phi_c);
            }
        }
        return 6.0 * sum;            // 2 / c_s^2  with  c_s^2 = 1/3
    }

    // ------------------------------------------------------------------
    // ∇(∇²ρ)   : gradient of the Laplacian
    // ------------------------------------------------------------------
    std::array<double,3> grad_lap_rho(double& f0){
        //double lap_c = laplacian_rho(f0);
        size_t i     = &f0 - lattice;
        auto [iX,iY, iZ] = i_to_xyz(i);

        double gx=0.0, gy=0.0, gz = 0.0;
        for (int k=0;k<19;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            int iz = iZ + c[k][2];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            iz = (iz + dim.nz) % dim.nz;
            int nb = xyz_to_i(ix,iy,iz);

            if (flag[nb] != CellType_laplace3D::bounce_back) {
                double lap_nb = laplacian_rho(lattice[i]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
                gz += t[k] * c[k][2] * lap_nb;
            }

            else{
                double lap_nb = laplacian_rho(lattice[nb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
                gz += t[k] * c[k][2] * lap_nb;
            }
        }
        return { 3.0*gx, 3.0*gy, 3.0 * gz };   // 1 / c_s^2 factor
    }



    // ------------------------------------------------------------------
    // ∇(∇²phi)   : gradient of the Laplacian
    // ------------------------------------------------------------------
    std::array<double,3> grad_lap_phi(double& f0){
        //double lap_c = laplacian_rho(f0);
        size_t i     = &f0 - lattice;
        auto [iX,iY, iZ] = i_to_xyz(i);

        double gx=0.0, gy=0.0, gz = 0.0;
        for (int k=0;k<19;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            int iz = iZ + c[k][2];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            iz = (iz + dim.nz) % dim.nz;
            int nb = xyz_to_i(ix,iy,iz);

            if (flag[nb] == CellType_laplace3D::bounce_back) {
                double lap_nb = laplacian_phi(lattice[i]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
                gz += t[k] * c[k][2] * lap_nb;
            }
            else{

                double lap_nb = laplacian_phi(lattice[nb]);
                gx += t[k] * c[k][0] * lap_nb;
                gy += t[k] * c[k][1] * lap_nb;
                gz += t[k] * c[k][2] * lap_nb;
            }
        }
        return { 3.0*gx, 3.0*gy, 3.0 * gz };   // 1 / c_s^2 factor
    }

    // ------------------------------------------------------------------
    // ∇ψ(rho)
    // ------------------------------------------------------------------
    std::array<double,3> grad_psi_rho(double& f0)
    {
        //double psi_c = psi_rho(f0);
        size_t i     = &f0 - lattice;
        auto [iX,iY, iZ] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0, gz = 0.0;
        for (int k=0;k<19;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            int iz = iZ + c[k][2];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            iz = (iz + dim.nz) % dim.nz;
            int nb = xyz_to_i(ix,iy,iz);

            if (flag[nb] == CellType_laplace3D::bounce_back) {
                double psi_nb = psi_rho(lattice[i]);
                gx += t[k] * c[k][0] * psi_nb;
                gy += t[k] * c[k][1] * psi_nb;
                gz += t[k] * c[k][2] * psi_nb;
            }
            else{
                double psi_nb = psi_rho(lattice[nb]);
                gx += t[k] * c[k][0] * psi_nb;
                gy += t[k] * c[k][1] * psi_nb;
                gz += t[k] * c[k][2] * psi_nb;
            }
        }
        return { 3.0*gx, 3.0*gy, 3.0 * gz };   // 1 / c_s^2 factor
    }
    
    
    // ------------------------------------------------------------------
    // ∇ψ(phi)
    // ------------------------------------------------------------------
    std::array<double,3> grad_psi_phi(double& f0)
    {
        //double psi_c = psi_rho(f0);
        size_t i     = &f0 - lattice;
        auto [iX,iY, iZ] = i_to_xyz(int(i));

        double gx=0.0, gy=0.0, gz = 0.0;
        for (int k=0;k<19;++k) {
            int ix = iX + c[k][0];
            int iy = iY + c[k][1];
            int iz = iZ + c[k][2];
            ix = (ix + dim.nx) % dim.nx;
            iy = (iy + dim.ny) % dim.ny;
            iz = (iz + dim.nz) % dim.nz;
            int nb = xyz_to_i(ix,iy,iz);

            if (flag[nb] == CellType_laplace3D::bounce_back) {
                double psi_nb = psi_phi(lattice[i]);
                gx += t[k] * c[k][0] * psi_nb;
                gy += t[k] * c[k][1] * psi_nb;
                gz += t[k] * c[k][2] * psi_nb;
            }
            else{
                double psi_nb = psi_phi(lattice[nb]);
                gx += t[k] * c[k][0] * psi_nb;
                gy += t[k] * c[k][1] * psi_nb;
                gz += t[k] * c[k][2] * psi_nb;
            }
        }
        return { 3.0*gx, 3.0*gy, 3.0 * gz };   // 1 / c_s^2 factor
    }
    
    // single-population stream
    void stream(int i,int k,int iX,int iY, int iZ,double pf,double pg)
    {
        int XX = iX + c[k][0];
        int YY = iY + c[k][1];
        int ZZ = iZ + c[k][2];

        XX = (XX + dim.nx) % dim.nx;
        YY = (YY + dim.ny) % dim.ny;
        ZZ = (ZZ + dim.nz) % dim.nz;

        size_t nb = xyz_to_i(XX,YY,ZZ);

        if (flag[nb] == CellType_laplace3D::bounce_back) {
            fout(i,opp[k]) = pf;
            gout(i,opp[k]) = pg;
        }
        else {
            fout(nb,k) = pf;
            gout(nb,k) = pg;
        }
    };

    // BGK collision for the pair (f,g)
    std::tuple<double,double,double,double>
    collideBgk(int i,int k,const std::array<double,3>& u,double rho,
               double phi,double   P,double usqr,
               std::array<double,3>& grad_psi_rho,
               std::array<double,3>& grad_psi_phi,
               //std::array<double,3>& grad_lap_rho,
               std::array<double,3>& grad_lap_phi)
    {
  
        double ck_u = c[k][0]*u[0] + c[k][1]*u[1] + c[k][2]*u[2];

        double eqf    = phi * t[k] * (1 + 3*ck_u + 4.5*ck_u*ck_u - usqr);   
        double eqf_op = eqf - 6*phi*t[k]*ck_u;                              

        double eqg    = t[k] * (P + (rho/3.0)*(3*ck_u + 4.5*ck_u*ck_u - usqr));
        double eqg_op = eqg - 6*(rho/3.0)*t[k]*ck_u;
        
        double e_u_x = c[k][0] - u[0];
        double e_u_x_op = c[opp[k]][0] - u[0];
        
        double e_u_y = c[k][1] - u[1];
        double e_u_y_op = c[opp[k]][1] - u[1];

        double e_u_z = c[k][2] - u[2];
        double e_u_z_op = c[opp[k]][2] - u[2];

        //double gamma0 = t[k] * phi;

        //double forcex = kappa * rho * grad_lap_rho[0];
        //double forcey = kappa * rho * grad_lap_rho[1]; // check eq 7.49

        double forcex = kappa * phi * grad_lap_phi[0];
        double forcey = kappa * phi * grad_lap_phi[1];
        double forcez = kappa * phi * grad_lap_phi[2];



        forcey += gravity * rho;

        double Ex = grad_psi_rho[0];
        double Ey = grad_psi_rho[1];
        double Ez = grad_psi_rho[2];
        
        
        double fg = (1. - 0.5 * omega ) * ((e_u_x * forcex + e_u_y  * forcey + e_u_z  * forcez) * eqf / phi)
                  + (1. - 0.5 * omega ) * ((e_u_x * -1 * Ex ) + (e_u_y * -1 * Ey ) + (e_u_z * -1 * Ez )) * (eqf/phi - t[k]);

        double ff = (1. - 0.5 * omega ) * (( e_u_x * -1 * grad_psi_phi[0] ) + ( e_u_y * -1 * grad_psi_phi[1] ) + ( e_u_z * -1 * grad_psi_phi[2] )) * 3. * eqf / rho;
        
        
        double fg_op = (1. - 0.5 * omega ) * ((e_u_x_op * forcex + e_u_y_op * forcey + e_u_z_op * forcez) * eqf_op / phi)
                     + (1. - 0.5 * omega ) * ((e_u_x_op * -1 * Ex + e_u_y_op * -1 * Ey + e_u_z_op * -1 * Ez)) * (eqf_op/phi - t[k]);

        double ff_op = (1. - 0.5 * omega ) * (( e_u_x_op * -1 * grad_psi_phi[0] ) + ( e_u_y_op * -1 * grad_psi_phi[1] ) + ( e_u_z_op * -1 * grad_psi_phi[2] )) * 3. * eqf_op / rho;

        double pf    = (1. - omega ) * fin(i,k)      + omega*eqf + ff;
        double pg    = (1. - omega ) * gin(i,k)      + omega*eqg + fg;

        double pf_op = (1. - omega ) * fin(i,opp[k]) + omega*eqf_op + ff_op;
        double pg_op = (1. - omega ) * gin(i,opp[k]) + omega*eqg_op + fg_op;

        return {pf, pg, pf_op, pg_op};
    }

    // main per-cell operator
    void operator()(double& f0)
    {
        int i = &f0 - lattice;
        if (flag[i] == CellType_laplace3D::bulk) {
        
        auto  u   = velocity(f0);         
        double P  = total_P(f0);          
        auto [phi,/*P_term*/__] = macro_phi_P(f0);
        double rho = total_rho(f0);


        
        auto gpsi_rho = grad_psi_rho (f0);
        auto glap_rho = grad_lap_rho (f0);
        auto glap_phi = grad_lap_phi (f0);
        auto gpsi_phi = grad_psi_phi (f0);
        double forcex = kappa * phi * glap_phi[0];
        double forcey = kappa * phi * glap_phi[1];
        double forcez = kappa * phi * glap_phi[2];

        forcey += gravity * rho;

        double Ex = gpsi_rho[0];
        double Ey = gpsi_rho[1];
        double Ez = gpsi_rho[2];
        
        double usqr = 1.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

        auto [iX,iY,iZ] = i_to_xyz(i);

        for (int k = 0; k < 9; ++k) {
            auto [pf, pg, pf_op, pg_op] = collideBgk(i,k,u,rho,phi,P,usqr,gpsi_rho,gpsi_phi,glap_phi);
            stream(i,     k , iX,iY, iZ, pf   , pg   );
            stream(i, opp[k], iX,iY, iZ, pf_op, pg_op);
        }

        // rest population (k = 4)
          for ( int k :{9} ){
            double eqf0 = phi * t[k] * (1. - usqr);
            double eqg0 = t[k] * (P  - (rho/3.0)*usqr);

            double fg0 = (1. - 0.5 * omega) * -1 * (u[0] * forcex + u[1] * forcey + u[2] * forcez ) * eqf0/phi
                       + (1. - 0.5 * omega) * -1 * (u[0] * -1 * Ex + u[1] * -1 * Ey + u[2] * -1 * Ez) * (eqf0/phi - t[k]);

            double ff0 = (1. - 0.5 * omega) * -3. * eqf0 * ( u[0] * -1 * gpsi_phi[0] + u[1] * -1 * gpsi_phi[1] + u[2] * -1 * gpsi_phi[2]) / rho;


            fout(i,k) = (1-omega)*fin(i,k) + omega*eqf0 + ff0;
            gout(i,k) = (1-omega)*gin(i,k) + omega*eqg0 + fg0;

          }
        }
    }
};



// VTK output (integrated with new accessors)
void saveVtkFields_laplace3D(LBM_laplace3D& lbm, int time_iter, double dx = 0.) {
    using namespace std;
    Dim_laplace3D const& dim = lbm.dim;
    if (dx == 0.0) dx = 1.0 / dim.nx;

    stringstream ss;
    ss << "sol_" << setw(7) << setfill('0') << time_iter << ".vtk";
    ofstream os(ss.str());
    os << "# vtk DataFile Version 2.0" << endl;
    os << "iteration " << time_iter << endl;
    os << "ASCII" << endl << endl;
    os << "DATASET STRUCTURED_POINTS" << endl;
    os << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dim.nz << endl;
    os << "ORIGIN 0 0 0" << endl;
    os << "SPACING " << dx << " " << dx << " " << dx << endl << endl;
    os << "POINT_DATA " << dim.nx*dim.ny*dim.nz << endl ;


    // phi
    os << "SCALARS phi float 1" << endl;
    os << "LOOKUP_TABLE default" << endl;
    for (int z = dim.nz-1; z >=0; --z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = lbm.xyz_to_i(x, y, z);
                auto [phi,P] = lbm.macro_phi_P(lbm.lattice[i]);
                if (lbm.flag[i] == CellType_laplace3D::bulk) {
                    double phim = 0.;
                    }
                os << phi << " ";
                }
            os << endl;
            }
        os << endl;
    }
    os << endl;


    // Presssure
    os << "SCALARS Pressure float 1" << endl;
    os << "LOOKUP_TABLE default" << endl;
    for (int z = dim.nz-1; z >=0; --z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                double dens ;
                int i = lbm.xyz_to_i(x, y, z);
                dens = lbm.total_P(lbm.lattice[i]);
                if (lbm.flag[i] == CellType_laplace3D::bulk) {
                    double densm = 0.;
                    }
                os << dens << " ";
                }
            os << endl;
            }
        os << endl;
    }
    os << endl;


   /* // Velocity_x
    os << "VECTORS velocity float" << endl;
    for (int z = dim.nz-1; z >=0; --z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                double vx = 0., vy = 0. , vz = 0.;
                int i = lbm.xyz_to_i(x, y, z);
                auto u = lbm.velocity(lbm.lattice[i]);
                vx = u[0];
                vy = u[1];
                vz = u[2];
                }
                os << vx << " " << vy << " " << vz << " 0\n";
            }
            os << endl;
        }
        os << endl;
    //}
    //os << endl ;*/



    /*// laplacian rho
    os << "SCALARS laplacian float" << endl;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = lbm.xyz_to_i(x, y);
                double lp = 0.0;
                //if (lbm.flag[i] == CellType_laplace3D::bulk) {
                    auto u = lbm.laplacian_phi(lbm.lattice[i]);
                    lp = u;
                //}
                os << lp << endl;
            }
            os << endl;
        }
        os << endl;
    }
    os << endl ;*/


    // Wall flag matrix
    os << "SCALARS Flag int 1" << endl ;
    os << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dim.nz-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY ,iZ);
                if (lbm.flag[i] == CellType_laplace3D::bounce_back) {
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


// Energy computation

double computeEnergy_laplace3D(LBM_laplace3D& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int z = dim.nz-1; z >=0; --z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = lbm.xyz_to_i(x, y,z);
                if (lbm.flag[i] == CellType_laplace3D::bulk) {
                    auto u = lbm.velocity(lbm.lattice[i]);
                    energy += u[0]*u[0] + u[1]*u[1] + u[2] * u[2];
                }
            }
        }
    }
    return 0.5 * energy / (dim.nx * dim.ny * dim.nz);
}


// Geometry initialization

void inigeom_laplace3D(LBM_laplace3D& lbm) {
    Dim_laplace3D const& dim = lbm.dim;
    for (int z = dim.nz-1; z >=0; --z) {
        for (int y = 0; y < dim.ny; ++y) {
            for (int x = 0; x < dim.nx; ++x) {
                int i = lbm.xyz_to_i(x, y,z);
                if (y == dim.ny+1000 ) {lbm.flag[i] = CellType_laplace3D::bounce_back;
                  for (int k = 0 ; k<19 ; ++k){
                    lbm.fin(i,k) = 0.;
                    //lbm.feq(i,k) = 0.;

                    lbm.gin(i,k) = 0.;
                    //lbm.geq(i,k) = 0.;
                  }
                }
                else {lbm.flag[i] = CellType_laplace3D::bulk;}
            }
        }
    }
}


// ─────────────────────────   main driver   ──────────────────────────────────
void laplace3D()
{
    ifstream contfile("../apps/Config_Files/config_laplace3D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. It should be named "
          "\"config_laplace3D.txt\" in Files_Config.");
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

        // 3) convert using the right function
        
            if (param == "Re") {          Re       = stod(value);}
            else if (param == "ulb"){    ulb      = stod(value);}
            else if (param == "N"){      N        = std::stoi(value);}
            else if (param == "max_t"){   max_t    = stod(value);}
            else if (param == "out_freq"){out_freq = stoi(value);}
            else if (param == "vtk_freq"){vtk_freq = stoi(value);}
            else if (param == "phi_l"){   phi_l    = stod(value);}
            else if (param == "phi_g"){   phi_g    = stod(value);}
            else if (param == "rho_l"){   rho_l    = stod(value);}
            else if (param == "rho_g"){   rho_g    = stod(value);}
            else if (param == "a") {      a        = stod(value);}
            else if (param == "b"){       b        = stod(value);}
            else if (param == "kappa"){   kappa    = stod(value);}
            else if (param == "gravity"){ gravity  = stod(value);}
            else{
                cerr << "Warning: unknown parameter \"" << param << "\"\n";}
        

    }

    
    using CellData = typename LBM_laplace3D::CellData;

    // lattice parameters
    Dim_laplace3D dim {N, N, N};
    auto [nu,omega,dx,dt] = lbParameters_laplace3D(ulb,N,Re);
    printParameters_laplace3D(dim,Re,omega,ulb,N,max_t,nu);

    // allocate
    vector<CellData> lattice_vect(LBM_laplace3D::sizeOfLattice(dim.nelem));
    CellData *lattice = &lattice_vect[0];
    
    vector<CellType_laplace3D> flag_vect(dim.nelem);
    CellType_laplace3D* flag = &flag_vect[0];
    
    vector<int> parity_vect {0};        
    int* parity = &parity_vect[0]; 

    auto [c_vect, opp_vect, t_vect] = d3q19_constants_laplace3D();

    LBM_laplace3D lbm{lattice, flag, parity, &c_vect[0], &opp_vect[0], &t_vect[0], omega , phi_l, phi_g, rho_l, rho_g, a , b, kappa, gravity, dim};
  
 
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });     

    inigeom_laplace3D(lbm);

    auto [start, clock_iter] = restartClock_laplace3D();
    ofstream efile("energy.dat");

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0 && time_iter >= 0) {
           saveVtkFields_laplace3D(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0 && time_iter >= 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]" << endl;
            double energy = computeEnergy_laplace3D(lbm) *dx*dx / (dt*dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            efile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
        }
        
              
        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        
        // After a collision-streaming cycle, swap the parity for the next iteration.
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_laplace3D(start, clock_iter, dim.nelem);

}
