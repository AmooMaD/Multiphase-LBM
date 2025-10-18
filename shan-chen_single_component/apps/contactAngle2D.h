#pragma once

#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <algorithm>
#include <execution>
#include <chrono>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace std::chrono;

// D2Q9 lattice
//   3  6  7
//    \ | /
//   0--4--5
//    / | \
//   2  1  8

enum class CellType_contactAngle2D : uint8_t { bounce_back, bulk };

inline auto d2q9_constants_contactAngle2D() {
    vector<array<int,2>> c_vect = {
        {-1,  0}, { 0, -1}, {-1, -1}, {-1,  1}, { 0,  0},
        { 1,  0}, { 0,  1}, { 1,  1}, { 1, -1},
    };
    vector<int> opp_vect = {5, 6, 7, 8, 4, 0, 1, 2, 3};
    vector<double> t_vect = {
        1./9., 1./9., 1./36., 1./36.,
        4./9.,
        1./9., 1./9., 1./36., 1./36.
    };
    return make_tuple(c_vect, opp_vect, t_vect);
}

struct Dim_contactAngle2D {
    Dim_contactAngle2D(int nx_, int ny_)
        : nx(nx_), ny(ny_), nelem((size_t)nx*ny), npop(9*nelem) {}
    int nx, ny;
    size_t nelem, npop;
};

inline auto lbParameters_contactAngle2D(double ulb, int lref, double Re) {
    double nu = ulb * lref / Re;
    double omega = 1. / (3.*nu + 0.5);
    double dx = 1. / lref;
    double dt = dx * ulb;
    return make_tuple(nu, omega, dx, dt);
}

void printParameters_contactAngle2D(const Dim_contactAngle2D &dim, double Re, double omega,
                                    double ulb, int N, double max_t , double nu)
{
    cout << "contact Angle 2-D (Yuan–CS) problem\n"
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

inline auto restartClock_contactAngle2D() {
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TimePoint>
void printMlups_contactAngle2D(TimePoint start, int clock_iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double mlups = (double)(nelem * clock_iter) / duration.count();
    cout << "result: " << duration.count()/1e6 << " seconds\n";
    cout << "result: " << setprecision(4) << mlups << " MLUPS\n";
}

// ======================== Yuan–CS single-component SC (D2Q9) ========================
struct LBM_contactAngle2D {
    using CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem; }

    // memory & topo
    CellData* lattice;
    CellType_contactAngle2D* flag;
    int* parity;
    std::array<int,2>* c;
    int* opp;
    double* t;

    // params
    double omega;
    double rhol, rhog;
    double rho_w;          // wall density for wetting
    double a, b, R;        // EOS params
    double TT0;            // reduced T/T0 (book)
    double TT;             // absolute T used in EOS (TT0*Tc)
    double gravity;        // +y body force
    double RR;             // droplet radius
    Dim_contactAngle2D dim;

    // helpers
    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
        return std::make_tuple(iX, iY);
    }
    size_t xyz_to_i (int x, int y) {
        return (y + dim.ny * x);
    }

    // populations (non-const)
    double& f(int i, int k)    { return lattice[k*dim.nelem + i]; }
    double& fin(int i, int k)  { return lattice[*parity*dim.npop + k*dim.nelem + i]; }
    double& fout(int i, int k) { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    // init: droplet resting on bottom wall (like contact-angle)
    void iniLattice(CellData& f0) {
        size_t i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz((int)i);

        int x_c = dim.nx / 2;
        int y_c = 5;
        double dx = double(iX) - double(x_c);
        double dy = double(iY) - double(y_c);
        double rho = (dx*dx + dy*dy <= RR*RR) ? rhol : rhog;

        for (int k = 0; k < 9; ++k) fin((int)i,k) = rho * t[k];
    }

    // macroscopic fields
    double density (double& f0) {
        int i = int(&f0 - lattice);
        double X_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double X_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);
        double X_0  = fin(i, 6) + fin(i, 1) + fin(i, 4);
        return X_M1 + X_P1 + X_0;
    }

    std::array<double,2> u_common (double& f0) {
        int i = int(&f0 - lattice);
        std::array<double, 2> u {0., 0.};
        double rho = std::max(density(f0), 1e-14);

        double X_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double X_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);
        double Y_M1 = fin(i, 1) + fin(i, 2) + fin(i, 8);
        double Y_P1 = fin(i, 3) + fin(i, 6) + fin(i, 7);

        u[0] = (X_P1 - X_M1) / rho;
        u[1] = (Y_P1 - Y_M1) / rho;
        return u;
    }

    // === Yuan–CS EOS (book) ===
    inline double cs2() const { return 1.0/3.0; }

    inline double Z_yuan(double rho) const {
        const double d = (1.0 - rho);
        const double frac = (4.0*rho - 2.0*rho*rho) / (d*d*d);
        return 1.0 + frac;
    }

    inline double P_eos_rho(double rho) const {
        return rho * R * TT * Z_yuan(rho) - a * rho * rho;
    }

    inline double G1_sign(double rho) const {
        // exactly like book: choose branch so that P matches via psi
        const double s = R*TT*Z_yuan(rho) - a*rho - cs2();
        return (s > 0.0) ? cs2() : -cs2();
    }

    inline double psi_yuan_from_rho(double rho) const {
        const double P   = P_eos_rho(rho);
        const double G1  = G1_sign(rho);
        const double val = 6.0 * (P - cs2()*rho) / G1;
        return (val > 0.0) ? std::sqrt(val) : 0.0;
    }

    // pressure (for output)
    inline double pressure_node(double& f0) {
        double rho = density(f0);
        double psi = psi_yuan_from_rho(rho);
        double G1  = G1_sign(rho);
        return cs2()*rho + (G1/6.0)*psi*psi;
    }

/*

    // total force = fluid–fluid + wall–fluid + gravity (book’s calcu_Fxy)
    std::array<double,2> force(double& f0) {
        int i = int(&f0 - lattice);
        auto [iX,iY] = i_to_xyz(i);

        const double rho_c = density(f0);
        const double psi_c = psi_yuan_from_rho(rho_c);
        const double G1    = G1_sign(rho_c);

        // wall pseudo-potential with SAME branch (like book)
        const double psi_w = psi_yuan_from_rho(rho_w);

        double sum_ff_x = 0.0, sum_ff_y = 0.0;   // fluid–fluid
        double sum_bb_x = 0.0, sum_bb_y = 0.0;   // wall vector sum

        for (int k=0; k<9; ++k) {
            int XX = (iX + c[k][0] + dim.nx) % dim.nx;
            int YY = iY + c[k][1];
            size_t nb = xyz_to_i(XX,YY);

            if (flag[nb] == CellType_contactAngle2D::bounce_back) {
                // like the Fortran: only the direction vectors enter
                sum_bb_x += t[k] * c[k][0];
                sum_bb_y += t[k] * c[k][1];
            } else {
                double& f0_nb = lattice[nb];
                double rho_nb = density(f0_nb);
                double psi_nb = psi_yuan_from_rho(rho_nb);
                sum_ff_x += t[k] * c[k][0] * psi_nb;
                sum_ff_y += t[k] * c[k][1] * psi_nb;
            }
        }

        // Final forces (book):
        //   F_ff = -G1 * psi_c * sum_{k} w_k e_k psi_nb
        //   S_w  = -G1 * psi_c * psi_w * sum_{k to-wall} w_k e_k
        double Fx = -G1 * psi_c * sum_ff_x  +  (-G1 * psi_c * psi_w * sum_bb_x);
        double Fy = -G1 * psi_c * sum_ff_y  +  (-G1 * psi_c * psi_w * sum_bb_y);

        // gravity (body force)
        Fy += gravity * rho_c;

        return {Fx, Fy};
    }

*/


    // total force = fluid–fluid + wall–fluid  (Fortran calcu_Fxy style)
    std::array<double,2> force(double& f0) {
        const int i = int(&f0 - lattice);
        auto [iX,iY] = i_to_xyz(i);

        const double rho_c = density(f0);
        if (rho_c <= 0.0) return {0.0, 0.0};

        // local branch (same as Fortran reusing G1 at the current node)
        const double G1c   = G1_sign(rho_c);
        const double psi_c = psi_yuan_from_rho(rho_c);

        // wall pseudo-potential with the SAME branch as the current node
        const double Zw    = Z_yuan(rho_w);
        const double val_w = 6.0 * rho_w * (R*TT*Zw - a*rho_w - cs2()) / G1c;
        const double psi_w = (val_w > 0.0) ? std::sqrt(val_w) : 0.0;

        double sum_ff_x = 0.0, sum_ff_y = 0.0; // fluid–fluid accumulator
        double sum_bb_x = 0.0, sum_bb_y = 0.0; // wall-vector accumulator

        for (int k = 0; k < 9; ++k) {
            const int XX = (iX + c[k][0] + dim.nx) % dim.nx;
            const int YY = (iY + c[k][1] + dim.ny) % dim.ny;  // wrap y like the book
            const size_t nb = xyz_to_i(XX, YY);

            if (flag[nb] == CellType_contactAngle2D::bounce_back) {
                // like Fortran: accumulate only w_k * e_k for wall directions
                sum_bb_x += t[k] * c[k][0];
                sum_bb_y += t[k] * c[k][1];
            } else {
                // fluid neighbor: accumulate w_k * e_k * psi_nb
                double& f_nb  = lattice[nb];
                const double rho_nb = density(f_nb);
                const double psi_nb = psi_yuan_from_rho(rho_nb);
                sum_ff_x += t[k] * c[k][0] * psi_nb;
                sum_ff_y += t[k] * c[k][1] * psi_nb;
            }
        }

        // final forces (exactly the book’s form)
        double Fx = -G1c * psi_c * sum_ff_x  +  (-G1c * psi_c * psi_w * sum_bb_x);
        double Fy = -G1c * psi_c * sum_ff_y  +  (-G1c * psi_c * psi_w * sum_bb_y);

        // NOTE: Fortran calcu_Fxy has no gravity term here.
        // If you want gravity, add Fy += gravity * rho_c outside this routine.
        return {Fx, Fy};
    }


    // velocity shifts: “upr” = u + (Fx+Sx)/2/rho, and tau-shift in feq (book)
    std::array<double,2> u_actual (double& f0) {
        double rho = std::max(density(f0), 1e-14);
        auto u     = u_common(f0);
        auto F     = force(f0);
        return { u[0] + 0.5 * F[0] / rho, u[1] + 0.5 * F[1] / rho };
    }
    std::array<double,2> u_collide (double& f0) {
        double tau = 1.0/omega;
        double rho = std::max(density(f0), 1e-14);
        auto u     = u_common(f0);
        auto F     = force(f0);
        return { u[0] + tau * F[0] / rho, u[1] + tau * F[1] / rho };
    }

    // streaming: periodic x; y has bounce-back walls at y=0, y=ny-1 (no wrap)
    void stream(size_t i, int k, int iX, int iY, double pop_out) {
        int x2 = (iX + c[k][0] + dim.nx) % dim.nx;
        int y2 = iY + c[k][1];
        size_t nb = xyz_to_i(x2,y2);
        if(flag[nb]==CellType_contactAngle2D::bounce_back){
            fout((int)i, opp[k]) = pop_out;    // simple on-site BB
        } else {
            fout(nb, k) = pop_out;
        }
    }

    auto collideBgk(int i, int k, double rho, double usqr, const array<double,2>& uc) {
        const double ck_u = c[k][0]*uc[0] + c[k][1]*uc[1];
        const double eq   = rho * t[k] * (1. + 3.*ck_u + 4.5*ck_u*ck_u - usqr);
        const double eqop = eq - 6.0 * rho * t[k] * ck_u;

        const double out     = (1. - omega) * fin(i, k)      + omega * eq;
        const double out_opp = (1. - omega) * fin(i, opp[k]) + omega * eqop;
        return std::make_pair(out, out_opp);
    }

    void operator() (double& f0) {
        int i = int(&f0 - lattice);
        if (flag[i] != CellType_contactAngle2D::bulk) return;

        auto [iX, iY] = i_to_xyz(i);
        double rho = std::max(density(f0), 1e-14);

        auto uc = u_collide(f0);                   // tau-shifted velocity
        double usqr = 1.5*(uc[0]*uc[0] + uc[1]*uc[1]);

        // pair directions
        for (int k = 0; k < 4; ++k) {
            auto [po, po_opp] = collideBgk(i, k, rho, usqr, uc);
            stream(i, k,      iX, iY, po);
            stream(i, opp[k], iX, iY, po_opp);
        }
        // center
        {
            int k=4;
            double eq = rho * t[k] * (1. - usqr);
            fout(i, k) = (1. - omega) * fin(i, k) + omega * eq;
        }
    }
};

// ============================ I/O & utilities ============================
void saveVtkFields_contactAngle2D(LBM_contactAngle2D& lbm, int time_iter, double dx=0.) {
    auto& dim=lbm.dim;
    if(dx==0.) dx=1./dim.nx;
    int dimZ=1;
    stringstream ss; ss<<"sol_"<<setfill('0')<<setw(7)<<time_iter<<".vtk";
    ofstream vtk(ss.str());
    vtk<<"# vtk DataFile Version 2.0\n";
    vtk<<"iteration "<<time_iter<<"\nASCII\n\n";
    vtk<<"DATASET STRUCTURED_POINTS\n";
    vtk<<"DIMENSIONS "<<dim.nx<<" "<<dim.ny<<" "<<dimZ<<"\n";
    vtk<<"ORIGIN 0 0 0\n";
    vtk<<"SPACING "<<dx<<" "<<dx<<" "<<dx<<"\n\n";
    vtk<<"POINT_DATA "<<dim.nx*dim.ny*dimZ<<"\n";

    // density
    vtk<<"SCALARS Density float 1\nLOOKUP_TABLE default\n";
    for(int y=0;y<dim.ny;++y){
        for(int x=0;x<dim.nx;++x){
            size_t i=lbm.xyz_to_i(x,y);
            double d = (lbm.flag[i]==CellType_contactAngle2D::bounce_back) ? 0.0
                                                                           : lbm.density(lbm.lattice[i]);
            vtk<<d<<" ";
        }
        vtk<<"\n";
    }
    vtk<<"\n";

    // pressure
    vtk<<"SCALARS Pressure float 1\nLOOKUP_TABLE default\n";
    for(int y=0;y<dim.ny;++y){
        for(int x=0;x<dim.nx;++x){
            size_t i=lbm.xyz_to_i(x,y);
            double p = (lbm.flag[i]==CellType_contactAngle2D::bounce_back) ? 0.0
                                                                           : lbm.pressure_node(lbm.lattice[i]);
            vtk<<p<<" ";
        }
        vtk<<"\n";
    }
    vtk<<"\n";

    // force
    vtk<<"VECTORS Force float\n";
    for(int y=0;y<dim.ny;++y){
        for(int x=0;x<dim.nx;++x){
            size_t i=lbm.xyz_to_i(x,y);
            auto F = lbm.force(lbm.lattice[i]);
            vtk<<F[0]<<" "<<F[1]<<" "<<0.0<<"\n";
        }
    }
    vtk<<"\n";
}



double totalMass(LBM_contactAngle2D& lbm, bool ignore_solids = true) {
    const auto& dim = lbm.dim;
    double M = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            if (ignore_solids && lbm.flag[i] == CellType_contactAngle2D::bounce_back) continue;
            M += lbm.density(lbm.lattice[i]);
        }
    }
    return M;
}


double computeEnergy_contactAngle2D(LBM_contactAngle2D& lbm) {
    const auto& dim = lbm.dim;
    double E = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_contactAngle2D::bulk) {
                auto u = lbm.u_actual(lbm.lattice[i]);
                E += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * E / (dim.nx * dim.ny);
}

void inigeom_contactAngle2D(LBM_contactAngle2D& lbm) {
    Dim_contactAngle2D const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == 0 || y == dim.ny -1) { // only bottom wall
                lbm.flag[i] = CellType_contactAngle2D::bounce_back;
                for (int k = 0 ; k<9 ; ++k) lbm.f(i,k) = 0.;
            } else {
                lbm.flag[i] = CellType_contactAngle2D::bulk;
            }
        }
    }
}



// --- Contact angle measurement (base/height method) ---
// Uses a density threshold rho_cut = 0.5*(rho_l+rho_g).
// Base width is measured on the first fluid row above the bottom wall.
// Height is measured along xmid = nx/2.
// Prints Base, Height, and Contact Angle (deg).

void calculateContactAngle(LBM_contactAngle2D& lbm,
                           double rho_l, double rho_g)
{
    const auto& dim = lbm.dim;
    const double rho_cut = 0.5*(rho_l + rho_g);
    const double PI = 3.14159265358979323846;

    // 1) find the first non-solid row above the bottom wall
    int base_y = 2.;
    while (base_y < dim.ny &&
           lbm.flag[ lbm.xyz_to_i(0, base_y) ] == CellType_contactAngle2D::bounce_back)
        ++base_y;

    if (base_y >= dim.ny-1) {
        std::cout << "ContactAngle: no fluid row found above wall.\n";
        return;
    }

    // helper to get density on the base row with periodic x
    auto rhoAtBase = [&](int x) -> double {
        int xx = (x % dim.nx + dim.nx) % dim.nx;
        size_t i = lbm.xyz_to_i(xx, base_y);
        return lbm.density(lbm.lattice[i]);
    };

    // 2) choose vertical line at xmid ~ center of domain
    const int xmid = dim.nx / 2;

    // 3) scan left/right from xmid on the base row until density < threshold
    int left = xmid, right = xmid;
    while (left  > 0           && rhoAtBase(left -1)  > rho_cut) --left;
    while (right < dim.nx - 1  && rhoAtBase(right +1) > rho_cut) ++right;
    const int base = std::max(0, right - left + 1);

    // 4) measure height along xmid
    int height = 0;
    for (int y = base_y; y < dim.ny; ++y) {
        size_t i = lbm.xyz_to_i(xmid, y);
        if (lbm.flag[i] == CellType_contactAngle2D::bounce_back) break;
        if (lbm.density(lbm.lattice[i]) > rho_cut) ++height;
        else break;
    }

    if (height <= 0 || base <= 1) {
        std::cout << "ContactAngle: droplet not detected (Base="<<base
                  <<", Height="<<height<<")\n";
        return;
    }

    // 5) circle geometry: R = (4h^2 + b^2) / (8h), theta = atan((b/2)/(R-h))
    const double h  = static_cast<double>(height);
    const double b  = static_cast<double>(base);
    const double R  = (4.0*h*h + b*b) / (8.0*h);
    double theta    = std::atan( (0.5*b) / (R - h) ) * 180.0 / PI;
    if (theta < 0.0) theta += 180.0;

    std::cout << std::setprecision(8)
              << "Base="   << base
              << " Height="<< height
              << " ContactAngle=" << theta << " deg\n";

    // (optional) append to file
    static std::ofstream ca_log("contact_angle.dat", std::ios::app);
    if (ca_log) ca_log << base << " " << height << " " << theta << "\n";
}


/*

// --- Contact angle measurement (base/height method) ---
// Uses a density threshold rho_cut = 0.5*(rho_l+rho_g).
// Base width is measured on the first fluid row above the bottom wall.
// Height is measured along xmid = nx/2.
// Prints Base, Height, and Contact Angle (deg).
// --- robust contact-angle (auto-raises the base row) ---
void calculateContactAngle(LBM_contactAngle2D& lbm,
                           double rho_l, double rho_g,
                           int y_guard = 1,          // skip this many rows above the wall (1 or 2 are common)
                           int max_rows_to_try = 10, // how many rows upward we'll try
                           int min_base_width = 3)   // reject tiny noisy segments
{
    const auto& dim = lbm.dim;
    const double PI = 3.14159265358979323846;
    const double rho_cut = 0.5*(rho_l + rho_g);  // threshold; tweak (e.g., 0.3*) if needed

    // 1) find first fluid row above the bottom wall
    int first_fluid_y = 0;
    while (first_fluid_y < dim.ny &&
           lbm.flag[ lbm.xyz_to_i(0, first_fluid_y) ] == CellType_contactAngle2D::bounce_back)
        ++first_fluid_y;

    if (first_fluid_y >= dim.ny-1) {
        std::cout << "ContactAngle: no fluid row found above wall.\n";
        return;
    }

    auto rhoAt = [&](int x, int y) -> double {
        int xx = (x % dim.nx + dim.nx) % dim.nx;
        if (y < 0 || y >= dim.ny) return 0.0;
        size_t i = lbm.xyz_to_i(xx, y);
        return (lbm.flag[i]==CellType_contactAngle2D::bounce_back) ? 0.0
                                                                   : lbm.density(lbm.lattice[i]);
    };

    // search base row upwards until we find a usable liquid run
    int base_y = std::min(first_fluid_y + y_guard, dim.ny-2);
    int best_left = -1, best_right = -1;

    auto find_longest_segment = [&](int y, int& L, int& R) -> int {
        int best_len = 0; L = R = -1;
        int run_start = -1;
        for (int x=0; x<dim.nx; ++x) {
            if (rhoAt(x,y) > rho_cut) {
                if (run_start == -1) run_start = x;
            } else if (run_start != -1) {
                int len = x - run_start;
                if (len > best_len) { best_len=len; L=run_start; R=x-1; }
                run_start = -1;
            }
        }
        // close trailing run
        if (run_start != -1) {
            int len = dim.nx - run_start;
            if (len > best_len) { best_len=len; L=run_start; R=dim.nx-1; }
        }
        return best_len;
    };

    int base_width = 0;
    for (int tries = 0; tries < max_rows_to_try; ++tries) {
        base_width = find_longest_segment(base_y, best_left, best_right);
        if (base_width >= min_base_width) break;       // found a usable base
        ++base_y;                                      // raise the base row and try again
        if (base_y >= dim.ny-1) break;
    }

    if (base_width < min_base_width) {
        std::cout << "ContactAngle: no liquid footprint found (tried "
                  << max_rows_to_try << " rows starting at y=" << first_fluid_y + y_guard << ").\n";
        return;
    }

    // 2) use the center of the base as x for height measurement
    const int x_center = (best_left + best_right) / 2;

    // 3) measure height from base_y upwards at x_center
    int height = 0;
    for (int y = base_y; y < dim.ny; ++y) {
        if (rhoAt(x_center, y) > rho_cut) ++height;
        else break;
    }

    if (height <= 0) {
        std::cout << "ContactAngle: height is zero at x_center="<<x_center
                  << " base_y="<<base_y<<".\n";
        return;
    }

    // 4) circle fit geometry
    const double b = static_cast<double>(base_width);
    const double h = static_cast<double>(height);
    const double R  = (4.0*h*h + b*b) / (8.0*h);
    double theta    = std::atan( (0.5*b) / (R - h) ) * 180.0 / PI;
    if (theta < 0.0) theta += 180.0;

    std::cout << std::setprecision(8)
              << "[CA] y_guard="<< y_guard
              << " base_y="<< base_y
              << " Base="<< base_width
              << " Height="<< height
              << " ContactAngle="<< theta << " deg\n";

    static std::ofstream ca_log("contact_angle.dat", std::ios::app);
    if (ca_log) ca_log << base_y << " " << base_width << " " << height << " " << theta << "\n";
}

 */

// ============================== driver ===============================
void contactAngle2D() {
    ifstream contfile("../apps/Config_Files/config_contactAngle2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. It should be named "
          "\"config_contactAngle2D.txt\" in Files_Config.");
    }

    // defaults
    double Re=60, ulb=0.1, max_t=10.0;
    int    N=100, out_freq=400, vtk_freq=400;

    double rhol=1.0, rhog=0.1, rho_w=0.12;   // coexisting densities, wall density
    double a=1.0, b=4.0, R=1.0, TT0=0.875;   // EOS params
    double gravity=0.0, tau_in=-1.0;         // body force, optional tau
    double RR= 100;                        // droplet radius
    int    k_index = -1;                     // 1..12 (book table); -1 = ignore

    // read config (param value per line)
    string line, param, value;
    while (getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        if(!(contfile >> param >> value)) break;

        if (param=="Re") Re=stod(value);
        else if (param=="ulb") ulb=stod(value);
        else if (param=="N") N=stoi(value);
        else if (param=="max_t") max_t=stod(value);
        else if (param=="out_freq") out_freq=stoi(value);
        else if (param=="vtk_freq") vtk_freq=stoi(value);

        else if (param=="rhol") rhol=stod(value);
        else if (param=="rhog") rhog=stod(value);
        // inside the while (getline...) parsing:
        else if (param=="rho_w" || param=="rhow") rho_w = stod(value);

        // optional: allow direct TT instead of TT0*Tc
        //else if (param=="TT") { double TT_in = stod(value); TT0 = TT_in; /* if you want to treat as absolute, also set lbm.TT=TT_in later */ }


        else if (param=="a") a=stod(value);
        else if (param=="b") b=stod(value);
        else if (param=="R") R=stod(value);
        else if (param=="TT0") TT0=stod(value);
        else if (param=="k_index") k_index=stoi(value); // 1..12 -> table mode

        else if (param=="gravity") gravity=stod(value);
        else if (param=="tau") tau_in=stod(value);
        else if (param=="RR") RR=stod(value);
    }

    /*
    // optional “book table” for (TT0, rhol, rhog)
    const double TT0W[12] = {0.975,0.95,0.925,0.9,0.875,0.85,0.825,0.8,0.775,0.75,0.7,0.65};
    const double RHW [12] = {0.16 ,0.21,0.23 ,0.247,0.265,0.279,0.29 ,0.314,0.30 ,0.33,0.36,0.38};
    const double RLW [12] = {0.08 ,0.067,0.05 ,0.0405,0.038,0.032,0.025,0.0245,0.02,0.015,0.009,0.006};

    if (k_index>=1 && k_index<=12) {
        TT0  = TT0W[k_index-1];
        rhol = RHW [k_index-1];
        rhog = RLW [k_index-1];
        cout<<"[table] k_index="<<k_index<<" -> TT0="<<TT0<<" rhol="<<rhol<<" rhog="<<rhog<<"\n";
    }
*/
    Dim_contactAngle2D dim {2*N, N};  // like your CA code: wider domain helps

    // tau/omega/nu/dx/dt
    double nu=0.0, omega=1.0, dx=1.0/N, dt=dx*ulb;
    if (tau_in > 0.0) {
        const double tau = tau_in;
        omega = 1.0 / tau;
        nu    = (tau - 0.5) / 3.0;
    } else {
        tie(nu, omega, dx, dt) = lbParameters_contactAngle2D(ulb, N, Re);
    }
    printParameters_contactAngle2D(dim, Re, omega, ulb, N, max_t, nu);

    // allocate
    vector<LBM_contactAngle2D::CellData> lattice_vect(LBM_contactAngle2D::sizeOfLattice(dim.nelem));
    auto *lattice = lattice_vect.data();

    vector<CellType_contactAngle2D> flag_vect(dim.nelem);
    auto* flag = flag_vect.data();

    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    auto[c, opp, t] = d2q9_constants_contactAngle2D();

    // build LBM object
    LBM_contactAngle2D lbm{
    lattice, flag, parity, &c[0], &opp[0], &t[0],
    omega, rhol, rhog, rho_w,
    a, b, R, TT0, /*TT*/0.0,
    gravity, RR, dim
    };

    // Yuan: TT = TT0 * Tc,  Tc = 0.3773*a/(b*R)
    const double Tc = 0.3773 * a / (b * R);
    lbm.TT = lbm.TT0 * Tc;   // absolute T used in EOS

    std::cout << std::setprecision(6)
    << "CS params: a="<<a<<" b="<<b<<" R="<<R<<"\n"
    << "TT0 (reduced)="<<TT0<<"  Tc="<<Tc<<"  TT (abs)="<<lbm.TT<<"\n"
    << "rho_l="<<rhol<<"  rho_g="<<rhog<<"  rho_w="<<rho_w<<"  RR="<<RR<<"\n"
    << "gravity="<<gravity<<"\n";



    // (handy sanity prints)
    auto psi = [&](double r){ return lbm.psi_yuan_from_rho(r); };
    std::cout << "psi(rho_l)="<<psi(rhol)
            << " psi(rho_g)="<<psi(rhog)
            << " psi(rho_w)="<<psi(rho_w) << "\n"
            << "G1(rho_l)="<<lbm.G1_sign(rhol)
            << " G1(rho_g)="<<lbm.G1_sign(rhog) << "\n";

    // init
    for_each(lattice, lattice + dim.nelem, [&lbm](double& f0){ lbm.iniLattice(f0); });
    inigeom_contactAngle2D(lbm);

    // time loop
    auto[start, clock_iter] = restartClock_contactAngle2D();
    ofstream energyfile("energy.dat");

    std::ofstream mass_log("mass.dat");
    double M0 = -1.0;

    const int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_contactAngle2D(lbm, time_iter, dx);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]\n";


            calculateContactAngle(lbm, rhol, rhog);


            const double M  = totalMass(lbm);
            if (M0 < 0.0) M0 = M;
            const double rel = (M - M0) / M0 * 100.0;

            std::cout << std::setprecision(12)
                    << "[Mass] M="<< M << "   ΔM/M0=" << std::setprecision(6) << rel << "%\n";
            if (mass_log) mass_log << std::setprecision(16) << time_iter*dt << " " << M << "\n";



            double energy = computeEnergy_contactAngle2D(lbm) * dx*dx / (dt*dt);
            cout << "Average energy: " << setprecision(10) << energy << "\n";
            energyfile << setw(10) << time_iter * dt << setw(16) << setprecision(10) << energy << "\n";
        }

        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_contactAngle2D(start, clock_iter, dim.nelem);
}
