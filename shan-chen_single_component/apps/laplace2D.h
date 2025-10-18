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

// D2Q9 Lattice Boltzmann Velocities:
//   3  6  7
//    \ | /
//   0--4--5
//    / | \
//   2  1  8

enum class CellType_Laplace2D : uint8_t { bounce_back, bulk };

inline auto d2q9_constants_Laplace2D() {
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

// Simulation dimensions
struct Dim_Laplace2D {
    Dim_Laplace2D(int nx_, int ny_)
        : nx(nx_), ny(ny_), nelem((size_t)nx*ny), npop(9*nelem) {}
    int nx, ny;
    size_t nelem, npop;
};

// LBM parameters (if tau not given)
inline auto lbParameters_Laplace2D(double ulb, int lref, double Re) {
    double nu = ulb * lref / Re;
    double omega = 1. / (3.*nu + 0.5);
    double dx = 1. / lref;
    double dt = dx * ulb;
    return make_tuple(nu, omega, dx, dt);
}

void printParameters_Laplace2D(const Dim_Laplace2D &dim, double Re, double omega,
                               double ulb, int N, double max_t , double nu)
{
    cout << "Laplace 2D problem\n"
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

inline auto restartClock_Laplace2D() {
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TimePoint>
void printMlups_Laplace2D(TimePoint start, int clock_iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double mlups = (double)(nelem * clock_iter) / duration.count();
    cout << "result: " << duration.count()/1e6 << " seconds" << endl;
    cout << "result: " << setprecision(4) << mlups << " MLUPS" << endl;
};

// --------------------------
// Yuan–CS single-component SC (D2Q9), contactAngle-style
// --------------------------
struct LBM_Laplace2D {
    using CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem; }

    // memory & topo
    CellData* lattice;
    CellType_Laplace2D* flag;
    int* parity;
    std::array<int,2>* c;
    int* opp;
    double* t;

    // hydro / params
    double omega;
    double rhol;
    double rhog;
    double rho_w;          // wall density (unused if no walls)
    double a;              // CS a
    double b;              // CS b
    double R;              // gas constant
    double TT0;            // reduced temp T/T0 (book-style)
    double TT;             // absolute “T” used in EOS (TT0*Tc)
    double gravity;        // +y body force
    Dim_Laplace2D dim;

    // helpers
    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
        return std::make_tuple(iX, iY);
    };
    size_t xyz_to_i (int x, int y) {
        return (y + dim.ny * x);
    };

    // populations (non-const, like your contactAngle2D)
    double& f(int i, int k)    { return lattice[k*dim.nelem + i]; }
    double& fin(int i, int k)  { return lattice[*parity*dim.npop + k*dim.nelem + i]; }
    double& fout(int i, int k) { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    // initialization: circular droplet (centered)
    void iniLattice(CellData& f0) {
        size_t i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(i);

        double cx = double(dim.nx) / 2.0;
        double cy = double(dim.ny) / 2.0;
        double Rdrop = 10;

        double dx = double(iX) - cx;
        double dy = double(iY) - cy;
        double rho = (dx*dx + dy*dy <= Rdrop*Rdrop) ? rhol : rhog;

        for (int k = 0; k < 9; ++k) fin(i,k) = rho * t[k];
    }

    // macroscopic fields
    auto density (double& f0) {
        auto i = &f0 - lattice;
        double X_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double X_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);
        double X_0  = fin(i, 6) + fin(i, 1) + fin(i, 4);
        return X_M1 + X_P1 + X_0;
    }

    auto u_common (double& f0) {
        auto i = &f0 - lattice;
        std::array<double, 2> u {0., 0.};
        double rho = max(density(f0), 1e-14);

        double X_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double X_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);

        double Y_M1 = fin(i, 1) + fin(i, 2) + fin(i, 8);
        double Y_P1 = fin(i, 3) + fin(i, 6) + fin(i, 7);

        u[0] = (X_P1 - X_M1) / rho;
        u[1] = (Y_P1 - Y_M1) / rho;
        return u;
    }

    // === Yuan–CS EOS pieces ===
    inline double cs2() const { return 1.0/3.0; }

    inline double Z_yuan(double rho) {
        const double d = (1.0 - rho);
        const double frac = (4.0*rho - 2.0*rho*rho) / (d*d*d);
        return 1.0 + frac;
    }

    inline double P_eos_rho(double rho) {
        return rho * R * TT * Z_yuan(rho) - a * rho * rho;
    }

    inline double G1_sign(double rho) {
        const double s = R*TT*Z_yuan(rho) - a*rho - cs2();
        return (s > 0.0) ? cs2() : -cs2();
    }

    inline double psi_yuan_from_rho(double rho) {
        const double P = P_eos_rho(rho);
        const double G1 = G1_sign(rho);
        const double val = 6.0 * (P - cs2()*rho) / G1;
        return (val > 0.0) ? std::sqrt(val) : 0.0;
    }

    // === total force: fluid–fluid + wall (if any) + gravity ===
    std::array<double,2> force(double& f0) {
        auto i = &f0 - lattice;
        auto [iX,iY] = i_to_xyz(i);

        const double rho_c = density(f0);
        const double psi_c = psi_yuan_from_rho(rho_c);
        const double G1    = G1_sign(rho_c);

        double sum_ff_x = 0.0, sum_ff_y = 0.0;
        double sum_bb_x = 0.0, sum_bb_y = 0.0;

        // wall psi (if you later set walls)
        const double psi_w = psi_yuan_from_rho(rho_w);

        for (int k=0; k<9; ++k) {
            int XX = (iX + c[k][0] + dim.nx) % dim.nx;
            int YY = (iY + c[k][1] + dim.ny) % dim.ny;
            size_t nb = xyz_to_i(XX,YY);

            if (flag[nb] == CellType_Laplace2D::bounce_back) {
                // only vector sum enters wall term
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

        // fluid–fluid (note the MINUS sign)
        double Fx = -G1 * psi_c * sum_ff_x;
        double Fy = -G1 * psi_c * sum_ff_y;

        // wall–fluid (zero if no walls flagged)
        Fx += -G1 * psi_c * psi_w * sum_bb_x;
        Fy += -G1 * psi_c * psi_w * sum_bb_y;

        // gravity
        Fy += gravity * rho_c;

        return {Fx, Fy};
    }

    // shifted velocities
    auto u_eq (double& f0) {
        double tau = 1. / omega;
        double rho = max(density(f0), 1e-14);
        auto u     = u_common(f0);
        auto F     = force(f0);
        return array<double,2>{ u[0] + tau * F[0] / rho, u[1] + tau * F[1] / rho };
    }
    auto u_actual (double& f0) {
        double rho = max(density(f0), 1e-14);
        auto u     = u_common(f0);
        auto F     = force(f0);
        return array<double,2>{ u[0] + 0.5 * F[0] / rho, u[1] + 0.5 * F[1] / rho };
    }

    // stream (periodic in both x,y for Laplace test)
    void stream(size_t i, int k, int iX, int iY, double pop_out) {
        int x2 = (iX + c[k][0] + dim.nx) % dim.nx;
        int y2 = (iY + c[k][1] + dim.ny) % dim.ny;

        size_t nb = xyz_to_i(x2,y2);
        if(flag[nb]==CellType_Laplace2D::bounce_back){
            fout(i,opp[k])=pop_out;
        } else {
            fout(nb,k)=pop_out;
        }
    };

    auto collideBgk(int i, int k, double rho, double usqr, array<double,2>& ueq) {
        double pop_out, pop_out_opp;

        const double ck_u = c[k][0]*ueq[0] + c[k][1]*ueq[1];
        const double eq   = rho * t[k] * (1. + 3.*ck_u + 4.5*ck_u*ck_u - usqr);
        const double eqop = eq - 6.0 * rho * t[k] * ck_u; // opp

        pop_out     = (1. - omega) * fin(i, k)      + omega * eq;
        pop_out_opp = (1. - omega) * fin(i, opp[k]) + omega * eqop;

        return std::make_pair(pop_out, pop_out_opp);
    }

    void operator() (double& f0) {
        auto i = &f0 - lattice;
        if (flag[i] != CellType_Laplace2D::bulk) return;

        auto [iX, iY] = i_to_xyz(i);
        double rho = max(density(f0), 1e-14);
        auto ueq   = u_eq(f0);
        double usqr = 1.5*(ueq[0]*ueq[0] + ueq[1]*ueq[1]);

        // paired streaming
        for (int k = 0; k < 4; ++k) {
            auto [pop_out, pop_out_opp] = collideBgk(i, k, rho, usqr, ueq);
            stream(i, k,      iX, iY, pop_out);
            stream(i, opp[k], iX, iY, pop_out_opp);
        }
        // center
        {
            int k=4;
            double eq = rho * t[k] * (1. - usqr);
            fout(i, k) = (1. - omega) * fin(i, k) + omega * eq;
        }
    }

    inline double pressure_node(double& f0) {
        // lattice form identical to EOS if discrete forcing consistent:
        // p = cs2*rho + G1/6 * psi^2
        double rho = density(f0);
        double psi = psi_yuan_from_rho(rho);
        double G1  = G1_sign(rho);
        return (1.0/3.0)*rho + (1.0/6.0)*G1*psi*psi;
    }
};

// ---------- VTK ----------
void saveVtkFields_Laplace2D(LBM_Laplace2D& lbm, int time_iter, double dx=0.) {
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

    vtk<<"SCALARS Density float 1\nLOOKUP_TABLE default\n";
    for(int y=0;y<dim.ny;++y){
        for(int x=0;x<dim.nx;++x){
            size_t i=lbm.xyz_to_i(x,y);
            double d=lbm.density(lbm.lattice[i]);
            if(lbm.flag[i]==CellType_Laplace2D::bounce_back) d=0.;
            vtk<<d<<" ";
        }
        vtk<<"\n";
    }
    vtk<<"\n";

    vtk<<"SCALARS Pressure float 1\nLOOKUP_TABLE default\n";
    for(int y=0;y<dim.ny;++y){
        for(int x=0;x<dim.nx;++x){
            size_t i=lbm.xyz_to_i(x,y);
            double p = (lbm.flag[i]==CellType_Laplace2D::bounce_back)?0.0:lbm.pressure_node(lbm.lattice[i]);
            vtk<<p<<" ";
        }
        vtk<<"\n";
    }
    vtk<<"\n";

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

// energy with u_actual (like your contactAngle2D)
double computeEnergy_Laplace2D(LBM_Laplace2D& lbm) {
    const auto& dim = lbm.dim;
    double E = 0.0;
    for (int y=0; y<dim.ny; ++y)
      for (int x=0; x<dim.nx; ++x) {
        int i = lbm.xyz_to_i(x,y);
        if (lbm.flag[i]==CellType_Laplace2D::bulk) {
            auto u = lbm.u_actual(lbm.lattice[i]);
            E += u[0]*u[0] + u[1]*u[1];
        }
      }
    return 0.5 * E / (dim.nx*dim.ny);
}

double totalMass_Laplace2D(LBM_Laplace2D& lbm, bool ignore_solids = true) {
    const auto& dim = lbm.dim;
    double M = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            if (ignore_solids && lbm.flag[i] == CellType_Laplace2D::bounce_back) continue;
            M += lbm.density(lbm.lattice[i]);
        }
    }
    return M;
}


// periodic everywhere (good for Laplace)
void inigeom_Laplace2D(LBM_Laplace2D& lbm) {
    Dim_Laplace2D const& dim = lbm.dim;
    for (int y=0; y<dim.ny; ++y)
      for (int x=0; x<dim.nx; ++x) {
        int i = lbm.xyz_to_i(x,y);
        lbm.flag[i] = CellType_Laplace2D::bulk;
      }
}
void Laplace2D() {
    ifstream contfile("../apps/Config_Files/config_Laplace2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument("Config file not found. It should be named \"config_Laplace2D.txt\" in Files_Config.");
    }

    // --- config defaults (same names as before) ---
    double Re=60, ulb=0.1, max_t=10.0, rhol=1.0, rhog=0.1, rho_w=0.12;
    double a=1.0, b=4.0, R=1.0, TT0=0.875, gravity=0.0, tau_in=-1.0, g_IGNORED=0.0;
    int N=100, out_freq=400, vtk_freq=400;

    string line, value, param;
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
        else if (param=="rho_w") rho_w=stod(value);
        else if (param=="g") g_IGNORED=stod(value); // not used in Yuan path
        else if (param=="a") a=stod(value);
        else if (param=="b") b=stod(value);
        else if (param=="R") R=stod(value);
        else if (param=="TT0") TT0=stod(value);
        else if (param=="gravity") gravity=stod(value);
        else if (param=="tau") tau_in=stod(value);
    }

    Dim_Laplace2D dim {N, N};

    // compute nu, omega, dx, dt
    double nu=0.0, omega=1.0, dx=1.0/N, dt=dx*ulb;
    if (tau_in > 0.0) {
        const double tau = tau_in;
        omega = 1.0 / tau;
        nu    = (tau - 0.5) / 3.0;
    } else {
        tie(nu, omega, dx, dt) = lbParameters_Laplace2D(ulb, N, Re);
    }
    printParameters_Laplace2D(dim, Re, omega, ulb, N, max_t, nu);

    // allocate
    vector<LBM_Laplace2D::CellData> lattice_vect(LBM_Laplace2D::sizeOfLattice(dim.nelem));
    auto *lattice = &lattice_vect[0];

    vector<CellType_Laplace2D> flag_vect(dim.nelem);
    auto* flag = &flag_vect[0];

    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    auto[c, opp, t] = d2q9_constants_Laplace2D();

    LBM_Laplace2D lbm{lattice, flag, parity, &c[0], &opp[0], &t[0],
                      omega, rhol, rhog, rho_w, a, b, R, TT0, /*TT*/0.0,
                      gravity, dim};

    // Yuan: TT = TT0 * Tc,  Tc = 0.3773*a/(b*R)
    const double Tc = 0.3773 * a / (b * R);
    lbm.TT = lbm.TT0 * Tc;

    // init
    for_each(lattice, lattice + dim.nelem, [&lbm](double& f0){ lbm.iniLattice(f0); });
    inigeom_Laplace2D(lbm);

    // time loop
    auto[start, clock_iter] = restartClock_Laplace2D();
    ofstream energyfile("energy.dat");
    ofstream mass_log("mass.dat");

    double M0 = -1.0;


    const int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields_Laplace2D(lbm, time_iter, dx);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]\n";
            double energy = computeEnergy_Laplace2D(lbm) * dx*dx / (dt*dt);
            cout << "Average energy: " << setprecision(8) << energy << "\n";
            energyfile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << "\n";


            const double M  = totalMass_Laplace2D(lbm);
            if (M0 < 0.0) M0 = M;
            const double rel = (M - M0) / M0 * 100.0;
            std::cout << std::setprecision(12)
                      << "[Mass] M="<< M << "   ΔM/M0=" << std::setprecision(6) << rel << "%\n";
            if (mass_log) mass_log << std::setprecision(16) << time_iter*dt << " " << M << "\n";
        }

        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_Laplace2D(start, clock_iter, dim.nelem);
}
