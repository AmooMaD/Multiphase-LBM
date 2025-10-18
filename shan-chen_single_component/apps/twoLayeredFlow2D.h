// LayeredTwoPhase_SC_CS_Yuan.cpp
// Single-component pseudopotential (Shan–Chen) LBM with Carnahan–Starling (Yuan) EOS
// D2Q9, BGK, periodic x, bounce-back walls at y=0, y=Ny-1.
// Constant attractive coupling G<0, EOS→psi mapping with constant pressure shift p_shift.
// Includes layered (liquid near walls, gas in middle) initialization and uniform body-force drive (gx).
//
// ──────────────────────────────────────────────────────────────────────────────
// Example config file (place next to the executable as: config_twoLayeredFlow2D.txt)
// ──────────────────────────────────────────────────────────────────────────────
// # grid/time
// N           100
// ulb         0.1
// Re          60
// max_t       200.0
// out_freq    500
// vtk_freq    1000
//
// # EOS (Yuan–CS)
// a           1.0
// b           4.0
// R           1.0
// TT0         0.875
//
// # densities (targets for initialization; the model will relax near coexistence)
// rhol        0.265
// rhog        0.038
// rho_w       0.265      # wet the walls with liquid
//
// # interface shaping
// h_lower     0.30       # thickness of each near-wall liquid layer as fraction of H (0..0.5)
// w_int       4          # interface half-thickness in lattice nodes (3–6 is safer than 1–2)
//
// # relaxation (optional)
// #tau         1.0
//
// # body force (prefer uniform gx for Poiseuille-like drive)
// gx          1e-7
// gy          0.0
// Gx_const    0.0
//
// # model coupling (optional overrides; otherwise defaults)
// #G          -1.0       # global attractive coupling
//
// ──────────────────────────────────────────────────────────────────────────────
// Recommended presets (rule-of-thumb, adjust if needed):
//
// 1) Moderate ratio ~7–8:
//    TT0=0.875, a=1.0, b=4.0, R=1.0, rhol≈0.26, rhog≈0.038, rho_w≈rhol, w_int=4, G=-1.
//
// 2) Ratio ~15–20:
//    TT0≈0.85,  a=0.8 (thicker interface, calmer), b=4.0, R=1.0,
//    rhol≈0.28, rhog≈0.02, rho_w≈rhol, w_int=5, G=-1.
//
// 3) Ratio ~50+:
//    TT0≈0.80,  a≈0.6–0.7 (even thicker to tame spurious currents), b=4.0, R=1.0,
//    rhol≈0.30, rhog≈0.006–0.01, rho_w≈rhol, w_int=6, G=-1.
//
// Notes:
//  • Decreasing TT0 increases density ratio; reducing "a" thickens the interface and improves stability.
//  • Keep velocities small (Ma≪1); prefer gx drive and set Gx_const=0 for layered Poiseuille.
//  • p_shift is auto-computed so ψ stays real in [rhog,rhol].
//  • Pressure written to VTK is the thermodynamic EOS pressure P_eos(rho).
// ──────────────────────────────────────────────────────────────────────────────

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

// D2Q9 lattice directions (cx,cy) and weights t[k]
inline auto d2q9_constants() {
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

struct Dim { Dim(int nx_, int ny_) : nx(nx_), ny(ny_), nelem((size_t)nx*ny), npop(9*nelem) {} int nx,ny; size_t nelem,npop; };

inline auto lbParameters(double ulb, int lref, double Re) {
    double nu = ulb * lref / Re;            // kinematic viscosity (LU)
    double omega = 1. / (3.*nu + 0.5);      // BGK relaxation
    double dx = 1. / lref;
    double dt = dx * ulb;                   // time step implied by ulb
    return make_tuple(nu, omega, dx, dt);
}

void printParameters(const Dim &dim, double Re, double omega,
                     double ulb, int N, double max_t , double nu,
                     double h_lower, int w_int, double G) {
    cout << "Two-Layered Flow 2-D (Yuan–CS)\n"
         << "N      = " << N      << '\n'
         << "nx     = " << dim.nx << '\n'
         << "ny     = " << dim.ny << '\n'
         << "Re     = " << Re     << '\n'
         << "omega  = " << omega  << '\n'
         << "tau    = " << 1. / omega  << '\n'
         << "nu     = " << nu  << '\n'
         << "ulb    = " << ulb    << '\n'
         << "max_t  = " << max_t  << '\n'
         << "h_lower (frac of H) = " << h_lower << '\n'
         << "w_int (nodes) = " << w_int << '\n'
         << "G (coupling) = " << G << '\n';
}

inline auto restartClock(){ return make_pair(high_resolution_clock::now(), 0); }

template<class TimePoint>
void printMlups(TimePoint start, int clock_iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double mlups = (double)(nelem * clock_iter) / duration.count();
    cout << "result: " << duration.count()/1e6 << " seconds\n";
    cout << "result: " << setprecision(4) << mlups << " MLUPS\n";
}

enum class CellType : uint8_t { bounce_back, bulk };

// ======================== Yuan–CS single-component SC (D2Q9) ========================
struct LBM {
    using CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem; }

    // memory & topo
    CellData* lattice;
    CellType* flag;
    int* parity;
    std::array<int,2>* c; int* opp; double* t;

    // params
    double omega;          // BGK relaxation parameter
    double rhol, rhog;     // target (initial) densities
    double rho_w;          // wall density for wetting
    double a, b, R;        // EOS params (Carnahan–Starling)
    double TT0;            // reduced T/Tc (Yuan’s notation)
    double TT;             // absolute T used in EOS (TT0*Tc)
    double gx, gy;         // body force per unit mass (uniform)
    double G;              // global attractive coupling (<0)
    double p_shift;        // constant pressure shift for ψ mapping
    Dim dim;

    // helpers
    inline double cs2() const { return 1.0/3.0; }
    auto i_to_xyz (int i) { int iX = i / (dim.ny); int iY = i % (dim.ny); return std::make_tuple(iX, iY); }
    size_t xyz_to_i (int x, int y) { return (y + dim.ny * x); }

    // populations (non-const)
    double& f(int i, int k)    { return lattice[k*dim.nelem + i]; }
    double& fin(int i, int k)  { return lattice[*parity*dim.npop + k*dim.nelem + i]; }
    double& fout(int i, int k) { return lattice[(1-*parity)*dim.npop + k*dim.nelem + i]; }

    // EOS: Yuan’s Carnahan–Starling form
    inline double Z_yuan(double rho) const {
        const double d = (1.0 - rho);
        const double frac = (4.0*rho - 2.0*rho*rho) / (d*d*d);
        return 1.0 + frac;
    }
    inline double P_eos_rho(double rho) const {
        return rho * R * TT * Z_yuan(rho) - a * rho * rho;
    }

    // ψ(ρ) from EOS with constant-G mapping and constant pressure shift
    // p_SC = c_s^2 ρ + (G c_s^2 / 2) ψ^2  ⇒  ψ^2 = 2 (c_s^2 ρ - (P_eos + p_shift)) / (G c_s^2)
    inline double psi_from_rho(double rho) const {
        const double P = P_eos_rho(rho) + p_shift;
        const double S = cs2()*rho - P;                 // S must be ≥ 0 in [ρg, ρl]
        if (S <= 0.0) return 0.0;                       // clipped; p_shift should avoid this
        return std::sqrt( 2.0 * S / (std::abs(G) * cs2()) ); // use |G| since G<0
    }

    // pressure for VTK: report thermodynamic EOS pressure (not mechanical lattice pressure)
    inline double pressure_node(double& f0) {
        double rho = density(f0);
        return P_eos_rho(rho);
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

    // force: fluid–fluid + wall–fluid + uniform body force; constant G everywhere
    std::array<double,2> force(double& f0) {
        const int i = int(&f0 - lattice);
        auto [iX,iY] = i_to_xyz(i);
        const double rho_c = density(f0);
        if (rho_c <= 0.0) return {0.0, 0.0};

        const double psi_c = psi_from_rho(rho_c);
        const double psi_w = psi_from_rho(rho_w); // same mapping for wall

        double sum_ff_x = 0.0, sum_ff_y = 0.0; // fluid–fluid accumulator
        double sum_bb_x = 0.0, sum_bb_y = 0.0; // wall-vector accumulator

        for (int k = 0; k < 9; ++k) {
            const int XX = (iX + c[k][0] + dim.nx) % dim.nx;
            const int YY = iY + c[k][1];

            // treat out-of-domain in y as wall contribution
            if (YY < 0 || YY >= dim.ny) {
                sum_bb_x += t[k] * c[k][0];
                sum_bb_y += t[k] * c[k][1];
                continue;
            }

            const size_t nb = xyz_to_i(XX, YY);
            if (flag[nb] == CellType::bounce_back) {
                sum_bb_x += t[k] * c[k][0];
                sum_bb_y += t[k] * c[k][1];
            } else {
                double& f_nb  = lattice[nb];
                const double psi_nb = psi_from_rho( density(f_nb) );
                sum_ff_x += t[k] * c[k][0] * psi_nb;
                sum_ff_y += t[k] * c[k][1] * psi_nb;
            }
        }

        double Fx = -G * psi_c * sum_ff_x  +  (-G * psi_c * psi_w * sum_bb_x);
        double Fy = -G * psi_c * sum_ff_y  +  (-G * psi_c * psi_w * sum_bb_y);

        // uniform body force (prefer gx for layered Poiseuille)
        Fx += gx;
        Fy += gy;

        return {Fx, Fy};
    }

    // velocity shifts
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

    // streaming: periodic x; y has bounce-back walls at y=0, y=ny-1
    void stream(size_t i, int k, int iX, int iY, double pop_out) {
        int x2 = (iX + c[k][0] + dim.nx) % dim.nx;
        int y2 = iY + c[k][1];
        size_t nb = xyz_to_i(x2,y2);
        if(flag[nb]==CellType::bounce_back){
            fout((int)i, opp[k]) = pop_out; // on-site BB
        } else {
            fout(nb, k) = pop_out;
        }
    }

    auto collideBgk(int i, int k, double rho, double usqr, const array<double,2>& uc) {
        const double ck_u = c[k][0]*uc[0] + c[k][1]*uc[1];
        const double eq   = rho * t[k] * (1. + 3.*ck_u + 4.5*ck_u*ck_u - usqr);
        const double eqop = eq - 6.0 * rho * t[k] * ck_u; // paired opposite
        const double out     = (1. - omega) * fin(i, k)      + omega * eq;
        const double out_opp = (1. - omega) * fin(i, opp[k]) + omega * eqop;
        return std::make_pair(out, out_opp);
    }

    void operator() (double& f0) {
        int i = int(&f0 - lattice);
        if (flag[i] != CellType::bulk) return;

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

    // ---- initialization: liquid near walls, gas in the middle (two interfaces) ----
    // h_lower: fraction (0..0.5) of channel height occupied by each near-wall liquid layer
    // w_int  : interface half-thickness in lattice nodes (tanh smoothing)
    void iniLattice_layers(CellData& f0, double h_lower, int w_int) {
        const size_t i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz((int)i);

        const double H      = double(dim.ny - 1);
        const double y_low  = std::clamp(h_lower, 0.0, 0.5) * H;   // bottom layer height
        const double y_high = H - y_low;                           // top layer start
        const double w      = std::max(1, w_int);

        const double yy = double(iY);
        const double s_bottom = 0.5 * (1.0 - std::tanh((yy - y_low ) / w)); // ~1 below y_low, 0 above
        const double s_top    = 0.5 * (1.0 + std::tanh((yy - y_high) / w)); // ~1 above y_high, 0 below

        double s_liq = s_bottom + s_top;           // liquid near both walls
        s_liq = std::clamp(s_liq, 0.0, 1.0);
        const double s_gas = 1.0 - s_liq;

        const double rho = s_liq * rhog + s_gas * rhol; // CORRECT: liquid at walls, gas in middle
        for (int k = 0; k < 9; ++k) fin((int)i, k) = rho * t[k];
    }
};

// ============================ I/O & utilities ============================
void saveVtkFields(LBM& lbm, int time_iter, double dx=0.) {
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
            double d = (lbm.flag[i]==CellType::bounce_back) ? 0.0 : lbm.density(lbm.lattice[i]);
            vtk<<d<<" ";
        }
        vtk<<"\n";
    }
    vtk<<"\n";

    // pressure (thermodynamic EOS)
    vtk<<"SCALARS Pressure float 1\nLOOKUP_TABLE default\n";
    for(int y=0;y<dim.ny;++y){
        for(int x=0;x<dim.nx;++x){
            size_t i=lbm.xyz_to_i(x,y);
            double p = (lbm.flag[i]==CellType::bounce_back) ? 0.0 : lbm.pressure_node(lbm.lattice[i]);
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

    // velocity (upr = u + 0.5*F/rho)
    vtk<<"VECTORS Velocity float\n";
    for (int y=0; y<dim.ny; ++y){
        for (int x=0; x<dim.nx; ++x){
            size_t i = lbm.xyz_to_i(x,y);
            if (lbm.flag[i]==CellType::bounce_back){
                vtk<<0.0<<" "<<0.0<<" "<<0.0<<"\n";
            } else {
                auto u = lbm.u_actual(lbm.lattice[i]);
                vtk<<u[0]<<" "<<u[1]<<" "<<0.0<<"\n";
            }
        }
    }
    vtk<<"\n";
}

double totalMass(LBM& lbm, bool ignore_solids = true) {
    const auto& dim = lbm.dim;
    double M = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            if (ignore_solids && lbm.flag[i] == CellType::bounce_back) continue;
            M += lbm.density(lbm.lattice[i]);
        }
    }
    return M;
}

double computeEnergy(LBM& lbm) {
    const auto& dim = lbm.dim;
    double E = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType::bulk) {
                auto u = lbm.u_actual(lbm.lattice[i]);
                E += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * E / (dim.nx * dim.ny);
}

void inigeom(LBM& lbm) {
    Dim const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == 0 || y == dim.ny -1) { // top & bottom walls
                lbm.flag[i] = CellType::bounce_back;
                for (int k = 0 ; k<9 ; ++k) lbm.f(i,k) = 0.;
            } else {
                lbm.flag[i] = CellType::bulk;
            }
        }
    }
}

// ============================== driver ===============================
void twoLayeredFlow2D() {
    ifstream contfile("../apps/Config_Files/config_twoLayeredFlow2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. Expected: config_twoLayeredFlow2D.txt in current directory.");
    }

    // defaults
    double Re=60, ulb=0.1, max_t=10.0; int N=100, out_freq=400, vtk_freq=400;
    double rhol=1.0, rhog=0.1, rho_w=0.12;   // coexisting densities, wall density
    double a=1.0, b=4.0, R=1.0, TT0=0.875;   // EOS params
    double tau_in=-1.0;                      // optional override for tau
    // interface controls
    double h_lower = 0.3;    // bottom layer thickness (fraction of channel height)
    int    w_int   = 4;      // interface half-thickness in nodes (tanh width)
    double gx = 0.0, gy = 0.0;   // body-force drive (prefer gx)
    double Gx_const = 0.0;       // not used; kept for compatibility
    double G = -1.0;             // global attractive coupling

    // read config (param value per line; '#' starts comment)
    string line, param, value;
    while (getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        istringstream iss(line);
        if (!(iss >> param >> value)) continue;
        if (param=="Re") Re=stod(value);
        else if (param=="ulb") ulb=stod(value);
        else if (param=="N") N=stoi(value);
        else if (param=="max_t") max_t=stod(value);
        else if (param=="out_freq") out_freq=stoi(value);
        else if (param=="vtk_freq") vtk_freq=stoi(value);
        else if (param=="rhol") rhol=stod(value);
        else if (param=="rhog") rhog=stod(value);
        else if (param=="rho_w" || param=="rhow") rho_w = stod(value);
        else if (param=="a") a=stod(value);
        else if (param=="b") b=stod(value);
        else if (param=="R") R=stod(value);
        else if (param=="TT0") TT0=stod(value);
        else if (param=="tau") tau_in=stod(value);
        else if (param=="gx") gx = stod(value);
        else if (param=="gy") gy = stod(value);
        else if (param=="Gx_const") Gx_const = stod(value);
        else if (param=="h_lower") h_lower=stod(value);
        else if (param=="w_int")   w_int=stoi(value);
        else if (param=="G") G = stod(value);
    }

    // domain: a bit wider in x for stratified layers
    Dim dim {10, N+1};

    // tau/omega/nu/dx/dt
    double nu=0.0, omega=1.0, dx=1.0/N, dt=dx*ulb;
    tie(nu, omega, dx, dt) = lbParameters(ulb, N, Re);
    if (tau_in>0.0) omega = 1.0/tau_in;

    printParameters(dim, Re, omega, ulb, N, max_t, nu, h_lower, w_int, G);

    // allocate
    vector<LBM::CellData> lattice_vect(LBM::sizeOfLattice(dim.nelem));
    auto *lattice = lattice_vect.data();
    vector<CellType> flag_vect(dim.nelem); auto* flag = flag_vect.data();
    vector<int> parity_vect {0}; int* parity = &parity_vect[0];
    auto[c, opp, t] = d2q9_constants();

    // build LBM object
    LBM lbm{
        lattice, flag, parity, &c[0], &opp[0], &t[0],
        omega, rhol, rhog, rho_w,
        a, b, R, TT0, /*TT*/0.0,
        gx, gy,
        G, /*p_shift*/0.0,
        dim
    };

    // Yuan: Tc = 0.3773 * a / (b * R),  TT = TT0 * Tc
    const double Tc = 0.3773 * a / (b * R);
    lbm.TT = lbm.TT0 * Tc;

    cout << std::setprecision(6)
         << "CS params: a="<<a<<" b="<<b<<" R="<<R<<"\n"
         << "TT0 (reduced)="<<TT0<<"  Tc="<<Tc<<"  TT (abs)="<<lbm.TT<<"\n"
         << "rho_l="<<rhol<<"  rho_g="<<rhog<<"  rho_w="<<rho_w<<"\n";

    // choose p_shift so S(ρ)=cs2*ρ - (P(ρ)+p_shift) ≥ 0 on [ρg, ρl]
    auto max_S_neg = [&lbm](double rmin, double rmax) {
        double worst = -1e30; int Ns = 600;
        for (int s=0; s<=Ns; ++s) {
            double r = rmin + (rmax - rmin) * (double(s)/Ns);
            double S = lbm.cs2()*r - lbm.P_eos_rho(r);
            worst = std::max(worst, -S); // if S is negative, -S positive increases worst
        }
        return std::max(0.0, worst);
    };
    lbm.p_shift = max_S_neg(rhog, rhol) + 1e-12; // small cushion
    cout << "p_shift = " << std::setprecision(12) << lbm.p_shift << "\n";

    // sanity prints
    auto psi = [&](double r){ return lbm.psi_from_rho(r); };
    cout << "psi(rho_l)="<<psi(rhol)
         << " psi(rho_g)="<<psi(rhog)
         << " psi(rho_w)="<<psi(rho_w) << "\n";

    // init populations & geometry
    for_each(lattice, lattice + dim.nelem, [&lbm,h_lower,w_int](double& f0){ lbm.iniLattice_layers(f0, h_lower, w_int); });
    inigeom(lbm);

    // time loop
    auto[start, clock_iter] = restartClock();
    ofstream energyfile("energy.dat");
    ofstream mass_log("mass.dat");
    double M0 = -1.0;

    const int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0) {
            saveVtkFields(lbm, time_iter, dx);
        }
        if (out_freq != 0 && time_iter % out_freq == 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]\n";

            const double M  = totalMass(lbm);
            if (M0 < 0.0) M0 = M;
            const double rel = (M - M0) / M0 * 100.0;

            std::cout << std::setprecision(12)
                      << "[Mass] M="<< M << "   ΔM/M0=" << std::setprecision(6) << rel << "%\n";
            if (mass_log) mass_log << std::setprecision(16) << time_iter*dt << " " << M << "\n";

            double energy = computeEnergy(lbm) * dx*dx / (dt*dt);
            cout << "Average energy: " << setprecision(10) << energy << "\n";
            energyfile << setw(10) << time_iter * dt << setw(16) << setprecision(10) << energy << "\n";
        }

        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);
        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups(start, clock_iter, dim.nelem);
}
