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
#include <execution>   // C++17 parallel algorithms (par_unseq)
#include <functional>
#include <chrono>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <numeric>     // iota

using namespace std;
using namespace std::chrono;

// ─────────────────────────   cell-level helpers   ───────────────────────────
enum class CellType_YL2D : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants (rest at k=4; opp-pairing) ─────
// (Eq. 8) directions & (He–Luo) weights
inline auto d2q9_constants_YL2D()
{
    // order: k = 0..8 ; opp = {5,6,7,8,4,0,1,2,3}; k=4 is rest
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
struct Dim_YL2D {
    Dim_YL2D(int nx_, int ny_)
        : nx(nx_), ny(ny_),
          nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny_)),
          npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

// ─────────────────────────   timing helpers   ──────────────────────────────
inline auto restartClock_YL2D() { return make_pair(high_resolution_clock::now(), 0); }

template<class TP>
inline void printMlups_YL2D(TP start, int iter, size_t nelem)
{
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    // MLUPS = million lattice updates per second (one collide+stream per cell)
    double mlups = (double(nelem) * double(iter)) / double(us.count());
    cout << "Runtime: " << us.count()/1e6 << " s\n";
    cout << "Throughput: " << setprecision(5) << mlups << " MLUPS\n";
}

// ─────────────────────────   solver functor (BGK)   ────────────────────────
// Implements Fakhari et al. (2017) conservative phase-field + velocity-based hydro.
// Inline comments reference Eq. numbers from the paper.
struct LBM_Young_Laplace2D
{
    using  CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 4 * 9 * nelem; }

    // raw buffers: [ h_in | h_out | g_in | g_out ], each 9*nelem
    CellData*                 lattice;
    CellType_YL2D*            flag;
    int*                      parity;   // 0/1 ping-pong
    std::array<int,2>*        c;        // e_α  (Eq. 8)
    int*                      opp;      // opposite index, for pairing
    double*                   t;        // w_α  (weights, Eq. 8)

    Dim_YL2D                  dim;

    // Physical & model parameters (match Fortran defaults)
    double Rhol  = 0.001;      // light phase density
    double Rhoh  = 1.0;        // heavy phase density
    double Sigma = 0.01;       // surface tension σ
    double W     = 4.0;        // interface thickness ξ (lu)
    double M     = 0.02;       // mobility M  (Eq. 11)
    double tau   = 0.8;        // hydrodynamic BGK relaxation τ  (Eq. 21)
    double s8    = 1.0/0.8;    // = 1/τ used in code (BGK collision, Eq. 26)

    // Conservative phase-field coefficients β, κ (Eq. 5)
    double Beta  = 12.0 * Sigma / W;     // β = 12 σ / ξ
    double kappa = 1.5  * Sigma * W;     // κ = 3 σ ξ / 2

    // convenience
    static constexpr double cs2   = 1.0/3.0;  // c_s^2
    static constexpr double epsC  = 1e-30;
    double dRho3 = (Rhoh - Rhol)/3.0;   // helper used like Fortran for F_p

    // ───────── array accessors with ping-pong (short k-loop via opp pairing) ─────────
    inline double& fin (int i,int k){ return lattice[               (*parity)*9*dim.nelem + k*dim.nelem + i]; }
    inline double& fout(int i,int k){ return lattice[((1-*parity))*9*dim.nelem + k*dim.nelem + i]; }
    inline double& gin (int i,int k){ return lattice[2*9*dim.nelem + (*parity)*9*dim.nelem + k*dim.nelem + i]; }
    inline double& gout(int i,int k){ return lattice[2*9*dim.nelem + ((1-*parity))*9*dim.nelem + k*dim.nelem + i]; }

    // ───────── index helpers ─────────
    inline size_t xyz_to_i (int X, int Y) const { return size_t(Y) + size_t(dim.ny) * size_t(X); }
    inline auto   i_to_xyz (int i)        const { int X = i / dim.ny; int Y = i % dim.ny; return std::make_tuple(X, Y); }
    inline int wrapX(int X)  const { int nx=dim.nx; int v = X % nx; return v<0? v+nx : v; }
    inline int wrapY(int Y)  const { int ny=dim.ny; int v = Y % ny; return v<0? v+ny : v; }

    // ───────── macroscopic fields (one value per cell) ─────────
    vector<double> C;      // phase field φ (Eq. 12)
    vector<double> P;      // normalized pressure p* (sum g_α ; Eq. 32a)
    vector<double> Rho;    // mixture density ρ = ρ_L + (φ-φ_L)(ρ_H-ρ_L) (Eq. 13; here φ_L=0, φ_H=1)
    vector<double> Ux, Uy; // velocity (Eq. 32b)
    vector<double> mu;     // chemical potential μ_φ (Eq. 5)
    vector<double> DcDx, DcDy; // ∇φ  (Eqs. 34, used also in Eqs. 4,7,33,35)
    vector<double> ni,  nj;    // n = ∇φ / |∇φ|  (used in Eq. 7)

    LBM_Young_Laplace2D(CellData* lat_,
                        CellType_YL2D* flag_,
                        int* parity_,
                        std::array<int,2>* c_,
                        int* opp_,
                        double* t_,
                        Dim_YL2D dim_)
      : lattice(lat_), flag(flag_), parity(parity_), c(c_), opp(opp_), t(t_), dim(dim_),
        C(dim.nelem,0.0), P(dim.nelem,0.0), Rho(dim.nelem,0.0),
        Ux(dim.nelem,0.0), Uy(dim.nelem,0.0),
        mu(dim.nelem,0.0), DcDx(dim.nelem,0.0), DcDy(dim.nelem,0.0),
        ni(dim.nelem,0.0), nj(dim.nelem,0.0) {}

    // =====================================================================
    // Initialization  — tanh bubble profile (Eq. 2, choose + for bubble)
    // Also set P ← P − φ σ / (R (ρ/3)) like the Fortran bubble line.
    // =====================================================================
    void iniCell(int i)
    {
        auto [X,Y] = i_to_xyz(i);
        double xc = double(dim.nx)/2.0 - 0.5;
        double yc = double(dim.ny)/2.0 - 0.5;
        double R0 = double(dim.nx) / 8.0;                // R = L0/8
        double r  = std::sqrt( (X - xc)*(X - xc) + (Y - yc)*(Y - yc) );
        double phi = 0.5 - 0.5 * std::tanh( 2.0*(R0 - r)/W );   // bubble (φ_L=0, φ_H=1), Eq. (2)

        // u=0 ⇒ Γ-w term zero initially (Eq. 10)
        for (int kq=0;kq<9;++kq) {
            fin (i,kq) = phi * t[kq];    // h_eq with Γ = w, Fφ omitted at t=0 (Eqs. 9-10)
            gin (i,kq) = 0.0;            // p* starts 0
        }

        C[i]   = phi;                    // Eq. (12)
        Rho[i] = Rhol + phi*(Rhoh - Rhol); // Eq. (13)
        Ux[i] = Uy[i] = 0.0;
        P[i]   = 0.0;

        // Laplace-pressure correction like Fortran init (bubble)
        double prho = (Rho[i] + 1e-12)/3.0;
        double corr = (phi * Sigma / R0) / prho;  // sign “−” for bubble below
        P[i] -= corr;

        // initialize g to p*w + (Γ-w) with u=0 ⇒ (Γ-w)=0
        for (int kq=0;kq<9;++kq) gin(i,kq) = P[i] * t[kq];
    }

    // =====================================================================
    // Helpers for distributions
    // =====================================================================
    // (Eq. 10) Γ_α − w_α = w_α [ 3 e·u + 4.5 (e·u)^2 − 1.5 u^2 ]
    inline void GaWa_fromU(double U,double V, double Ga_Wa[9])  const {
        double U2 = U*U + V*V;
        for (int kq=0;kq<9;++kq){
            double eU = c[kq][0]*U + c[kq][1]*V;
            Ga_Wa[kq] = t[kq] * ( 3.0*eU + 4.5*eU*eU - 1.5*U2 );
        }
    }

    // BGK stress from g^neq (exclude rest k=4 to match Fortran) — used in Eq. (30) form
    inline void StressTensor_BGK( const double gneq[9], double& sxx,double& sxy,double& syy) const {
        sxx = sxy = syy = 0.0;
        for (int kq=0;kq<9;++kq){
            if (kq==4) continue;
            sxx += gneq[kq] * c[kq][0] * c[kq][0];
            sxy += gneq[kq] * c[kq][0] * c[kq][1];
            syy += gneq[kq] * c[kq][1] * c[kq][1];
        }
    }

    // Viscous force F_μ (BGK form consistent with Eq. 30; same scaling as Fortran)
    inline void ViscousForce_BGK(double tau_, double dcdx, double dcdy,
                                  const double gneq[9], double& FmX,double& FmY) const {
        double sxx,sxy,syy; StressTensor_BGK(gneq,sxx,sxy,syy);
        double fac = (0.5 - tau_) / tau_;
        double dR  = (Rhoh - Rhol);
        FmX = fac * (sxx*dcdx + sxy*dcdy) * dR;   // Eq. (30) + Eq. (33)
        FmY = fac * (sxy*dcdx + syy*dcdy) * dR;
    }

    // ─────────────────────────  streaming (periodic)  ──────────────────────
    inline void stream_pair(int i,int k,int X,int Y,double hk,double gk)
    {
        int XX = wrapX(X + c[k][0]);
        int YY = wrapY(Y + c[k][1]);
        size_t nb = xyz_to_i(XX,YY);
        fout(nb,k) = hk;
        gout(nb,k) = gk;
    }

    // =====================================================================
    // Collide + stream at one cell (uses stored fields)  — h: (Eq. 6–11), g: (Eq. 14–20, 32)
    // We run with paired k/opp[k] (short inner loop) and then handle k=4 (rest).
    // =====================================================================
    void collide_stream_at(int i)
    {
        if (flag[i] != CellType_YL2D::bulk) return;
        auto [X,Y] = i_to_xyz(i);

        // local fields
        double Cc   = C[i];
        double Rhoc = Rho[i];
        double Pn   = P[i];
        double U    = Ux[i];
        double V    = Uy[i];
        double dCx  = DcDx[i];
        double dCy  = DcDy[i];
        double muc  = mu[i];
        double ni_  = ni[i];
        double nj_  = nj[i];

        // Γ_α − w_α from current u (Eq. 10)
        double Ga_Wa[9]; GaWa_fromU(U,V,Ga_Wa);
        double GammaK[9]; for (int kq=0;kq<9;++kq) GammaK[kq] = t[kq] + Ga_Wa[kq];

        // ===== h (phase) LBE: Eq. (6) with F^φ_α (Eq. 7), h̄^eq (Eq. 9)
        double eFh[9], hlp_h[9], heqk[9];
        double shape = ( 1.0 - 4.0*(Cc - 0.5)*(Cc - 0.5) ) / W;  // bracket in Eq. (7) / ξ
        for (int kq=0;kq<9;++kq){
            double proj = c[kq][0]*ni_ + c[kq][1]*nj_;            // e_α · (∇φ/|∇φ|) in Eq. (7)
            eFh[kq]   = shape * proj;
            hlp_h[kq] = t[kq] * eFh[kq];
            heqk[kq]  = Cc * GammaK[kq] - 0.5 * hlp_h[kq];        // h̄^eq_α (Eq. 9)
        }
        double wc = 1.0 / (0.5 + 3.0*M);                          // τ_φ via M (Eq. 11) ⇒ w_c = 1/(0.5+3M)

        // ===== g (hydro) LBE: Eq. (14) with F_α (Eq. 15), ḡ^eq (Eq. 16–17)
        // Extra forces (Eq. 18): Fs=μ_φ∇φ (Eq. 4), Fp (Eq. 19) with ∇ρ=(ρH-ρL)∇φ (Eq. 33), Fμ (Eq. 20 via Eq. 30)
        double FpX = - Pn * dRho3 * dCx;                          // Eq. (19) + (33), using Fortran's dRho3 helper
        double FpY = - Pn * dRho3 * dCy;

        double geqk[9]; for (int kq=0;kq<9;++kq) geqk[kq] = Pn * t[kq] + Ga_Wa[kq];   // Eq. (17)
        double gneq[9]; for (int kq=0;kq<9;++kq) gneq[kq] = gin(i,kq) - geqk[kq];
        double FmX, FmY; ViscousForce_BGK(tau, dCx, dCy, gneq, FmX, FmY);            // Eq. (30)

        double Fx = muc*dCx + FpX + FmX;                           // total F (Eq. 18), Fs uses Eq. (4)
        double Fy = muc*dCy + FpY + FmY;

        // Guo-style half-step (leading-order of Eq. 15 with trapezoidal correction used in Eq. 16)
        double hlp_g[9], geq_corr[9];
        for (int kq=0;kq<9;++kq){
            double eF = c[kq][0]*Fx + c[kq][1]*Fy;
            hlp_g[kq]    = 3.0 * t[kq] * eF / (Rhoc + epsC);       // δt * wα e·F / (ρ c_s^2), with c_s^2=1/3 ⇒ *3
            geq_corr[kq] = geqk[kq] - 0.5*hlp_g[kq];               // ḡ^eq (Eq. 16)
        }

        // ====================== COLLISION + STREAM (paired) ======================
        auto collide_h = [&](int kq) {
            return (1.0 - wc)*fin(i,kq) + wc*heqk[kq] + hlp_h[kq];              // Eq. (6)
        };
        auto collide_g = [&](int kq) {
            return (1.0 - s8)*gin(i,kq) + s8*geq_corr[kq] + hlp_g[kq];          // Eq. (14) with BGK (Eq. 26)
        };

        for (int kq=0;kq<4;++kq){
            int ko = opp[kq];
            double hk   = collide_h(kq),   hkop = collide_h(ko);
            double gk   = collide_g(kq),   gkop = collide_g(ko);
            stream_pair(i, kq, X,Y, hk,   gk  );
            stream_pair(i, ko, X,Y, hkop, gkop);
        }
        // rest k=4 streams to itself
        {
            int k0=4;
            fout(i,k0) = collide_h(k0);
            gout(i,k0) = collide_g(k0);
        }
    }

    // =====================================================================
    // Post-stream macroscopic updates (Fortran order)
    //   φ,ρ (Eq. 12–13) → ∇φ (Eq. 34) → μ_φ (Eq. 5 with ∇²φ from Eq. 35)
    //   n=∇φ/|∇φ| → p* (Eq. 32a) → u (Eq. 32b) using Fp (19), Fμ (30) and Fs (4)
    // =====================================================================
    void update_fields()
    {
        // φ and ρ from h (Eq. 12, 13)
        for (int X=0; X<dim.nx; ++X)
        for (int Y=0; Y<dim.ny; ++Y){
            int i = xyz_to_i(X,Y);
            double phi=0.0; for (int kq=0;kq<9;++kq) phi += fin(i,kq);
            C[i]   = phi;
            Rho[i] = Rhol + phi*(Rhoh - Rhol);
        }

        // helper to fetch φ with periodic wrap
        auto C_at = [&](int X,int Y){
            return C[ xyz_to_i(wrapX(X), wrapY(Y)) ];
        };

        // ∇φ via isotropic centered differences (Eq. 34) — implemented with the 9-point isotropic stencil (Fortran)
        for (int X=0; X<dim.nx; ++X)
        for (int Y=0; Y<dim.ny; ++Y){
            int i = xyz_to_i(X,Y);
            double cE=C_at(X+1,Y), cW=C_at(X-1,Y), cN=C_at(X,Y+1), cS=C_at(X,Y-1);
            double cNE=C_at(X+1,Y+1), cNW=C_at(X-1,Y+1), cSE=C_at(X+1,Y-1), cSW=C_at(X-1,Y-1);
            DcDx[i] = (cE - cW)/3.0 + ( cSE + cNE - cSW - cNW )/12.0;  // same as Fortran Isotropic_Gradient
            DcDy[i] = (cN - cS)/3.0 + ( cNW + cNE - cSW - cSE )/12.0;
        }

        // μ_φ = 4β φ(φ−1)(φ−0.5) − κ ∇²φ  (Eq. 5),  ∇²φ by 9-pt Laplacian (Eq. 35 implemented as in Fortran)
        for (int X=0; X<dim.nx; ++X)
        for (int Y=0; Y<dim.ny; ++Y){
            int i = xyz_to_i(X,Y);
            double cC=C_at(X,Y);
            double cE=C_at(X+1,Y), cW=C_at(X-1,Y), cN=C_at(X,Y+1), cS=C_at(X,Y-1);
            double cNE=C_at(X+1,Y+1), cNW=C_at(X-1,Y+1), cSE=C_at(X+1,Y-1), cSW=C_at(X-1,Y-1);
            double D2C = ( cSW + cSE + cNW + cNE + 4.0*(cS + cW + cE + cN) - 20.0*cC ) / 6.0;
            mu[i] = 4.0*Beta * cC*(cC-1.0)*(cC-0.5) - kappa * D2C;
        }

        // n = ∇φ / |∇φ|  (used in Eq. 7)
        for (int i=0;i<(int)dim.nelem;++i){
            double g2 = DcDx[i]*DcDx[i] + DcDy[i]*DcDy[i] + 1e-32;
            double inv = 1.0 / std::sqrt(g2);
            ni[i] = DcDx[i] * inv;
            nj[i] = DcDy[i] * inv;
        }

        // p* (Eq. 32a), then u (Eq. 32b with F total from Eqs. 4,19,30 and ∇ρ=(ρH−ρL)∇φ, Eq. 33)
        for (int X=0; X<dim.nx; ++X)
        for (int Y=0; Y<dim.ny; ++Y){
            int i = xyz_to_i(X,Y);

            double pstar=0.0; for (int kq=0;kq<9;++kq) pstar += gin(i,kq);
            P[i] = pstar;                                        // Eq. (32a)

            double FpX = - P[i] * dRho3 * DcDx[i];               // Eq. (19) + (33)
            double FpY = - P[i] * dRho3 * DcDy[i];

            double Ga_Wa[9]; GaWa_fromU(Ux[i],Uy[i],Ga_Wa);
            double geqk[9]; for (int kq=0;kq<9;++kq) geqk[kq] = P[i]*t[kq] + Ga_Wa[kq]; // Eq. (17)
            double gneq[9]; for (int kq=0;kq<9;++kq) gneq[kq] = gin(i,kq) - geqk[kq];

            double FmX, FmY; ViscousForce_BGK(tau, DcDx[i], DcDy[i], gneq, FmX, FmY); // Eq. (30)

            double Fx = mu[i]*DcDx[i] + FpX + FmX;               // Eq. (18), Fs from Eq. (4)
            double Fy = mu[i]*DcDy[i] + FpY + FmY;

            double mx=0.0,my=0.0;  // ∑ gα eα
            for (int kq=0;kq<9;++kq){
                mx += gin(i,kq) * c[kq][0];
                my += gin(i,kq) * c[kq][1];
            }
            Ux[i] = mx + 0.5*Fx / (Rho[i] + epsC);               // Eq. (32b)
            Uy[i] = my + 0.5*Fy / (Rho[i] + epsC);
        }
    }
};

// ─────────────────────────  VTK output (same payload as Fortran) ─────────
inline void saveVtkFields_Young_Laplace2D(LBM_Young_Laplace2D& lbm, int it)
{
    Dim_YL2D const& dim = lbm.dim;
    double dx = 1.0; int dimZ = 1;

    stringstream ss;
    ss << "sol_" << setw(7) << setfill('0') << it << ".vtk";
    ofstream os(ss.str());
    os << "# vtk DataFile Version 2.0\n";
    os << "iteration " << it << "\n";
    os << "ASCII\n\n";
    os << "DATASET STRUCTURED_POINTS\n";
    os << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dimZ << "\n";
    os << "ORIGIN 0 0 0\n";
    os << "SPACING " << dx << " " << dx << " " << dx << "\n\n";
    os << "POINT_DATA " << dim.nx * dim.ny * dimZ << "\n";

    os << "SCALARS phi float 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){
        for (int x=0;x<dim.nx;++x){
            int i = lbm.xyz_to_i(x,y);
            os << float(lbm.C[i]) << " ";
        } os << "\n";
    } os << "\n";

    os << "SCALARS Pressure float 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){
        for (int x=0;x<dim.nx;++x){
            int i = lbm.xyz_to_i(x,y);
            os << float(lbm.P[i]) << " ";
        } os << "\n";
    } os << "\n";

    os << "VECTORS velocity float\n";
    for (int y=0;y<dim.ny;++y){
        for (int x=0;x<dim.nx;++x){
            int i = lbm.xyz_to_i(x,y);
            os << float(lbm.Ux[i]) << " " << float(lbm.Uy[i]) << " 0\n";
        } os << "\n";
    } os << "\n";

    os << "SCALARS Flag int 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){
        for (int x=0;x<dim.nx;++x){
            int i = lbm.xyz_to_i(x,y);
            os << (lbm.flag[i]==CellType_YL2D::bounce_back?1:0) << " ";
        } os << "\n";
    } os << "\n";
}

// Energy / Mass logging (just helpers)
inline double computeEnergy_Young_Laplace2D(LBM_Young_Laplace2D& lbm) {
    auto& dim = lbm.dim;
    double E = 0.0;
    for (int y=0; y<dim.ny; ++y)
    for (int x=0; x<dim.nx; ++x) {
        int i = lbm.xyz_to_i(x,y);
        double vx = lbm.Ux[i], vy = lbm.Uy[i];
        E += vx*vx + vy*vy;
    }
    return 0.5 * E / (dim.nx * dim.ny);
}
inline double totalMass_Young_Laplace2D(LBM_Young_Laplace2D& lbm) {
    auto& dim = lbm.dim;
    double M = 0.0;
    for (int y = 0; y < dim.ny; ++y)
    for (int x = 0; x < dim.nx; ++x) {
        size_t i = lbm.xyz_to_i(x, y);
        M += lbm.Rho[i];
    }
    return M;
}

// Geometry (all bulk, periodic domain)
inline void inigeom_Young_Laplace2D(LBM_Young_Laplace2D& lbm) {
    for (int y=0; y<lbm.dim.ny; ++y)
    for (int x=0; x<lbm.dim.nx; ++x)
        lbm.flag[ lbm.xyz_to_i(x,y) ] = CellType_YL2D::bulk;
}

// ─────────────────────────   driver (BGK)   ─────────────────────────────────
// Reads config, runs tf steps; exports VTK/energy/mass
inline void Young_Laplace2D()
{
    // ---------- Read config ----------
    ifstream contfile("../apps/Config_Files/config_laplace2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. Expected \"config_laplace2D.txt\" in ../apps/Config_Files/");
    }

    // Defaults = Fortran
    int    N = 128, tf = 10000, out_freq = tf/10, vtk_freq = tf/10;
    double Sigma=0.01, W=4.0, M=0.02;
    double RhoL=0.001, RhoH=1.0, tau=0.8;

    string param, value, line;
    while (getline(contfile, line)) {
        if (auto pos = line.find('#'); pos != string::npos) line.erase(pos);
        if (line.empty()) continue;
        stringstream ls(line);
        ls >> param >> value;
        if (!ls) continue;

        if      (param == "N")         N        = stoi(value);
        else if (param == "tf")        tf       = stoi(value);
        else if (param == "out_freq")  out_freq = stoi(value);
        else if (param == "vtk_freq")  vtk_freq = stoi(value);
        else if (param == "Sigma")     Sigma    = stod(value);
        else if (param == "W")         W        = stod(value);
        else if (param == "M")         M        = stod(value);
        else if (param == "RhoL")      RhoL     = stod(value);
        else if (param == "RhoH")      RhoH     = stod(value);
        else if (param == "tau")       tau      = stod(value);
        else {
            cerr << "Warning: unknown parameter \"" << param << "\"\n";
        }
    }

    Dim_YL2D dim {N, N};

    // ---------- Allocate ----------
    vector<CellData> lattice_vect(LBM_Young_Laplace2D::sizeOfLattice(dim.nelem), 0.0);
    CellData *lattice = lattice_vect.data();

    vector<CellType_YL2D> flag_vect(dim.nelem, CellType_YL2D::bulk);
    CellType_YL2D* flag = flag_vect.data();

    int p_store = 0;
    int* parity = &p_store;

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_YL2D();

    LBM_Young_Laplace2D lbm{ lattice, flag, parity,
                             c_vect.data(), opp_vect.data(), t_vect.data(),
                             dim };

    // Set params from config (and derived constants like the Fortran module)
    lbm.Sigma = Sigma; lbm.W = W; lbm.M = M;
    lbm.Rhol  = RhoL;  lbm.Rhoh = RhoH;
    lbm.tau   = tau;   lbm.s8   = 1.0 / tau;
    lbm.Beta  = 12.0 * lbm.Sigma / lbm.W;
    lbm.kappa = 1.5  * lbm.Sigma * lbm.W;
    lbm.dRho3 = (lbm.Rhoh - lbm.Rhol)/3.0;

    // ---------- Init ----------
    for (size_t i=0;i<dim.nelem;++i) lbm.iniCell((int)i);
    inigeom_Young_Laplace2D(lbm);
    lbm.update_fields();

    // Prebuild a random-access index vector so par_unseq is guaranteed to parallelize
    vector<int> cell_index(dim.nelem);
    std::iota(cell_index.begin(), cell_index.end(), 0);

    // ---------- Run ----------
    auto [start, clock_iter] = restartClock_YL2D();
    ofstream efile("energy.dat");
    ofstream mass_log("mass.dat");
    double M0 = -1.0;

    for (int it = 0; it <= tf; ++it)
    {
        if (vtk_freq != 0 && it % vtk_freq == 0) {
            saveVtkFields_Young_Laplace2D(lbm, it);
        }
        if (out_freq != 0 && it % out_freq == 0) {
            cout << "it = " << setw(7) << it
                 << "   [" << fixed << setprecision(1) << (100.0*it/double(tf)) << "%]" << endl;
            double energy = computeEnergy_Young_Laplace2D(lbm);
            cout << "Average kinetic energy: " << setprecision(8) << energy << endl;
            if (efile) efile << setw(10) << it << setw(16) << setprecision(8) << energy << "\n";

            double Mcur  = totalMass_Young_Laplace2D(lbm);
            if (M0 < 0.0) M0 = Mcur;
            double rel = (Mcur - M0) / M0 * 100.0;
            cout << std::setprecision(12)
                 << "[Mass] M="<< Mcur << "   ΔM/M0=" << std::setprecision(6) << rel << "%\n";
            if (mass_log) mass_log << it << " " << std::setprecision(16) << Mcur << "\n";
        }

        // collide + stream over cells — par_unseq (no OpenMP). Scatter writes go to distinct (nb,k) ⇒ no races.
        std::for_each(std::execution::par_unseq,
                      cell_index.begin(), cell_index.end(),
                      [&](int i){ lbm.collide_stream_at(i); });

        // ping-pong swap
        *parity = 1 - *parity;
        ++clock_iter;

        // post-stream macros
        lbm.update_fields();
    }

    printMlups_YL2D(start, clock_iter, dim.nelem);
}
