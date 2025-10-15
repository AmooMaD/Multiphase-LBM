#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
using std::size_t;

// -------------------- Lattice/domain --------------------
static constexpr int    L0   = 128;
static constexpr int    Nx   = L0;
static constexpr int    Ny   = L0;
static constexpr int    X0   = L0/2 + 1;
static constexpr int    Y0   = L0/2 + 1;
static constexpr double cs2  = 1.0/3.0;
static constexpr double dt   = 1.0;
static constexpr double dx   = 1.0;

// -------------------- Time control ----------------------
static constexpr int tf   = 100000;
static constexpr int step = tf/100;

// -------------------- Physical params -------------------
// Densities (light/heavy):
static constexpr double rhoL  = 0.001;     // light
static constexpr double rhoH  = 1.0;       // heavy

// Dynamic viscosities mu_L, mu_H (choose your physics):
// If you want uniform kinematic nu, set mu{L,H} = rho{L,H} * nu_target.
static constexpr double nu_target = 0.1;   // example kinematic viscosity
static constexpr double muL = rhoL * nu_target;
static constexpr double muH = rhoH * nu_target;

// Surface tension & interface thickness
static constexpr double Sigma = 0.01;
static constexpr double W_    = 4.0;           // ξ
// Phase-field constants (β=12σ/ξ, κ=3σξ/2)
static constexpr double Beta  = 12.0 * Sigma / W_;
static constexpr double kappa = 1.5  * Sigma * W_;

// Phase-field mobility M and its relaxation time τ_φ (M = τ_φ c_s^2, dt=1)
static constexpr double Mmob    = 0.02;
static constexpr double tau_phi = Mmob / cs2;
static constexpr double w_c     = 1.0 / (0.5 + tau_phi); // BGK weight for h

// Geometry & init
//static constexpr double R           = L0/8.0;
//static constexpr bool   DROP_INSIDE = true;  // true: heavy drop, false: light bubble
// Geometry & init
static constexpr double R           = L0/8.0;

// --- NEW: rectangle full sizes (in lattice units)
static constexpr double RECT_WX = 2.0 * R;          // full width
static constexpr double RECT_WY = (M_PI/2.0) * R;   // full height  (≈ area-match with circle)

// choose which phase is inside (unchanged)
static constexpr bool   DROP_INSIDE = true;


// Body force (gravity) in lattice units: F_b = rho * g
static constexpr double GRAV_Y = 0.;   // downward acceleration
// (change sign to flip direction; magnitude must be small for stability)

// -------------------- D2Q9 ------------------------------
static const int    ex[9]  = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
static const int    ey[9]  = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
static const double wv[9]  = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0 };

    inline size_t idx(int k, int x, int y){
        return size_t(k)*(Nx+2)*(Ny+2) + size_t(x)*(Ny+2) + size_t(y);
    }

    // -------------------- Fields ----------------------------
    std::vector<double> h(9*(Nx+2)*(Ny+2), 0.0);
    std::vector<double> g(9*(Nx+2)*(Ny+2), 0.0);

    double phi [Nx+2][Ny+2]{};      // phase field
    double rho [Nx+2][Ny+2]{};      // density
    double P   [Nx+2][Ny+2]{};      // reduced pressure p* (sum g_k)
    double Ux  [Nx+2][Ny+2]{};
    double Uy  [Nx+2][Ny+2]{};

    // gradients & normals of φ
    double dphix[Nx+2][Ny+2]{};
    double dphiy[Nx+2][Ny+2]{};
    double ni   [Nx+2][Ny+2]{};
    double nj   [Nx+2][Ny+2]{};

    // chemical potential μ_φ
    double mu_phi[Nx+2][Ny+2]{};

    // paper’s dynamic viscosity and *local* tau (for g)
    double MU  [Nx+2][Ny+2]{};
    double TAU [Nx+2][Ny+2]{};

    // scratch
    double Gamma[9]{}, Ga_Wa[9]{}, heq_bar[9]{}, geq[9]{}, hlp[9]{};

    // -------------------- Helpers ---------------------------
    void periodic_fill(double A[Nx+2][Ny+2]){
        for(int x=0;x<=Nx+1;++x){ A[x][0]=A[x][Ny]; A[x][Ny+1]=A[x][1]; }
        for(int y=0;y<=Ny+1;++y){ A[0][y]=A[Nx][y]; A[Nx+1][y]=A[1][y]; }
    }
    void Boundary_Conditions_f(std::vector<double>& f){
        for(int k=0;k<9;++k){
            for(int y=0;y<=Ny+1;++y){ f[idx(k,0,y)]=f[idx(k,Nx,y)]; f[idx(k,Nx+1,y)]=f[idx(k,1,y)]; }
            for(int x=0;x<=Nx+1;++x){ f[idx(k,x,0)]=f[idx(k,x,Ny)]; f[idx(k,x,Ny+1)]=f[idx(k,x,1)]; }
        }
    }
    void Streaming(std::vector<double>& f){
        static double fnew[9][Nx+2][Ny+2];
        for(int y=1;y<=Ny;++y) for(int x=1;x<=Nx;++x) for(int a=1;a<=8;++a){
            fnew[a][x][y] = f[idx(a, x-ex[a], y-ey[a])];
        }
        for(int y=1;y<=Ny;++y) for(int x=1;x<=Nx;++x) for(int a=1;a<=8;++a){
            f[idx(a,x,y)] = fnew[a][x][y];
        }
    }
    void Equilibrium_new(double U, double V){
        double U2 = U*U+V*V;
        for(int a=0;a<9;++a){
            double eU = ex[a]*U + ey[a]*V;
            Ga_Wa[a] = wv[a]*( eU*(3.0 + 4.5*eU) - 1.5*U2 );
        }
    }
    void iso_grad_phi(){
        for(int y=1;y<=Ny;++y){
            for(int x=1;x<=Nx;++x){
                dphix[x][y] = (phi[x+1][y]-phi[x-1][y])/3.0
                + (phi[x+1][y-1]+phi[x+1][y+1]-phi[x-1][y-1]-phi[x-1][y+1])/12.0;
                dphiy[x][y] = (phi[x][y+1]-phi[x][y-1])/3.0
                + (phi[x-1][y+1]+phi[x+1][y+1]-phi[x-1][y-1]-phi[x+1][y-1])/12.0;
            }
        }
    }
    void normals_from_grad(){
        for(int y=1;y<=Ny;++y){
            for(int x=1;x<=Nx;++x){
                double s = std::sqrt(dphix[x][y]*dphix[x][y] + dphiy[x][y]*dphiy[x][y] + 1e-32);
                ni[x][y] = dphix[x][y]/s;
                nj[x][y] = dphiy[x][y]/s;
            }
        }
    }
    void chemical_potential(){
        const double phiL=0.0, phiH=1.0, phi0=0.5*(phiL+phiH);
        for(int y=1;y<=Ny;++y){
            for(int x=1;x<=Nx;++x){
                double lap = ( phi[x-1][y-1]+phi[x+1][y-1]+phi[x-1][y+1]+phi[x+1][y+1]
                +4.0*(phi[x][y-1]+phi[x-1][y]+phi[x+1][y]+phi[x][y+1])
                -20.0*phi[x][y] ) / 6.0;
                mu_phi[x][y] = 4.0*Beta*(phi[x][y]-phiL)*(phi[x][y]-phiH)*(phi[x][y]-phi0) - kappa*lap;
            }
        }
    }
    void stress_from_gneq(const double gneq[9], double& sxx, double& sxy, double& syy){
        sxx=sxy=syy=0.0;
        for(int a=1;a<=8;++a){ sxx += gneq[a]*ex[a]*ex[a]; sxy += gneq[a]*ex[a]*ey[a]; syy += gneq[a]*ey[a]*ey[a]; }
    }
    void viscous_force_paper(double tau_loc, double dpx, double dpy,
                             const double gneq[9], double& Fmx, double& Fmy){
        double sxx,sxy,syy; stress_from_gneq(gneq,sxx,sxy,syy);
        double fac = - (tau_loc) / (tau_loc + 0.5) * (rhoH - rhoL);
        Fmx = fac * (sxx*dpx + sxy*dpy);
        Fmy = fac * (sxy*dpx + syy*dpy);
                             }

                             // -------------------- Initialization --------------------
                             void initialize(){
                                 // phase-field profile (tanh): choose drop vs bubble
                                 /*for(int y=0;y<=Ny+1;++y){
                                     for(int x=0;x<=Nx+1;++x){
                                         double dx = x - (X0-0.5);
                                         double dy = y - (Y0-0.5);
                                         double r  = std::sqrt(dx*dx + dy*dy);
                                         double arg= 2.0*(R - r)/W_;
                                         double phi_drop   = 0.5 + 0.5*std::tanh(arg); // ~1 inside
                                         double phi_bubble = 0.5 - 0.5*std::tanh(arg); // ~0 inside
                                         phi[x][y] = DROP_INSIDE ? phi_drop : phi_bubble;
                                         Ux[x][y]=Uy[x][y]=0.0;
                                         P [x][y]=0.0;
                                     }

                                 } */
                             // phase-field profile: *rectangle* with tanh smoothing (interface thickness W_)


                                for(int y=0;y<=Ny+1;++y){
                                    for(int x=0;x<=Nx+1;++x){
                                        const double xc = (X0 - 0.5);
                                        const double yc = (Y0 - 0.5);

                                        // distances from rectangle center
                                        double dx = std::abs(x - xc) - 0.5 * RECT_WX;
                                        double dy = std::abs(y - yc) - 0.5 * RECT_WY;

                                        // signed distance to axis-aligned rectangle:
                                        // outside: max(dx,dy) > 0 (positive), inside: both <=0 so max<=0 (negative)
                                        double sdist_outside = std::max(dx, dy);

                                        // we want positive "inside" like the circle used (R - r): flip sign
                                        double d_rect = -sdist_outside;

                                        // tanh profile: φ ≈ 1 inside (drop) or 0 inside (bubble), transition ~ W_
                                        double arg = (2.0 * d_rect) / W_;
                                        double phi_drop   = 0.5 + 0.5 * std::tanh(arg); // ~1 inside rectangle
                                        double phi_bubble = 0.5 - 0.5 * std::tanh(arg); // ~0 inside rectangle

                                        phi[x][y] = DROP_INSIDE ? phi_drop : phi_bubble;

                                        Ux[x][y] = 0.0;
                                        Uy[x][y] = 0.0;
                                        P [x][y] = 0.0;
                                    }
                                }


                                 // rho from φ (linear interpolation)
                                 for(int y=1;y<=Ny;++y) for(int x=1;x<=Nx;++x){
                                     rho[x][y] = rhoL + (phi[x][y]-0.0)*(rhoH - rhoL);
                                 }
                                 periodic_fill(phi);

                                 // μ_φ, ∇φ, normals
                                 chemical_potential();
                                 iso_grad_phi();
                                 normals_from_grad();

                                 // dynamic viscosity μ via Eq.(24); local τ via Eq.(25)
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){
                                         MU[x][y]  = muL + (phi[x][y]-0.0) * (muH - muL);
                                         double r  = std::max(rho[x][y], 1e-12);
                                         TAU[x][y] = MU[x][y] / (r * cs2);
                                     }
                                 }

                                 // preload pressure ± phi * Sigma/R / (rho*cs2)
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){
                                         double r = std::max(rho[x][y],1e-12);
                                         double preload = phi[x][y] * Sigma / R / (r*cs2);
                                         if (DROP_INSIDE) P[x][y] += preload; else P[x][y] -= preload;
                                     }
                                 }

                                 // Build initial distributions
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){
                                         // h: heq_bar = φ Γ - 0.5 Fφ
                                         Equilibrium_new(0.0,0.0);
                                         for(int a=0;a<9;++a) Gamma[a] = Ga_Wa[a] + wv[a];

                                         double s = std::sqrt(dphix[x][y]*dphix[x][y] + dphiy[x][y]*dphiy[x][y] + 1e-32);
                                         for(int a=0;a<9;++a){
                                             double proj = (ex[a]*dphix[x][y] + ey[a]*dphiy[x][y]) / s;
                                             double shape= (1.0 - 4.0*std::pow(phi[x][y]-0.5,2.0))/W_;
                                             double Fphi = dt * shape * wv[a] * proj;
                                             hlp[a]      = Fphi;
                                             heq_bar[a]  = phi[x][y]*Gamma[a] - 0.5*Fphi;
                                             h[idx(a,x,y)] = heq_bar[a];
                                         }

                                         // g: geq = P w + (Γ - w); set g = geq
                                         for(int a=0;a<9;++a){
                                             g[idx(a,x,y)] = P[x][y]*wv[a] + (Gamma[a] - wv[a]);
                                         }
                                     }
                                 }
                             }

                             // -------------------- Collision/Streaming --------------------
                             void collide(){
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){

                                         // --- local τ for g (BGK) ---
                                         double tau_g = TAU[x][y];
                                         double inv_tau_star = 1.0 / (tau_g + 0.5); // 1/(τ+1/2) per paper

                                         // --- Equilibrium_new at stored U ---
                                         Equilibrium_new(Ux[x][y], Uy[x][y]);
                                         for(int a=0;a<9;++a) Gamma[a] = Ga_Wa[a] + wv[a];

                                         // === h: BGK with conservative forcing ===
                                         double s = std::sqrt(dphix[x][y]*dphix[x][y] + dphiy[x][y]*dphiy[x][y] + 1e-32);
                                         for(int a=0;a<9;++a){
                                             double proj = (ex[a]*dphix[x][y] + ey[a]*dphiy[x][y]) / s;
                                             double shape= (1.0 - 4.0*std::pow(phi[x][y]-0.5,2.0))/W_;
                                             double Fphi = dt * shape * wv[a] * proj;
                                             hlp[a]      = Fphi;
                                             heq_bar[a]  = phi[x][y]*Gamma[a] - 0.5*Fphi;
                                         }
                                         for(int a=0;a<9;++a){
                                             double hxy = h[idx(a,x,y)];
                                             h[idx(a,x,y)] = (1.0 - w_c)*hxy + w_c*heq_bar[a] + hlp[a];
                                         }

                                         // === g: BGK with Guo forcing, F_p, F_mu and body force F_b ===
                                         // F_p = - p* ( (rhoH-rhoL) * cs2 ) ∇φ  (since ∇ρ = (rhoH-rhoL)∇φ, cs2=1/3)
                                         double dRho3 = (rhoH - rhoL) * cs2;
                                         double FpX = - P[x][y] * dRho3 * dphix[x][y];
                                         double FpY = - P[x][y] * dRho3 * dphiy[x][y];

                                         // geq
                                         for(int a=0;a<9;++a) geq[a] = P[x][y]*wv[a] + (Gamma[a] - wv[a]);

                                         // gneq
                                         double gneq[9]; for(int a=0;a<9;++a) gneq[a] = g[idx(a,x,y)] - geq[a];

                                         // F_mu (paper prefactor uses *local* tau)
                                         double FmX, FmY;
                                         viscous_force_paper(tau_g, dphix[x][y], dphiy[x][y], gneq, FmX, FmY);

                                         // body force (gravity)
                                         double FbX = 0.0;
                                         double FbY = GRAV_Y * rho[x][y];

                                         // total force
                                         double Fx = mu_phi[x][y]*dphix[x][y] + FpX + FmX + FbX;
                                         double Fy = mu_phi[x][y]*dphiy[x][y] + FpY + FmY + FbY;

                                         double rho_safe = std::max(rho[x][y], 1e-12);
                                         for(int a=0;a<9;++a){
                                             double eF  = ex[a]*Fx + ey[a]*Fy;
                                             double guo = dt * wv[a] * (eF) / (rho_safe*cs2); // Guo forcing term
                                             double geq_bar = geq[a] - 0.5*guo;
                                             // BGK with local (τ+1/2)
                                             g[idx(a,x,y)] += ( - (g[idx(a,x,y)] - geq_bar) * inv_tau_star ) + guo;
                                         }
                                     }
                                 }
                             }

                             void stream_and_macro(){
                                 // periodic BC on populations
                                 Boundary_Conditions_f(h);
                                 Boundary_Conditions_f(g);

                                 // streaming
                                 Streaming(h);
                                 Streaming(g);

                                 // φ and ρ from h
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){
                                         double sumh=0.0;
                                         for(int a=0;a<9;++a) sumh += h[idx(a,x,y)];
                                         phi[x][y] = sumh;
                                         rho[x][y] = rhoL + (phi[x][y]-0.0)*(rhoH - rhoL);
                                     }
                                 }
                                 periodic_fill(phi);

                                 chemical_potential();
                                 iso_grad_phi();
                                 normals_from_grad();

                                 // update MU, TAU (local τ!)
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){
                                         MU[x][y]  = muL + (phi[x][y]-0.0) * (muH - muL);
                                         double r  = std::max(rho[x][y], 1e-12);
                                         TAU[x][y] = MU[x][y] / (r * cs2);
                                     }
                                 }

                                 // macroscopic P, U (Guo half-step uses total force incl. gravity)
                                 for(int y=1;y<=Ny;++y){
                                     for(int x=1;x<=Nx;++x){
                                         double sumg=0.0; for(int a=0;a<9;++a) sumg += g[idx(a,x,y)];
                                         P[x][y] = sumg;

                                         // Rebuild forces for half-step velocity
                                         double tau_g = TAU[x][y];
                                         double dRho3 = (rhoH - rhoL) * cs2;
                                         double FpX = - P[x][y] * dRho3 * dphix[x][y];
                                         double FpY = - P[x][y] * dRho3 * dphiy[x][y];

                                         Equilibrium_new(Ux[x][y], Uy[x][y]);
                                         for(int a=0;a<9;++a) geq[a] = P[x][y]*wv[a] + (Ga_Wa[a] + wv[a] - wv[a]); // = P w + Ga_Wa

                                         double gneq[9]; for(int a=0;a<9;++a) gneq[a] = g[idx(a,x,y)] - geq[a];
                                         double FmX,FmY; viscous_force_paper(tau_g, dphix[x][y], dphiy[x][y], gneq, FmX, FmY);

                                         // body force again
                                         double FbX = 0.0;
                                         double FbY = GRAV_Y * rho[x][y];

                                         double Fx = mu_phi[x][y]*dphix[x][y] + FpX + FmX + FbX;
                                         double Fy = mu_phi[x][y]*dphiy[x][y] + FpY + FmY + FbY;

                                         double gx = g[idx(1,x,y)] - g[idx(3,x,y)] + g[idx(5,x,y)] - g[idx(6,x,y)] - g[idx(7,x,y)] + g[idx(8,x,y)];
                                         double gy = g[idx(2,x,y)] - g[idx(4,x,y)] + g[idx(5,x,y)] + g[idx(6,x,y)] - g[idx(7,x,y)] - g[idx(8,x,y)];
                                         double r  = std::max(rho[x][y],1e-12);
                                         Ux[x][y] = gx + 0.5*Fx / r;
                                         Uy[x][y] = gy + 0.5*Fy / r;
                                     }
                                 }
                             }

                             // -------------------- VTK output ------------------------
                             void writeVTK(int t){
                                 const double sp = 1.0/double(L0);
                                 std::ostringstream ss; ss<<"sol_"<<std::setw(7)<<std::setfill('0')<<t<<".vtk";
                                 std::ofstream os(ss.str());
                                 if(!os){ std::cerr<<"Cannot open "<<ss.str()<<"\n"; return; }
                                 os << "# vtk DataFile Version 2.0\n";
                                 os << "Conservative PF LBM (paper + gravity) t="<<t<<"\nASCII\n\n";
                                 os << "DATASET STRUCTURED_POINTS\n";
                                 os << "DIMENSIONS "<<Nx<<" "<<Ny<<" 1\n";
                                 os << "ORIGIN 0 0 0\n";
                                 os << "SPACING "<<sp<<" "<<sp<<" "<<sp<<"\n\n";
                                 os << "POINT_DATA "<<Nx*Ny<<"\n";

                                 os << "SCALARS phi float 1\nLOOKUP_TABLE default\n";
                                 for(int y=1; y<=Ny; ++y){ for(int x=1; x<=Nx; ++x) os<<(float)phi[x][y]<<" "; os<<"\n"; } os<<"\n";

                                 os << "SCALARS density float 1\nLOOKUP_TABLE default\n";
                                 for(int y=1; y<=Ny; ++y){ for(int x=1; x<=Nx; ++x) os<<(float)rho[x][y]<<" "; os<<"\n"; } os<<"\n";

                                 os << "SCALARS pressure float 1\nLOOKUP_TABLE default\n";
                                 for(int y=1; y<=Ny; ++y){ for(int x=1; x<=Nx; ++x) os<<(float)P[x][y]<<" "; os<<"\n"; } os<<"\n";

                                 os << "SCALARS pressure_times_rho_cs2 float 1\nLOOKUP_TABLE default\n";
                                 for(int y=1; y<=Ny; ++y){ for(int x=1; x<=Nx; ++x) os<<(float)(P[x][y]*rho[x][y]*cs2)<<" "; os<<"\n"; } os<<"\n";

                                 os << "VECTORS velocity float\n";
                                 for(int y=1; y<=Ny; ++y){ for(int x=1; x<=Nx; ++x) os<<(float)Ux[x][y]<<" "<<(float)Uy[x][y]<<" 0\n"; } os<<"\n";
                             }

                             // -------------------- Main ------------------------------
                             int main(){
                                 if(!(rhoH>rhoL)){ std::cerr<<"ERROR: require rhoH>rhoL (heavy minus light)\n"; return 1; }
                                 std::cout<<(DROP_INSIDE? "Case: HEAVY DROP inside LIGHT ambient\n"
                                 : "Case: LIGHT BUBBLE inside HEAVY ambient\n");
                                 std::cout<<"rhoL="<<rhoL<<", rhoH="<<rhoH
                                 <<", muL="<<muL<<", muH="<<muH
                                 <<", Sigma="<<Sigma<<", W="<<W_
                                 <<", g_y="<<GRAV_Y<<"\n";

                                 // init
                                 initialize();

                                 std::cout << "\n   tf    Sigma     W      M      R\n";
                                 std::cout << std::setw(6)<<tf
                                 << std::fixed<<std::setprecision(4)<<std::setw(8)<<Sigma
                                 << std::setprecision(1)<<std::setw(7)<<W_
                                 << std::setprecision(3)<<std::setw(8)<<Mmob
                                 << std::setw(8)<<R << "\n\n";

                                 std::cout<<std::left<<std::setw(6)<<"t"
                                 <<std::right<<std::setw(12)<<"phi_min"
                                 <<std::setw(12)<<"phi_max"
                                 <<std::setw(12)<<"Ux_max"
                                 <<std::setw(12)<<"Uy_max"
                                 <<std::setw(12)<<"|U|_max"
                                 <<std::setw(12)<<"Mass_phi"<<"\n";

                                 const int vtk_every = step;

                                 for(int t=0;t<=tf;++t){
                                     if(t%step==0){
                                         // diagnostics
                                         double pmin=1e9,pmax=-1e9,uxm=0.0,uym=0.0,um=0.0,mass=0.0;
                                         for(int y=1;y<=Ny;++y) for(int x=1;x<=Nx;++x){
                                             pmin = std::min(pmin, phi[x][y]); pmax = std::max(pmax, phi[x][y]);
                                             uxm  = std::max(uxm, std::abs(Ux[x][y]));
                                             uym  = std::max(uym, std::abs(Uy[x][y]));
                                             um   = std::max(um, std::hypot(Ux[x][y],Uy[x][y]));
                                             mass+= phi[x][y];
                                         }
                                         std::cout<<std::setw(7)<<t
                                         <<std::fixed<<std::setprecision(6)
                                         <<std::setw(12)<<pmin<<std::setw(12)<<pmax
                                         <<std::scientific<<std::setprecision(3)
                                         <<std::setw(12)<<uxm<<std::setw(12)<<uym<<std::setw(12)<<um
                                         <<std::fixed<<std::setprecision(2)
                                         <<std::setw(12)<<mass<<"\n";
                                         writeVTK(t);
                                     }

                                     // One LB step:
                                     collide();
                                     stream_and_macro();
                                 }

                                 std::cout<<"\nDone.\n";
                                 return 0;
                             }
