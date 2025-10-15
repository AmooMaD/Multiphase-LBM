// -------------- BEGIN PulsatileBloodFlow2D.h (complete) --------------
#pragma once
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
#include <limits>

using namespace std;
using namespace std::chrono;

// ─────────────────────────   cell-level helpers   ───────────────────────────
enum class CellType_PulsatileBloodFlow2D : uint8_t { bounce_back, bulk };
using  CellData = double;

// ─────────────────────────   D2Q9 constants (your signature)   ─────────────
inline auto d2q9_constants_PulsatileBloodFlow2D()
{
    vector<array<int,2>> c_vect = {
        {-1,  0}, { 0, -1}, {-1, -1}, {-1,  1}, { 0,  0},
        { 1,  0}, { 0,  1}, { 1,  1}, { 1, -1}
    };
    vector<int>    opp_vect = { 5, 6, 7, 8, 4, 0, 1, 2, 3 };
    vector<double> t_vect   = { 1./9., 1./9., 1./36., 1./36., 4./9., 1./9., 1./9., 1./36., 1./36. };
    return make_tuple(c_vect, opp_vect, t_vect);
}

// Abbas directions (I indexing): 0=rest, 1=E,2=N,3=W,4=S,5=NE,6=NW,7=SW,8=SE
static constexpr int exI[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
static constexpr int eyI[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
static constexpr int jB_I[9] = {0,3,4,1,2,7,8,5,6}; // opposite in I-space

// map from Abbas I to your k indexing
static constexpr int kFromI_table[9] = {
/*I=0..8*/ 4, /*E*/5, /*N*/6, /*W*/0, /*S*/1, /*NE*/7, /*NW*/3, /*SW*/2, /*SE*/8
};

// ─────────────────────────   geometry helpers   ────────────────────────────
struct Dim_PulsatileBloodFlow2D {
    Dim_PulsatileBloodFlow2D(int nx_, int ny_)
        : nx(nx_), ny(ny_),
          nelem(static_cast<size_t>(nx_) * static_cast<size_t>(ny)),
          npop (9 * nelem) {}
    int    nx, ny;
    size_t nelem, npop;
};

struct LBM_PulsatileBloodFlow2D; // fwd
void saveVtkFields_PulsatileBloodFlow2D(LBM_PulsatileBloodFlow2D& lbm, int time_iter, double dx = 0.);

// ─────────────────────────   LBM functor   ─────────────────────────────────
struct LBM_PulsatileBloodFlow2D
{
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem; }

    // raw buffers
    CellData*                      lattice;     // holds g-in/out
    CellType_PulsatileBloodFlow2D* flag;        // bulk / bounceback (walls)
    int*                           parity;      // in/out selector
    std::array<int,2>*             c;           // your c[k]
    int*                           opp;         // your opp[k]
    double*                        t;           // your t[k]
    Dim_PulsatileBloodFlow2D       dim;

    // accessors
    inline double& g   (int i,int k){ return lattice[                        k*dim.nelem + i]; }
    inline double& gin (int i,int k){ return lattice[*parity  *dim.npop   +  k*dim.nelem + i]; }
    inline double& gout(int i,int k){ return lattice[(1-*parity)*dim.npop   + k*dim.nelem + i]; }

    // index helpers
    inline size_t xyz_to_i (int x, int y) const { return (y + dim.ny * x); }
    inline auto   i_to_xyz (int i) const {
        int iX = i / (dim.ny);
        int iY = i % (dim.ny);
        return std::make_tuple(iX, iY);
    }
    inline int wrapX(int x) const { return (x + dim.nx) % dim.nx; }
    inline int wrapY(int y) const { return y; }

    // direction mapping
    inline int kFromI(int I) const { return kFromI_table[I]; }
    inline int kOppFromI(int I) const { return kFromI_table[jB_I[I]]; }

    // ------------------- physical & model parameters ------------------------
    const int    hsize = 1;
    const double Rho0  = 1.0 / pow(hsize,3);
    double tau   = 0.75;
    double s8    = 1.0 / tau;
    double s5    = 1.0;
    double S[9]  = {1,1,1,1, s5, 1, s5, s8, s8};

    bool   deformable = true;
    bool   is_severed = true;
    double alpha      = 0.01;

    double p0_in=0.0, p0_out=0.0, p_tissue=0.0, p_oscillatory=0.0, Delta_p=0.0;
    double omega=0.0;
    int    t_beat=0, t_propagation=0, t_start=0, t_sever=0;

    // diag
    double Umax=0.0, Re=0.0, Wo=0.0;

    // macroscopic fields
    vector<double> P, Ux, Uy;

    // moving-wall geometry
    vector<double> yr1, yr2, y1new, y2new, Vw1, Vw2;

    // Fobj with halo [0..nx+1]×[0..ny+1]
    vector<double> Fobj;
    inline double& F(int Xp, int Yp) { return Fobj[Xp*(dim.ny+2) + Yp]; }

    // border nodes
    struct Border_Node { int X, Y; double Delta[8]; };
    vector<Border_Node> Borders1, Borders2;
    int Nb1=0, Nb2=0;

    // bookkeeping like the original
    int FreshNodes=0, KilledNodes=0;

    inline int X0() const { return (dim.nx-1)/2; }
    inline int Y0() const { return (dim.ny-1)/2; }

    LBM_PulsatileBloodFlow2D(CellData* lat_,
                             CellType_PulsatileBloodFlow2D* flag_,
                             int* parity_,
                             std::array<int,2>* c_,
                             int* opp_,
                             double* t_,
                             Dim_PulsatileBloodFlow2D dim_)
      : lattice(lat_), flag(flag_), parity(parity_),
        c(c_), opp(opp_), t(t_), dim(dim_),
        P(dim.nelem,0.0), Ux(dim.nelem,0.0), Uy(dim.nelem,0.0),
        yr1(dim.nx,0.0), yr2(dim.nx,0.0), y1new(dim.nx,0.0), y2new(dim.nx,0.0),
        Vw1(dim.nx,0.0), Vw2(dim.nx,0.0),
        Fobj( (dim.nx+2)*(dim.ny+2), 1.0 )
    {}

    // ========================= INITIALIZATION ===============================
    void Setup_Simulation_Parameters() {
        if (t_beat <= 0) t_beat = max(1, dim.nx);
        omega = 2.0*3.141592653589793 / double(t_beat);

        if (p0_in == 0.0 && p0_out == 0.0) { p0_in=0.20; p0_out=0.19; }
        if (is_severed) { p0_in=0.02; p0_out=0.00; }
        p_tissue = p0_in;

        p_oscillatory = (p0_in - p0_out);
        if (is_severed) p_oscillatory *= 0.1;

        Delta_p = p0_out - p0_in;
        t_propagation = int((dim.nx - 1.)*sqrt(3.) - 1)*hsize;
        t_start = 2*t_propagation;
        t_sever = 0;

        double mu = Rho0*(tau-0.5)/3.0;
        Umax = -Delta_p / (dim.nx - 1) * pow(dim.ny - 2, 2.0) / (8.0 * mu);
        Re   = Rho0*Umax/mu * 0.5*(dim.ny-2);
        Wo   = 0.5*(dim.ny-2)*sqrt(omega*Rho0/mu);
    }

    void Initialize_Yr_and_Vw_and_p() { Get_Theoretical_Wall_Location_and_pressure(); y1new=yr1; y2new=yr2; fill(Vw1.begin(),Vw1.end(),0.0); fill(Vw2.begin(),Vw2.end(),0.0); }

    void Get_Theoretical_Wall_Location_and_pressure() {
        double yr1_in  = (Y0()+0.5) - (p0_in  - p_tissue)/alpha;
        double yr2_in  = (Y0()+0.5) + (p0_in  - p_tissue)/alpha;
        double yr1_out = (Y0()+0.5) - (p0_out - p_tissue)/alpha;
        double yr2_out = (Y0()+0.5) + (p0_out - p_tissue)/alpha;
        if (yr1_in<1 || yr2_in>dim.ny-2 || yr1_out<1 || yr2_out>dim.ny-2) throw runtime_error("Initial wall location out of bounds.");
        double R0=(yr2_in-yr1_in)/2.0, RL=(yr2_out-yr1_out)/2.0;
        for (int X=0; X<dim.nx; ++X) {
            double Rx4=(pow(RL,4)-pow(R0,4))*(double(X)/double(dim.nx-1))+pow(R0,4);
            double Rx=pow(Rx4,0.25);
            yr1[X]=(Y0()+0.5)-Rx; yr2[X]=(Y0()+0.5)+Rx;
            for (int Y=0; Y<dim.ny; ++Y) P[xyz_to_i(X,Y)] = (yr2[X] - (dim.ny-1 - 0.5))*alpha + p_tissue;
        }
    }

    void Initialize_P_U_g() {
        fill(Ux.begin(),Ux.end(),0.0);
        fill(Uy.begin(),Uy.end(),0.0);
        // seed u from dP/dx
        for (int X=0; X<dim.nx; ++X) {
            for (int Y=int(ceil(yr1[X]-0.01)); Y<=int(floor(yr2[X]+0.01)); ++Y) {
                int i = xyz_to_i(X,Y);
                double dpx=0.0;
                if (X==0) dpx=P[xyz_to_i(1,Y)]-P[i];
                else if (X==dim.nx-1) dpx=P[i]-P[xyz_to_i(X-1,Y)];
                else dpx=0.5*(P[xyz_to_i(X+1,Y)]-P[xyz_to_i(X-1,Y)]);
                double mu = Rho0*(tau-0.5)/3.0;
                Ux[i] = dpx/(2.0*mu)*( (Y-yr1[X])*(Y-yr2[X]) );
            }
        }
        // fill gin with equilibrium
        for (int X=0; X<dim.nx; ++X) for (int Y=0; Y<dim.ny; ++Y) {
            int i=xyz_to_i(X,Y);
            if (isSolidNode(X,Y)) { for (int k=0;k<9;++k) gin(i,k)=0.0; continue; }
            double geq[9]; Equilibrium_g(P[i],Ux[i],Uy[i],geq);
            for (int k=0;k<9;++k) gin(i,k)=geq[k];
        }
    }

    // compute macroscopic fields from PDFs (skip solids)
    void Macroscopic_Properties_g()
    {
        for (int X = 0; X < dim.nx; ++X) {
            for (int Y = 0; Y < dim.ny; ++Y) {
                int i = xyz_to_i(X,Y);
                if (isSolidNode(X,Y)) { P[i]=Ux[i]=Uy[i]=0.0; continue; }
                double p=0.0, ux=0.0, uy=0.0;
                for (int k=0;k<9;++k) p  += gin(i,k);
                for (int k=1;k<9;++k) { ux += gin(i,k) * c[k][0]; uy += gin(i,k) * c[k][1]; }
                P[i]  = p;
                Ux[i] = 3.0 * ux / Rho0;
                Uy[i] = 3.0 * uy / Rho0;
            }
        }
    }

    // move both walls based on pressure and rebuild geometry
    void Calculate_Pressure_and_Move_Walls(int /*t_iter*/)
    {
        Calculate_Pressure_and_Move_Wall_1_Bottom();
        Calculate_Pressure_and_Move_Wall_2_Top();

        Update_Fobj_for_Vessel_Walls();
        Find_or_Update_Boundary_Nodes();
    }

    // bottom wall update (yr1) with tighter clamp (≤0.25 cell/step)
    void Calculate_Pressure_and_Move_Wall_1_Bottom()
    {
        const double Yw1_const = 0.0;
        for (int Xf = 0; Xf < dim.nx; ++Xf) {
            double Ps = P[ xyz_to_i(Xf, Y0()) ] - p_tissue;
            double target = (Yw1_const + 0.5) - Ps / alpha;
            double d = target - yr1[Xf];
            double cap = 0.25;
            if (d >  cap) d =  cap;
            if (d < -cap) d = -cap;
            y1new[Xf] = yr1[Xf] + d;
        }
        for (int X = 0; X < dim.nx; ++X) { Vw1[X] = y1new[X] - yr1[X]; yr1[X] = y1new[X]; }
    }

    // top wall update (yr2) with tighter clamp
    void Calculate_Pressure_and_Move_Wall_2_Top()
    {
        const double Yw2_const = double(dim.ny - 1);
        for (int Xf = 0; Xf < dim.nx; ++Xf) {
            double Ps = P[ xyz_to_i(Xf, Y0()+1) ] - p_tissue;
            double target = (Yw2_const - 0.5) + Ps / alpha;
            double d = target - yr2[Xf];
            double cap = 0.25;
            if (d >  cap) d =  cap;
            if (d < -cap) d = -cap;
            y2new[Xf] = yr2[Xf] + d;
        }
        for (int X = 0; X < dim.nx; ++X) { Vw2[X] = y2new[X] - yr2[X]; yr2[X] = y2new[X]; }
    }

    // ====================== CURVED-WALL GEOMETRY ============================
    void Initialize_Fobj_for_Vessel_Walls() {
        int NX=dim.nx, NY=dim.ny;
        for (int X=0; X<NX; ++X) {
            for (int Y=-1; Y<=Y0(); ++Y)     F(X+1,Y+1) = (yr1[X] - (Y0()+0.5)) / (Y - (Y0()+0.5));
            for (int Y= Y0()+1; Y<NY+1; ++Y) F(X+1,Y+1) = (yr2[X] - (Y0()+0.5)) / (Y - (Y0()+0.5));
        }
        for (int Y=0; Y<NY+2; ++Y) { F(0,Y)=2.0*F(1,Y)-F(2,Y); F(NX+1,Y)=2.0*F(NX,Y)-F(NX-1,Y); }
        for (int X=0; X<NX; ++X) for (int Y=0; Y<NY; ++Y)
            flag[xyz_to_i(X,Y)] = (F(X+1,Y+1) < 1.0 ? CellType_PulsatileBloodFlow2D::bounce_back
                                                    : CellType_PulsatileBloodFlow2D::bulk);
    }

    inline bool isSolidNode(int X, int Y) const { return flag[xyz_to_i(X,Y)] == CellType_PulsatileBloodFlow2D::bounce_back; }
    static inline void Find_Delta(int mA, double mB, double Y1, double &Delta) {
        Delta = 1.0 - std::abs(Y1 / (mA - mB)); if (Delta < 0) Delta = 0;
    }

    void Find_or_Update_Boundary_Nodes() { Update_Boundary_Nodes_Bottom(); Update_Boundary_Nodes_Top(); }

    void Update_Boundary_Nodes_Bottom() {
        Borders1.clear(); Nb1=0;
        int X=0, Y=int(floor(yr1[X]));
        if (F(X+1,Y+1) >= 1) Y = Y - 1;
        auto pushNode=[&](int Xn,int Yn,double D[8]){
            Border_Node bn; bn.X=Xn; bn.Y=Yn; for(int i=0;i<8;++i) bn.Delta[i]=D[i]; Borders1.push_back(bn); ++Nb1;
        };
        { double D[8]; for (int i=0;i<8;++i) D[i]=2;
          if (F(X+2,Y+1) >= 1) Find_Delta(0, yr1[X+1]-yr1[X], yr1[X]-Y, D[0]);
          D[1] = 1 - (yr1[X] - Y);
          if (F(X+2,Y+2) >= 1) Find_Delta(1, yr1[X+1]-yr1[X], yr1[X]-Y, D[4]);
          pushNode(X,Y,D);
        }
        for (X=1; X<dim.nx-1; ++X) {
            int Yx=int(floor(yr1[X])); if (F(X+1,Yx+1) >= 1) Yx=Yx-1;
            if (Yx != Y) {
                double D[8]; for (int i=0;i<8;++i) D[i]=2;
                if (Yx > Y) { Find_Delta(-1, yr1[X]-yr1[X-1], yr1[X]-Y, D[5]); pushNode(X,Y,D); }
                else        { Find_Delta( 1, yr1[X]-yr1[X-1], yr1[X-1]-Yx, D[4]); pushNode(X-1,Yx,D); }
            }
            { double D[8]; for (int i=0;i<8;++i) D[i]=2;
              if (F(X+2,Yx+1) >= 1) Find_Delta(0, yr1[X+1]-yr1[X], yr1[X]-Yx, D[0]);
              D[1] = 1 - (yr1[X] - Yx);
              if (F(X,  Yx+1) >= 1) Find_Delta(0, yr1[X]-yr1[X-1], yr1[X]-Yx, D[2]);
              if (F(X+2,Yx+2) >= 1) Find_Delta(1, yr1[X+1]-yr1[X], yr1[X]-Yx, D[4]);
              if (F(X,  Yx+2) >= 1) Find_Delta(-1,yr1[X]-yr1[X-1], yr1[X]-Yx, D[5]);
              pushNode(X,Yx,D);
            }
            Y = Yx;
        }
        X = dim.nx-1;
        int Yx=int(floor(yr1[X])); if (F(X+1,Yx+1) >= 1) Yx=Yx-1;
        if (Yx != Y) {
            double D[8]; for (int i=0;i<8;++i) D[i]=2;
            if (Yx > Y) { Find_Delta(-1, yr1[X]-yr1[X-1], yr1[X]-Y, D[5]); pushNode(X,Y,D); }
            else        { Find_Delta( 1, yr1[X]-yr1[X-1], yr1[X-1]-Yx, D[4]); pushNode(X-1,Yx,D); }
        }
        { double D[8]; for (int i=0;i<8;++i) D[i]=2;
          D[1] = 1 - (yr1[X] - Yx);
          if (F(X,  Yx+1) >= 1) Find_Delta(0, yr1[X]-yr1[X-1], yr1[X]-Yx, D[2]);
          if (F(X,  Yx+2) >= 1) Find_Delta(-1,yr1[X]-yr1[X-1], yr1[X]-Yx, D[5]);
          pushNode(X,Yx,D);
        }
    }

    void Update_Boundary_Nodes_Top() {
        Borders2.clear(); Nb2=0;
        int X=0, Y=int(ceil(yr2[X])); if (F(X+1,Y+1) >= 1) Y=Y+1;
        auto pushNode=[&](int Xn,int Yn,double D[8]){
            Border_Node bn; bn.X=Xn; bn.Y=Yn; for(int i=0;i<8;++i) bn.Delta[i]=D[i]; Borders2.push_back(bn); ++Nb2;
        };
        { double D[8]; for (int i=0;i<8;++i) D[i]=2;
          if (F(X+2,Y+1) >= 1) Find_Delta(0, yr2[X+1]-yr2[X], yr2[X]-Y, D[0]);
          D[3] = 1 - (Y - yr2[X]);
          if (F(X+2,Y  ) >= 1) Find_Delta(-1,yr2[X+1]-yr2[X], yr2[X]-Y, D[7]);
          pushNode(X,Y,D);
        }
        int Yprev=Y;
        for (X=1; X<dim.nx-1; ++X) {
            int Yx=int(ceil(yr2[X])); if (F(X+1,Yx+1) >= 1) Yx=Yx+1;
            if (Yx != Yprev) {
                double D[8]; for (int i=0;i<8;++i) D[i]=2;
                if (Yx > Yprev) { Find_Delta(-1,yr2[X]-yr2[X-1], yr2[X-1]-Yx, D[7]); pushNode(X-1,Yx,D); }
                else            { Find_Delta( 1,yr2[X]-yr2[X-1], yr2[X]-Yprev, D[6]); pushNode(X,Yprev,D); }
            }
            { double D[8]; for (int i=0;i<8;++i) D[i]=2;
              if (F(X+2,Yx+1) >= 1) Find_Delta(0, yr2[X+1]-yr2[X], yr2[X]-Yx, D[0]);
              if (F(X,  Yx+1) >= 1) Find_Delta(0, yr2[X]-yr2[X-1], yr2[X]-Yx, D[2]);
              D[3] = 1 - (Yx - yr2[X]);
              if (F(X,  Yx  ) >= 1) Find_Delta(1, yr2[X]-yr2[X-1], yr2[X]-Yx, D[6]);
              if (F(X+2,Yx  ) >= 1) Find_Delta(-1,yr2[X+1]-yr2[X], yr2[X]-Yx, D[7]);
              pushNode(X,Yx,D);
            }
            Yprev=Yx;
        }
        X = dim.nx-1;
        int Yx=int(ceil(yr2[X])); if (F(X+1,Yx+1) >= 1) Yx=Yx+1;
        if (Yx != Yprev) {
            double D[8]; for (int i=0;i<8;++i) D[i]=2;
            if (Yx > Yprev) { Find_Delta(-1,yr2[X]-yr2[X-1], yr2[X-1]-Yx, D[7]); pushNode(X-1,Yx,D); }
            else            { Find_Delta( 1,yr2[X]-yr2[X-1], yr2[X]-Yprev, D[6]); pushNode(X,Yprev,D); }
        }
        { double D[8]; for (int i=0;i<8;++i) D[i]=2;
          if (F(X,  Yx+1) >= 1) Find_Delta(0, yr2[X]-yr2[X-1], yr2[X]-Yx, D[2]);
          D[3] = 1 - (Yx - yr2[X]);
          if (F(X,  Yx  ) >= 1) Find_Delta(1,  yr2[X]-yr2[X-1], yr2[X]-Yx, D[6]);
          pushNode(X,Yx,D);
        }
    }

    void Update_Fobj_for_Vessel_Walls() {
        int NX=dim.nx, NY=dim.ny;
        vector<double> Fold=Fobj;
        Initialize_Fobj_for_Vessel_Walls();
        int c1=0,c2=0;
        for (int X=1; X<=NX; ++X) for (int Y=1; Y<=NY; ++Y) {
            if (Fold[X*(NY+2)+Y] < 1 && F(X,Y) >= 1) {
                ++c1;
                int Ffrac[3][3]; for (int i=-1;i<=1;++i) for(int j=-1;j<=1;++j)
                    Ffrac[i+1][j+1] = int( Fold[(X+i)*(NY+2)+(Y+j)] );
                Fill_Fluid_Node(X-1,Y-1,Ffrac);
            }
            if (Fold[X*(NY+2)+Y] >= 1 && F(X,Y) < 1) ++c2;
        }
        FreshNodes = c1; KilledNodes=c2;
    }

    void Fill_Fluid_Node(int X, int Y, int Ffrac[][3]) {
        if (X==0)             Fill_Inlet_node(X,Y);
        else if (X==dim.nx-1) Fill_Outlet_node(X,Y);
        else                  Fill_Fluid_Node_g_new(X,Y,Ffrac);
        Fresh_Macroscopic_Values(X,Y);
    }

    void Fill_Inlet_node(int X, int Y) {
        auto cpRow=[&](int Ys, int Yd){
            int is=xyz_to_i(X,Ys), id=xyz_to_i(X,Yd);
            for (int I=0; I<9; ++I) gin(id, kFromI(I)) = gin(is, kFromI(I));
        };
        if (Y < Y0()) cpRow(Y+1, Y); else cpRow(Y-1, Y);
    }

    void Fill_Outlet_node(int X, int Y) {
        auto cpRow=[&](int Ys, int Yd){
            int is=xyz_to_i(X,Ys), id=xyz_to_i(X,Yd);
            for (int I=0; I<9; ++I) gin(id, kFromI(I)) = gin(is, kFromI(I));
        };
        if (Y < Y0()) cpRow(Y+1, Y); else cpRow(Y-1, Y);
    }

    // ---- robust fresh-node seeding helper (fixes SumFrac==0) ----
    void seed_from_nearest_fluid(int X, int Y)
    {
        static const int dx[8] = { 1,-1, 0, 0, 1, 1,-1,-1 };
        static const int dy[8] = { 0, 0, 1,-1, 1,-1, 1,-1 };

        int i_dst = xyz_to_i(X,Y);
        bool any = false; int cnt = 0; double acc[9] = {0};

        // first ring
        for (int n=0;n<8;++n) {
            int Xn = X + dx[n], Yn = Y + dy[n];
            if (Xn<0 || Xn>=dim.nx || Yn<0 || Yn>=dim.ny) continue;
            if (isSolidNode(Xn,Yn)) continue;
            int i_src = xyz_to_i(Xn,Yn);
            for (int k=0;k<9;++k) acc[k] += gin(i_src,k);
            any = true; ++cnt;
        }
        // outward shells
        for (int R=2; !any && R<=4; ++R) {
            for (int sx=-R; sx<=R; ++sx) {
                int sy_top =  R - std::abs(sx);
                int sy_bot = -sy_top;
                for (int sy : {sy_top, sy_bot}) {
                    int Xn = X + sx, Yn = Y + sy;
                    if (Xn<0 || Xn>=dim.nx || Yn<0 || Yn>=dim.ny) continue;
                    if (isSolidNode(Xn,Yn)) continue;
                    int i_src = xyz_to_i(Xn,Yn);
                    for (int k=0;k<9;++k) acc[k] += gin(i_src,k);
                    any = true; ++cnt;
                }
            }
        }
        if (any && cnt>0) {
            for (int k=0;k<9;++k) gin(i_dst,k) = acc[k] / double(cnt);
        } else {
            double P0 = P[i_dst]; double geqv[9]; Equilibrium_g(P0, 0.0, 0.0, geqv);
            for (int k=0;k<9;++k) gin(i_dst,k) = geqv[k];
        }
    }

    void Fill_Fluid_Node_g_new(int X, int Y, int Ffrac[][3])
    {
        int SumFrac = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                SumFrac += Ffrac[i][j];

        int id = xyz_to_i(X,Y);

        if (SumFrac == 0) { seed_from_nearest_fluid(X, Y); return; }

        for (int I = 0; I < 9; ++I) {
            if (Ffrac[1 - exI[I]][1 - eyI[I]] != 1) {
                double acc = 0.0;
                acc += gin(xyz_to_i(X-1,Y-1), kFromI(I)) * Ffrac[0][0];
                acc += gin(xyz_to_i(X  ,Y-1), kFromI(I)) * Ffrac[1][0];
                acc += gin(xyz_to_i(X+1,Y-1), kFromI(I)) * Ffrac[2][0];
                acc += gin(xyz_to_i(X-1,Y  ), kFromI(I)) * Ffrac[0][1];
                acc += gin(xyz_to_i(X+1,Y  ), kFromI(I)) * Ffrac[2][1];
                acc += gin(xyz_to_i(X-1,Y+1), kFromI(I)) * Ffrac[0][2];
                acc += gin(xyz_to_i(X  ,Y+1), kFromI(I)) * Ffrac[1][2];
                acc += gin(xyz_to_i(X+1,Y+1), kFromI(I)) * Ffrac[2][2];
                gin(id, kFromI(I)) = acc / double(SumFrac);
            }
        }
    }

    void Fresh_Macroscopic_Values(int X, int Y) {
        int i=xyz_to_i(X,Y);
        double p=0, ux=0, uy=0;
        for (int I=0; I<9; ++I) p+=gin(i,kFromI(I));
        for (int I=1; I<9; ++I){ ux += gin(i,kFromI(I))*exI[I]; uy += gin(i,kFromI(I))*eyI[I]; }
        P[i]=p; Ux[i]=3*ux/Rho0; Uy[i]=3*uy/Rho0;
    }

    // ======================= COLLISION / STREAM / BC ========================
    void Equilibrium_g(double P_, double U, double V, double geq[9]) const {
        double U2=U*U+V*V;
        for (int k=0;k<9;++k){
            double eU = c[k][0]*U + c[k][1]*V;
            geq[k] = t[k]*( P_ + Rho0/3.0*( eU*(3.0 + 4.5*eU) - 1.5*U2 ) );
        }
    }

    static void CONVERT(const double IN[9], double OUT[9]) {
        OUT[0]= IN[0]+IN[1]+IN[2]+IN[3]+IN[4]+IN[5]+IN[6]+IN[7]+IN[8];
        OUT[1]=-IN[1]-IN[2]-IN[3]-IN[4]+2*(IN[5]+IN[6]+IN[7]+IN[8]) - 4*IN[0];
        OUT[2]= (IN[5]+IN[6]+IN[7]+IN[8]) - 2*(IN[1]+IN[2]+IN[3]+IN[4]) + 4*IN[0];
        OUT[3]= IN[1]-IN[3]+IN[5]-IN[6]-IN[7]+IN[8];
        OUT[4]= IN[5]-IN[6]-IN[7]+IN[8] - 2*(IN[1]-IN[3]);
        OUT[5]= IN[2]-IN[4]+IN[5]+IN[6]-IN[7]-IN[8];
        OUT[6]= IN[5]+IN[6]-IN[7]-IN[8] - 2*(IN[2]-IN[4]);
        OUT[7]= IN[1]-IN[2]+IN[3]-IN[4];
        OUT[8]= IN[5]-IN[6]+IN[7]-IN[8];
    }
    static void RECONVERT(const double IN[9], double OUT[9]) {
        double C0=IN[0]/9.0, C7=IN[7]/4.0, C8=IN[8]/4.0;
        OUT[0]= C0 - (IN[1]-IN[2])/9.0;
        OUT[1]= C0 - (IN[1]+2*IN[2])/36.0 + (IN[3]-IN[4])/6.0 + C7;
        OUT[2]= C0 - (IN[1]+2*IN[2])/36.0 + (IN[5]-IN[6])/6.0 - C7;
        OUT[3]= C0 - (IN[1]+2*IN[2])/36.0 - (IN[3]-IN[4])/6.0 + C7;
        OUT[4]= C0 - (IN[1]+2*IN[2])/36.0 - (IN[5]-IN[6])/6.0 - C7;
        OUT[5]= C0 + (IN[2]+2*IN[1])/36.0 + (IN[3]+IN[5])/6.0 + (IN[4]+IN[6])/12.0 + C8;
        OUT[6]= C0 + (IN[2]+2*IN[1])/36.0 - (IN[3]-IN[5])/6.0 - (IN[4]-IN[6])/12.0 - C8;
        OUT[7]= C0 + (IN[2]+2*IN[1])/36.0 - (IN[3]+IN[5])/6.0 - (IN[4]+IN[6])/12.0 + C8;
        OUT[8]= C0 + (IN[2]+2*IN[1])/36.0 + (IN[3]-IN[5])/6.0 + (IN[4]-IN[6])/12.0 - C8;
    }

    void MRT_Collision(int X, int Y) {
        int i=xyz_to_i(X,Y); if (isSolidNode(X,Y)) return;
        double geqv[9]; Equilibrium_g(P[i],Ux[i],Uy[i],geqv);
        double tmp[9]; for (int k=0;k<9;++k) tmp[k]=gin(i,k)-geqv[k];
        double m[9]; CONVERT(tmp,m);
        for (int q=0;q<9;++q) m[q]*=S[q];
        double dpost[9]; RECONVERT(m,dpost);
        for (int k=0;k<9;++k) gout(i,k)=gin(i,k)-dpost[k];
    }

    void BGK_Collision(int X, int Y) {
        int i=xyz_to_i(X,Y); if (isSolidNode(X,Y)) return;
        double geqv[9]; Equilibrium_g(P[i],Ux[i],Uy[i],geqv);
        double s = 1.0/tau;
        for (int k=0;k<9;++k) gout(i,k) = gin(i,k)*(1.0 - s) + s*geqv[k];
    }

    // PRE-stream curved BC (keep name; now robust)
    void Boundary_Conditions() { Bouzidi_quadratic(); }

    void Bouzidi_quadratic() {
        auto inDom=[&](int Xp,int Yp){ return (Xp>=0 && Xp<dim.nx && Yp>=0 && Yp<dim.ny); };
        auto apply_for_borders = [&](const vector<Border_Node>& B){
            for (const auto& bn : B) {
                int X=bn.X, Y=bn.Y;

                // IMPORTANT: our arrays have no halo — skip out-of-domain solid border nodes
                if (!inDom(X,Y)) continue;

                for (int I=1; I<=8; ++I) {
                    double D = bn.Delta[I-1];
                    if (D >= 1.0) continue;
                    int jI = jB_I[I];
                    int kI = kFromI(I), kJ = kFromI(jI);

                    int X1=X+exI[I], Y1=Y+eyI[I];
                    int X2=X1+exI[I], Y2=Y1+eyI[I];
                    int X3=X2+exI[I], Y3=Y2+eyI[I];

                    if (!inDom(X1,Y1)) continue; // nothing we can do safely
                    if (!inDom(X2,Y2)) { X2=X1; Y2=Y1; }
                    if (!inDom(X3,Y3)) { X3=X1; Y3=Y1; }
                    if (!inDom(X3,Y3)) { X3=X2; Y3=Y2; }

                    // ensure fluid neighbors by Fobj
                    if (F(X2+1,Y2+1) < 1) { X2=X1; Y2=Y1; }
                    if (F(X3+1,Y3+1) < 1) { X3=X2; Y3=Y2; }

                    int b  = xyz_to_i(X ,Y );
                    int n1 = xyz_to_i(X1,Y1);
                    int n2 = xyz_to_i(X2,Y2);
                    int n3 = xyz_to_i(X3,Y3);

                    if (D < 0.5) {
                        gout(b,kI) = gout(n1,kJ) * (1 + 2*D) * D
                                   + gout(n2,kJ) * (1 - 2*D) * (1 + 2*D)
                                   - gout(n3,kJ) * (1 - 2*D) * D;
                    } else {
                        if (!inDom(X2,Y2)) { X2=X1; Y2=Y1; n2=n1; }
                        gout(b,kI) = ( gout(n1,kJ)
                                     - gout(n1,kI) * (1 - 2*D) * (1 + 2*D)
                                     + gout(n2,kI) * (1 - 2*D) * D ) / ( D*(1 + 2*D) );
                    }
                }
            }
        };
        apply_for_borders(Borders1);
        apply_for_borders(Borders2);
    }

    void Streaming() {
        // pull
        vector<double> tmp(9);
        for (int X=0; X<dim.nx; ++X) for (int Y=0; Y<dim.ny; ++Y) {
            int i=xyz_to_i(X,Y);
            for (int k=0;k<9;++k) {
                int XX = wrapX(X - c[k][0]);
                int YY = wrapY(Y - c[k][1]);
                int src = xyz_to_i(XX,YY);
                tmp[k] = gout(src,k);
            }
            for (int k=0;k<9;++k) gin(i,k)=tmp[k];
        }
    }

    void Inlet_ZouHe(int t_iter) {
        double Pin = p0_in;
        if (t_iter >= t_start) Pin = p0_in + p_oscillatory * sin(omega*(t_iter + 1 - t_start));

        int X=0;
        int ylo=int(ceil(yr1[0]-0.01));
        int yhi=int(floor(yr2[0]+0.01));
        ylo = max(ylo, 0); yhi = min(yhi, dim.ny-1);
        for (int Y=ylo; Y<=yhi; ++Y) {
            int i=xyz_to_i(X,Y);
            double g0 = gin(i,kFromI(0));
            double g2 = gin(i,kFromI(2));
            double g3 = gin(i,kFromI(3));
            double g4 = gin(i,kFromI(4));
            double g6 = gin(i,kFromI(6));
            double g7 = gin(i,kFromI(7));

            double Uin = Pin - g0 - g2 - 2*g3 - g4 - 2*g6 - 2*g7;
            Uin = Uin * 3.0 / Rho0;

            gin(i,kFromI(1)) = g3 + 2.0*Rho0/9.0 * Uin;
            gin(i,kFromI(5)) = Rho0/18.0 * Uin - 0.5*(g2 - g4) + g7;
            gin(i,kFromI(8)) = Rho0/18.0 * Uin + 0.5*(g2 - g4) + g6;
        }
    }

    void Outlet_ZouHe(int t_iter) {
        double Pout = p0_out;
        if (t_iter >= t_start + t_propagation) Pout = p0_out + p_oscillatory * sin(omega*(t_iter + 1 - t_start - t_propagation));
        if (t_iter > t_sever) Pout = 0;

        int X=dim.nx-1;
        int ylo=int(ceil(yr1[dim.nx-1]-0.01));
        int yhi=int(floor(yr2[dim.nx-1]+0.01));
        ylo = max(ylo, 0); yhi = min(yhi, dim.ny-1);
        for (int Y=ylo; Y<=yhi; ++Y) {
            int i=xyz_to_i(X,Y);
            double g0 = gin(i,kFromI(0));
            double g1 = gin(i,kFromI(1));
            double g2 = gin(i,kFromI(2));
            double g4 = gin(i,kFromI(4));
            double g5 = gin(i,kFromI(5));
            double g8 = gin(i,kFromI(8));

            double Uout = g0 + 2*g1 + g2 + g4 + 2*g5 + 2*g8 - Pout;
            Uout = Uout * 3.0 / Rho0;

            gin(i,kFromI(3)) = g1 - 2.0*Rho0/9.0 * Uout;
            gin(i,kFromI(6)) = -Rho0/18.0 * Uout - 0.5*(g2 - g4) + g8;
            gin(i,kFromI(7)) = -Rho0/18.0 * Uout + 0.5*(g2 - g4) + g5;
        }
    }

    // ======================= functor (per cell collision) ===================
    void operator()(double& f0) {
        int i = &f0 - lattice;
        auto [X,Y] = i_to_xyz(i);
        MRT_Collision(X,Y); // or BGK_Collision(X,Y);
    }
};

// ───────────────────────── VTK output ─────────────────────────────────
void saveVtkFields_PulsatileBloodFlow2D(LBM_PulsatileBloodFlow2D& lbm, int time_iter, double dx) {
    Dim_PulsatileBloodFlow2D const& dim = lbm.dim;
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

    os << "SCALARS P float 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){ for (int x=0;x<dim.nx;++x){ int i=lbm.xyz_to_i(x,y); os << float(lbm.P[i]) << " "; } os << "\n"; } os << "\n";
    os << "SCALARS Ux float 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){ for (int x=0;x<dim.nx;++x){ int i=lbm.xyz_to_i(x,y); os << float(lbm.Ux[i]) << " "; } os << "\n"; } os << "\n";
    os << "SCALARS Uy float 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){ for (int x=0;x<dim.nx;++x){ int i=lbm.xyz_to_i(x,y); os << float(lbm.Uy[i]) << " "; } os << "\n"; } os << "\n";
    os << "SCALARS Flag int 1\nLOOKUP_TABLE default\n";
    for (int y=0;y<dim.ny;++y){ for (int x=0;x<dim.nx;++x){ int i=lbm.xyz_to_i(x,y); os << (lbm.flag[i]==CellType_PulsatileBloodFlow2D::bounce_back?1:0) << " "; } os << "\n"; } os << "\n";
}

// ───────────────────────── timing helpers ──────────────────────────────
inline auto restartClock_PulsatileBloodFlow2D() { return make_pair(high_resolution_clock::now(), 0); }
template<class TP>
inline void printMlups_PulsatileBloodFlow2D(TP start, int iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto us = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * iter) / us.count();
    cout << "Runtime: " << us.count()/1e6 << " s\n";
    cout << "Throughput: " << setprecision(4) << mlups << " MLUPS\n";
}

// ───────────────────────── main driver ─────────────────────────────────
void PulsatileBloodFlow2D()
{
    int N   = 64;
    Dim_PulsatileBloodFlow2D dim{ 1 + 10*(N-2), N };

    vector<CellData> lattice_vect(LBM_PulsatileBloodFlow2D::sizeOfLattice(dim.nelem));
    CellData *lattice = lattice_vect.data();

    vector<CellType_PulsatileBloodFlow2D> flag_vect(dim.nelem, CellType_PulsatileBloodFlow2D::bulk);
    auto* flag = flag_vect.data();

    vector<int> parity_vect {0};
    int* parity = parity_vect.data();

    auto [c_vect, opp_vect, t_vect] = d2q9_constants_PulsatileBloodFlow2D();

    LBM_PulsatileBloodFlow2D lbm{
        lattice, flag, parity,
        c_vect.data(), opp_vect.data(), t_vect.data(), dim
    };

    lbm.tau        = 0.75;
    lbm.s8         = 1.0/lbm.tau;
    lbm.s5         = 1.0;
    { double Svec[9] = {1,1,1,1, lbm.s5, 1, lbm.s5, lbm.s8, lbm.s8};
      std::copy(std::begin(Svec), std::end(Svec), lbm.S); }

    lbm.deformable = true;
    lbm.is_severed = true;
    lbm.alpha      = 0.01;
    lbm.p0_in      = 0.20;
    lbm.p0_out     = 0.19;
    lbm.t_beat     = max(1, dim.nx);

    lbm.Setup_Simulation_Parameters();
    lbm.Initialize_Yr_and_Vw_and_p();
    lbm.Initialize_Fobj_for_Vessel_Walls();
    lbm.Find_or_Update_Boundary_Nodes();
    lbm.Initialize_P_U_g();

    int tf   = lbm.t_beat + 2*lbm.t_propagation;
    int step = max(1, tf/100);

    auto [start, clock_iter] = restartClock_PulsatileBloodFlow2D();

    for (int t = 0; t <= tf; ++t) {
        // 1) collision (parallel)
        for_each(std::execution::par_unseq, lattice, lattice + dim.nelem, lbm);

        // 2) BCs before streaming (curved walls) — now robust to borders
        lbm.Boundary_Conditions();

        // 3) streaming (pull)
        lbm.Streaming();

        // 4) inlet/outlet after streaming
        lbm.Inlet_ZouHe(t);
        lbm.Outlet_ZouHe(t);

        // 5) macroscopic fields
        lbm.Macroscopic_Properties_g();

        // 6) wall motion
        if (lbm.deformable) lbm.Calculate_Pressure_and_Move_Walls(t);

        if (t % step == 0) {
            saveVtkFields_PulsatileBloodFlow2D(lbm, t);
            std::cout << "t="<<t<<" / "<<tf<<"\n";
        }

        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_PulsatileBloodFlow2D(start, clock_iter, dim.nelem);
}
// -------------- END PulsatileBloodFlow2D.h (complete) --------------
