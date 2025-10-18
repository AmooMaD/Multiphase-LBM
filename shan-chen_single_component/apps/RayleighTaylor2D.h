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
#include <stdlib.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
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

enum class CellType_RayleighTaylor2D : uint8_t { bounce_back, bulk };

inline auto d2q9_constants_RayleighTaylor2D() {
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
struct Dim_RayleighTaylor2D {
    Dim_RayleighTaylor2D(int nx_, int ny_)
        : nx(nx_), ny(ny_), nelem((size_t)nx*ny), npop(9*nelem) {}
    int nx, ny;
    size_t nelem, npop;
};

// LBM parameters: returns nu, omega, dx, dt
inline auto lbParameters_RayleighTaylor2D(double ulb, int lref, double Re) {
    double nu = ulb * lref / Re;
    double omega = 1. / (3.*nu + 0.5);
    double dx = 1. / lref;
    double dt = dx * ulb;
    return make_tuple(nu, omega, dx, dt);
}

void printParameters_RayleighTaylor2D(const Dim_RayleighTaylor2D &dim, double Re, double omega,
                                             double ulb, int N, double max_t , double nu)
{

    cout << "Rayleigh Taylor 2D problem\n"
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

inline auto restartClock_RayleighTaylor2D() {
    return make_pair(high_resolution_clock::now(), 0);
}

template<class TimePoint>
void printMlups_RayleighTaylor2D(TimePoint start, int clock_iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double mlups = (double)(nelem * clock_iter) / duration.count();
    cout << "result: " << duration.count()/1e6 << " seconds" << endl;
    cout << "result: " << setprecision(4) << mlups << " MLUPS" << endl;
};

struct LBM_RayleighTaylor2D {
    using CellData = double;
    static size_t sizeOfLattice(size_t nelem) { return 2 * 9 * nelem; }

    CellData* lattice;
    CellType_RayleighTaylor2D* flag;
    int* parity;
    std::array<int,2>* c;
    int* opp;
    double* t;
    double omega;
    double rhol;
    double rhog;
    double rhow;
    double g;      // fluid-fluid interaction
    double a;
    double b;
    double gravity;
    Dim_RayleighTaylor2D dim;

    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);
        int remainder = i % (dim.ny);
        int iY = remainder;
        return std::make_tuple(iX, iY);
    };

    size_t xyz_to_i (int x, int y) {
        return (y + dim.ny * x);
    };

    double& f(int i, int k) {
        return lattice[k*dim.nelem + i];
    }
    double& fin(int i, int k) {
        return lattice[*parity*dim.npop + k*dim.nelem + i];
    }
    double& fout(int i, int k) {
        return lattice[(1-*parity)*dim.npop + k*dim.nelem + i];
    }


   void iniLattice(CellData& f0) {
        // 1) compute linear index i and (iX, iY)
        size_t i = &f0 - lattice;
        auto [iX, iY] = i_to_xyz(int(i));

        // 2) compute the “mean” interface height as before
        double x = static_cast<double>(iX);
        double interface = (static_cast<double>(dim.ny) / 2.0)
                        + (static_cast<double>(dim.nx)) * 0.1 * std::cos(2.0 * M_PI * x / (static_cast<double>(dim.nx - 1)));

        // width parameter for the tanh transition
        double w = 2.5;

        // 3) smooth φ with a tanh profile across the interface
        //    φ = ½(φ_l + φ_g) + ½(φ_l – φ_g)·tanh[(y – interface)/(2·w)]
        double y = static_cast<double>(iY);
        double rho = 0.5 * (rhol + rhog)
                + 0.5 * (rhol - rhog) * std::tanh((y - interface) / (2.0 * w));


        // 6) initialize equilibrium populations (U=0) using φ as local “rho”
        for (int k = 0; k < 9; ++k) {
            fin(i, k) = rho * t[k];
        }
    }

    auto density (double& f0) {
        auto i = &f0 - lattice;
        double rho;
            double X_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
            double X_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);
            double X_0  = fin(i, 6) + fin(i, 1) + fin(i, 4);

            rho = X_M1 + X_P1 + X_0;

        return rho;
    }


    auto u_common (double& f0) {
        auto i = &f0 - lattice;
        std::array<double, 2> u {0., 0.};

        double rho = density(f0);

        double X_M1 = fin(i, 0) + fin(i, 2) + fin(i, 3);
        double X_P1 = fin(i, 5) + fin(i, 7) + fin(i, 8);

        double Y_M1 = fin(i, 1) + fin(i, 2) + fin(i, 8);
        double Y_P1 = fin(i, 3) + fin(i, 6) + fin(i, 7);

        u[0] = X_P1 - X_M1;
        u[1] = Y_P1 - Y_M1;

        u[0] /= rho;
        u[1] /= rho;

        return u;
    }

    double psi_rho(double dens) {
        return 1-exp(-dens);
    }




    double P_eos(double& f0) {
        int   i         = &f0 - lattice;
        auto [iX, iY]   = i_to_xyz(i);

        double rho = density(f0);
        double rt = b* rho /4 ;

        return (rho /3.) * (1. + rt + rt*rt - rt *rt *rt ) / ( ( 1- rt ) *  ( 1- rt ) *  ( 1- rt ) ) - a*rho * rho;
    }

  /*
    double P_eos(double& f0) {
        int   i         = &f0 - lattice;
        auto [iX, iY]   = i_to_xyz(i);

        double rho = density(f0);
        double Tc = 0.1961 ;
        double T = 0.85 * Tc ;

        return rho * T / (1 - b*rho) - (a*rho * rho) / ( sqrt (T) * ( 1 + b* rho ));

    }

    */


    double psi_sc(double& f0) {
        int   i         = &f0 - lattice;
        auto [iX, iY]   = i_to_xyz(i);

        double rho      = density(f0);
        double pressure = P_eos(f0);
        return sqrt( 6.0*(pressure - rho/3.0) / g );
    }


	/// force function (fluid-fluid interactions)
	std::array<double,2> force_ff(double& f0) {
	    // compute center index and its coords
	    auto i = &f0 - lattice;
	    auto [iX, iY] = i_to_xyz(i);

	    double fx = 0.0;
	    double fy = 0.0;
        double rho_c = density(f0);
        //double psi_c = psi_sc(f0);
        double psi_c = psi_rho(rho_c);

	    // loop over all 9 lattice directions
	    for (int k = 0; k < 9; ++k) {
            // periodic boundaries in x and y
            int XX = iX + c[k][0];
            XX = (XX + dim.nx) % dim.nx;
            int YY = iY + c[k][1];
            //YY = (YY + dim.ny) % dim.ny;
            int nb = xyz_to_i(XX, YY);

            double rho_nb;
            // bounce-back (wall) handling
            if (flag[nb] == CellType_RayleighTaylor2D::bounce_back) {

                // reflect direction when hitting a wall
                int XXX = iX - c[k][0];
                XXX = (XXX + dim.nx) % dim.nx;
                int YYY = iY - c[k][1];
                //YYY = (YYY + dim.ny) % dim.ny;
                int nbb = xyz_to_i(XXX, YYY);
                double rho_nbb = density(lattice[nbb]);

                //double psi_nb = psi_sc(lattice[nbb]);
                double psi_nb = psi_rho(rho_nbb);
                // accumulate interaction force
                fx += t[k] * c[k][0] * psi_nb;
                fy += t[k] * c[k][1] * psi_nb;
            }

            else {
                rho_nb = density(lattice[nb]);
                //double psi_nb = psi_sc(lattice[nb]);
                double psi_nb = psi_rho(rho_nb);
                // accumulate interaction force
                fx += t[k] * c[k][0] * psi_nb;
                fy += t[k] * c[k][1] * psi_nb;
            }
	    }

	    fx *= -g * psi_c;
	    fy *= -g * psi_c;


        fy += gravity * rho_c;

	    return {fx, fy};
	}



	/// force function (fluid-solid interactions)
	std::array<double,2> force_fw(double& f0) {
	    // compute center index and its coords
	    auto i = &f0 - lattice;
	    auto [iX, iY] = i_to_xyz(i);

	    double fx = 0.0;
	    double fy = 0.0;
        double rho_c = density(f0);
        double psi_c = psi_rho(rho_c);


	    // loop over all 9 lattice directions
	    for (int k = 0; k < 9; ++k) {
            // periodic boundaries in x and y
            int XX = iX + c[k][0];
            XX = (XX + dim.nx) % dim.nx;
            int YY = iY + c[k][1];
            int nb = xyz_to_i(XX, YY);

            double rho_nb;
            // bounce-back (wall) handling
            if (flag[nb] == CellType_RayleighTaylor2D::bounce_back) {

                // reflect direction when hitting a wall
                int XXX = iX + c[k][0];
                XXX = (XXX + dim.nx) % dim.nx;
                int YYY = iY + c[k][1];
                //YYY = (YYY + dim.ny) % dim.ny;
                //int nbb = xyz_to_i(XXX, YYY);
                rho_nb = density(lattice[nb]);

                double psi_nb = psi_rho(rho_nb);
                // accumulate interaction force
                fx += t[k] * c[k][0] * psi_nb;
                fy += t[k] * c[k][1] * psi_nb;
            }

	    }

	    fx *= -g * psi_c * 0.;                                                     // for now, we eliminate the effect of the wall on the fluid
	    fy *= -g * psi_c * 0.;                                                     // for now, we eliminate the effect of the wall on the fluid

	    return {fx, fy};
	}


    auto u_eq (double& f0) {
        auto i = &f0 - lattice;
        double tau = 1. / omega;
        double rho = density(f0);
        auto u    = u_common(f0);
        auto FF = force_ff(f0);
        auto FW = force_fw(f0);
        array<double,2> u_equilibruim = { u[0] +  (FF[0] + FW[0]) / (2 * rho), u[1] + (FF[1] + FW[1]) / (2 * rho) };
        return u_equilibruim;
    }


    void stream(size_t i, int k, int iX, int iY, double pop_out) {
        int x2 = iX + c[k][0];
        int y2 = iY + c[k][1];
        x2 = (x2 + dim.nx) % dim.nx;
        //y2 = (y2 + dim.ny) % dim.ny;

        size_t nb=xyz_to_i(x2,y2);
        if(flag[nb]==CellType_RayleighTaylor2D::bounce_back){
            fout(i,opp[k])=pop_out;
        }
        else{
            fout(nb,k)=pop_out;
        }
    };

    auto collideBgk(int i, int k, double rho, double usqreq,
        array<double,2>& u_eq, array<double, 2> FF, array<double, 2> FW) {

        double pop_out;
        double pop_out_opp;
        std::array<double, 2> ueq;

        ueq[0] = u_eq[0];
        ueq[1] = u_eq[1];

        double F_x = FF[0];
        double F_y = FF[1];

        //F_x += FW[0];
        //F_y += FW[1];

        double e_u_x = c[k][0] - u_eq[0];
        double e_u_x_op = c[opp[k]][0] - u_eq[0];
        double e_u_y = c[k][1] - u_eq[1];
        double e_u_y_op = c[opp[k]][1] - u_eq[1];

        double ck_ueq = c[k][0] * ueq[0] + c[k][1] * ueq[1];

        double eq_f = rho * t[k] * (1. + 3. * ck_ueq + 4.5 * ck_ueq * ck_ueq - usqreq);
        double eq_fopp = eq_f - 6. * rho * t[k] * ck_ueq;

        double total_F = t[k] * ( 1 - 0.5 * omega ) * ( ( 3 *  e_u_x + 9 * ck_ueq * c[k][0] ) * F_x + ( 3 *  e_u_y + 9 * ck_ueq * c[k][1] ) * F_y ) ;
        double total_Fopp = t[k] * ( 1 - 0.5 * omega ) * ( ( 3 *  e_u_x_op - 9 * ck_ueq * c[opp[k]][0] ) * F_x + ( 3 *  e_u_y_op - 9 * ck_ueq * c[opp[k]][1] ) * F_y ) ;

        pop_out = (1. - omega) * fin(i, k) + omega * eq_f + total_F;
        pop_out_opp = (1. - omega) * fin(i, opp[k]) + omega * eq_fopp + total_Fopp;

        return std::make_pair(pop_out, pop_out_opp);
    }

    void operator() (double& f0) {
        auto i = &f0 - lattice;

        if (flag[i] == CellType_RayleighTaylor2D::bulk) {
            auto rho = density(f0);
            auto[iX, iY] = i_to_xyz(i);
            auto ueq = u_eq(f0);
            auto FF = force_ff(f0);
            auto FW = force_fw(f0);

            double usqreq = 1.5 * ( ueq[0] * ueq[0] + ueq[1] * ueq[1] );

            for (int k = 0; k < 4; ++k) {
                auto[pop_out, pop_out_opp] = collideBgk(i, k, rho, usqreq, ueq, FF, FW) ;
                stream(i, k, iX, iY, pop_out);
                stream(i, opp[k], iX, iY, pop_out_opp);
            }

            for (int k: {4}) {
                double eq_f = rho * t[k] * (1. - usqreq);

                double total_F_center = t[k] * ( 1 - 0.5 * omega ) * ( (-3. *  ueq[0] * (FW[0] + FF[0]) ) + ( -3. * ueq[1] * (FW[1] + FF[1]) ) );

                fout(i, k) =  (1. - omega) * fin(i, k) + omega * eq_f + total_F_center;

            }
        }
    }
};


void saveVtkFields_RayleighTaylor2D(LBM_RayleighTaylor2D& lbm, int time_iter, double dx=0.) {
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
            double d=lbm.density(lbm.lattice[i]); 
            if(lbm.flag[i]==CellType_RayleighTaylor2D::bounce_back) {
                d=0.;
            }
            vtk<<d<<" ";
        }
        vtk<<"\n";
    } 
    vtk<<"\n";


    vtk << "VECTORS Force_ff float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            auto ff = lbm.force_ff(lbm.lattice[i]);
            double force_x = ff[0];
            double force_y = ff[1];
            double force_z = 0.0;  // <-- supply a dummy z-component

            // Now write three numbers per point: fx, fy, fz
            vtk << force_x << " " << force_y << " " << force_z << "\n";
        }
    }
    vtk << "\n";


    vtk << "VECTORS Force_fw float\n";
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            size_t i = lbm.xyz_to_i(x, y);
            auto fw = lbm.force_fw(lbm.lattice[i]);
            double force_x = fw[0];
            double force_y = fw[1];
            double force_z = 0.0;  // <-- supply a dummy z-component

            // Now write three numbers per point: fx, fy, fz
            vtk << force_x << " " << force_y << " " << force_z << "\n";
        }
    }
    vtk << "\n";


}


double computeEnergy_RayleighTaylor2D(LBM_RayleighTaylor2D& lbm) {
    const auto& dim = lbm.dim;
    double energy = 0.0;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (lbm.flag[i] == CellType_RayleighTaylor2D::bulk) {
                auto u = lbm.u_eq(lbm.lattice[i]);
                energy += u[0]*u[0] + u[1]*u[1];
            }
        }
    }
    return 0.5 * energy / (dim.nx * dim.ny);
}



// Geometry initialization

void inigeom_RayleighTaylor2D(LBM_RayleighTaylor2D& lbm) {
    Dim_RayleighTaylor2D const& dim = lbm.dim;
    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);
            if (y == 0 || y == dim.ny-1) {lbm.flag[i] = CellType_RayleighTaylor2D::bounce_back;
              for (int k = 0 ; k<9 ; ++k){
                lbm.f(i,k) = 0.;

              }
            }
            else {lbm.flag[i] = CellType_RayleighTaylor2D::bulk;}
        }
    }
}


/*
#include <random>

void inigeom_RayleighTaylor2D(LBM_RayleighTaylor2D& lbm, double porosity = 0.85) {
    Dim_RayleighTaylor2D const& dim = lbm.dim;

    // Random number generator with fixed seed for reproducibility
    std::mt19937 gen(42);
    std::bernoulli_distribution dist(porosity);

    for (int y = 0; y < dim.ny; ++y) {
        for (int x = 0; x < dim.nx; ++x) {
            int i = lbm.xyz_to_i(x, y);

            // Always solid boundaries at top and bottom
            if (y == 0 || y == dim.ny - 1) {
                lbm.flag[i] = CellType_RayleighTaylor2D::bounce_back;
                for (int k = 0; k < 9; ++k) {
                    lbm.fin(i, k) = 0.;
                }
            }
            else {
                // Randomly assign interior cell as fluid or solid
                if (dist(gen)) {
                    lbm.flag[i] = CellType_RayleighTaylor2D::bulk;
                } else {
                    lbm.flag[i] = CellType_RayleighTaylor2D::bounce_back;
                    for (int k = 0; k < 9; ++k) {
                        lbm.fin(i, k) = 0.;
                    }
                }
            }
        }
    }
}
*/



void RayleighTaylor2D() {
    ifstream contfile("../apps/Config_Files/config_RayleighTaylor2D.txt");
    if (!contfile.is_open()) {
        throw invalid_argument(
          "Config file not found. It should be named "
          "\"config_rayleighTaylor2D.txt\" in Files_Config.");
    }
    double Re = 0, ulb = 0, max_t = 0, rhol = 0, rhog = 0, rhow = 0, g = 0, a=0., b=0. ,gravity =0.;
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
            else if (param == "ulb"){     ulb      = stod(value);}
            else if (param == "N"){       N        = std::stoi(value);}
            else if (param == "max_t"){   max_t    = stod(value);}
            else if (param == "out_freq"){out_freq = stoi(value);}
            else if (param == "vtk_freq"){vtk_freq = stoi(value);}
            else if (param == "rhol"){     rhol    = stod(value);}
            else if (param == "rhog"){     rhog    = stod(value);}
            else if (param == "rhow") {rhow        = stod(value);}
            else if (param == "g"){       g        = stod(value);}
            else if (param == "a") {      a        = stod(value);}
            else if (param == "b"){       b        = stod(value);}
            else if (param == "gravity"){       gravity        = stod(value);}
            else{
                cerr << "Warning: unknown parameter \"" << param << "\"\n";}
        

    }

    using CellData = typename LBM_RayleighTaylor2D::CellData;

    Dim_RayleighTaylor2D dim {N , 4 * N +2};

    auto[nu, omega, dx, dt] = lbParameters_RayleighTaylor2D(ulb, N , Re);
    printParameters_RayleighTaylor2D(dim,Re,omega,ulb,N,max_t,nu);

    vector<CellData> lattice_vect(LBM_RayleighTaylor2D::sizeOfLattice(dim.nelem));
    CellData *lattice = &lattice_vect[0];

    vector<CellType_RayleighTaylor2D> flag_vect(dim.nelem);
    CellType_RayleighTaylor2D* flag = &flag_vect[0];

    vector<int> parity_vect {0};
    int* parity = &parity_vect[0];

    auto[c, opp, t] = d2q9_constants_RayleighTaylor2D();

    LBM_RayleighTaylor2D lbm{lattice, flag, parity, &c[0], &opp[0], &t[0], omega, rhol,
            rhog, rhow, g, a, b, gravity, dim};

    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });

    inigeom_RayleighTaylor2D(lbm);

    auto[start, clock_iter] = restartClock_RayleighTaylor2D();

    ofstream energyfile("energy.dat");

    int max_time_iter = static_cast<int>(max_t / dt);
    for (int time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (vtk_freq != 0 && time_iter % vtk_freq == 0 && time_iter >= 0) {
           saveVtkFields_RayleighTaylor2D(lbm, time_iter);
        }
        if (out_freq != 0 && time_iter % out_freq == 0 && time_iter >= 0) {
            cout << "Saving profiles at iteration " << time_iter
                 << ", t = " << setprecision(4) << time_iter * dt << setprecision(3)
                 << " [" << time_iter * dt / max_t * 100. << "%]" << endl;
            double energy = computeEnergy_RayleighTaylor2D(lbm) *dx*dx / (dt*dt);
            cout << "Average energy: " << setprecision(8) << energy << endl;
            energyfile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
        }

        for_each(execution::par_unseq, lattice, lattice + dim.nelem, lbm);

        *parity = 1 - *parity;
        ++clock_iter;
    }

    printMlups_RayleighTaylor2D(start, clock_iter, dim.nelem);
}
