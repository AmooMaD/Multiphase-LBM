/*******************************************************************************
# CooLBM SOFTWARE LIBRARY

# Copyright ©️ 2025 CORIA Lab., CNRS UMR 6614
# Contact: Dr. Mostafa Safdari Shadloo, msshadloo@coria.fr

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# In case of use of CooLBM, please properly cite the following articles:
# [1] R. Alamian, A. K. Nayak, M. S. Shadloo, CooLBM: A Collaborative Open-Source Reactive Multi-Phase/Component Simulation Code via Lattice Boltzmann Method, doi.org/10.48550 arXiv.2502.12955.
# [2] R. Alamian, M. Sawaf, C. Stockinger, A. Hadjadj, J. Latt, M. S. Shadloo, Modeling soot filter regeneration process through surface-reactive flow in porous media using iterative lattice Boltzmann method, Energy, 2024 (289), 129980.
# [3] C. Stockinger, A. Raiolo, R. Alamian, A. Hadjadj, U. Nieken, M. S. Shadloo, Lattice Boltzmann simulations of heterogeneous combustion reactions for application in porous media, Engineering Analysis with Boundary Elements, 2024 (166) 105817.
# [4] A. K. Nayak, A. Singh, M. Mesgarpour, M. S. Shadloo, A Numerical Investigation of Particle Deposition on a Substrate, doi.org/10.48550/arXiv.2502.12719.
# ******************************************************************************/

//C Code for 2-dimensional LBM with D2Q9 scheme, selfcontained
//C Basis ist der MRT2 Code von Rechnung 3 vom 31.05.2022


#include <vector>       /// Vector is a sequential container to store elements and not index based. ector is dynamic in nature so, size increases with insertion of elements.
#include <array>        /// Array stores a fixed-size sequential collection of elements of the same type and it is index based. As array is fixed size, once initialized can’t be resized.
#include <string>
#include <iostream>     /// Header that defines the standard input/output stream objects (cin, cout, cerr, clog).
#include <fstream>       /// Input/output stream class to operate on files.
#include <iomanip>      /// is a library that is used to manipulate the output of C++ program. Using C++, header providing parametric manipulators
#include <algorithm>    /// defines a collection of functions especially designed to be used on ranges of elements (for_each)
#include <execution>    /// This header is part of the algorithm library (seq, par, par_unseq, unseq)
#include <chrono>       /// The chrono library, a flexible collection of types that track time with varying degrees of precision (clocks, time points, durations)
#include <cmath>        /// Header <cmath> declares a set of functions to compute common mathematical operations and transformations.
#include <tuple>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

double Re        = 200.;  // Reynolds number                                        //C the first 5 global variables, which are declared here, are the inputs for the function runCavitytwoPop, which is called
double ulb       = 0.002;  // Velocity in lattice units                              //C in the main-function of the program
int N            = 16;   // Number of nodes in x-direction //C used to be 128
int N_X = 420;      //C 420         (davor 200)     120
int N_Y = 80;       //C 80                          80
bool benchmark   = false;  // Run in benchmark mode ?
int out_freq  = 1000;      // Non-benchmark mode: Frequency in LU for output of terminal message and profiles (use 0 for no messages) 20 for compressible
int data_freq = 0;        // Non-benchmark mode: Frequency in LU of full data dump (use 0 for no data dump)
int vtk_freq = 1000;       //C was added by Claudius; frequency for the output in vtk files       50 for compressible
int bench_ini_iter = 1000; // Benchmark mode: Number of warmup iterations
int bench_max_iter = 2000; // Benchmark mode: Total number of iteration
bool periodic = false;
int q_value = 9;
int d_value = 2;
double omega_glob = 1.0;        //C war 1.8, führte aber zu Instabilitäten
double Faktor = 1.;       //C Faktor for relaxation times in MRT of flow field
double Faktor2= 1.;         //C Faktor for relaxation times in MRT of the Concentrations
double Faktor3= 1.;         //C Faktor for relaxation times in MRT of the Temperature

bool first_step = true;

double CS2 = 1./3.;         //C lattice speed of sound squared

//C parameters of the simulation

double max_t     = 487717.0;              // number of time steps   //C Konvergenz bei ca. 90% von 30000 steps      //C value according to paper 487717.0
double delta_t_lu = 1.;                 //C size of the time-step [t-lu]
double delta_t_ph = 1./48771706.22;     //C size of the time-step [s]



//C model parameters

double D_n          = 1.;           //C diffusion coefficient of component n; sollte eigentlich ein Vektor sein, falls versch. Komponenten versch. Diff-koeff. haben
double D_O2         = 0.1568739;    //C diffusion coefficient O2
double D_CO2        = 0.1425984;    //C diffusion coefficient CO2
double alpha_gas    = 0.2337421;    //C Wärmeleitfähigkeit für Gas
double alpha_solid  = 0.0278153;    //C Temperaturleitfähigkeit a  =^= im Englischen thermal diffusivity alpha
double ny_gas       = 0.1673593;    //C kinemat viscosity v
double lambda_gas   = 83.251305;

double Tau_flow     = 1.0;                //C (ny_gas   /CS2)+0.5;
double Tau_O2       = 1.021;              //C (D_O2     /CS2)+0.5;
double Tau_CO2      = 0.973;              //C (D_CO2    /CS2)+0.5;
double Tau_T        = 1.198;              //C (alpha_gas/CS2)+0.5;
double Tau_T_solid  = 0.583;              //C (alpha_solid/CS2)+0.5;

double cp_gas   = 356.16739;    //C                     spezifische Wärmekapazität of the gas
double cp_solid = 235.4266;     //C                     specific heat capacity of the coke
double rho_gas  = 1.;           //C [m-lu/s-lu^3]       density of the gas
double rho_solid = 556.694;     //C [m-lu/s-lu^3]       density of the coke
double M_O2     = 32.;          //C [m-lu/mol-lu]       molar mass of Oxygen
double M_CO2    = 44.;          //C [m-lu/mol-lu]       molar mass of carbon dioxyde

double Sigma = (rho_solid*cp_solid)/(rho_gas*cp_gas);

double R_id     = 2701.8026;    //C ideale Gaskonstante     [lu]

//C parameters of the reaction

int stoich_O2  = -1;
int stoich_CO2 = 1;
double Prae_exp_factor = 1.992343666625*pow(10,5);  // [lu]
double E_akt   = 5.511041294*pow(10,4);                       // [lu]
double delta_hr = 1.8132054257*pow(10,5);        //C1.633259243791*pow(10,5);                      // [lu]   1.633259243791*pow(10,5) pow(10,4) zu viel wurde nach 17% schon 1,8 warm
                                                    //C     1.633259243791*pow(10,3) ein bissle zu wenig; bei t=0.006s war T 1.065 und Y_O2 noch zu hoch vor erstem obstacles

//C Reaktionsparameter in SI-Einheiten

double E_akt_SI = 131.09;           //C [kJ/mol]
double A_SI     = 9.717*pow(10,6);  //C [m/s]
double hr_SI    = 388.5;            //C [kJ/mol]
double R_SI     = 8.314;            //C [J/(mol*K)]
double dx_SI    = 1.002*pow(10,-6); //C [m]
double dt_SI    = 2.05*pow(10,-8);  //C [s]
double rho_SI   = 4.5;              //C [kg/m^3]
double cp_SI    = 1.096;            //C [kJ/(kg*K)]
double M_O2_SI  = 32.;               //C [g/mol]

//C parameters of the inlet

double u_lb         = 0.00011741168;              // [lu]  0.00011741168   //C laut Timan 0.0001166      //C bei 0.2 gibt es die Probleme mit T-divergenz am Rechten Rand des Obstacles
double T_inlet      = 1.;                      // [lu]
double Y_O2_inlet   = 0.22;                    // [lu]
double Y_CO2_inlet  = 0.0;                     // [lu]
double rho_inlet    = 1.;                      // [lu]

//C Values at boundary conditions

double T_links  = 0.0;
double T_rechts = 0.0;

//C parameters for initialization of the domain

double T_ini    = 1.0;          //C Temperatur Gas
double T_ini2   = 2.0;          //C temperatur solid, only in combination with obstacles_temp
double rho_ini  = 1.;
double Y_O2_ini = 0.;
double Y_CO2_ini = 0.;
double T_ref        = T_ini;


bool Obstacles = true; 
bool ObstaclesTest = false;
bool Obstacle1 = false;   
bool Konjug_Waermetransport = true;         //C toggles conjugated heat transfer in the stream_temp


//C choice of konjugated heat transfer BC

bool GUO2015 = false;
// this works
bool LI2014  = true;           //C bounce-back mit extratermL
bool HUBER2015 = false;          //C normales Treatment mit force-term

int NC = 2;                 //C number of components in the fluid flow


//C choice if single relaxation time or multi relaxation time model
bool MRT = true;

//C choice of test cases
bool Cavity             = false;
bool Couette            = false;
bool Poiseuille         = false;
bool Inflow_Outflow     = true;             //C simulation of flow with inflow/outflow BC rather than force-driven
bool Multi_Component    = true;            //C enables multiple components for the gas phase
bool Temperature_coupling = false;      //C toggles beta in f_eq and C_dach in Kollisionsschritt für flowfield f
bool Solid_reaction     = true;             //C toggles the reaction at the interface of solid/gas


//C Vorkonfigurierte Testcases
bool Reaktionstestcase  = true;
bool Flowfield          = false;
bool Testsimulation = false;

bool Vorwaertsdiff  = false;
bool Zentraldiff    = false;

//C MRT Parameters for the flow field
double w_q      = 1.0;            //C free parameter to tune
double w_eps    = 1.;          //C free parameter to tune
double w_e      = 1.0;            //C connected to bulk viscosity     //C in general, decrease           bei 0.5 und 1.2 explosion, zwischen 0.7 und 1 gehts; konvergiert schneller bei 0.7
double w_ny     = 1./Tau_flow;       //1./Tau_flow;  //C connected to shear viscosity

//C MRT Parameters for Component O2
double w_qO2    = 1.;
double w_epsO2  = 1.;
double w_eO2    = 1.;
double w_nyO2   = 1./Tau_O2;

//C MRT Parameters for Component CO2
double w_qCO2   = 1.;
double w_epsCO2 = 1.;
double w_eCO2   = 1.;
double w_nyCO2  = 1./Tau_CO2;

//C MRT Parameters for Temperature
double w_qT = 1.;
double w_epsT = 1.;
double w_eT = 1.;
double w_nyT = 1./Tau_T;

//C MRT Parameters for Temperature solid
double w_qT_solid = 1.;
double w_epsT_solid = 1.;
double w_eT_solid = 1.;
double w_nyT_solid = 1./Tau_T_solid;

//C configuration of boundary conditions
bool x_noslip = false;
bool y_noslip = true;
bool x_periodic = false;
bool y_periodic = false;
bool x_inflow_outflow = true;
bool x_freeslip_wall = false;                   //C toggles free-slip wall BC; implemented with specular reflection
bool y_freeslip_wall = false;  
bool links_dirichlet_temp = false;
bool rechts_dirichlet_temp = false;

bool inlet_parabolic = true;
bool inlet_constant  = false;

double dpdx = 0.;           //C 0.00005;
double dpdy = 0.;

bool External_force = false;
double efx = 0;
double efy = 0;

bool timestep_stop = false;
bool print_temp    = true;     //C prints thermodynamical Mitteltemperatur


//C Variables to configure reading in variables or saving them (-> for reading in those variables in the next simulation)
bool Einleseoption_f = true;        //C true enables the reading in of the distribution function f for the flow field
bool Ausgabeoption_f = false;       //C true enables printing the values of f into a file (for reading in the next simulation)




//C u_lb auf 0 setzen und Konv auf true, um reines Diffusionsproblem zu haben
bool Konv = true;          //C used to pass the konv information into the operator function
double Konvergenzabweichung = 0.00001;          //C 0.00001 for u_lb = 0.0002


vector<double> f_ini(N_X*N_Y*q_value);          //C vector which saves the input data
double& f_input (int i, int k) {                //C function with which the vector is called  
        return f_ini[k * N_X*N_Y + i];
}



vector <vector<double>> Inflow_vel(N_Y, vector<double>(d_value,0));

vector<double> F_glob_component(N_X*N_Y*NC*q_value,0);    
vector<double> F_glob_temperature(N_X*N_Y*q_value,0);   
vector<double> rho_glob_prev(N_X*N_Y,1.);           //C hier ganz wichtig, den Vektor mit dem Wert der Initialdichte des Systems zu initialisieren!!! (sonst Probleme mit 1. Zeitschritt mit Zeitableitung in Fq2)
vector<double> q_glob(N_X*N_Y,0.);           //C hier ganz wichtig, den Vektor mit dem Wert der Initialdichte des Systems zu initialisieren!!! (sonst Probleme mit 1. Zeitschritt mit Zeitableitung in Fq2)
vector<double> g_post_coll(N_X*N_Y*q_value,0.); 

double& F_O2_alt (int i, int k) {
        return F_glob_component[i + k*N_X*N_Y];
}

double& F_CO2_alt (int i, int k) {
        return F_glob_component[i + k*N_X*N_Y + N_X*N_Y*q_value];
}

double& F_T_alt (int i, int k) {
        return F_glob_temperature[i + k*N_X*N_Y];
}

double& rho_i_prev(int i){
    return rho_glob_prev[i];
}
double& q_i(int i){
    return q_glob[i];
}

double& g_coll (int i, int k) {                 //C saves post-collision populations at interface for conjugated heat transfer according to Li 2014
        return g_post_coll[i + k*N_X*N_Y];
}

double TC;


enum class CellType : uint8_t { bounce_back, bulk, specular_reflection, reactive_obstacle};        /// An enumeration is a distinct type whose value is restricted to a range of values (see below for details), which may include 
                                                            /// several explicitly named constants ("enumerators")
                                                            /// unit8_t : unsigned integer type with width of exactly 8 bits
inline auto d2q9_constants() {                             //// C++ provides an inline functions to reduce the function call overhead. Inline function may increase efficiency if it is small.
    // The discrete velocities of the d3q19 mesh.           /// auto: Deduces the type of a declared variable from its initialization expression.
    vector<array<int, 2>> c_vect = {                       //C std::array encapsulates fixed-size arrays, in this case of length 3
        {0, 0},
        {1, 0}, {0, 1},                  //C -> a vector containing arrays of size 3 is created
        {-1,0}, {0, -1}, 
        { 1, 1}, { -1, 1}, 
        {-1,-1}, { 1,-1}, 
    };

    //C Transformation matrix for MRT
    vector<array<int, 9>> M = {                       
        { 1,  1,  1,  1,  1,  1,  1,  1,  1}, 
        {-4, -1, -1, -1, -1,  2,  2,  2,  2},                 
        { 4, -2, -2, -2, -2,  1,  1,  1,  1}, 
        { 0,  1,  0, -1,  0,  1, -1, -1,  1}, 
        { 0, -2,  0,  2,  0,  1, -1, -1,  1},
        { 0,  0,  1,  0, -1,  1,  1, -1, -1}, 
        { 0,  0, -2,  0,  2,  1,  1, -1, -1}, 
        { 0,  1, -1,  1, -1,  0,  0,  0,  0}, 
        { 0,  0,  0,  0,  0,  1, -1,  1, -1}, 
    };


    vector<vector<double>> M_inv = {                       
        { 1./9., -1./9.,  1./9.,  0,     0,       0,     0,      0,   0}, 
        { 1./9., -1/36., -1/18., 1/6., -1/6.,     0,     0,     1/4., 0},                 
        { 1/9.,  -1/36., -1/18.,  0,     0,      1/6., -1/6.,  -1/4., 0}, 
        { 1/9.,  -1/36., -1/18.,-1/6.,  1/6.,     0,     0,     1/4., 0}, 
        { 1/9.,  -1/36., -1/18.,  0,     0,     -1/6.,  1/6.,  -1/4., 0},
        { 1/9.,  1/18.,   1/36.,  1/6.,  1/12.,  1/6.,   1/12.,  0,  1/4.}, 
        { 1/9.,  1/18.,   1/36., -1/6., -1/12.,  1/6.,   1/12.,  0, -1/4.}, 
        { 1/9.,  1/18.,   1/36., -1/6., -1/12., -1/6.,  -1/12.,  0,  1/4.}, 
        { 1/9.,  1/18.,   1/36.,  1/6.,  1/12., -1/6.,  -1/12.,  0, -1/4.}, 
    };


    //vector<double> S = { 0, w_e, w_eps, 0, w_q, 0, w_q, w_ny, w_ny};               //C defines the arrangement of the velocity vectors, which ones are opposite of each othe

    vector<double> S = { 1*Faktor, w_e*Faktor, w_eps*Faktor, 1*Faktor, w_q*Faktor, 1*Faktor, w_q*Faktor, w_ny*Faktor, w_ny*Faktor};               //C defines the arrangement of the velocity vectors, which ones are opposite of each othe

    vector<double> S_GO2        = { 1*Faktor2, w_eO2*Faktor2,        w_epsO2*Faktor2,        1*Faktor2, w_qO2*Faktor2,        1*Faktor2, w_qO2*Faktor2,       w_nyO2*Faktor2,         w_nyO2*Faktor2}; 
    vector<double> S_GCO2       = { 1*Faktor2, w_eCO2*Faktor2,       w_epsCO2*Faktor2,       1*Faktor2, w_qCO2*Faktor2,       1*Faktor2, w_qCO2*Faktor2,      w_nyCO2*Faktor2,        w_nyCO2*Faktor2};  
    vector<double> S_T          = { w_nyT, w_nyT,        w_nyT,         w_nyT, w_nyT,        w_nyT, w_nyT,        w_nyT,          w_nyT};  
    vector<double> S_T_solid    = { w_nyT_solid, w_nyT_solid,   w_nyT_solid,   w_nyT_solid, w_nyT_solid,   w_nyT_solid, w_nyT_solid,  w_nyT_solid,   w_nyT_solid};  
    //vector<double> S_T          = { 1*Faktor3, w_eT*Faktor3,         w_epsT*Faktor3,         1*Faktor3, w_qT*Faktor3,         1*Faktor3, w_qT*Faktor3,        w_nyT*Faktor3,          w_nyT*Faktor3};  
    //vector<double> S_T_solid    = { 1*Faktor3, w_eT_solid*Faktor3,   w_epsT_solid*Faktor3,   1*Faktor3, w_qT_solid*Faktor3,   1*Faktor3, w_qT_solid*Faktor3,   w_nyT_solid*Faktor3,   w_nyT_solid*Faktor3};  

    // The opposite of a given direction.
    vector<int> opp_vect =
        { 0,3,4,1,2,7,8,5,6};               //C defines the arrangement of the velocity vectors, which ones are opposite of each other

    // The lattice weights.
    vector<double> t_vect =
        {
            4./9.,
            1./9., 1./9., 1./9., 1./9.,
            1./36., 1./36., 1./36., 1./36.
        };
    return make_tuple(c_vect, opp_vect, t_vect, M, M_inv, S, S_GO2, S_GCO2, S_T, S_T_solid);            //C std::make_tuple creates a tuple object; the target type is deduced from the types of the arguments
}                                                           //C access to the tuple by std::get<variable>(tuple_name)

// Stores the number of elements in a rectangular-shaped simulation.
struct Dim {
    Dim(int nx_, int ny_)                                                                  //C Dim::DIM can be called and given three inputs, which are the number of lattice nodes in each direction
        : nx(nx_), ny(ny_),                                                                //C die übergebenen Variablen are renamed and can be accesed by i.e. DIM::nx            
          nelem(static_cast<size_t>(nx) * static_cast<size_t>(ny)),       ///size_t : unsigned integer type       //C DIM::nelem returns the total number of lattice nodes
          npop(q_value * nelem)                                                                          //C DIM::npop returns the total number of populations, i.e. for D3Q19 19 times the number of nodes
    { }
    int nx, ny;
    size_t nelem, npop;                                                                             //C is a typedef for the size of an array
};

// Compute lattice-unit variables and discrete space and time step for a given lattice
// velocity (acoustic scaling).
auto lbParameters(double ulb, int lref, double Re) {
    double nu = ulb * static_cast<double>(lref) / Re;
    //double omega = 1. / (3. * nu + 0.5);                                //C does it need adaption to D2Q9?
    double dx = 1.;
    //double dt = dx * ulb;
    double omega = omega_glob;                  //C omega = 1.0 funktioniert ganz gut
    double dt = 1;
    return make_tuple(nu, omega, dx, dt);
}

// Print the simulation parameters to the terminal.
void printParameters(bool benchmark, double Re, double omega, double ulb, int N, double max_t, double dt) {            /// condition ? result1 : result2  /// If condition is true, the entire expression
                                                                                                            /// evaluates to result1, and otherwise to result2.
    cout << "2D Soot Combustion Testcase " << (benchmark ? "benchmark" : "production") << " mode" << endl;
    cout << "NX = " << N_X << "NY = " << N_Y << endl;
    cout << "Tau_flow = " << Tau_flow << endl;
    cout << "Tau_O2 = " << Tau_O2 << endl;
    cout << "Tau_CO2 = " << Tau_CO2 << endl;
    cout << "Tau_T_gas = " << Tau_T << endl;
    cout << "Tau_T_solid = " << Tau_T_solid << endl;
    cout << "u_inlet_max = " << u_lb << endl;
    cout << "dt = " << dt << endl;
    if (benchmark) {
        cout << "Now running " << bench_ini_iter << " warm-up iterations." << endl;
    }
    else {
        cout << "max_t = " << max_t << endl;
    }    
    printf("Tau_O2: %lf \t Tau:CO2: %lf \n",Tau_O2, Tau_CO2);
}

// Return a new clock for the current time, for benchmarking.
auto restartClock() {
    return make_pair(high_resolution_clock::now(), 0);
}

// Compute the time elapsed since a starting point, and the corresponding
// performance of the code in Mega Lattice site updates per second (MLups).
template<class TimePoint>           ///Function templates are special functions that can operate with generic types. This allows us to create a function 
                                    /// template whose functionality can be adapted to more than one type or class without repeating the entire code for each type.
void printMlups(TimePoint start, int clock_iter, size_t nelem) {
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double mlups = static_cast<double>(nelem * clock_iter) / duration.count();

    cout << "Benchmark result: " << setprecision(4) << mlups << " MLUPS" << endl;
}

// Instances of this class are function objects: the function-call operator executes a collision-streaming cycle.

/// CellData names the type of a single data element provided to the for_each algorithm, which must be chosen according to the data layout.
struct LBM {
    using CellData = double;        /// the name 'CellData' is now an alias for  double //C using is a type alias declaration -> can be used as synonym for the type, which is written
    static size_t sizeOfLattice(size_t nelem) { return 2 * q_value * nelem; }        //C function which returns the total number of entries needed for the whole lattice (with 19 velocities) and two populations
                                                                                //C size_t is an unsigned integer type
    /// A pointer to the full lattice is captured to allow access to the neighboring cells. In the case of the soa data alignment, the full lattice is also required to gather all populations of the current cell.
    CellData* lattice;
    CellType* flag;         /// enum class CellType : uint8_t { bounce_back, bulk }
    int* parity;    
    int delta_x;        
    std::array<int, 2>* c;  /// lattice velocities      //C has to be modified for D2Q9
    int* opp;
    double* S;
    double* S_GO2;
    double* S_GCO2;
    double* S_T;
    double* S_T_solid;
    double* t;              /// lattice weights
    std::array<int, 9>*  M;
    vector<double>*  M_inv;
    double omega;           /// relaxation parameter
    Dim dim;      
    double delta_t;          /// nx x ny x nz dimensions     //C from now on, it can be called as LBM.dim


    //void (*Boundary_Condition_Temperature)(int);
    // Convert linear index to Cartesian indices.               //C the whole thing depends on the order in which the directions are ordered in the 1d vector
    auto i_to_xyz (int i) {
        int iX = i / (dim.ny);    //C using type int cuts off the decimal points -> i / (ny*nz) results in the x position
        int remainder = i % (dim.ny);      //C remainder gives the position in the y-z-plane
        int iY = remainder;              //C dividing the remainder by nz and cutting off the decimal points results in the y position               //C NOTE: it starts counting from ZERO
                    
        return std::make_tuple(iX, iY);
    };

    // Convert Cartesian indices to linear index.
    size_t xyz_to_i (int x, int y) {
        return (y + dim.ny * x);                       //C self explanatory 
    };

    // The "f" operator always refers to the population with parity 0, in which
    // the momentum-exchange term is stored.
    double& f (int i, int k) {
        return lattice[k * dim.nelem + i];
    }

    // Get the pre-collision populations of the current time step (their location depends on parity).
    double& fin (int i, int k) {
        return lattice[*parity * dim.npop + k * dim.nelem + i];                     //C parity is the variable which states wether the current population is in the
    }     //C dim.npop is the ammount of memory needed for one population           //C first or second population of the two-population scheme
    
    // Get the post-collision, streamed populations at the end of the current step.
    double& fout (int i, int k) {
        return lattice[(1 - *parity) * dim.npop + k * dim.nelem + i];                           //C here it is 1-parity because the post-collision popilations are always stored
    }

    double& g_O2 (int i, int k) {
        return lattice[k * dim.nelem + i    +sizeOfLattice(dim.nelem)];
    }

    // Get the pre-collision populations of the current time step (their location depends on parity).
    double& gin_O2 (int i, int k) {
        return lattice[*parity * dim.npop + k * dim.nelem + i   +sizeOfLattice(dim.nelem)];                     //C parity is the variable which states wether the current population is in the
    }     //C dim.npop is the ammount of memory needed for one population           //C first or second population of the two-population scheme
    
    // Get the post-collision, streamed populations at the end of the current step.
    double& gout_O2 (int i, int k) {
        return lattice[(1 - *parity) * dim.npop + k * dim.nelem + i +sizeOfLattice(dim.nelem)];                           //C here it is 1-parity because the post-collision popilations are always stored
    }
    
    double& g_CO2 (int i, int k) {
        return lattice[k * dim.nelem + i    +2*sizeOfLattice(dim.nelem)];
    }

    // Get the pre-collision populations of the current time step (their location depends on parity).
    double& gin_CO2 (int i, int k) {
        return lattice[*parity * dim.npop + k * dim.nelem + i   +2*sizeOfLattice(dim.nelem)];                     //C parity is the variable which states wether the current population is in the
    }     //C dim.npop is the ammount of memory needed for one population           //C first or second population of the two-population scheme
    
    // Get the post-collision, streamed populations at the end of the current step.
    double& gout_CO2 (int i, int k) {
        return lattice[(1 - *parity) * dim.npop + k * dim.nelem + i +2*sizeOfLattice(dim.nelem)];                           //C here it is 1-parity because the post-collision popilations are always stored
    }


    double& g_T (int i, int k) {
        return lattice[k * dim.nelem + i    +3*sizeOfLattice(dim.nelem)];
    }

    // Get the pre-collision populations of the current time step (their location depends on parity).
    double& gin_T (int i, int k) {
        return lattice[*parity * dim.npop + k * dim.nelem + i   +3*sizeOfLattice(dim.nelem)];                     //C parity is the variable which states wether the current population is in the
    }     //C dim.npop is the ammount of memory needed for one population           //C first or second population of the two-population scheme
    
    // Get the post-collision, streamed populations at the end of the current step.
    double& gout_T (int i, int k) {
        return lattice[(1 - *parity) * dim.npop + k * dim.nelem + i +3*sizeOfLattice(dim.nelem)];                           //C here it is 1-parity because the post-collision popilations are always stored
    }
    

    //C i iterates through the nodes and k iterates through the directions  at one node   //C in the jeweils other population (parity), compared to the pre-coll-pops
    //C here k element of [1,19], i element of [0,dim.nelem]
    // Initialize the lattice with zero velocity and density 1.
    //C adaption to D2Q9 necessary 
    auto iniLattice (double& f0) {
        auto i = &f0 - lattice;         //C defines the index of the lattice node based on the allocated memory
        //cout << "i= " << i << "f0 wo=" << f0 << "f0 with" << &f0 << "lattice=" << lattice << endl;

        //C initialization of the flow field by readin in from file or initialyzing as density 1 and velocity 0
        if(Einleseoption_f){
            for(int k=0;k<q_value;k++){
                fin(i,k)  = f_input(i,k);
                fout(i,k) = f_input(i,k);
            }
        }
        else{
            for(int k=0;k<q_value;k++){
                fin(i, k)  = t[k]*rho_inlet;           //C lattice gets initiated with the values of the weighting function
                fout(i, k) = t[k]*rho_inlet;           //C lattice gets initiated with the values of the weighting function
            }
        }

        //C initialization of the temperature field and the mass fractions of the components
        for (int k = 0; k < q_value; ++k) {
            gin_O2(i,k)     = t[k]*Y_O2_ini;
            gin_CO2(i,k)    = t[k]*Y_O2_ini;
            gout_O2(i,k)    = t[k]*Y_CO2_ini;
            gout_CO2(i,k)   = t[k]*Y_CO2_ini;
            gin_T(i,k) =t[k]*T_ini; 
            gout_T(i,k)=t[k]*T_ini; 
        }

        auto[iX,iY] = i_to_xyz(i);


        
    };

    // Compute the macroscopic variables density and velocity on a cell.
    //C adaption to D2Q9 necessary
    auto macro (double& f0) {
        auto i = &f0 - lattice;             //C is used to iterate through the lattice
        double X_M1 = fin(i, 6) + fin(i, 3) + fin(i, 7);            /// in negative X direction //C lay in y-z-plane at the left side of the cube
        double X_P1 = fin(i, 5) + fin(i, 1) + fin(i, 8);       /// in positive X direction //C lay in y-z-plane at the right side of the cube
        double X_0  = fin(i, 2) + fin(i, 0)+ fin(i, 4);     /// with X=0 direction //C lay in y-z-plane in the middle of the cube

        double Y_M1 = fin(i, 7) + fin(i, 4) + fin(i, 8);
        double Y_P1 = fin(i, 6) + fin(i, 2) + fin(i, 5);


        double rho;
        rho = X_M1 + X_P1 + X_0;                                                                     //C sum over all f's

        std::array<double, 2> u{ (X_P1 - X_M1+efx/2.) / rho, (Y_P1 - Y_M1+efy/2.) / rho};           //C rho*u_vect=sum(f_i*e_i)
        
        return std::make_pair(rho, u);
    }

    auto macroKondition (double& f0) {
        auto i = &f0 - lattice;             //C is used to iterate through the lattice
        double cp_lok;
        double X_M1 = fin(i, 6) + fin(i, 3) + fin(i, 7);            /// in negative X direction //C lay in y-z-plane at the left side of the cube
        double X_P1 = fin(i, 5) + fin(i, 1) + fin(i, 8);       /// in positive X direction //C lay in y-z-plane at the right side of the cube
        double X_0  = fin(i, 2) + fin(i, 0)+ fin(i, 4);     /// with X=0 direction //C lay in y-z-plane in the middle of the cube

        double rho;
        if(flag[i] == CellType::bulk){
            rho = X_M1 + X_P1 + X_0;                                                                     //C sum over all f's
            cp_lok = cp_gas;
        }
        else if(flag[i] == CellType::reactive_obstacle){
            rho = rho_solid;                                                                     //C sum over all f's
            cp_lok = cp_solid;
        }


        return rho*cp_lok;
    }

    auto macroKonz (double& f0) {
        auto i = &f0 - lattice;            
        double YO2=0, YCO2=0;                                                                  

        for(int k=0;k<q_value; k++){
            YO2  = YO2  + gin_O2(i,k);
            YCO2 = YCO2 + gin_CO2(i,k);
        }        
        
        return std::make_pair(YO2,YCO2);
    }

    double macroTemp (double& f0){
        auto i = &f0 - lattice;
        double T=0;

        for(int k=0;k<q_value; k++){
            T  = T  + gin_T(i,k);
        }
        return T;
    }


    void external_forces (double& f0){
        int i = &f0 - lattice;

        vector <double> force (d_value);

        force[0] = dpdx;
        force[1] = dpdy;

        efx = force[0];
        efy = force[1];

        //return force;
    }

    auto Mass_fraction_forcing(int i){          //C forcing term incorporates the thermal expansion into the model


            auto [iX,iY] = i_to_xyz(i);

            vector <double> gradient_u    (d_value);
            vector <double> gradient_rho  (d_value);
            vector <double> gradient_YO2  (d_value);
            vector <double> gradient_YCO2 (d_value);

            vector<double> F_n (NC);
            
            auto[rho_i,     u_i]   = macro(lattice[i    ]);
            auto[YO2_i,  YCO2_i]   = macroKonz(lattice[i]);


                int Xe = iX+1;
                int Xw = iX-1;
                int Yn = iY+1;
                int Ys = iY-1;
                int i_nord      = xyz_to_i(iX, Yn);
                int i_ost       = xyz_to_i(Xe, iY);
                int i_sued       = xyz_to_i(iX, Ys);
                int i_west      = xyz_to_i(Xw, iY);

                auto[rho_nord,  u_nord]  = macro(lattice[i_nord]);
                auto[rho_sued,  u_sued]  = macro(lattice[i_sued]);
                auto[rho_ost,   u_ost]   = macro(lattice[i_ost ]);
                auto[rho_west,  u_west]  = macro(lattice[i_west]);


                auto[YO2_nord,  YCO2_nord]  = macroKonz(lattice[i_nord]);
                auto[YO2_sued,  YCO2_sued]  = macroKonz(lattice[i_sued]);
                auto[YO2_ost,   YCO2_ost]   = macroKonz(lattice[i_ost ]);
                auto[YO2_west,  YCO2_west]  = macroKonz(lattice[i_west]);


                gradient_rho[0] = (rho_ost - rho_west) / 2.;
                gradient_rho[1] = (rho_nord - rho_sued)/2.;

                gradient_u[0] = (u_ost[0]-u_west[0])/2.;                                  //C derivative of u_x by x and u_y by y
                gradient_u[1] = (u_nord[1]-u_sued[1])/2.; 

                gradient_YO2[0]=       (YO2_ost  -YO2_west)/2.;
                gradient_YO2[1]=       (YO2_nord -YO2_sued)/2.;
                gradient_YCO2[0]=      (YCO2_ost -YCO2_west)/2.;
                gradient_YCO2[1]=      (YCO2_nord-YCO2_sued)/2.;

                

            

                //C fluid node with wall to the south, gradient with forward difference scheme
                if(flag[i_sued] == CellType::bounce_back || flag[i_sued] == CellType::reactive_obstacle){
                    int i_nord_nord      = xyz_to_i(iX, Yn+1);
                    auto[rho_nord_nord,  u_nord_nord   ]  = macro(    lattice[i_nord_nord]);
                    auto[YO2_nord_nord,  YCO2_nord_nord]  = macroKonz(lattice[i_nord_nord]);
                    gradient_rho[1] = ( -3.*rho_i  +4*rho_nord  - rho_nord_nord  )/2.;
                    gradient_u[1]   = ( -3.*u_i[1] +4*u_nord[1] - u_nord_nord[1] )/2.;
                    gradient_YO2[1] = ( -3.*YO2_i  +4*YO2_nord  - YO2_nord_nord  )/2.;   
                    gradient_YCO2[1]= ( -3.*YCO2_i +4*YCO2_nord - YCO2_nord_nord )/2.;      
                }
                //C fluid node with wall to the north, gradient with backward difference scheme
                if(flag[i_nord] ==  CellType::bounce_back || flag[i_nord] == CellType::reactive_obstacle){
                    int i_sued_sued      = xyz_to_i(iX, Ys-1);
                    auto[rho_sued_sued,  u_sued_sued   ]  = macro(    lattice[i_sued_sued]);
                    auto[YO2_sued_sued,  YCO2_sued_sued]  = macroKonz(lattice[i_sued_sued]);

                    gradient_rho[1] = ( 3.*rho_i  -4*rho_sued  + rho_sued_sued  )/2.;
                    gradient_u[1]   = ( 3.*u_i[1] -4*u_sued[1] + u_sued_sued[1] )/2.;
                    gradient_YO2[1] = ( 3.*YO2_i  -4*YO2_sued  + YO2_sued_sued  )/2.;   
                    gradient_YCO2[1]= ( 3.*YCO2_i -4*YCO2_sued + YCO2_sued_sued )/2.; 
                }
                //C fluid node with wall to the east, gradient with forward difference scheme
                if(flag[i_west] ==  CellType::bounce_back || flag[i_west] == CellType::reactive_obstacle){
                    int i_ost_ost      = xyz_to_i(Xe+1, iY);
                    auto[rho_ost_ost,  u_ost_ost   ]  = macro(    lattice[i_ost_ost]);
                    auto[YO2_ost_ost,  YCO2_ost_ost]  = macroKonz(lattice[i_ost_ost]);

                    gradient_rho[0] = ( -3.*rho_i  +4*rho_ost  - rho_ost_ost  )/2.;
                    gradient_u[0]   = ( -3.*u_i[0] +4*u_ost[0] - u_ost_ost[0] )/2.;
                    gradient_YO2[0] = ( -3.*YO2_i  +4*YO2_ost  - YO2_ost_ost  )/2.;   
                    gradient_YCO2[0]= ( -3.*YCO2_i +4*YCO2_ost - YCO2_ost_ost )/2.; 
                }
                //C fluid node with wall to the west, gradient with backward difference scheme
                if(flag[i_ost] ==  CellType::bounce_back || flag[i_ost] == CellType::reactive_obstacle){
                    int i_west_west      = xyz_to_i(Xw-1, iY);
                    auto[rho_west_west,  u_west_west   ]  = macro(    lattice[i_west_west]);
                    auto[YO2_west_west,  YCO2_west_west]  = macroKonz(lattice[i_west_west]);

                    gradient_rho[0] = ( 3.*rho_i  -4*rho_west  + rho_west_west  )/2.;
                    gradient_u[0]   = ( 3.*u_i[0] -4*u_west[0] + u_west_west[0] )/2.;
                    gradient_YO2[0] = ( 3.*YO2_i  -4*YO2_west  + YO2_west_west  )/2.;   
                    gradient_YCO2[0]= ( 3.*YCO2_i -4*YCO2_west + YCO2_west_west )/2.; 
                }

                /* if(iX == dim.nx-1 && flag[i]==CellType::bulk){
                    int i_west_west      = xyz_to_i(Xw-1, iX);
                    auto[rho_west_west,  u_west_west   ]  = macro(    lattice[i_west_west]);
                    auto[YO2_west_west,  YCO2_west_west]  = macroKonz(lattice[i_west_west]);

                    gradient_rho[0] = ( 3.*rho_i  -4*rho_west  + rho_west_west  )/2.;
                    gradient_u[0]   = ( 3.*u_i[0] -4*u_west[0] + u_west_west[0] )/2.;
                    gradient_YO2[0] = ( 3.*YO2_i  -4*YO2_west  + YO2_west_west  )/2.;   
                    gradient_YCO2[0]= ( 3.*YCO2_i -4*YCO2_west + YCO2_west_west )/2.; 
                } */
                if(iX == dim.nx-1){
                    gradient_YO2[0] = 0.;
                    gradient_YCO2[0]= 0.;
                    gradient_u[0]   = 0.;
                } 


            //TODO Gradienten an den 1. Fluidnodes mit Vorwärts/Rückwärts berechnen
            
            //C gradient for lower wall


        /*             if(flag[i_nord] == CellType::bounce_back || flag[i_sued] == CellType::bounce_back){
                gradient_YO2[1] = 0.;
                gradient_YCO2[1]= 0.;
                gradient_u[0]   = 0.;
            }
            if(iX == dim.nx-1){
                gradient_YO2[0] = 0.;
                gradient_YCO2[0]= 0.;
                gradient_u[0]   = 0.;
            } */

            double F_O2  = (D_O2/rho_i) *(gradient_YO2[0]*gradient_rho[0]  + gradient_YO2[1] *gradient_rho[1])  + YO2_i *(gradient_u[0] +gradient_u[1]);
            double F_CO2 = (D_CO2/rho_i)*(gradient_YCO2[0]*gradient_rho[0] + gradient_YCO2[1]*gradient_rho[1])  + YCO2_i*(gradient_u[0] +gradient_u[1]);


           /*  if(iY == 1 || iY == dim.ny-2){
            printf("iX: %d \t iY: %d \t F_O2: %lf \t F_CO2: %lf \n", iX,iY,F_O2,F_CO2);
            printf("dRho(0): %lf \t dRho(1): %lf\n", gradient_rho[0], gradient_rho[1]);
            printf("dYO2(0): %lf \t dYO2(1): %lf\n", gradient_YO2[0], gradient_YO2[1]);
            printf("dYCO2(0): %lf \t dYCO2(1): %lf\n", gradient_YCO2[0], gradient_YCO2[1]);
            printf("du(0): %lf \t du(1): %lf\n", gradient_u[0], gradient_u[1]);


            } */
            //printf("dim.nx: %d \t dim.ny: %d \n",dim.nx,dim.ny);

            /* if(iX == 1 || iX == dim.nx-1){
                printf("iX: %d \t iY: %d \t F_O2: %lf \t F_CO2: %lf \n", iX,iY,F_O2,F_CO2);
                printf("dRho(0): %lf \t dRho(1): %lf\n", gradient_rho[0], gradient_rho[1]);
                printf("dYO2(0): %lf \t dYO2(1): %lf\n", gradient_YO2[0], gradient_YO2[1]);
                printf("dYCO2(0): %lf \t dYCO2(1): %lf\n", gradient_YCO2[0], gradient_YCO2[1]);
                printf("du(0): %lf \t du(1): %lf\n", gradient_u[0], gradient_u[1]);
            } */



            return make_tuple(F_O2, F_CO2);
    }

// not working
    auto Temperature_forcing(int i){          //C forcing term enforces correct temperature diffusion


        auto [iX,iY] = i_to_xyz(i);

        vector <double> gradient_u    (d_value);
        vector <double> gradient_rho  (d_value);
        vector <double> gradient_T  (d_value);

        vector<double> F_n (NC);
        
        auto[rho_i,u_i]   = macro(lattice[i    ]);
        double T_i        = macroTemp(lattice[i]);


            int Xe = iX+1;
            int Xw = iX-1;
            int Yn = iY+1;
            int Ys = iY-1;
            int i_nord      = xyz_to_i(iX, Yn);
            int i_ost       = xyz_to_i(Xe, iY);
            int i_sued       = xyz_to_i(iX, Ys);
            int i_west      = xyz_to_i(Xw, iY);

            auto[rho_nord,  u_nord]  = macro(lattice[i_nord]);
            auto[rho_sued,  u_sued]  = macro(lattice[i_sued]);
            auto[rho_ost,   u_ost]   = macro(lattice[i_ost ]);
            auto[rho_west,  u_west]  = macro(lattice[i_west]);


            double T_nord  = macroTemp(lattice[i_nord]);
            double T_sued  = macroTemp(lattice[i_sued]);
            double T_ost   = macroTemp(lattice[i_ost ]);
            double T_west  = macroTemp(lattice[i_west]);


            gradient_rho[0] = (rho_ost - rho_west) / 2.;
            gradient_rho[1] = (rho_nord - rho_sued)/2.;

            gradient_u[0] = (u_ost[0]-u_west[0])/2.;                                  //C derivative of u_x by x and u_y by y
            gradient_u[1] = (u_nord[1]-u_sued[1])/2.; 

            gradient_T[0]=       (T_ost  -T_west)/2.;
            gradient_T[1]=       (T_nord -T_sued)/2.;


        

            //C fluid node with wall to the south, gradient with forward difference scheme
            if(flag[i_sued] == CellType::bounce_back || flag[i_sued] == CellType::reactive_obstacle){
                int i_nord_nord      = xyz_to_i(iX, Yn+1);
                auto[rho_nord_nord,u_nord_nord]  = macro(    lattice[i_nord_nord]);
                double T_nord_nord               = macroTemp(lattice[i_nord_nord]);
                gradient_rho[1] = ( -3.*rho_i  +4*rho_nord  - rho_nord_nord  )/2.;
                gradient_u[1]   = ( -3.*u_i[1] +4*u_nord[1] - u_nord_nord[1] )/2.;
                gradient_T[1]   = ( -3.*T_i    +4*T_nord    - T_nord_nord    )/2.;   
            }
            //C fluid node with wall to the north, gradient with backward difference scheme
            if(flag[i_nord] ==  CellType::bounce_back || flag[i_nord] == CellType::reactive_obstacle){
                int i_sued_sued      = xyz_to_i(iX, Ys-1);
                auto[rho_sued_sued,u_sued_sued]  = macro(    lattice[i_sued_sued]);
                double T_sued_sued               = macroTemp(lattice[i_sued_sued]);

                gradient_rho[1] = ( 3.*rho_i  -4*rho_sued  + rho_sued_sued  )/2.;
                gradient_u[1]   = ( 3.*u_i[1] -4*u_sued[1] + u_sued_sued[1] )/2.;
                gradient_T[1]   = ( 3.*T_i    -4*T_sued    + T_sued_sued    )/2.;   
            }
            //C fluid node with wall to the east, gradient with forward difference scheme
            if(flag[i_west] ==  CellType::bounce_back || flag[i_west] == CellType::reactive_obstacle){
                int i_ost_ost      = xyz_to_i(Xe+1, iY);
                auto[rho_ost_ost,u_ost_ost]  = macro(    lattice[i_ost_ost]);
                double T_ost_ost             = macroTemp(lattice[i_ost_ost]);

                gradient_rho[0] = ( -3.*rho_i  +4*rho_ost  - rho_ost_ost  )/2.;
                gradient_u[0]   = ( -3.*u_i[0] +4*u_ost[0] - u_ost_ost[0] )/2.;
                gradient_T[0]   = ( -3.*T_i    +4*T_ost    - T_ost_ost    )/2.;   
            }
            //C fluid node with wall to the west, gradient with backward difference scheme
            if(flag[i_ost] ==  CellType::bounce_back || flag[i_ost] == CellType::reactive_obstacle){
                if(iX != dim.nx-1){
                    int i_west_west      = xyz_to_i(Xw-1, iY);
                    auto[rho_west_west,u_west_west]  = macro(    lattice[i_west_west]);
                    double T_west_west               = macroTemp(lattice[i_west_west]);

                    gradient_rho[0] = ( 3.*rho_i  -4*rho_west  + rho_west_west  )/2.;
                    gradient_u[0]   = ( 3.*u_i[0] -4*u_west[0] + u_west_west[0] )/2.;
                    gradient_T[0]   = ( 3.*T_i    -4*T_west    + T_west_west    )/2.;  
                }
            }

            if(iX == dim.nx-1){
            gradient_rho[0] = 0.;
            gradient_T[0] = 0.;
            gradient_u[0]   = 0.;
            } 
        double alpha = lambda_gas/(rho_i*cp_gas);
        //double F_Q1  = 0;     //TODO hier noch die Reaktionswärme implementieren
        double F_Q2  = (1./rho_i*cp_gas) *(  cp_gas*gradient_rho[0]*(alpha*gradient_T[0]-T_i*u_i[0]) + cp_gas*gradient_rho[1]*(alpha*gradient_T[1]-T_i*u_i[1])  )        -   (T_i/(rho_i*cp_gas))*((rho_i*cp_gas - rho_i_prev(i)*cp_gas))/delta_t        ;
        
        rho_i_prev(i) = rho_i;

        double F_T = /*F_Q1+*/  F_Q2;

        return F_T;
    }

    auto Temperature_forcing_neu(int i, double rho_i, std::array<double, 2> const& u_i , double T_i, double usqr, vector <double> g_eq_T_i){          //C forcing term enforces correct temperature diffusion; Correct implementation accoring to Karani (2015)


        auto [iX,iY] = i_to_xyz(i);

        vector <double> gradient_inv_rho_cp    (d_value,0.);           //C gradient of the product of the density and specific heat capacity
        vector <double> gradient_rho_cp    (d_value,0.);           //C gradient of the product of the density and specific heat capacity
        vector <double> gradient_T    (d_value,0.);           //C gradient of the product of the density and specific heat capacity
        vector <double> q_vector  (d_value);                    //C total heat flux (can be calculated locally in LBM)
        vector <double> u_lokal  (d_value,0.);                    //C total heat flux (can be calculated locally in LBM)

        double cp_local = 0., tau_T_local=0., rho_local = 0.;
        double cp_ost   = 0., rho_ost = 0.;
        double cp_west   = 0., rho_west = 0.;
        double cp_nord  = 0., rho_nord = 0.;        
        double cp_sued  = 0., rho_sued = 0.;
        double alpha_lokal = 0.;

        if(flag[i] == CellType::bulk){
            cp_local = cp_gas;
            tau_T_local = Tau_T;
            rho_local = rho_i;
            u_lokal[0] = u_i[0];
            u_lokal[1] = u_i[1];
            alpha_lokal = alpha_gas;
        }
        else if(flag[i] == CellType::reactive_obstacle){
            cp_local = cp_solid;
            tau_T_local = Tau_T_solid;            
            rho_local = rho_solid;
            u_lokal[0] = 0.;
            u_lokal[1] = 0.;
            alpha_lokal = alpha_solid;
        }

        int Xe = iX+1;
        int Xw = iX-1;
        int Yn = iY+1;
        int Ys = iY-1;
        int i_nord      = xyz_to_i(iX, Yn);
        int i_ost       = xyz_to_i(Xe, iY);
        int i_sued      = xyz_to_i(iX, Ys);
        int i_west      = xyz_to_i(Xw, iY);

        double T_ost  = macroTemp(lattice[i_ost]);
        double T_west = macroTemp(lattice[i_west]);
        double T_nord = macroTemp(lattice[i_nord]);
        double T_sued = macroTemp(lattice[i_sued]);

        gradient_T[0]=       (T_ost  -T_west)/2.;
        gradient_T[1]=       (T_nord -T_sued)/2.;



        q_vector[0] = rho_local * cp_local * (    (1-(1/(2*tau_T_local)))*(  gin_T(i,1)-g_eq_T_i[1] + gin_T(i,5)-g_eq_T_i[5] + gin_T(i,8)-g_eq_T_i[8] - (gin_T(i,3)-g_eq_T_i[3]) - (gin_T(i,6)-g_eq_T_i[6]) - (gin_T(i,7)-g_eq_T_i[7])  )       +u_lokal[0]*T_i );
        q_vector[1] = rho_local * cp_local * (    (1-(1/(2*tau_T_local)))*(  gin_T(i,2)-g_eq_T_i[2] + gin_T(i,5)-g_eq_T_i[5] + gin_T(i,6)-g_eq_T_i[6] - (gin_T(i,4)-g_eq_T_i[4]) - (gin_T(i,7)-g_eq_T_i[7]) - (gin_T(i,8)-g_eq_T_i[8])  )       +u_lokal[1]*T_i );
        
        //q_vector[0] = rho_local * cp_local * (  -alpha_lokal*gradient_T[0]        +u_lokal[0]*T_i );
        //q_vector[1] = rho_local * cp_local * (  -alpha_lokal*gradient_T[1]        +u_lokal[1]*T_i );

        //q_vector[0] = (1./(rho_local * cp_local)) * (  alpha_lokal*gradient_T[0]        -u_lokal[0]*T_i );
        //q_vector[1] = (1./(rho_local * cp_local)) * (  alpha_lokal*gradient_T[1]        -u_lokal[1]*T_i );


        gradient_rho_cp[0] = (macroKondition(lattice[i_ost]) -macroKondition(lattice[i_west]))/2.;
        gradient_rho_cp[1] = (macroKondition(lattice[i_nord])-macroKondition(lattice[i_sued]))/2.;
        

        
        if(flag[i]==CellType::bulk && flag[i_nord]==CellType::reactive_obstacle && flag[i_sued]!=CellType::reactive_obstacle){
            cp_nord = cp_solid;
            rho_nord = rho_solid;  

            double rho_cp_avg_nord = (rho_local*cp_local + rho_nord*cp_nord)/2.;
            gradient_inv_rho_cp[1] = (  (1/rho_cp_avg_nord)   -   (1/(rho_local*cp_local))  )  /(delta_x/2.);
        }
        else if(flag[i]==CellType::bulk && flag[i_sued]==CellType::reactive_obstacle && flag[i_nord]!=CellType::reactive_obstacle){
            cp_sued = cp_solid;
            rho_sued = rho_solid;  

            double rho_cp_avg_sued = (rho_local*cp_local + rho_sued*cp_sued)/2.;
            gradient_inv_rho_cp[1] = (  (1/(rho_local*cp_local))   -   (1/rho_cp_avg_sued)  )  /(delta_x/2.);
        }
        
        if(flag[i]==CellType::bulk && flag[i_ost]==CellType::reactive_obstacle && flag[i_west]!=CellType::reactive_obstacle){
            cp_ost = cp_solid;
            rho_ost = rho_solid;  

            double rho_cp_avg_ost = (rho_local*cp_local + rho_ost*cp_ost)/2.;
            gradient_inv_rho_cp[0] = (  (1/rho_cp_avg_ost)   -   (1/(rho_local*cp_local))  )  /(delta_x/2.);
        }
        else if(flag[i]==CellType::bulk && flag[i_west]==CellType::reactive_obstacle && flag[i_ost]!=CellType::reactive_obstacle){
            cp_west = cp_solid;
            rho_west = rho_solid;  

            double rho_cp_avg_west = (rho_local*cp_local + rho_west*cp_west)/2.;
            gradient_inv_rho_cp[0] = (  (1/(rho_local*cp_local))   -   (1/rho_cp_avg_west)  )  /(delta_x/2.);
        }
        

        //C set gradient of (rho*cp) to zero at the outside of the domain
            if(iX == dim.nx-2){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            }
            if(iY == dim.ny-2 || iY == 1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            }
            if(iX == 1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            } 
        

        double F_T = q_vector[0]*gradient_inv_rho_cp[0] + q_vector[1]*gradient_inv_rho_cp[1];

        return F_T;
    }

    auto Temperature_forcing_2(int i, double rho_i, std::array<double, 2> const& u_i , double T_i, double usqr, vector <double> g_eq_T_i){          //C forcing term enforces correct temperature diffusion; Correct implementation accoring to Karani (2015)


        auto [iX,iY] = i_to_xyz(i);

        vector <double> gradient_inv_rho_cp    (d_value);           //C gradient of the product of the density and specific heat capacity
        vector <double> q_vector  (d_value);                    //C total heat flux (can be calculated locally in LBM)
        vector <double> u_lokal  (d_value,0.);                    //C total heat flux (can be calculated locally in LBM)

        double cp_local = 0., tau_T_local=0., rho_local = 0.;
        double cp_ost   = 0., rho_ost = 0.;
        double cp_west   = 0., rho_west = 0.;
        double cp_nord  = 0., rho_nord = 0.;        
        double cp_sued  = 0., rho_sued = 0.;

        if(flag[i] == CellType::bulk){
            cp_local = cp_gas;
            tau_T_local = Tau_T;
            rho_local = rho_gas;
            u_lokal[0] = u_i[0];
            u_lokal[1] = u_i[1];
        }
        else if(flag[i] == CellType::reactive_obstacle){
            cp_local = cp_solid;
            tau_T_local = Tau_T_solid;            
            rho_local = rho_solid;
            u_lokal[0] = 0.;
            u_lokal[1] = 0.;
        }

        q_vector[0] = rho_local * cp_local * (    (1-(1/(2*tau_T_local)))*(  gin_T(i,1)-g_eq_T_i[1] + gin_T(i,5)-g_eq_T_i[5] + gin_T(i,8)-g_eq_T_i[8] - (gin_T(i,3)-g_eq_T_i[3]) - (gin_T(i,6)-g_eq_T_i[6]) - (gin_T(i,7)-g_eq_T_i[7])  )       +u_lokal[0]*T_i );
        q_vector[1] = rho_local * cp_local * (    (1-(1/(2*tau_T_local)))*(  gin_T(i,2)-g_eq_T_i[2] + gin_T(i,5)-g_eq_T_i[5] + gin_T(i,6)-g_eq_T_i[6] - (gin_T(i,4)-g_eq_T_i[4]) - (gin_T(i,7)-g_eq_T_i[7]) - (gin_T(i,8)-g_eq_T_i[8])  )       +u_lokal[1]*T_i );



            int Xe = iX+1;
            int Xw = iX-1;
            int Yn = iY+1;
            int Ys = iY-1;
            int i_nord      = xyz_to_i(iX, Yn);
            int i_ost       = xyz_to_i(Xe, iY);
            int i_sued      = xyz_to_i(iX, Ys);
            int i_west      = xyz_to_i(Xw, iY);

            if(flag[i_nord] == CellType::bulk){
                cp_nord = cp_gas;
                rho_nord = rho_gas;  
            }
            else if(flag[i_nord] == CellType::reactive_obstacle){
                cp_nord = cp_solid;
                rho_nord = rho_solid;  
            }
            if(flag[i_ost] == CellType::bulk){
                cp_ost = cp_gas;
                rho_ost = rho_gas;  
            }
            else if(flag[i_ost] == CellType::reactive_obstacle){
                cp_ost = cp_solid;
                rho_ost = rho_solid;  
            }
            if(flag[i_sued] == CellType::bulk){
                cp_sued = cp_gas;
                rho_sued = rho_gas;  
            }
            else if(flag[i_sued] == CellType::reactive_obstacle){
                cp_sued = cp_solid;
                rho_sued = rho_solid;  
            }
            if(flag[i_west] == CellType::bulk){
                cp_west = cp_gas;
                rho_west = rho_gas;  
            }
            else if(flag[i_west] == CellType::reactive_obstacle){
                cp_west = cp_solid;
                rho_west = rho_solid;  
            }




            double rho_cp_avg_nord = (rho_local*cp_local + rho_nord*cp_nord)/2.;
            double rho_cp_avg_ost  = (rho_local*cp_local + rho_ost*cp_ost)/2.;
            double rho_cp_avg_sued = (rho_local*cp_local + rho_sued*cp_sued)/2.;
            double rho_cp_avg_west  = (rho_local*cp_local + rho_west*cp_west)/2.;
            if(Vorwaertsdiff){
                gradient_inv_rho_cp[0] = (  (1/rho_cp_avg_ost)   -   (1/(rho_local*cp_local))  )  /(delta_x/2.);
                gradient_inv_rho_cp[1] = (  (1/rho_cp_avg_nord)  -   (1/(rho_local*cp_local))  )  /(delta_x/2.);
            }            
            if(Zentraldiff){
                gradient_inv_rho_cp[0] = (  (1/rho_cp_avg_ost)    -   (1/rho_cp_avg_west)  )  /(delta_x);
                gradient_inv_rho_cp[1] = (  (1/rho_cp_avg_nord)  -   (1/rho_cp_avg_sued)  )  /(delta_x);
            }




        //C set gradient of (rho*cp) to zero at the outside of the domain
            if(iX == dim.nx-2){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            }
            if(iY == dim.ny-2 || iY == 1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            }
            if(iX == 1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            } 
        

        double F_T = q_vector[0]*gradient_inv_rho_cp[0] + q_vector[1]*gradient_inv_rho_cp[1];

        return F_T;
    }

    auto Temperature_forcing_solid(int i){          //C forcing term incorporates the thermal expansion into the model


        auto [iX,iY] = i_to_xyz(i);

        vector <double> gradient_T  (d_value);

        vector<double> F_n (NC);
        
        double T_i        = macroTemp(lattice[i]);


            int Xe = iX+1;
            int Xw = iX-1;
            int Yn = iY+1;
            int Ys = iY-1;
            int i_nord      = xyz_to_i(iX, Yn);
            int i_ost       = xyz_to_i(Xe, iY);
            int i_sued       = xyz_to_i(iX, Ys);
            int i_west      = xyz_to_i(Xw, iY);

            double T_nord  = macroTemp(lattice[i_nord]);
            double T_sued  = macroTemp(lattice[i_sued]);
            double T_ost   = macroTemp(lattice[i_ost ]);
            double T_west  = macroTemp(lattice[i_west]);


            gradient_T[0]=       (T_ost  -T_west)/2.;
            gradient_T[1]=       (T_nord -T_sued)/2.;


        

            //C fluid node with wall to the south, gradient with forward difference scheme
            if(flag[i_sued] == CellType::bounce_back || flag[i_sued] == CellType::reactive_obstacle){
                int i_nord_nord      = xyz_to_i(iX, Yn+1);
                double T_nord_nord               = macroTemp(lattice[i_nord_nord]);

                gradient_T[1]   = ( -3.*T_i    +4*T_nord    - T_nord_nord    )/2.;   
            }
            //C fluid node with wall to the north, gradient with backward difference scheme
            if(flag[i_nord] ==  CellType::bounce_back || flag[i_nord] == CellType::reactive_obstacle){
                int i_sued_sued      = xyz_to_i(iX, Ys-1);
                double T_sued_sued               = macroTemp(lattice[i_sued_sued]);

                gradient_T[1]   = ( 3.*T_i    -4*T_sued    + T_sued_sued    )/2.;   
            }
            //C fluid node with wall to the east, gradient with forward difference scheme
            if(flag[i_west] ==  CellType::bounce_back || flag[i_west] == CellType::reactive_obstacle){
                int i_ost_ost      = xyz_to_i(Xe+1, iY);
                double T_ost_ost             = macroTemp(lattice[i_ost_ost]);

                gradient_T[0]   = ( -3.*T_i    +4*T_ost    - T_ost_ost    )/2.;   
            }
            //C fluid node with wall to the west, gradient with backward difference scheme
            if(flag[i_ost] ==  CellType::bounce_back || flag[i_ost] == CellType::reactive_obstacle){
                if(iX != dim.nx-1){
                    int i_west_west      = xyz_to_i(Xw-1, iY);
                    double T_west_west               = macroTemp(lattice[i_west_west]);

                    gradient_T[0]   = ( 3.*T_i    -4*T_west    + T_west_west    )/2.;  
                }
            }

            if(iX == dim.nx-1){
            gradient_T[0] = 0.;
            } 
        //double alpha = alpha_solid;
        double F_Q1  = 0;     //TODO hier noch die Reaktionswärme implementieren
        double F_Q2  =  -(T_i/(rho_solid*cp_solid))*((rho_solid*cp_solid - rho_solid*cp_solid))/delta_t;          //C (1./rho_i*cp_gas) *(  cp_gas*gradient_rho[0]*(alpha*gradient_T[0]) + cp_gas*gradient_rho[1]*(alpha*gradient_T[1])  )
        
        //rho_i_prev(i) = rho_i;

        double F_T = F_Q1 + F_Q2;

        return F_T;
    }

    auto Temperature_forcing_solid_neu(int i, double T_i, vector <double> g_eq_T_i){          //C forcing term enforces correct temperature diffusion; Correct implementation accoring to Karani (2015)


        auto [iX,iY] = i_to_xyz(i);

        vector <double> gradient_inv_rho_cp    (d_value);           //C gradient of the product of the density and specific heat capacity
        vector <double> q_vector  (d_value);                    //C total heat flux (can be calculated locally in LBM)
        vector <double> u_i  (d_value,0.);                    //C total heat flux (can be calculated locally in LBM)
        double cp_local = 0., tau_T_local=0., rho_local = 0.;
        double cp_ost   = 0., rho_ost = 0.;
        double cp_west   = 0., rho_west = 0.;
        double cp_nord  = 0., rho_nord = 0.;        
        double cp_sued  = 0., rho_sued = 0.;

            cp_local = cp_solid;
            tau_T_local = Tau_T_solid;            
            rho_local = rho_solid;

        q_vector[0] = rho_local * cp_local * (    (1-(1/(2*tau_T_local)))*(  gin_T(i,1)-g_eq_T_i[1] + gin_T(i,5)-g_eq_T_i[5] + gin_T(i,8)-g_eq_T_i[8] - (gin_T(i,3)-g_eq_T_i[3]) - (gin_T(i,6)-g_eq_T_i[6]) - (gin_T(i,7)-g_eq_T_i[7])  )       +u_i[0]*T_i );
        q_vector[1] = rho_local * cp_local * (    (1-(1/(2*tau_T_local)))*(  gin_T(i,2)-g_eq_T_i[2] + gin_T(i,5)-g_eq_T_i[5] + gin_T(i,6)-g_eq_T_i[6] - (gin_T(i,4)-g_eq_T_i[4]) - (gin_T(i,7)-g_eq_T_i[7]) - (gin_T(i,8)-g_eq_T_i[8])  )       +u_i[1]*T_i );



            int Xe = iX+1;
            int Xw = iX-1;
            int Yn = iY+1;
            int Ys = iY-1;
            int i_nord      = xyz_to_i(iX, Yn);
            int i_ost       = xyz_to_i(Xe, iY);
            int i_sued      = xyz_to_i(iX, Ys);
            int i_west      = xyz_to_i(Xw, iY);

        if(flag[i]==CellType::reactive_obstacle && flag[i_nord]==CellType::bulk && flag[i_sued]==CellType::reactive_obstacle){
            cp_nord = cp_gas;
            auto[rho_i,u_lok] = macro(lattice[i_nord]);
            rho_nord = rho_i;  

            double rho_cp_avg_nord = (rho_local*cp_local + rho_nord*cp_nord)/2.;
            gradient_inv_rho_cp[1] = (  (1/rho_cp_avg_nord)   -   (1/(rho_local*cp_local))  )  /(delta_x/2.);
        }
        else if(flag[i]==CellType::reactive_obstacle && flag[i_sued]==CellType::bulk && flag[i_nord]==CellType::reactive_obstacle){
            cp_sued = cp_gas;
            auto[rho_i,u_lok] = macro(lattice[i_sued]);
            rho_sued = rho_i;  

            double rho_cp_avg_sued = (rho_local*cp_local + rho_sued*cp_sued)/2.;
            gradient_inv_rho_cp[1] = (  (1/(rho_local*cp_local))   -   (1/rho_cp_avg_sued)  )  /(delta_x/2.);
        }
        
        if(flag[i]==CellType::reactive_obstacle && flag[i_ost]==CellType::bulk && flag[i_west]==CellType::reactive_obstacle){
            cp_ost = cp_gas;
            auto[rho_i,u_lok] = macro(lattice[i_ost]);
            rho_ost = rho_i;  

            double rho_cp_avg_ost = (rho_local*cp_local + rho_ost*cp_ost)/2.;
            gradient_inv_rho_cp[0] = (  (1/rho_cp_avg_ost)   -   (1/(rho_local*cp_local))  )  /(delta_x/2.);
        }
        else if(flag[i]==CellType::reactive_obstacle && flag[i_west]==CellType::bulk && flag[i_ost]==CellType::reactive_obstacle){
            cp_west = cp_gas;
            auto[rho_i,u_lok] = macro(lattice[i_west]);
            rho_west = rho_i;  

            double rho_cp_avg_west = (rho_local*cp_local + rho_west*cp_west)/2.;
            gradient_inv_rho_cp[0] = (  (1/(rho_local*cp_local))   -   (1/rho_cp_avg_west)  )  /(delta_x/2.);
        }


        //C set gradient of (rho*cp) to zero at the outside of the domain
            if(iX == dim.nx-1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            }
            if(iY == dim.ny-2 || iY == 1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            }
            if(iX == 1){
                gradient_inv_rho_cp[0] = 0.;
                gradient_inv_rho_cp[1] = 0.;
            } 
        

        double F_T = q_vector[0]*gradient_inv_rho_cp[0] + q_vector[1]*gradient_inv_rho_cp[1];

        return F_T;
    }

    auto Q_xy(int i){
        double Qx=0;
        double Qy=0;

        auto[Y_O2, Y_CO2]   = macroKonz(lattice[i]);
        double T_i          = macroTemp(lattice[i]);
        auto[rho,u]         = macro(lattice[i]);


        //double r_mass = 0;      //C mass specific ideal gas constant
        //r_mass = R_id* ( (Y_O2/M_O2) + (Y_CO2/M_CO2) );                   //TODO hier nochmal nachprüfen, ob man das wirklich so machen kann bzw. was der Unterschied ist (hinsichtlich der Zahlenwerte)
        //double Teta = (r_mass*T_i)/(CS2);
        double Teta = T_i / T_ref;


        Qx = rho*u[0]*(1-Teta-u[0]*u[0]);
        Qy = rho*u[1]*(1-Teta-u[1]*u[1]);


        return std::make_pair(Qx,Qy);
    }

    void Specular(int i, int k, double pop_out){

        auto[iX,iY] = i_to_xyz(i);

        if(iY == dim.ny-2){
            if(k==2){
                fout(i,opp[k]) = pop_out;

            }
            else if(k==5){
                int i_out = xyz_to_i(iX+1,iY);
                fout(i_out,8) = pop_out;

            }
            else if(k== 6){
                int i_out = xyz_to_i(iX-1,iY);
                fout(i_out,7) = pop_out;
            }
        }
        else if(iY == 1){
            if(k==4){
                fout(i,opp[k]) = pop_out;
            }
            else if(k==8){
                int i_out = xyz_to_i(iX+1,iY);
                fout(i_out,5) = pop_out;
            }
            else if(k== 7){
                int i_out = xyz_to_i(iX-1,iY);
                fout(i_out,6) = pop_out;
            }

        }
        else{
            printf("hallo1234\n");
        }


    }

    
    auto get_normal_vect( int i, int k, int iX, int iY){          //C works, is tested 
            vector<int>z(q_value,0.); 
            vector<double>n(d_value,0.); 
            int erg = 0, laufw=0;

            for(int k_lok=0; k_lok<q_value; k_lok++){
                int XX = iX + c[k_lok][0];                          //C position of the current step iX is being altered by the direction c
                int YY = iY + c[k_lok][1];
                size_t nb = xyz_to_i(XX, YY); 
                if(flag[nb]==CellType::reactive_obstacle){
                    z[laufw]=k_lok;
                    laufw +=1;
                }
            }
            for(int k_lok=0;k_lok<q_value;k_lok++){               
                erg +=z[k_lok];
            }


            if(k==1){
                n[0]=-1;       //c[opp[0]][0];
                n[1]=0;  
            }
            else if(k==2){
                n[0]=0;       //c[opp[0]][0];
                n[1]=-1;  
            }
            else if(k==3){
                n[0]=1;       //c[opp[0]][0];
                n[1]=0;  
            }
            else if(k==4){
                n[0]=0;       //c[opp[0]][0];
                n[1]=1;  
            }
            else if(k==5){
                if(erg==14){
                    n[0]=-1;       
                    n[1]=0;  
                }
                else if(erg==13){
                    n[0]=0;      
                    n[1]=-1;  
                }
                else if(erg==23){
                    n[0]=0;      
                    n[1]=-1;  
                }
                else if(erg==22){
                    n[0]=-1;       
                    n[1]=-1;  
                }
                else if(erg==25){
                    n[0]=-1;      
                    n[1]=0;  
                }
                else if(erg==5){
                    n[0]=-1;       
                    n[1]=-1;  
                }
                else if(erg==8){
                    n[0]=-1;       
                    n[1]=-1;  
                }
                else if(erg==20){
                    n[0]=-1;       
                    n[1]=-1;  
                }
                else if(erg==17){
                    n[0]=-1;      
                    n[1]=-1;  
                }
                else if(erg==33){
                    n[0]=-1;      
                    n[1]=-1;  
                }
                else if(erg==35){
                    n[0]=0;       
                    n[1]=-1;  
                }
                else if(erg==32){
                    n[0]=-1;       
                    n[1]=-1;  
                }
                else if(erg==34){
                    n[0]=-1;       
                    n[1]=0;  
                }
            }
            else if(k==6){
                int desx = iX + c[1][0];                          //C position of the current step iX is being altered by the direction c
                int desy = iY + c[1][1];
                size_t des = xyz_to_i(desx, desy); 
                if(erg==16){
                    n[0]=1;       
                    n[1]=0;  
                }
                else if(erg==13){
                    n[0]=0;      
                    n[1]=-1;  
                }
                else if(erg==23){
                    n[0]=1;      
                    n[1]=-1;  
                }
                else if(erg==22 && flag[des]==CellType::reactive_obstacle){       
                    n[0]=0;       
                    n[1]=-1;  
                }
                else if(erg==28){
                    n[0]=1;      
                    n[1]=0;  
                }
                else if(erg==6){
                    n[0]=1;       
                    n[1]=-1;  
                }
                else if(erg==11){
                    n[0]=1;       
                    n[1]=-1;  
                }
                else if(erg==22 && flag[des]!=CellType::reactive_obstacle){           
                    n[0]=1;       
                    n[1]=-1;  
                }
                else if(erg==17){
                    n[0]=1;      
                    n[1]=-1;  
                }
                else if(erg==33){
                    n[0]=0;      
                    n[1]=-1;  
                }
                else if(erg==35){
                    n[0]=1;       
                    n[1]=-1;  
                }
                else if(erg==32){
                    n[0]=1;       
                    n[1]=-1;  
                }
                else if(erg==34){
                    n[0]=1;       
                    n[1]=0;  
                }
            }
            else if(k==7){
                int desx = iX + c[2][0];                          //C position of the current step iX is being altered by the direction c
                int desy = iY + c[2][1];
                size_t des = xyz_to_i(desx, desy); 
                if(erg==19){
                    n[0]=0;       
                    n[1]=1;  
                }
                else if(erg==16){
                    n[0]=1;      
                    n[1]=0;  
                }
                else if(erg==23 && flag[des] == CellType::reactive_obstacle){
                    n[0]=1;      
                    n[1]=0;  
                }
                else if(erg==28){
                    n[0]=1;       
                    n[1]=1;  
                }
                else if(erg==25){
                    n[0]=0;      
                    n[1]=1;  
                }
                else if(erg==7){
                    n[0]=1;       
                    n[1]=1;  
                }
                else if(erg==14){
                    n[0]=1;       
                    n[1]=1;  
                }
                else if(erg==22){
                    n[0]=1;       
                    n[1]=1;  
                }
                else if(erg==23 && flag[des] != CellType::reactive_obstacle){
                    n[0]=1;      
                    n[1]=1;  
                }
                else if(erg==33){
                    n[0]=0;      
                    n[1]=1;  
                }
                else if(erg==35){
                    n[0]=1;       
                    n[1]=1;  
                }
                else if(erg==32){
                    n[0]=1;       
                    n[1]=0;  
                }
                else if(erg==34){
                    n[0]=1;       
                    n[1]=1;  
                }
            }
            else if(k==8){
                if(erg==14){
                    n[0]=-1;       
                    n[1]=0;  
                }
                else if(erg==19){
                    n[0]=0;      
                    n[1]=1;  
                }
                else if(erg==22){
                    n[0]=-1;      
                    n[1]=0;  
                }
                else if(erg==28){
                    n[0]=0;       
                    n[1]=1;  
                }
                else if(erg==25){
                    n[0]=-1;      
                    n[1]=1;  
                }
                else if(erg==8){
                    n[0]=-1;       
                    n[1]=1;  
                }
                else if(erg==13){
                    n[0]=-1;       
                    n[1]=1;  
                }
                else if(erg==20){
                    n[0]=-1;       
                    n[1]=1;  
                }
                else if(erg==23){
                    n[0]=-1;      
                    n[1]=1;  
                }
                else if(erg==33){
                    n[0]=-1;      
                    n[1]=1;  
                }
                else if(erg==35){
                    n[0]=0;       
                    n[1]=1;  
                }
                else if(erg==32){
                    n[0]=-1;       
                    n[1]=0;  
                }
                else if(erg==34){
                    n[0]=-1;       
                    n[1]=1;  
                }
            }

            return n;
    }


    auto Reaction_Interface(int i, int k, int iX, int iY, double T_lok, double Y_O2_lok, double Y_CO2_lok){

        double Reaktionsterm=0;

        int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
        int YY = iY + c[k][1];

        int i_nb = xyz_to_i(XX,YY);

        double Y_O2_interface=0., Y_CO2_interface=0.;

        auto n_vect = get_normal_vect(i,k,iX,iY);
        double ck_n = c[k][0]*n_vect[0] + c[k][1]*n_vect[1];

        Y_O2_interface  = (D_O2*Y_O2_lok)   /   (D_O2 + (0.5*ck_n*delta_x*stoich_O2*Prae_exp_factor*exp((-E_akt)/(R_id*T_lok)) ) );
        Y_CO2_interface = Y_CO2_lok - (  0.5*ck_n*delta_x*Prae_exp_factor*exp((-E_akt)/(R_id*T_lok))*Y_O2_interface*M_CO2  )/(D_CO2*M_O2);
        
        return std::make_pair(Y_O2_interface, Y_CO2_interface);
    }



    int Vorzeichen(double expression){
        int vz = 0.;

        if(expression > 0){
            vz = 1.;
        }
        if(expression < 0){
            vz = -1.;
        }
        if(expression==0){
            vz = 0.;
        }

    return vz;
    }

    // Execute the streaming step on a cell.
    /// popk = fk       on fluid cells in pre-collision state (except for the AA-pattern)
    /// popk = −6 * tk * ρw * (ck · uw)       on solid cells (Momentum Exchange for Bounce Back Moving Walls)

    /// The following notation is used: pop denotes the 19 variables of the local cell, at position x, and nb_pop the variables 
    /// of the neighbor cell in direction k, at position x + ck (e.g. nb_popk- denotes the variable of index k- of the cell at position x + ck).

    //C this function streams the population from the position i in direction k, it gets called k times in the function operator()
    void stream (int i, int k, int iX, int iY, double pop_out) {
        int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
        int YY = iY + c[k][1];
        if(x_periodic){
            if (iX == 0 || iX == dim.nx-1) {
                XX = (XX + dim.nx) % dim.nx;
            }
        }
        if(y_periodic){
            if(iY == 0 || iY == dim.ny-1){
                YY = (YY + dim.ny) % dim.ny;
            }
        }

        if(x_inflow_outflow && iX == dim.nx-1){
            if(k==5 || k==1 || k==8){
            }
            else{
                size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
                if (flag[nb] == CellType::bounce_back || flag[nb] == CellType::reactive_obstacle /* || flag[nb]== CellType::periodic */ ) {        //C if it is a solid cell -> wall interaction via function f
                    fout(i, opp[k]) = pop_out + f(nb, k);       //C here is the implementation of the bounce back boundary condition (switching of the f's)
                }                                               //C and adding the momentum exchange term f, in case the boundary is moving
                else if(flag[nb] == CellType::specular_reflection){
                    Specular(i,k,pop_out);
                }
                else {
                    fout(nb, k)     = pop_out;              //C pop_out = (1. - omega) * fin(i, k) + omega * eq;
                }
            }
        }
        else{
            //printf("Test123 iX: %d \t iY: %d \t XX:%d \t YY: %d \n", iX,iY,XX,YY);
            size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
            if (flag[nb] == CellType::bounce_back || flag[nb] == CellType::reactive_obstacle /*|| flag[nb]== CellType::periodic*/ ) {        //C if it is a solid cell -> wall interaction via function f
                fout(i, opp[k]) = pop_out + f(nb, k);       //C here is the implementation of the bounce back boundary condition (switching of the f's)
            }                                               //C and adding the momentum exchange term f, in case the boundary is moving
            else if(flag[nb] == CellType::specular_reflection){
                //printf("specular iX: %d \t iY: %d \t XX:%d \t YY: %d \n", iX,iY,XX,YY);
                Specular(i,k,pop_out);

            }
            else {
                fout(nb, k)     = pop_out;              //C pop_out = (1. - omega) * fin(i, k) + omega * eq;
            }
        }
    };

    void streamFlow (int i, int k, int iX, int iY, double pop_out) {
        int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
        int YY = iY + c[k][1];
        if(x_periodic){
            if (iX == 0 || iX == dim.nx-1) {
                XX = (XX + dim.nx) % dim.nx;
            }
        }
        if(y_periodic){
            if(iY == 0 || iY == dim.ny-1){
                YY = (YY + dim.ny) % dim.ny;
            }
        }

        if(x_inflow_outflow && iX == dim.nx-1){
            if(k==5 || k==1 || k==8){
            }
            else{
                size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
                if (flag[nb] == CellType::bounce_back  || flag[nb] == CellType::reactive_obstacle /* || flag[nb]== CellType::periodic */ ) {        //C if it is a solid cell -> wall interaction via function f
                    fout(i, opp[k]) = pop_out + f(nb, k);       //C here is the implementation of the bounce back boundary condition (switching of the f's)
                }                                               //C and adding the momentum exchange term f, in case the boundary is moving
                else if(flag[nb] == CellType::specular_reflection){
                    Specular(i,k,pop_out);
                }
                else {
                    fout(nb, k)     = pop_out;              //C pop_out = (1. - omega) * fin(i, k) + omega * eq;
                }
            }
        }
        else{
            size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
            if (flag[nb] == CellType::bounce_back || flag[nb] == CellType::reactive_obstacle /*|| flag[nb]== CellType::periodic*/ ) {        //C if it is a solid cell -> wall interaction via function f
                fout(i, opp[k]) = pop_out + f(nb, k);       //C here is the implementation of the bounce back boundary condition (switching of the f's)
            }                                               //C and adding the momentum exchange term f, in case the boundary is moving
            else if(flag[nb] == CellType::specular_reflection){
                Specular(i,k,pop_out);
            }
            else {
                fout(nb, k)     = pop_out;              //C pop_out = (1. - omega) * fin(i, k) + omega * eq;
            }
        }
    };

    auto streamKonz (int i, int k, int iX, int iY,double pop_out_g_O2, double pop_out_g_CO2, double T, double Y_O2, double Y_CO2) {
        double Y_O2_int = 0, Y_CO2_int = 0;
        
        int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
        int YY = iY + c[k][1];
        if(x_periodic){
            if (iX == 0 || iX == dim.nx-1) {
                XX = (XX + dim.nx) % dim.nx;
            }
        }
        if(y_periodic){
            if(iY == 0 || iY == dim.ny-1){
                YY = (YY + dim.ny) % dim.ny;
            }
        }

        if(x_inflow_outflow && iX == dim.nx-1){         //C boundary condition am rechten Ende der Domain
            if(k==5 || k==1 || k==8){
            }
            else{
                size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
                if (flag[nb] == CellType::bounce_back /* || flag[nb]== CellType::periodic */ ) {        //C if it is a solid cell -> wall interaction via function f
                    gout_O2(i,  opp[k]) = pop_out_g_O2 + g_O2(nb,k);
                    gout_CO2(i, opp[k]) = pop_out_g_CO2+ g_CO2(nb,k);
                }
                else if(flag[nb] == CellType::reactive_obstacle && Solid_reaction){
                    //TODO 20.12.2022 hier weitermachen mit option reaction/no reaction
                    auto [Y_O2_interface, Y_CO2_interface] = Reaction_Interface(i,k,iX,iY,T,Y_O2,Y_CO2);

                    gout_O2(i,  opp[k]) = -pop_out_g_O2   + 2*t[k]*Y_O2_interface;
                    gout_CO2(i, opp[k]) = -pop_out_g_CO2  + 2*t[k]*Y_CO2_interface;
                    Y_O2_int  = Y_O2_interface;
                    Y_CO2_int = Y_CO2_interface;
                }                                             //C and adding the momentum exchange term f, in case the boundary is moving
                else {
                    gout_O2(nb, k)  = pop_out_g_O2;
                    gout_CO2(nb, k) = pop_out_g_CO2;
                }
            }
        }
        else{
            size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
            if (flag[nb] == CellType::bounce_back /*|| flag[nb]== CellType::periodic*/ ) {        //C if it is a solid cell -> wall interaction via function f
                gout_O2(i, opp[k])  = pop_out_g_O2+ g_O2(nb,k);
                gout_CO2(i, opp[k]) = pop_out_g_CO2+ g_CO2(nb,k);
            }
            else if(flag[nb] == CellType::reactive_obstacle){           //C anti-bounce-back implementation of the component-sink caused by reaction
                if(Solid_reaction){
                    auto [Y_O2_interface, Y_CO2_interface] = Reaction_Interface(i,k,iX,iY,T,Y_O2,Y_CO2);

                    gout_O2(i,  opp[k]) = -pop_out_g_O2   + 2*t[k]*Y_O2_interface;
                    gout_CO2(i, opp[k]) = -pop_out_g_CO2  + 2*t[k]*Y_CO2_interface;
                    Y_O2_int  = Y_O2_interface;
                }
                else{
                    gout_O2(i, opp[k])  = pop_out_g_O2 + g_O2(nb,k);
                    gout_CO2(i, opp[k]) = pop_out_g_CO2 + g_CO2(nb,k);
                }
            }                                             
            else {
                gout_O2(nb, k)  = pop_out_g_O2;
                gout_CO2(nb, k) = pop_out_g_CO2;
            }
        }
    
    
    return Y_O2_int;
    };

    void streamTemp (int i, int k, int iX, int iY, double pop_out_T, double rho) {
        int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
        int YY = iY + c[k][1];
        if(x_periodic){
            if (iX == 0 || iX == dim.nx-1) {
                XX = (XX + dim.nx) % dim.nx;
            }
        }
        if(y_periodic){
            if(iY == 0 || iY == dim.ny-1){
                YY = (YY + dim.ny) % dim.ny;
            }
        }
        
        if(x_inflow_outflow){
            if(iX == dim.nx-1){
                if(k==5 || k==1 || k==8){
                }
                else{
                    size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
                    if (flag[nb] == CellType::reactive_obstacle&& Konjug_Waermetransport) {        //C if it is a solid cell -> wall interaction via function f
                        //C         gout_T(i,opp[k]) = ((1.-Sigma)/(1.+Sigma))*pop_out_T + (2.*Sigma/(1+Sigma))*gout_T(nb,opp[k]);
                        if(LI2014){
                            //gout_T(i,opp[k]) = ((1.-Sigma)/(1.+Sigma))*pop_out_T + (2.*Sigma/(1+Sigma))*gout_T(nb,opp[k]);
                            g_coll(i,k) = pop_out_T;
                        }
                        else if(HUBER2015){
                            gout_T(nb,k) = pop_out_T;       //*(rho*cp_gas)/(rho_solid*cp_solid);
                        }
                        else{
                            gout_T(nb,k) = pop_out_T;       //*(rho*cp_gas)/(rho_solid*cp_solid);
                        }
                    }                                              
                    else if(flag[nb] == CellType::bulk) {
                        gout_T(nb, k) = pop_out_T;
                    }
                    else {
                        gout_T(nb, k) = pop_out_T;
                    }
                }
            }
            else{
                //printf("Test123 iX: %d \t iY: %d \t XX:%d \t YY: %d \n", iX,iY,XX,YY);
                size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
                    if (flag[nb] == CellType::reactive_obstacle && Konjug_Waermetransport) {        //C if it is a solid cell -> wall interaction via function f
                        int vz = 0;
                        //vz = Vorzeichen(pop_out_T - gin_T(nb,k));
                        //gout_T(nb,k) = vz*pop_out_T*(rho*cp_gas)/(rho_solid*cp_solid) + gin_T(nb,k);
                        //printf("iX: %d \t iY: %d \t k: %d \t pop_out_T: %lf \t gin(nb,k):%lf \t gout(nb,k): %lf \n",iX,iY,k, pop_out_T,gin_T(nb,k),gout_T(nb,k) );
                        
                        if(LI2014){
                            //gout_T(i,opp[k]) = ((1.-Sigma)/(1.+Sigma))*pop_out_T + (2.*Sigma/(1+Sigma))*gout_T(nb,opp[k]);
                            g_coll(i,k) = pop_out_T;
                        }
                        else if(HUBER2015){
                            gout_T(nb,k) = pop_out_T;       //*(rho*cp_gas)/(rho_solid*cp_solid);
                        }
                        else{
                            gout_T(nb,k) = pop_out_T;
                        }
                        //char w;
                        //printf("Press c to continue\n");
                        //scanf("%c", &w);
                        //while(getchar()!='c'){}
                    }                                              
                    else if(flag[nb] == CellType::bulk) {
                        gout_T(nb, k) = pop_out_T;
                    }
                    else{
                        gout_T(nb, k) = pop_out_T;
                    }
            }
        }
        else{
            size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
            if(flag[nb] == CellType::bounce_back){
                gout_T(i,opp[k]) = pop_out_T;
            }
            else if( flag[i] == CellType::bulk && flag[nb] == CellType::reactive_obstacle && Konjug_Waermetransport){
                //gout_T(i,opp[k]) = ((1.-Sigma)/(1.+Sigma))*pop_out_T + (2.*Sigma/(1+Sigma))*gout_T(nb,opp[k]);
                if(LI2014){
                    g_coll(i,k) = pop_out_T;
                }
                else if(HUBER2015){
                    gout_T(nb,k) = pop_out_T;   //gin_T(nb,k) + vz*pop_out_T*(rho_solid*cp_solid)/(rho_gas*cp_gas);
                }
                else{
                    gout_T(nb,k) = pop_out_T;   //gin_T(nb,k) + vz*pop_out_T*(rho_solid*cp_solid)/(rho_gas*cp_gas); 
                }
            }
            else{
                gout_T(nb, k) = pop_out_T;
            }
        }
    };

    void streamTemp_solid (int i, int k, int iX, int iY, double pop_out_T) {
        int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
        int YY = iY + c[k][1];

        size_t nb = xyz_to_i(XX, YY);               //C nb is the index for the position after the streaming
        //auto[rho_gas_bulk,u] = macro(lattice[nb]);


        //if(flag[nb] == CellType::bulk){
            //int vz = 0;
            //vz = Vorzeichen(pop_out_T - gin_T(nb,k));
        //    gout_T(nb,k) = pop_out_T;   //gin_T(nb,k) + vz*pop_out_T*(rho_solid*cp_solid)/(rho_gas*cp_gas);
        //}
        if(flag[i] == CellType::reactive_obstacle && flag[nb] == CellType::bulk &&  Konjug_Waermetransport){      //C Konjugierter Wärmetransport Li 2014
            //int vz = 0;
            //vz = Vorzeichen(pop_out_T - gin_T(nb,k));
            //gout_T(i,opp[k]) = -((1.-Sigma)/(1.+Sigma))*pop_out_T + (2./(1+Sigma))*gout_T(nb,opp[k]);                               //C hier evtl Problem, dass Population von Gas noch aus altem Zeitschritt (pre-collision), durch Parallelisierung
            if(HUBER2015){
                gout_T(nb,k) = pop_out_T;   //gin_T(nb,k) + vz*pop_out_T*(rho_solid*cp_solid)/(rho_gas*cp_gas);
            }
            else if(LI2014){
                //gout_T(i,opp[k]) = -((1.-Sigma)/(1.+Sigma))*pop_out_T + (2./(1+Sigma))*gout_T(nb,opp[k]);     
                g_coll(i,k) = pop_out_T;
            }
            else{
                gout_T(nb,k) = pop_out_T;   //gin_T(nb,k) + vz*pop_out_T*(rho_solid*cp_solid)/(rho_gas*cp_gas); 
            }
        }
        else if(flag[nb] == CellType::bounce_back){
            gout_T(i,opp[k]) = pop_out_T;            
        }
        else{
            gout_T(nb, k) = pop_out_T;//*(rho_solid*cp_solid)/(rho_gas_bulk*cp_gas);
        }
    };

    void Local_Specular(int i){

            auto [iX,iY] = i_to_xyz(i);

            if(iY == dim.ny-1){

                fin(i,8) = fin(i,5);
                fin(i,4) = fin(i,2);
                fin(i,7) = fin(i,6);
            }

            if(iY==0){
                fin(i,5) = fin(i,8);
                fin(i,2) = fin(i,4);
                fin(i,6) = fin(i,7);
            }
    }

    
    void Boundaries(int i){             //C external boundary condition flow

        auto [iX,iY] = i_to_xyz(i);
        vector<double> f_eq(q_value);
        vector<double> s_i(q_value);

        // outlet boundary condition - FLOW - 
        if(iX==dim.nx-1 && iY<dim.ny-2 && iY > 1 ){
            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);

            auto[rho1,u1] = macro(lattice[j1]);
            auto[rho2,u2] = macro(lattice[j2]);
            /* for(int k=0; k<q_value; k++){
                fin(i,k) = (1./3.)*(4*fin(j1,k)-fin(j2,k));

            } */

            
            
            //fin(i,6) = (1./3.)*(4*fin(j1,6)-fin(j2,6));
            //fin(i,3) = (1./3.)*(4*fin(j1,3)-fin(j2,3));
            //fin(i,7) = (1./3.)*(4*fin(j1,7)-fin(j2,7));
           


            //C interpolation from paper with temperature boundary conditions
             
            double rho_lok = (1./3.)*(4*rho1-rho2);
            double ux = (1./3.)*(4*u1[0]-u2[0]);
            double uy = (1./3.)*(4*u1[1]-u2[1]);

            double usqr =  (ux * ux + uy * uy);
            for(int k=0; k<q_value;k++){
                double ck_u = c[k][0] * ux + c[k][1] * uy;
                f_eq[k] = rho_lok * t[k] * (1+ 3. * ck_u + 4.5 * ck_u * ck_u - 1.5*usqr);
            }

            
            fin(i, 3) = f_eq[3] + 2. / 3. * rho_lok * ux + 2. / 3. * (f_eq[3] - fin(i, 1) + f_eq[7] - fin(i, 5)
            + f_eq[6] - fin(i, 8));


            fin(i, 6) = f_eq[6] - t[6] * (((rho_lok * ux - fin(i, 1) - fin(i, 5) - fin(i, 8) + f_eq[6] + f_eq[3] + f_eq[7]) 
            / (t[3] + t[7] + t[6])) - ((rho_lok * uy - fin(i, 2) - fin(i, 5) + fin(i, 8) + fin(i, 4) - f_eq[6] + f_eq[7]) 
            / (t[7] + t[6])));


            fin(i, 7) = f_eq[7] - t[7] * (((rho_lok * ux - fin(i, 1) - fin(i, 5) - fin(i, 8) + f_eq[6] + f_eq[3] + f_eq[7]) 
                        / (t[3] + t[7] + t[6])) + ((rho_lok * uy - fin(i, 2) - fin(i, 5) + fin(i, 8) + fin(i, 4) - f_eq[6] 
                        + f_eq[7]) / (t[7] + t[6])));




            /* fin(i,6) = fin(j1,6);
            fin(i,3) = fin(j1,3);
            fin(i,7) = fin(j1,7); */

        }
        else if(iX == dim.nx-1 && iY==1){               //C Auslass Ecke unten
            
            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);
            auto[rho1,u1] = macro(lattice[j1]);
            auto[rho2,u2] = macro(lattice[j2]);
            
            double rho_lok = (1./3.)*(4*rho1-rho2);
            double ux = (1./3.)*(4*u1[0]-u2[0]);
            double uy = (1./3.)*(4*u1[1]-u2[1]);

            fin(i,8) = (rho_lok + rho_lok*ux - (2./3.)*rho_lok*uy - fin(i,0) - 2.*(fin(i,1) + fin(i,4) + fin(i,8)) )/2.;
            fin(i,6) = fin(i,8) - (1./6.)*rho_lok*ux + (1./6.)*rho_lok*uy;
            fin(i,3) = fin(i,1) - (2./3.)*rho_lok*ux;
            fin(i,7) = fin(i,5) - (1./6.)*rho_lok*ux + (1./6.)*rho_lok*uy;
            fin(i,2) = fin(i,4) + (2./3.)*rho_lok*uy;

        }
        else if(iX == dim.nx-1 && iY==dim.ny-2){        //C Auslass Ecke oben
            
            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);
            auto[rho1,u1] = macro(lattice[j1]);
            auto[rho2,u2] = macro(lattice[j2]);
            
            double rho_lok = (1./3.)*(4*rho1-rho2);
            double ux = (1./3.)*(4*u1[0]-u2[0]);
            double uy = (1./3.)*(4*u1[1]-u2[1]);



            fin(i,8) = (rho_lok + rho_lok*ux + (2./3.)*rho_lok*uy - fin(i,0) - 2.*(fin(i,1) + fin(i,2) + fin(i,5)) )/2.;
            fin(i,6) = fin(i,8) - (1./6.)*rho_lok*ux + (1./6.)*rho_lok*uy;
            fin(i,3) = fin(i,1) - (2./3.)*rho_lok*ux;
            fin(i,7) = fin(i,5) - (1./6.)*rho_lok*ux + (1./6.)*rho_lok*uy;
            fin(i,4) = fin(i,2) - (2./3.)*rho_lok*uy;

        }

 
    }

    void Boundaries_Comp(int i){        //C external boundary condition compontents
        auto [iX,iY] = i_to_xyz(i);

        vector <double> g_eq_O2 (q_value);
        vector <double> g_eq_CO2 (q_value);

        // top wall boundary condition - Concentration O2/CO2 - entsprechend Anti-Bounce-Back Methode - zero gradient
        if(iY == dim.ny-2 && y_noslip){

            int j1 = xyz_to_i(iX,iY-1);
            int j2 = xyz_to_i(iX,iY-2);

            auto[Y_O2_1, Y_CO2_1] = macroKonz(lattice[j1]);
            auto[Y_O2_2, Y_CO2_2] = macroKonz(lattice[j2]);

            double Y_O2_lok  = (1./3.)*(4*Y_O2_1 -Y_O2_2 );
            double Y_CO2_lok = (1./3.)*(4*Y_CO2_1-Y_CO2_2);

            gin_O2(i,7) = -gin_O2(i,5) + 2*t[7]*Y_O2_lok;
            gin_O2(i,4) = -gin_O2(i,2) + 2*t[2]*Y_O2_lok;
            gin_O2(i,8) = -gin_O2(i,6) + 2*t[6]*Y_O2_lok;
            
            gin_CO2(i,7) = -gin_CO2(i,5) + 2*t[7]*Y_CO2_lok;
            gin_CO2(i,4) = -gin_CO2(i,2) + 2*t[2]*Y_CO2_lok;;
            gin_CO2(i,8) = -gin_CO2(i,6) + 2*t[6]*Y_CO2_lok;;
        }


        // bottom wall boundary condition - Concentration O2/CO2 - entsprechend Anti-Bounce-Back Methode in Krüger Buch Kapitel 8.5 - zero gradient
        if(iY == 1 && y_noslip){

            int j1 = xyz_to_i(iX,iY+1);
            int j2 = xyz_to_i(iX,iY+2);

            auto[Y_O2_1, Y_CO2_1] = macroKonz(lattice[j1]);
            auto[Y_O2_2, Y_CO2_2] = macroKonz(lattice[j2]);

            double Y_O2_lok  = (1./3.)*(4*Y_O2_1 -Y_O2_2 );
            double Y_CO2_lok = (1./3.)*(4*Y_CO2_1-Y_CO2_2);

            gin_O2(i,6) = -gin_O2(i,8) + 2*t[8]*Y_O2_lok;
            gin_O2(i,2) = -gin_O2(i,4) + 2*t[4]*Y_O2_lok;
            gin_O2(i,5) = -gin_O2(i,7) + 2*t[7]*Y_O2_lok;
            
            gin_CO2(i,6) = -gin_CO2(i,8) + 2*t[8]*Y_CO2_lok;
            gin_CO2(i,2) = -gin_CO2(i,4) + 2*t[4]*Y_CO2_lok;;
            gin_CO2(i,5) = -gin_CO2(i,7) + 2*t[7]*Y_CO2_lok;;
        }


        // inlet dirichlet boundary condition - Concentration O2/CO2 - 
        if(iX == 1 && Inflow_Outflow){

            double Y_O2_lok  = Y_O2_inlet;
            double Y_CO2_lok = Y_CO2_inlet;

            gin_O2(i,5) = -gin_O2(i,7) + 2*t[7]*Y_O2_lok;
            gin_O2(i,1) = -gin_O2(i,3) + 2*t[3]*Y_O2_lok;
            gin_O2(i,8) = -gin_O2(i,6) + 2*t[6]*Y_O2_lok;
            
            gin_CO2(i,5) = -gin_CO2(i,7) + 2*t[7]*Y_CO2_lok;;
            gin_CO2(i,1) = -gin_CO2(i,3) + 2*t[3]*Y_CO2_lok;;
            gin_CO2(i,8) = -gin_CO2(i,6) + 2*t[6]*Y_CO2_lok;;
        }



        // outlet zero gradient boundary condition - Concentration O2/CO2 - 
        if(iX == dim.nx-1 && iY != 1 && iY != dim.ny-2 && Inflow_Outflow){
            

            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);

            auto[Y_O2_1, Y_CO2_1] = macroKonz(lattice[j1]);
            auto[Y_O2_2, Y_CO2_2] = macroKonz(lattice[j2]);

            double Y_O2_lok  = (1./3.)*(4*Y_O2_1 -Y_O2_2 );
            double Y_CO2_lok = (1./3.)*(4*Y_CO2_1-Y_CO2_2);

            gin_O2(i,6) = -gin_O2(i,8) + 2*t[8]*Y_O2_lok;
            gin_O2(i,3) = -gin_O2(i,1) + 2*t[1]*Y_O2_lok;
            gin_O2(i,7) = -gin_O2(i,5) + 2*t[7]*Y_O2_lok;
            
            gin_CO2(i,6) = -gin_CO2(i,8) + 2*t[8]*Y_CO2_lok;
            gin_CO2(i,3) = -gin_CO2(i,1) + 2*t[1]*Y_CO2_lok;;
            gin_CO2(i,7) = -gin_CO2(i,5) + 2*t[7]*Y_CO2_lok;
        }

        //C outlet untere Ecke  2 6 3 5 7
        if(iX == dim.nx-1 && iY == 1 && Inflow_Outflow){


            auto [rho, u] = macro(lattice[i]);
            auto [YO2, YCO2] = macroKonz(lattice[i]);
            double usqr = 1.5 * (u[0] * u[0] + u[1] * u[1]);
            for(int k=0; k<q_value; k++){
                double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
                double temp;
                temp = t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);
                
                g_eq_O2[k]  = YO2  * temp;  
                g_eq_CO2[k]  = YCO2 * temp; 
            }


            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);

            auto[Y_O2_1, Y_CO2_1] = macroKonz(lattice[j1]);
            auto[Y_O2_2, Y_CO2_2] = macroKonz(lattice[j2]);

            double Y_O2_lok  = (1./3.)*(4*Y_O2_1 -Y_O2_2 );
            double Y_CO2_lok = (1./3.)*(4*Y_CO2_1-Y_CO2_2);

            double Y_O2_epsilon  = gin_O2(i,0)  + gin_O2(i,1)  + g_eq_O2[2]  + g_eq_O2[3]  + gin_O2(i,4)  + g_eq_O2[5]  + g_eq_O2[6]  + g_eq_O2[7]  + gin_O2(i,8);
            double Y_CO2_epsilon = gin_CO2(i,0) + gin_CO2(i,1) + g_eq_CO2[2] + g_eq_CO2[3] + gin_CO2(i,4) + g_eq_CO2[5] + g_eq_CO2[6] + g_eq_CO2[7] + gin_CO2(i,8);

            double GC_O2  = (Y_O2_lok  - Y_O2_epsilon)  / (t[2] + t[3] + t[6] + t[5] + t[7]);
            double GC_CO2 = (Y_CO2_lok - Y_CO2_epsilon) / (t[2] + t[3] + t[6] + t[5] + t[7]);

            gin_O2(i,6)  = g_eq_O2[6]  + t[6]*GC_O2;
            gin_O2(i,3)  = g_eq_O2[3]  + t[3]*GC_O2;
            gin_O2(i,7)  = g_eq_O2[7]  + t[7]*GC_O2;
            gin_O2(i,2)  = g_eq_O2[2]  + t[2]*GC_O2;
            gin_O2(i,5)  = g_eq_O2[5]  + t[5]*GC_O2;
            
            gin_CO2(i,6) = g_eq_CO2[6] + t[6]*GC_CO2;
            gin_CO2(i,3) = g_eq_CO2[3] + t[3]*GC_CO2;
            gin_CO2(i,7) = g_eq_CO2[7] + t[7]*GC_CO2;
            gin_CO2(i,2) = g_eq_CO2[2] + t[2]*GC_CO2;
            gin_CO2(i,5) = g_eq_CO2[5] + t[5]*GC_CO2;
        }

        //C outlet obere Ecke   3 4 7 6 8 
        if(iX == dim.nx-1 && iY == dim.ny-2 && Inflow_Outflow){



            auto [rho, u] = macro(lattice[i]);
            auto [YO2, YCO2] = macroKonz(lattice[i]);
            double usqr = 1.5 * (u[0] * u[0] + u[1] * u[1]);
            for(int k=0; k<q_value; k++){
                double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
                double temp;
                temp = t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);
                
                g_eq_O2[k]  = YO2  * temp;  
                g_eq_CO2[k]  = YCO2 * temp; 
            }


            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);

            auto[Y_O2_1, Y_CO2_1] = macroKonz(lattice[j1]);
            auto[Y_O2_2, Y_CO2_2] = macroKonz(lattice[j2]);

            double Y_O2_lok  = (1./3.)*(4*Y_O2_1 -Y_O2_2 );
            double Y_CO2_lok = (1./3.)*(4*Y_CO2_1-Y_CO2_2);

            double Y_O2_epsilon  = gin_O2(i,0)   + gin_O2(i,1)   + gin_O2(i,2)  + g_eq_O2[3]  + g_eq_O2[4]  + gin_O2(i,5)  + g_eq_O2[6]  + g_eq_O2[7]  + g_eq_O2[8];
            double Y_CO2_epsilon = gin_CO2(i,0)  + gin_CO2(i,1)  + gin_CO2(i,2) + g_eq_CO2[3] + g_eq_CO2[4] + gin_CO2(i,5) + g_eq_CO2[6] + g_eq_CO2[7] + g_eq_CO2[8];

            double GC_O2  = (Y_O2_lok  - Y_O2_epsilon)  / (t[3] + t[4] + t[6] + t[7] + t[8]);
            double GC_CO2 = (Y_CO2_lok - Y_CO2_epsilon) / (t[3] + t[4] + t[6] + t[7] + t[8]);

            gin_O2(i,6)  = g_eq_O2[6]  + t[6]*GC_O2;
            gin_O2(i,3)  = g_eq_O2[3]  + t[3]*GC_O2;
            gin_O2(i,7)  = g_eq_O2[7]  + t[7]*GC_O2;
            gin_O2(i,4)  = g_eq_O2[4]  + t[4]*GC_O2;
            gin_O2(i,8)  = g_eq_O2[8]  + t[8]*GC_O2;
            
            gin_CO2(i,6) = g_eq_CO2[6] + t[6]*GC_CO2;
            gin_CO2(i,3) = g_eq_CO2[3] + t[3]*GC_CO2;
            gin_CO2(i,7) = g_eq_CO2[7] + t[7]*GC_CO2;
            gin_CO2(i,4) = g_eq_CO2[4] + t[4]*GC_CO2;
            gin_CO2(i,8) = g_eq_CO2[8] + t[8]*GC_CO2;
        }




    }
    
    void Boundaries_Temp_Reaktion(int i){        //C external boundary condition temperature field
        auto [iX,iY] = i_to_xyz(i);



        // top wall boundary condition - Temperature - entsprechend Anti-Bounce-Back Methode
        if(iY == dim.ny-2 && y_noslip){

            int j1 = xyz_to_i(iX,iY-1);
            int j2 = xyz_to_i(iX,iY-2);

            double T_1 = macroTemp(lattice[j1]);
            double T_2 = macroTemp(lattice[j2]);

            double T_lok  = (1./3.)*(4*T_1 -T_2 );

            gin_T(i,7) = -gin_T(i,5) + 2*t[7]*T_lok;
            gin_T(i,4) = -gin_T(i,2) + 2*t[2]*T_lok;
            gin_T(i,8) = -gin_T(i,6) + 2*t[6]*T_lok;
            
        }

        // bottom wall boundary condition - Temperature - entsprechend Anti-Bounce-Back Methode in Krüger Buch Kapitel 8.5
        if(iY == 1 && y_noslip){

            int j1 = xyz_to_i(iX,iY+1);
            int j2 = xyz_to_i(iX,iY+2);

            double T_1 = macroTemp(lattice[j1]);
            double T_2 = macroTemp(lattice[j2]);

            double T_lok  = (1./3.)*(4*T_1 -T_2 );

            gin_T(i,6) = -gin_T(i,8) + 2*t[8]*T_lok;
            gin_T(i,2) = -gin_T(i,4) + 2*t[4]*T_lok;
            gin_T(i,5) = -gin_T(i,7) + 2*t[7]*T_lok;
            

        }

        // inlet dirichlet boundary condition - Temperature - 
        if(iX == 1 && x_inflow_outflow){

            double T_lok  = T_inlet;

            gin_T(i,5) = -gin_T(i,7) + 2*t[7]*T_lok;
            gin_T(i,1) = -gin_T(i,3) + 2*t[3]*T_lok;
            gin_T(i,8) = -gin_T(i,6) + 2*t[6]*T_lok;
            

        }

        // outlet zero gradient boundary condition - Temperature - 
        if(iX == dim.nx-1 && x_inflow_outflow){

            int j1 = xyz_to_i(iX-1,iY);
            int j2 = xyz_to_i(iX-2,iY);

            double T_1 = macroTemp(lattice[j1]);
            double T_2 = macroTemp(lattice[j2]);

            double Y_O2_lok  = (1./3.)*(4*T_1 -T_2 );

            gin_T(i,6) = -gin_T(i,8) + 2*t[8]*Y_O2_lok;
            gin_T(i,3) = -gin_T(i,1) + 2*t[1]*Y_O2_lok;
            gin_T(i,7) = -gin_T(i,5) + 2*t[7]*Y_O2_lok;
            
        }


        /*for(int k=0; k<q_value; k++){
            int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
            int YY = iY + c[k][1];
            size_t nb = xyz_to_i(XX, YY);  

            if(flag[nb] == CellType::reactive_obstacle &&  Konjug_Waermetransport){
                gin_T(i,opp[k]) = ((1.-Sigma)/(1.+Sigma))*gin_T(i,k) + (2.*Sigma/(1+Sigma))*gin_T(nb,opp[k]);
            }
        }*/



    }

    void Boundaries_Temp_Solid(int i){
        auto[iX,iY] = i_to_xyz(i);

        for(int k=0; k<q_value; k++){
            int XX = iX + c[k][0];                          //C position of the current step iX is being altered by the direction c
            int YY = iY + c[k][1];
            size_t nb = xyz_to_i(XX, YY);  

            if(flag[nb] == CellType::bulk &&  Konjug_Waermetransport && LI2014){
                gin_T(i,opp[k]) = -((1.-Sigma)/(1.+Sigma))*g_coll(i,k) + (2./(1.+Sigma))*g_coll(nb,opp[k]);     //C für Solid-node
                //gin_T(i,opp[k]) = -((1.-Sigma)/(1.+Sigma))*gin_T(i,k) + (2./(1.+Sigma))*gin_T(nb,opp[k]);     //C für Solid-node
                gin_T(nb,k) = ((1.-Sigma)/(1.+Sigma))*g_coll(nb,opp[k]) + (2.*Sigma/(1.+Sigma))*g_coll(i,k);     //C für gas-node
                //gin_T(nb,k) = ((1.-Sigma)/(1.+Sigma))*gin_T(nb,opp[k]) + (2.*Sigma/(1.+Sigma))*gin_T(i,k);     //C für gas-node
            }
        }
    }

    void Reaktion_Solid(int i){                 //C Implementierung der Reaktionswärme im Solid
        for(int k=0; k<q_value; k++){
            gin_T(i,k) = gin_T(i,k) + t[k]*q_i(i);
        }
        q_i(i) = 0.;
    }
   



    // Execute second-order BGK collision on a cell.
    auto collideBgk(int i, int k, double rho, std::array<double, 2> const& u, double usqr) {        /// usqr = 1.5 * (u.u)
        double ck_u = c[k][0] * u[0] + c[k][1] * u[1];         /// ck_u = c . u
        double pop_out;
        
        double eq = rho * t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr); 
        double source = (1-0.5*omega)*t[k]*(  (3.0*(c[k][0]-u[0])+9.0*(ck_u)*c[k][0])*efx   +   (3.0*(c[k][1]-u[1])+9.0*(ck_u)*c[k][1])*efy   );                                                       
        pop_out = (1. - omega) * fin(i, k) + omega * eq + source;                 //C omega = 1. / (3. * nu + 0.5); TODO mal checken, ob das 1/omega_lit ist
        
        return pop_out;
    }

    //C Multi relaxation time collision operator for a single phase flow
    auto collideMRT(int i, double rho, std::array<double, 2> const& u, double usqr) {        /// usqr = 1.5 * (u.u)
        
        
        vector <double> m       (9,0.);
        vector <double> m_eq    (9,0.);
        vector <double> m_coll  (9,0.);
        vector <double> f_eq    (9,0.);
        vector <double> f_coll  (9,0.);
        vector <double> pop_out (9,0.);
        vector <double> F_strich (9,0.);
        vector <double> m_F    (9,0.);
        vector <double> m_coll_F  (9,0.);
        vector <double> f_coll_F  (9,0.);




        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////Flow Field/////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //C calculation of equilibrium distribution functions
        for(int k=0; k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            f_eq[k] = rho * t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);  
        }

        


        //C transformation from population space into moment space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                m_eq[k] = m_eq[k] + M[k][j] * f_eq[j];
                m[k]    = m[k]    + M[k][j] * fin(i,j);
            }
        }
        
        //C calculating the collision step in moment space
        for(int k=0; k<q_value; k++){
            m_coll[k] = S[k] * (m[k] - m_eq[k]);
        }
        
        //C transformation from moment space back to population space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                f_coll[k] = f_coll[k] + M_inv[k][j]*m_coll[j];
            }
        }

        ////////////////external force/////////////////////////////////////////////////////////////////////////////////////////
        if(External_force){
            for(int k=0; k<q_value; k++){
            double ck_F = c[k][0] * efx + c[k][1] * efy;
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            F_strich[k] = t[k]*(  1/(CS2)*ck_F   + (1/pow(CS2,2))*ck_u*ck_F - (1/CS2)*(u[0]*efx + u[1]*efy)  );
            }

            for(int k=0; k<q_value; k++){
                for(int j=0; j<q_value; j++){
                    m_F[k] = m_F[k] + M[k][j] * F_strich[j];
                }
            }

            for(int k=0; k<q_value; k++){
                m_coll_F[k]   = m_F[k] - 0.5*S[k] * m_F[k];
            }

            for(int k=0; k<q_value; k++){
                for(int j=0; j<q_value; j++){
                    f_coll_F[k] = f_coll_F[k] + M_inv[k][j]*m_coll_F[j];
                }
            }
        }

        for(int k=0; k<q_value; k++){
                pop_out[k] = fin(i,k)-f_coll[k] + 0*f_coll_F[k];
        }
        



        return pop_out;
    }

    //C multi-relaxation time collision operator for single phase multi component flow
    auto collideMRTFlow(int i, double rho, std::array<double, 2> const& u, double usqr) {        /// usqr = 1.5 * (u.u)
        
        
        vector <double> m       (9,0.);
        vector <double> m_eq    (9,0.);
        vector <double> m_coll  (9,0.);

        vector <double> f_eq1    (9,0.);

        vector <double> f_eq    (9,0.);
        vector <double> f_coll  (9,0.);
        vector <double> pop_out (9,0.);
        vector <double> F_strich (9,0.);
        vector <double> m_F    (9,0.);
        vector <double> m_coll_F  (9,0.);
        vector <double> f_coll_F  (9,0.);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////Flow Field/////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto[iX1,iY1] = i_to_xyz(i);
        //C calculation of equilibrium distribution functions
        
        for(int k=0; k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            double ck_ck = c[k][0]*c[k][0] + c[k][1]*c[k][1];
            
            double T = macroTemp(lattice[i]);
            double Teta = T/T_ref;

            double beta = ((Teta-1)/(2*CS2))*( ck_ck - CS2*d_value + ck_u*( (ck_ck/CS2) - d_value - 2  )    );


            f_eq[k] = rho * t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);  

            f_eq1[k] = rho * t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr + TC*beta);  



            //if(iX1==1){
            //    printf("iX: %d \t iY: %d \t k: %d \t Teta: %lf \t Beta: %lf \t f_eq %lf \t f_eq_beta: %lf \n",iX1,iY1,k,Teta,beta,f_eq1[k], f_eq[k]);
            //}
        } 

        //C hardcode of f_eq nach Timan Lei Paper

        /*         double T = macroTemp(lattice[i]);
        double Teta = T/T_ref;

        f_eq[0] = rho;
        f_eq[1] = rho*( -4 + 3*pow(u[0],2) + 2*Teta );
        f_eq[2] = rho*(  3 - 3*pow(u[0],2) - 2*Teta );
        f_eq[3] = rho*(  u[0]);
        f_eq[4] = rho*( -2*u[0] + u[0]*Teta);
        f_eq[5] = rho*( u[1]);
        f_eq[6] = rho*( -2*u[1] + u[1]*Teta);
        f_eq[7] = rho*( pow(u[0],2) - pow(u[1],2));
        f_eq[8] = rho*( u[0]*u[1]); */

        //C transformation from population space into moment space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                m_eq[k] = m_eq[k] + M[k][j] * f_eq[j];
                m[k]    = m[k]    + M[k][j] * fin(i,j);
            }
        }
        
        //C calculating the collision step in moment space
        for(int k=0; k<q_value; k++){
            m_coll[k] = S[k] * (m[k] - m_eq[k]);
        }
        
        //C transformation from moment space back to population space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                f_coll[k] = f_coll[k] + M_inv[k][j]*m_coll[j];
            }
        }

        ////////////////external force - FLOW FIELD - /////////////////////////////////////////////////////////////////////////////////////////
        if(External_force){
            for(int k=0; k<q_value; k++){
            double ck_F = c[k][0] * efx + c[k][1] * efy;
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            F_strich[k] = t[k]*(  1/(CS2)*ck_F   + (1/pow(CS2,2))*ck_u*ck_F - (1/CS2)*(u[0]*efx + u[1]*efy)  );
            }

            for(int k=0; k<q_value; k++){
                for(int j=0; j<q_value; j++){
                    m_F[k] = m_F[k] + M[k][j] * F_strich[j];
                }
            }

            for(int k=0; k<q_value; k++){
                m_coll_F[k]   = m_F[k] - 0.5*S[k] * m_F[k];
            }

            for(int k=0; k<q_value; k++){
                for(int j=0; j<q_value; j++){
                    f_coll_F[k] = f_coll_F[k] + M_inv[k][j]*m_coll_F[j];
                }
            }
        }

      


        ///////////////////////////corrector term C_dach//////////////////////////////////////////////////////////////////


        vector <double> C_dach          (q_value,0.);
        vector <double> m_coll_C_dach   (q_value,0.);
        vector <double> f_coll_C        (q_value,0.);     

        auto[iX,iY] = i_to_xyz(i);
        int Xe = iX+1;
        int Xw = iX-1;
        int Yn = iY+1;
        int Ys = iY-1;
        int i_nord      = xyz_to_i(iX, Yn);
        int i_ost       = xyz_to_i(Xe, iY);
        int i_sued      = xyz_to_i(iX, Ys);
        int i_west      = xyz_to_i(Xw, iY);

        auto [Qx_i, Qy_i] = Q_xy(i);

        auto [Qx_nord, Qy_nord] = Q_xy(i_nord);
        auto [Qx_sued, Qy_sued] = Q_xy(i_sued);
        auto [Qx_ost,  Qy_ost]  = Q_xy(i_ost);
        auto [Qx_west, Qy_west] = Q_xy(i_west);

        double d_Qx_dx = (Qx_ost -Qx_west)/2.;
        double d_Qy_dy = (Qy_nord-Qy_sued)/2.;

        if(flag[i_nord] ==  CellType::bounce_back || flag[i_nord] == CellType::reactive_obstacle){
            int i_sued_sued      = xyz_to_i(iX, Ys-1);
            auto [Qx_sued_sued, Qy_sued_sued] = Q_xy(i_sued_sued);
            d_Qy_dy = (3.*Qy_i - 4.*Qy_sued + Qy_sued_sued) / 2.;
        }
        if(flag[i_sued] == CellType::bounce_back || flag[i_sued] == CellType::reactive_obstacle){
            int i_nord_nord     = xyz_to_i(iX, Yn+1);
            auto [Qx_nord_nord, Qy_nord_nord] = Q_xy(i_nord_nord); 
            d_Qy_dy = ( -3.*Qy_i + 4.*Qy_nord - Qy_nord_nord) / 2.;
        }
        if(flag[i_ost] ==  CellType::bounce_back || flag[i_ost] == CellType::reactive_obstacle){
            int i_west_west      = xyz_to_i(Xw-1, iY);
            auto[Qx_west_west, Qy_west_west] = Q_xy(i_west_west);
            d_Qx_dx = ( 3.*Qx_i - 4.*Qx_west + Qx_west_west)/2.;
        }
         if(flag[i_west] ==  CellType::bounce_back || flag[i_west] == CellType::reactive_obstacle){
            int i_ost_ost      = xyz_to_i(Xe+1, iY);
            auto[Qx_ost_ost,Qy_ost_ost] = Q_xy(i_ost_ost);
            d_Qx_dx = ( -3.*Qx_i + 4*Qx_ost - Qx_ost_ost) / 2.;
        }

        if(iX == dim.nx-1){
            d_Qx_dx = 0.;
        }


        C_dach[0] = 0;
        C_dach[1] =  3.*(d_Qx_dx+d_Qy_dy);
        C_dach[2] = -3.*(d_Qx_dx+d_Qy_dy);
        C_dach[3] = 0;
        C_dach[4] = 0;
        C_dach[5] = 0;
        C_dach[6] = 0;
        C_dach[7] = (d_Qx_dx-d_Qy_dy);
        C_dach[8] = 0;

        for(int k=0; k<q_value; k++){
                m_coll_C_dach[k]   = C_dach[k] - 0.5*S[k] * C_dach[k];
        }

        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                f_coll_C[k] = f_coll_C[k] + M_inv[k][j]*m_coll_C_dach[j];
            }
        }




        //C finaler Kollisionsschritt

        for(int k=0; k<q_value; k++){
            pop_out[k] = fin(i,k)-f_coll[k] + f_coll_F[k] + TC*f_coll_C[k];
        }

        
        return pop_out;
    }

    auto collideMRTComponents(int i, double rho, std::array<double, 2> const& u, double usqr, double YO2, double YCO2) {        /// usqr = 1.5 * (u.u)

        vector <double> g_eq_O2 (q_value);
        vector <double> g_eq_CO2 (q_value);
        vector <double> m_g_eq_O2 (q_value);
        vector <double> m_g_O2 (q_value);
        vector <double> m_g_eq_CO2 (q_value);
        vector <double> m_g_CO2 (q_value);
        vector <double> m_g_O2_coll (q_value);
        vector <double> m_g_CO2_coll (q_value);
        vector <double> g_O2_coll (q_value);
        vector <double> g_CO2_coll (q_value);
        vector <double> pop_out_g_O2(q_value);
        vector <double> pop_out_g_CO2 (q_value);
        vector <double> F_O2_strich(q_value, 0.);  
        vector <double> F_CO2_strich(q_value, 0.);  


        //C calculation of equilibrium distribution functions
        for(int k=0; k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            double temp;
            temp = t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);
            
            g_eq_O2[k]  = YO2  * temp;  
            g_eq_CO2[k]  = YCO2 * temp; 
        }

        
        //C transformation of populations g[n] to moment space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){

                m_g_O2[k]     = m_g_O2[k]     + M[k][j] * gin_O2(i,j);
                m_g_eq_O2[k]  = m_g_eq_O2[k]  + M[k][j] * g_eq_O2[j];

                m_g_CO2[k]     = m_g_CO2[k]     + M[k][j] * gin_CO2(i,j);
                m_g_eq_CO2[k]  = m_g_eq_CO2[k]  + M[k][j] * g_eq_CO2[j];

            }
        }


        //C collision in moment space for 
        for(int k=0; k<q_value; k++){
            m_g_O2_coll[k]    = S_GO2[k] * (m_g_O2[k]  - m_g_eq_O2[k]);
            m_g_CO2_coll[k]   = S_GCO2[k] * (m_g_CO2[k] - m_g_eq_CO2[k]);
        }
    

        //C transformation from moment space back to population space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                g_O2_coll[k]  = g_O2_coll[k]  + M_inv[k][j]*m_g_O2_coll[j];
                g_CO2_coll[k] = g_CO2_coll[k] + M_inv[k][j]*m_g_CO2_coll[j];
            }
        }


        //C force for the distribution function of the COMPONENTS

        auto [F_O2, F_CO2] = Mass_fraction_forcing(i);
        /*          double F_O2 = 0.;
                    double F_CO2 = 0.; */

        for(int k=0;k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            F_O2_strich[k]  = t[k]*F_O2* (1+ (1/CS2)*ck_u*((Tau_O2 -0.5)/Tau_O2));
            F_CO2_strich[k] = t[k]*F_CO2*(1+ (1/CS2)*ck_u*((Tau_CO2-0.5)/Tau_CO2)); 
        }

        auto[iX,iY] = i_to_xyz(i);
        //C actual collision calculation for g
        for(int k=0; k<q_value; k++){
            pop_out_g_O2[k]  = gin_O2(i,k)  - g_O2_coll[k]  + delta_t*F_O2_strich[k]  + 0.5*pow(delta_t,2)*((F_O2_strich[k] -F_O2_alt(i,k)) /delta_t);
            pop_out_g_CO2[k] = gin_CO2(i,k) - g_CO2_coll[k] + delta_t*F_CO2_strich[k] + 0.5*pow(delta_t,2)*((F_CO2_strich[k]-F_CO2_alt(i,k))/delta_t);
        }


        //C saving of the force for the next time step to calculate the temporal derivative
        for(int k=0; k<q_value; k++){
            F_O2_alt(i,k)  = F_O2_strich[k];
            F_CO2_alt(i,k) = F_CO2_strich[k];
        }


        return std::make_tuple(pop_out_g_O2, pop_out_g_CO2);
    }

    //C multi-relaxation time model for calculation of the temperature field
    auto collideMRTTemp(int i, double T,std::array<double, 2> const& u, double usqr, vector <double> Y_O2_interface, double rho){

        vector <double> g_eq_T (q_value);
        vector <double> m_g_eq_T (q_value,0.);
        vector <double> m_g_T (q_value, 0.);
        vector <double> m_g_T_coll (q_value);
        vector <double> g_T_coll (q_value);
        vector <double> pop_out_g_T(q_value);
        vector <double> F_T_strich(q_value, 0.);  


        //C calculation of equilibrium distribution functions
         for(int k=0; k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            g_eq_T[k]  = T  * t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);  
        }

        //C transformation of the populations into the moment space 
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){

                m_g_T[k]     = m_g_T[k]     + M[k][j] * gin_T(i,j);
                m_g_eq_T[k]  = m_g_eq_T[k]  + M[k][j] * g_eq_T[j];
            }
        }

        //C collision in moment space
        for(int k=0; k<q_value; k++){
            m_g_T_coll[k]    = S_T[k] * (m_g_T[k]  - m_g_eq_T[k]);
        }

        //C transformation back to population space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                g_T_coll[k]  = g_T_coll[k]  + M_inv[k][j]*m_g_T_coll[j];
            }
        }


        //C force for the distribution function of the Temperature
        double F_T = 0.;
        if(HUBER2015){
            F_T =  Temperature_forcing_neu(i, rho, u, T, usqr, g_eq_T);        //Temperature_forcing(i);    //TODO Problem bei hohen Inlet-Geschwindigkeiten: Temperature-Forcing führt zu Temp-divergenz an rechtem Rand der obstacles
        }
        double F_Q1 = 0.; double F_Q1_temp=0.;
        double F_r, Q;
        double F_Q = 0.; //double F_Q1 = 0.; 
        double F_Q2 = 0.; double F_Q3 = 0.; double F_Q4 = 0.; 
        //double F_Q1_temp=0.;
        double F_r1, F_r2, F_r3, F_r4;
        //double Q;
        if(Solid_reaction){
          //   (rho_solid*cp_solid);
            if(HUBER2015 || LI2014){
                F_r = Prae_exp_factor*exp(-E_akt/(R_id*T)) * (Y_O2_interface[1]+Y_O2_interface[2]+Y_O2_interface[3]+Y_O2_interface[4])*rho/M_O2;
                Q = F_r * delta_hr;
                F_Q1 = Q /(rho * cp_gas);    
            }
            if(GUO2015){
                F_r1 = Prae_exp_factor*exp(-E_akt/(R_id*T)) * (Y_O2_interface[1])*rho/M_O2;
                F_r2 = Prae_exp_factor*exp(-E_akt/(R_id*T)) * (Y_O2_interface[2])*rho/M_O2;
                F_r3 = Prae_exp_factor*exp(-E_akt/(R_id*T)) * (Y_O2_interface[3])*rho/M_O2;
                F_r4 = Prae_exp_factor*exp(-E_akt/(R_id*T)) * (Y_O2_interface[4])*rho/M_O2;
                Q = (F_r1 + F_r2 + F_r3 + F_r4 )* delta_hr;
                
                F_Q =  Q /(rho * cp_gas);              //   (rho_solid*cp_solid);
                F_Q1 = F_r1* delta_hr/(rho_solid * cp_solid);
                F_Q2 = F_r2* delta_hr/(rho_solid * cp_solid);
                F_Q3 = F_r3* delta_hr/(rho_solid * cp_solid);
                F_Q4 = F_r4* delta_hr/(rho_solid * cp_solid);
                auto [iX,iY] = i_to_xyz(i);
                int i_1 = xyz_to_i(iX+1,iY);
                int i_2 = xyz_to_i(iX,iY+1);
                int i_3 = xyz_to_i(iX-1,iY);
                int i_4 = xyz_to_i(iX,iY-1);
                //q_i(i) = F_Q;                  //TODO  ich muss eigentlich 4 Richtungen pro Zelle abspeichern, falls Geometrie zeitlich verändert

                q_i(i_1) += F_Q1;              
                q_i(i_2) += F_Q2;                 
                q_i(i_3) += F_Q3;                  
                q_i(i_4) += F_Q4;  
            }




            //F_r = A_SI*exp(-E_akt_SI*1000/(R_SI*T*773.))*(Y_O2_interface[1]+Y_O2_interface[2]+Y_O2_interface[3]+Y_O2_interface[4])*rho_SI/(M_O2_SI/1000.); //C [mol/(m^2*s)]
            //Q = F_r*hr_SI*pow(dx_SI,2);      //C [kJ/s]
            //F_Q1_temp = Q*dt_SI/(rho_SI*pow(dx_SI,3)*cp_SI);     //C[K]
            //F_Q1 = F_Q1_temp/773.;
        }
        if(HUBER2015 || LI2014){
            F_T = F_T + 1*F_Q1;
        }
        else {
            F_T = F_T;
        }


        for(int k=0;k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            F_T_strich[k]  = t[k]*F_T* (1+ (1/CS2)*ck_u*((Tau_T -0.5)/Tau_T));
            //F_T_strich[k] = 0.;
        }
        //C final collision equation
        double delta_T_reaktion = 0.;
        for(int k=0; k<q_value; k++){
            pop_out_g_T[k] = gin_T(i,k) - g_T_coll[k]  + 1*delta_t_lu *F_T_strich[k] + 1*0.5*pow(delta_t_lu,2)*((F_T_strich[k] -F_T_alt(i,k)) /delta_t_lu);
            delta_T_reaktion = delta_T_reaktion + 1*delta_t_lu *F_T_strich[k]  + 1*0.5*pow(delta_t_lu,2)*((F_T_strich[k] -F_T_alt(i,k)) /delta_t_lu);

        }



        //C saving of the force for the next time step to calculate the temporal derivative
        for(int k=0; k<q_value; k++){
            F_T_alt(i,k)  = F_T_strich[k];
        }


        return pop_out_g_T;
    }

    auto collideMRTTemp_WL(int i, double T,std::array<double, 2> const& u, double usqr, double rho){

        vector <double> g_eq_T (q_value);
        vector <double> m_g_eq_T (q_value,0.);
        vector <double> m_g_T (q_value, 0.);
        vector <double> m_g_T_coll (q_value);
        vector <double> g_T_coll (q_value);
        vector <double> pop_out_g_T(q_value);
        vector <double> F_T_strich(q_value, 0.);  


        //C calculation of equilibrium distribution functions
         for(int k=0; k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            g_eq_T[k]  = T  * t[k] * (1. + 3. * ck_u + 4.5 * ck_u * ck_u - usqr);  
        }

        //C transformation of the populations into the moment space 
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){

                m_g_T[k]     = m_g_T[k]     + M[k][j] * gin_T(i,j);
                m_g_eq_T[k]  = m_g_eq_T[k]  + M[k][j] * g_eq_T[j];
            }
        }

        //C collision in moment space
        for(int k=0; k<q_value; k++){
            m_g_T_coll[k]    = S_T[k] * (m_g_T[k]  - m_g_eq_T[k]);
        }

        //C transformation back to population space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                g_T_coll[k]  = g_T_coll[k]  + M_inv[k][j]*m_g_T_coll[j];
            }
        }


        //C force for the distribution function of the Temperature
        double F_T = 0.;
        if(HUBER2015){
            F_T =  Temperature_forcing_neu(i, rho, u, T, usqr, g_eq_T);        //Temperature_forcing(i);    //TODO Problem bei hohen Inlet-Geschwindigkeiten: Temperature-Forcing führt zu Temp-divergenz an rechtem Rand der obstacles
        }

        for(int k=0;k<q_value; k++){
            double ck_u = c[k][0] * u[0] + c[k][1] * u[1];
            F_T_strich[k]  = t[k]*F_T* (1+ (1/CS2)*ck_u*((Tau_T -0.5)/Tau_T));
            //F_T_strich[k] = 0.;
        }
        //C final collision equation
        double delta_T_reaktion = 0.;
        for(int k=0; k<q_value; k++){
            pop_out_g_T[k] = gin_T(i,k) - g_T_coll[k]  + 1*delta_t_lu *F_T_strich[k] + 1*0.5*pow(delta_t_lu,2)*((F_T_strich[k] -F_T_alt(i,k)) /delta_t_lu);
        }

        //C saving of the force for the next time step to calculate the temporal derivative
        for(int k=0; k<q_value; k++){
            F_T_alt(i,k)  = F_T_strich[k];
        }


        return pop_out_g_T;
    }

    auto collideMRTTemp_Solid(int i, double T){

        vector <double> g_eq_T (q_value);
        vector <double> m_g_eq_T (q_value,0.);
        vector <double> m_g_T (q_value, 0.);
        vector <double> m_g_T_coll (q_value);
        vector <double> g_T_coll (q_value);
        vector <double> pop_out_g_T(q_value);
        vector <double> F_T_strich(q_value, 0.);  


        //C calculation of equilibrium distribution functions
        for(int k=0; k<q_value; k++){
            g_eq_T[k]  = T  * t[k] * (1.);  
        }

        //C transformation of the populations into the moment space 
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){

                m_g_T[k]     = m_g_T[k]     + M[k][j] * gin_T(i,j);
                m_g_eq_T[k]  = m_g_eq_T[k]  + M[k][j] * g_eq_T[j];
            }
        }

        //C collision in moment space
        for(int k=0; k<q_value; k++){
            m_g_T_coll[k]    = S_T_solid[k] * (m_g_T[k]  - m_g_eq_T[k]);
        }

        //C transformation back to population space
        for(int k=0; k<q_value; k++){
            for(int j=0; j<q_value; j++){
                g_T_coll[k]  = g_T_coll[k]  + M_inv[k][j]*m_g_T_coll[j];
            }
        }


        //C force for the distribution function of the Temperature
        std::array<double, 2> u_local{0.,0.};
        double F_T = 0.;//Temperature_forcing_2(i,rho_solid, u_local, T, 0., g_eq_T);    //C Temperature forcing solid wird nicht benötigt, da sonst doppelt-gemoppelt
        for(int k=0;k<q_value; k++){
            F_T_strich[k]  = t[k]*F_T* (1);
        }
        auto[iX,iY] = i_to_xyz(i);
        //C final collision equation
        for(int k=0; k<q_value; k++){
            pop_out_g_T[k] = gin_T(i,k) - g_T_coll[k]+ 1*delta_t*F_T_strich[k];  //C    + 1*0.5*pow(delta_t,2)*((F_T_strich[k] -F_T_alt(i,k)) /delta_t);
        }

        //C saving of the force for the next time step to calculate the temporal derivative
        for(int k=0; k<q_value; k++){
            //F_T_alt(i,k)  = F_T_strich[k];
        }

        return pop_out_g_T;
    }


    /// To turn class instances into a function object, the function call operator is overloaded to implement a collision-streaming cycle for a single mesh grid cell.
    void operator() (double& f0) {              
        int i = &f0 - lattice;              //C giving index i by taking the difference of the two memory possitions, which are storen in a hexadecimal system              
        if (flag[i] == CellType::bulk) {        //C doing the operations only for the bulk cells, because the treatment of wall boundaries is treated inside of the streaming step
        
            auto[iX, iY] = i_to_xyz(i);     //C calculating the coordinates corresponding to the index i

            //Open_BC(i);                  //C outflow boundary condition

            Boundaries(i);                  //C outlet-flow and outlet corner treatment
            if(Konv==true){
                Boundaries_Comp(i);             //C zero-gradient for components at n,s,o,w boundary       
                Boundaries_Temp_Reaktion(i);    //C zero-gradient for temperature at top,bottom,outlet; dirichlet at inlet
            }
 
            if(y_freeslip_wall){
                Local_Specular(i);
            }

            external_forces(f0);
            auto[rho, u] = macro(f0);           //C calculating the macroscopic density and velocity
            double usqr = 1.5 * (u[0] * u[0] + u[1] * u[1]);



            vector <double> post_collision_populations;
            if(Reaktionstestcase){

                if(Konv==false){
                    post_collision_populations = collideMRTFlow(i, rho, u, usqr);    //C calculates the new population density functions after the collision step
                
                    for (int k = 1; k < q_value; ++k) {      
                        streamFlow(i, k, iX, iY, post_collision_populations[k]);              
                    }
                    for (int k: {0}) {                              //C this number has to be adapted to the respective model, i.e. D2Q9
                        fout(i, k) =  post_collision_populations[k];
                    }
                }
                else if(Konv==true){                    //C when the flow field is converged -> calculation of temperature and component fields

                    auto[YO2,YCO2] = macroKonz(f0);
                    double T       = macroTemp(f0);

                    auto pop_out                      = collideMRTFlow(  i,rho,u,usqr);
                    auto[pop_out_g_O2, pop_out_g_CO2] = collideMRTComponents(  i,rho,u,usqr,YO2, YCO2);
                    
                    vector <double> Y_O2_interface  (q_value, 0.);      //C saves the mass fraction of O2 at the interface between solid and gas phase

                    for(int k=0;k<q_value;k++){
                        // printf("iX:%d \t iY: %d \t k:%d \tginO2: %lf \tginCO2:%lf\n",iX,iY,k,gin_O2(i,k),gin_CO2(i,k));
                    }

                    for (int k = 1; k < q_value; ++k) {      
                        streamFlow(i, k, iX, iY, pop_out[k]); 
                        auto Y_O2_int = streamKonz(i, k, iX, iY, pop_out_g_O2[k], pop_out_g_CO2[k],T,YO2,YCO2); 
                        Y_O2_interface[k]  = Y_O2_int;
                    }
                    for (int k: {0}) {                              //C this number has to be adapted to the respective model, i.e. D2Q9
                        fout    (i,k) = pop_out[k];
                        gout_O2 (i,k) = pop_out_g_O2[k];
                        gout_CO2(i,k) = pop_out_g_CO2[k];
                    }

                    auto pop_out_T  = collideMRTTemp(i,T  ,u,usqr, Y_O2_interface, rho);

                    /* if(iX<74 && iX>70 && iY<55 && iY > 25){             //C Problem ist bei X=72 am größten
                        printf("iX:%d \t iY: %d \t\t T:%lf \n", iX, iY, T);
                    } */

                    for (int k = 1; k < q_value; ++k) {      
                        streamTemp(i,k,iX,iY,pop_out_T[k], rho);
                    }

                    //C treatment of the population remaining on the lattice node
                    for (int k: {0}) {                              //C this number has to be adapted to the respective model, i.e. D2Q9
                        gout_T  (i,k) = pop_out_T[k];
                    }
                }



            }

        }
        if(flag[i] == CellType::reactive_obstacle){
                auto[iX,iY] = i_to_xyz(i);


                if(LI2014 &! first_step){
                    Boundaries_Temp_Solid(i);           //C conjugate heat transfer
                }    
                if(Solid_reaction && GUO2015){
                    Reaktion_Solid(i);
                }   

                double T = macroTemp(f0);
                auto pop_out_T_solid = collideMRTTemp_Solid(i,T);

                for (int k = 1; k < q_value; ++k) {      
                    streamTemp_solid(i,k,iX,iY,pop_out_T_solid[k]);             
                }
                for (int k: {0}) {                              //C this number has to be adapted to the respective model, i.e. D2Q9
                    gout_T(i, k) =  pop_out_T_solid[k];
                }
        }
    }
};



struct functions{

    void Einlesefunktion_F(Dim dim_1){              //C function for reading in previous velocity solution for initialization
        
        int input_dimx, input_dimy, input_dim_k;
        Dim const& dim = dim;
        std::ifstream einlesedat("ini_f.dat");
        einlesedat >> input_dimx >> input_dimy >> input_dim_k;
        
        if(input_dimx!=dim_1.nx || input_dimy != dim_1.ny || input_dim_k != q_value){
            cout << "Dimensions of the initial values do not match the dimensions of the simulation";
            exit;
        }
        cout <<"dim_1.nelem"<<dim_1.nelem;
        cout <<"\n";

        double Platzhalter;
        for(int iX=0; iX<dim_1.nx; iX++){
            for(int iY=0; iY<dim_1.ny; iY++){
                int kl=iY+iX*dim_1.ny;
                for(int k=0; k<q_value; k++){
                    einlesedat >> Platzhalter;
                    f_input(kl,k) = Platzhalter;
                }
                
            // cout << ini_values[kl] << "\n";

            }
        }
    }

    void save_converged_f_solution(LBM& lbm){       //C saves velocity solution into file for initializiing new simulation with velocity profile from previous simulation
        Dim const& dim = lbm.dim;
        std::ofstream fini("ini_f.dat");
        std::setprecision(10);
        fini << dim.nx << " " << dim.ny << " " << q_value;
        fini << "\n";
        for (int iX = 0; iX <dim.nx ; iX++) {
            for (int iY = 0; iY < dim.ny; ++iY) {
                size_t i = lbm.xyz_to_i(iX, iY);
                for(int k=0; k<q_value; k++){
                    double f_akt = lbm.fin(i,k);
                    fini <<std::setprecision(15) << f_akt << " ";
                }
                fini << "\n";        
            }
            fini << "\n"; 

        }
    }


    void Conservation_check_output(LBM& lbm)        //C ulb is the velocity in lattice units; it is defined as a global variable and passed to this function
    {
               Dim const& dim = lbm.dim;                   //C gives back the dimensions of the problem
        ofstream inlet_mass         ("inlet_massflow.dat");
        ofstream inlet_mass_legend  ("inlet_massflow_names.dat");
        ofstream outlet_mass        ("outlet_massflow.dat");
        ofstream outlet_mass_legend ("outlet_massflow_names.dat");
        ofstream inlet_momentum     ("inlet_momentumflow.dat");
        ofstream outlet_momentum    ("outlet_momentumflow.dat");        
        ofstream inlet_momentum_legend ("inlet_momentumflow_names.dat");
        ofstream outlet_momentum_legend("outlet_momentumflow_names.dat");
        ofstream massflow_Komp_i_o  ("massflow_Komp_inlet_outlet.dat");


        int x1 = 1,         x2 = dim.nx - 1;     
        double massflow_x_inlet = 0., massflow_x_outlet = 0.;

        double lmfx=0.;

        //C mass flow inlet
        inlet_mass_legend << setw(20) << "Lattice-Number-in-Y-direction"<< "\t"
                << setw(20) << "Position"<< "\t" 
                << setw(20) << "Mass-flow-x-direction-inlet-[mlu/tlu]"<< "\t"
                << setw(20) << "Mass-flow-y-direction-inlet-[mlu/tlu]" <<  "\n";
        for (int iY = 1; iY < dim.ny - 1; ++iY) {           //C saves the velocity profile in the x direction
            size_t i1 = lbm.xyz_to_i(x1, iY);           //C index on the left side
                
                double width = static_cast<double>(dim.ny - 2);
                double pos = (static_cast<double>(iY) - 0.5) / width * 2. - 1.;
                auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                lmfx = rho1 * v1[0];
                massflow_x_inlet = massflow_x_inlet + lmfx;
                inlet_mass << setw(20) << setprecision(8) << iY << "\t"<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << pos << "\t"<< "\t"
                        << setw(20) << setprecision(8) << lmfx << "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[1] << "\n";
        }


        //C mass flow outlet
        outlet_mass_legend << setw(20) << "Lattice-Number-in-Y-direction"
                << setw(20) << "Position"<< "\t" << "\t"
                << setw(20) << "Mass-flow-x-direction-outlet-[mlu/tlu]"<< "\t"
                << setw(20) << "Mass-flow-y-direction-outlet-[mlu/tlu]" <<  "\n";
        for (int iY = 1; iY < dim.ny - 1; ++iY) {           //C saves the velocity profile in the x direction
            size_t i1 = lbm.xyz_to_i(x2, iY);           //C index on the right side
                
                double width = static_cast<double>(dim.ny - 2);
                double pos = (static_cast<double>(iY) - 0.5) / width * 2. - 1.;
                auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                lmfx = rho1 * v1[0];
                massflow_x_outlet = massflow_x_outlet + lmfx;

                outlet_mass << setw(20) << setprecision(8) << iY << "\t"<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << pos << "\t"<< "\t"
                        << setw(20) << setprecision(8) << lmfx << "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[1] << "\n";
        }
        double Abweichung = (massflow_x_inlet-massflow_x_outlet)/(massflow_x_inlet);
        printf("Abweichung Massenströme:%lf %\n",Abweichung*100);

        //C Massflow andere Formel

        double f_inflow=0., f_outflow=0.;
        int iX=1;
        for(int k=1; k<dim.nx-1;k++){
            int i = lbm.xyz_to_i(iX,k);
            f_inflow  = f_inflow  + lbm.fin(i,1) + lbm.fin(i,5) + lbm.fin(i,8);
            f_outflow = f_outflow + lbm.fin(i,3) + lbm.fin(i,6) + lbm.fin(i,7);
        }
        iX = dim.nx-1;
        for(int k=1; k<dim.nx-1;k++){
            int i = lbm.xyz_to_i(iX,k);
            f_inflow  = f_inflow  + lbm.fin(i,3) + lbm.fin(i,6) + lbm.fin(i,7);
            f_outflow = f_outflow + lbm.fin(i,1) + lbm.fin(i,5) + lbm.fin(i,8);
        }
        double Abweichung_anders = (f_outflow-f_inflow)/f_inflow;
        printf("Abweichung Massenströme anders:%lf %\n",Abweichung_anders*100);



        //C momentum flux inlet
        inlet_momentum_legend << setw(20) << "Lattice-Number-in-Y-direction"
                << setw(20) << "Position"<< "\t" << "\t"
                << setw(20) << "momentum-flow-x-direction-inlet-[mlu/tlu]"<< "\t"
                << setw(20) << "momentum-flow-y-direction-inlet-[mlu/tlu]" <<  "\n";
        for (int iY = 1; iY < dim.ny - 1; ++iY) {           //C saves the velocity profile in the x direction
            size_t i1 = lbm.xyz_to_i(x1, iY);           //C index on the left side
                
                double width = static_cast<double>(dim.ny - 2);
                double pos = (static_cast<double>(iY) - 0.5) / width * 2. - 1.;
                auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                inlet_momentum << setw(20) << setprecision(8) << iY << "\t"<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << pos << "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[0] * v1[0]<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[1] * v1[1]<< "\n";
        }

        //C momentum flux outlet
        outlet_momentum_legend << setw(20) << "Lattice-Number-in-Y-direction"
                << setw(20) << "Position"<< "\t" << "\t"
                << setw(20) << "momentum-flow-x-direction-outlet-[mlu/tlu]"<< "\t"
                << setw(20) << "momentum-flow-y-direction-outlet-[mlu/tlu]" <<  "\n";
        for (int iY = 1; iY < dim.ny - 1; ++iY) {           //C saves the velocity profile in the x direction
            size_t i1 = lbm.xyz_to_i(x2, iY);           //C index on the right side
                
                double width = static_cast<double>(dim.ny - 2);
                double pos = (static_cast<double>(iY) - 0.5) / width * 2. - 1.;
                auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                outlet_momentum << setw(20) << setprecision(8) << iY << "\t"<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << pos << "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[0] * v1[0]<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[1] * v1[1]<< "\n";
        }

        //C mass flux O2 inlet and outlet
        massflow_Komp_i_o << setw(20) << "Lattice Number in Y-direction"
                << setw(20) << "Position"<< "\t" << "\t"
                << setw(20) << "Mass flow O2 inlet x-direction [mlu O2/tlu]"<< "\t"<<  "\t"
                << setw(20) << "Mass flow O2 outlet x-direction [mlu O2/tlu]" <<  "\t"  <<  "\t"            
                << setw(20) << "Mass flow CO2 inlet x-direction [mlu CO2/tlu]"<< "\t"<<  "\t"
                << setw(20) << "Mass flow CO2 outlet x-direction [mlu CO2/tlu]" <<  "\n";
        for (int iY = 1; iY < dim.ny - 1; ++iY) {           //C saves the velocity profile in the x direction
            size_t i1 = lbm.xyz_to_i(x1, iY);           //C index on the inlet
            size_t i2 = lbm.xyz_to_i(x2, iY);           //C index on the outlet
                
                double width = static_cast<double>(dim.ny - 2);
                double pos = (static_cast<double>(iY) - 0.5) / width * 2. - 1.;
                auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                auto [rho2, v2] = lbm.macro(lbm.lattice[i2]);       //C macroscopic density and velocity on both sides of the centre line
                auto [Y_O2_inlet, Y_CO2_inlet]   = lbm.macroKonz(lbm.lattice[i1]);
                auto [Y_O2_outlet, Y_CO2_outlet] = lbm.macroKonz(lbm.lattice[i2]);


                massflow_Komp_i_o << setw(20) << setprecision(8) << iY << "\t"<< "\t"<< "\t"
                        << setw(20) << setprecision(8) << pos << "\t"<< "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[0]*Y_O2_inlet << "\t"<< "\t"<<  "\t"<<  "\t"<<  "\t"<<  "\t"
                        << setw(20) << setprecision(8) << rho2 * v2[0]*Y_O2_outlet << "\t"<< "\t"<<  "\t"<<  "\t"<<  "\t"<<  "\t"<<  "\t"
                        << setw(20) << setprecision(8) << rho1 * v1[0]*Y_CO2_inlet << "\t"<< "\t"<<  "\t"<<  "\t"<<  "\t"<<  "\t"<<  "\t"
                        << setw(20) << setprecision(8) << rho2 * v2[0]*Y_CO2_outlet << "\n";
        }

        //C temperature flux inlet


        //C temperature flux outlet




    }

    auto total_Energy(LBM & lbm){                   //C calculates the total thermal energy and mean temperature of the system
        double Energy_total = 0.;
        double T_average = 0.;
        double T_middle = 0.;
        for (int iX = 1; iX < lbm.dim.nx-1; ++iX) {   
            for (int iY = 1; iY < lbm.dim.ny-1; ++iY) {
                size_t i = lbm.xyz_to_i(iX, iY);
                double Temp_lokal = lbm.macroTemp(lbm.lattice[i]);
                T_average = T_average + Temp_lokal;
                if(lbm.flag[i] == CellType::bulk){
                    Energy_total = Energy_total + rho_gas*cp_gas*Temp_lokal;
                }
                else if(lbm.flag[i] == CellType::reactive_obstacle){
                    Energy_total = Energy_total + rho_solid*cp_solid*Temp_lokal;
                }
            }
        }
        T_average = T_average / (98.*98.);

        T_middle = Energy_total / (9204*rho_gas*cp_gas + 400 * rho_solid*cp_solid); 


        return std::make_tuple(Energy_total, T_average, T_middle);
    }

    void Middleline(LBM & lbm, double time){        //C save mittleline data in file for comparison with literature values
    
        Dim const& dim = lbm.dim;                   //C gives back the dimensions of the problem


        stringstream fNameStream;
        fNameStream << "velocity_middleline_" << setfill('0') << setw(7) << time << ".dat";                 //C creates the name of the file for the current time step
        string fName1 = fNameStream.str();
        ofstream fVelocity;
        fVelocity.open(fName1.c_str());  

        /*         fNameStream << "temperature_middleline_" << setfill('0') << setw(7) << time << ".dat";                 //C creates the name of the file for the current time step
        string fName2 = fNameStream.str();
        ofstream fTemperature;
        fTemperature.open(fName2.c_str());  

        fNameStream << "O2_middleline_" << setfill('0') << setw(7) << time << ".dat";                 //C creates the name of the file for the current time step
        string fName3 = fNameStream.str();
        ofstream fO2;
        fO2.open(fName3.c_str());  

        fNameStream << "CO2_middleline_" << setfill('0') << setw(7) << time << ".dat";                 //C creates the name of the file for the current time step
        string fName4 = fNameStream.str();
        ofstream fCO2;
        fCO2.open(fName4.c_str());   */


        
        int  y1 = dim.ny / 2,
             y2 = dim.ny / 2;
        
        if (dim.ny % 2 == 0) y1--;
        

        for (int iX = 0; iX <= dim.nx - 1; ++iX) {           //C saves the velocity profile in the x direction
            size_t i1 = lbm.xyz_to_i(iX, y1);           //C index on the left side
            size_t i2 = lbm.xyz_to_i(iX, y2);           //C index on the right side
            double ux_lok=0., uy_lok=0., T_lok=0., Y_O2_lok=0., Y_CO2_lok=0.;
            
            if(iX==0){
                double ux_lok1 = Inflow_vel[y1][0];
                double ux_lok2 = Inflow_vel[y2][0];
                double uy_lok1 = Inflow_vel[y1][1];
                double uy_lok2 = Inflow_vel[y2][1];

                ux_lok = (ux_lok1 + ux_lok2) / 2.;
                uy_lok = (uy_lok1 + uy_lok2) / 2.;

                T_lok = T_inlet;

                Y_O2_lok  = Y_O2_inlet;
                Y_CO2_lok = Y_CO2_inlet;
            }
            else{
                if(lbm.flag[i1] == CellType::bulk && lbm.flag[i2] == CellType::bulk){
                    auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                    auto [rho2, v2] = lbm.macro(lbm.lattice[i2]);

                    ux_lok = (v1[0] + v2[0]) / 2.;
                    uy_lok = (v1[1] + v2[1]) / 2.;
                
                    auto [Y_O2_lok1, Y_CO2_lok1] = lbm.macroKonz(lbm.lattice[i1]);
                    auto [Y_O2_lok2, Y_CO2_lok2] = lbm.macroKonz(lbm.lattice[i2]);

                    Y_O2_lok  = (Y_O2_lok1  + Y_O2_lok2)  / 2.;
                    Y_CO2_lok = (Y_CO2_lok1 + Y_CO2_lok2) / 2.;

                }
                else{
                    ux_lok = 0.;
                    uy_lok = 0.;

                    Y_O2_lok  = 0.;
                    Y_CO2_lok = 0.;
                }

                double T_lok1 = lbm.macroTemp(lbm.lattice[i1]);
                double T_lok2 = lbm.macroTemp(lbm.lattice[i2]);
                T_lok = (T_lok1 + T_lok2) / 2.;

            }
            
            
                double width = static_cast<double>(dim.nx - 2);
                double pos = (static_cast<double>(iX) - 0.5) / width * 2. - 1.;
                auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
                auto [rho2, v2] = lbm.macro(lbm.lattice[i2]);
                fVelocity << setw(20) << setprecision(8) << pos
                          << setw(20) << setprecision(8) << ux_lok       //C interpolation of the velocity in the centre line
                          << setw(20) << setprecision(8) << uy_lok       //C interpolation of the velocity in the centre line
                          << setw(20) << setprecision(8) << T_lok       //C interpolation of the velocity in the centre line
                          << setw(20) << setprecision(8) << Y_O2_lok       //C interpolation of the velocity in the centre line
                          << setw(20) << setprecision(8) << Y_CO2_lok    << "\n";

        }
    }



};


// Save the centerline velocity profiles into a text file to compare with reference values.
//C adaption to D2Q9
void saveProfiles(LBM& lbm, double ulb)   //C ulb is the velocity in lattice units; it is defined as a global variable and passed to this function
{
    Dim const& dim = lbm.dim;                   //C gives back the dimensions of the problem
    ofstream middlex("middlex.dat");
    ofstream middley("middley.dat");
    int x1 = dim.nx / 2, x2 = dim.nx / 2,
        y1 = dim.ny / 2, y2 = dim.ny / 2;
       

    if (dim.nx % 2 == 0) x1--;                  //C if the number of lattice points in one direction is an even number -> reduce this variable i.e. x1 by one -> x1 and x2 are the coordinates on both sides of the center 
    if (dim.ny % 2 == 0) y1--;
    

    for (int iX = 1; iX < dim.nx - 1; ++iX) {           //C saves the velocity profile in the x direction
        size_t i1 = lbm.xyz_to_i(iX, y1);           //C index on the left side
        size_t i2 = lbm.xyz_to_i(iX, y2);           //C index on the right side
        if (lbm.flag[i1] != CellType::bounce_back && lbm.flag[i2] != CellType::bounce_back) {  //C the cells on both sides of the centre line have to be fluid
            double width = static_cast<double>(dim.nx - 2);
            double pos = (static_cast<double>(iX) - 0.5) / width * 2. - 1.;
            auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);       //C macroscopic density and velocity on both sides of the centre line
            auto [rho2, v2] = lbm.macro(lbm.lattice[i2]);
            middlex << setw(20) << setprecision(8) << pos
                    << setw(20) << setprecision(8) << 0.5 * (v1[0] + v2[0]) / ulb       //C interpolation of the velocity in the centre line
                    << setw(20) << setprecision(8) << 0.5 * (v1[1] + v2[1]) / ulb<< "\n";
        }
    }

    for (int iY = 1; iY < dim.ny - 1; ++iY) {           //C saves the velocity profile in the y direction
        size_t i1 = lbm.xyz_to_i(x1, iY);
        size_t i2 = lbm.xyz_to_i(x2, iY);
        if (lbm.flag[i1] != CellType::bounce_back && lbm.flag[i2] != CellType::bounce_back) {
            double width = static_cast<double>(dim.ny - 2);
            double pos = (static_cast<double>(iY) - 0.5) / width * 2. - 1.;
            auto [rho1, v1] = lbm.macro(lbm.lattice[i1]);
            auto [rho2, v2] = lbm.macro(lbm.lattice[i2]);
            middley << setw(20) << setprecision(8) << pos
                    << setw(20) << setprecision(8) << 0.5 * (v1[0] + v2[0]) / ulb
                    << setw(20) << setprecision(8) << 0.5 * (v1[1] + v2[1]) / ulb<< "\n";
        }
    }
}

// Save the velocity and density into a text file to produce images in post-processing.
void saveSlice(LBM& lbm)
{
    Dim const& dim = lbm.dim;
    std::ofstream fvx("vx.dat");
    std::ofstream fvy("vy.dat");
    std::ofstream fv("v.dat");
    std::ofstream frho("rho.dat");
    for (int iX = dim.nx - 1; iX >= 0; --iX) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            size_t i = lbm.xyz_to_i(iX, iY);
            auto [rho, v] = lbm.macro(lbm.lattice[i]);              //C calculates the macroscopic variable v and rho
            if (lbm.flag[i] == CellType::bounce_back) {
                rho = 1.0;
                v = { lbm.f(i, 0) / (6. * lbm.t[0]),                //C factor of 6 comes from the momentum exchange formula
                      lbm.f(i, 1) / (6. * lbm.t[1]) };
            }
            fvx << v[0] << " ";
            fvy << v[1] << " ";
            fv << std::sqrt(v[0] * v[0] + v[1] * v[1]) << " ";        //C mean velocity
            frho << rho << " ";

        }
        fvx << "\n";
        fvy << "\n";
        fv << "\n";
        frho << "\n";
    }
}


void saveVtkFields(LBM& lbm, int time_iter, double dx = 0.)
{
    using namespace std;
    Dim const& dim = lbm.dim;
    if (dx == 0.) {
        dx = 1. / dim.nx;
    }

    int dimZ =1;

    stringstream fNameStream;
    fNameStream << "sol_" << setfill('0') << setw(7) << time_iter << ".vtk";                 //C creates the name of the file for the current time step
    string fName = fNameStream.str();
    ofstream fStream;
    fStream.open(fName.c_str());                                                            //C opens the file with the defined name
    fStream << "# vtk DataFile Version 2.0" << endl ;                                       //C start of: write head of the vtk file
    fStream << "iteration " << time_iter << endl ;
    fStream << "ASCII" << endl ;
    fStream << endl ;
    fStream << "DATASET STRUCTURED_POINTS" << endl ;
    fStream << "DIMENSIONS " << dim.nx << " " << dim.ny << " " << dimZ << endl ;
    fStream << "ORIGIN 0 0 0" << endl ;
    fStream << "SPACING " << dx << " " << dx << " " << dx << endl ;
    fStream << endl ;
    fStream << "POINT_DATA " << dim.nx*dim.ny*dimZ << endl ;                                //C end of: write head of the vtk file

    // Density or pressure
    fStream << "SCALARS Density float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                auto [rho, v] = lbm.macro(lbm.lattice[i]);
                if (lbm.flag[i] == CellType::bounce_back || lbm.flag[i] == CellType::reactive_obstacle) {
                    rho = 1.0;
                    v = { lbm.f(i, 0) / (6. * lbm.t[0]),
                          lbm.f(i, 1) / (6. * lbm.t[1])};
                }
                fStream << rho << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
    fStream << endl ;

    //C Temperature
    fStream << "SCALARS Temperature float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                double T = lbm.macroTemp(lbm.lattice[i]);

                fStream << T << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
    fStream << endl ; 

     // Mass fraction 1
    fStream << "SCALARS mass_fraction_O2 float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                double out;
                if(Reaktionstestcase){
                    auto [YO2,YCO2] = lbm.macroKonz(lbm.lattice[i]);
                    out = YO2;
                }
                else{
                    out = 0.;
                }
                
                if (lbm.flag[i] == CellType::bounce_back || lbm.flag[i] == CellType::reactive_obstacle) {
                    out = 0.;
                }
                fStream << out << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
    fStream << endl ;

    // Mass fraction 2
    fStream << "SCALARS mass_fraction_CO2 float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                                double out;
                if(Reaktionstestcase){
                    auto [YO2,YCO2] = lbm.macroKonz(lbm.lattice[i]);
                    out = YCO2;
                }
                else{
                    out = 0.;
                }
                if (lbm.flag[i] == CellType::bounce_back || lbm.flag[i] == CellType::reactive_obstacle) {
                    out = 0.;
                }
                fStream << out << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
    fStream << endl ;
    

    // velocity_X
    fStream << "SCALARS velocity_X float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                auto [rho, v] = lbm.macro(lbm.lattice[i]);
                if (lbm.flag[i] == CellType::bounce_back || lbm.flag[i] == CellType::reactive_obstacle) {
                    v = { lbm.f(i, 3) / (6. * lbm.t[3]),
                          lbm.f(i, 4) / (6. * lbm.t[4])};
                }
                fStream << v[0] << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
    fStream << endl ;

    // velocity_Y
    fStream << "SCALARS velocity_Y float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                auto [rho, v] = lbm.macro(lbm.lattice[i]);
                if (lbm.flag[i] == CellType::bounce_back || lbm.flag[i] == CellType::reactive_obstacle) {
                    v = { lbm.f(i, 3) / (6. * lbm.t[3]),
                          lbm.f(i, 4) / (6. * lbm.t[4])};
                }
                fStream << v[1] << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
    fStream << endl ;

    // velocity_Z
    fStream << "SCALARS velocity_Z float 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                auto [rho, v] = lbm.macro(lbm.lattice[i]);
                if (lbm.flag[i] == CellType::bounce_back || lbm.flag[i] == CellType::reactive_obstacle) {
                    v = { lbm.f(i, 3) / (6. * lbm.t[3]),
                          lbm.f(i, 4) / (6. * lbm.t[4])};
                }
                fStream << 0 << " ";
            }
            fStream << endl ;
        }
        fStream << endl ;
    }

    // Wall flag matrix
    fStream << "SCALARS Flag int 1" << endl ;
    fStream << "LOOKUP_TABLE default" << endl ;
    for (int iZ = dimZ-1; iZ >=0; --iZ) {
        for (int iY = 0; iY < dim.ny; ++iY) {
            for (int iX = 0; iX < dim.nx; ++iX) {
                size_t i = lbm.xyz_to_i(iX, iY);
                if (lbm.flag[i] == CellType::bounce_back) {
                    fStream << "1" << " ";
                }
                else if (lbm.flag[i] == CellType::specular_reflection) {
                    fStream << "3" << " ";
                }
                else if (lbm.flag[i] == CellType::reactive_obstacle) {
                    fStream << "2" << " ";
                }
                else {
                    fStream << "0" << " ";
                }
            }
            fStream << endl ;
        }
        fStream << endl ;
    }
}


// Compute the average kinetic energy in the domain.
double computeEnergy(LBM& lbm)
{                                                                                   //C computes kinetic energy by iterating over the whole domain
    Dim const& dim = lbm.dim;
    double energy = 0.;
    for (int iX = 0; iX < dim.nx; ++iX) {   
        for (int iY = 0; iY < dim.ny; ++iY) {
            
            size_t i = lbm.xyz_to_i(iX, iY);                                //C compute position in the vector based on the coordinates
            if (lbm.flag[i] != CellType::bounce_back && lbm.flag[i] != CellType::reactive_obstacle) {                         //C evaluates only the parts of the domain, which are considered to be filled by liquid
                auto[rho, v] = lbm.macro(lbm.lattice[i]);                       //C computes the macroscopic variables 
                energy += v[0] * v[0] + v[1] * v[1];              //C sums up the total kinetic energy for the whole system
            }
        }
    }
    energy *= 0.5;                                                                  //C multiply by 0.5 according to E=0.5*m*v²
    return energy;
}

void iniCavity(LBM& lbm, double ulb, vector<double> const& ulid) {
    Dim const& dim = lbm.dim;
    
    

    for (size_t i = 0; i < dim.nelem; ++i) {
        auto[iX, iY] = lbm.i_to_xyz(i);
        
        if(Cavity){
            if (iX == 0 || iX == dim.nx-1 || iY == dim.ny-1  || iY == 0) {
                
                if (iX == dim.nx-1) {
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = - 6. * lbm.t[k] * ulb * (
                            lbm.c[k][0] * ulid[0] + lbm.c[k][1] * ulid[1] );     //C adjustment of the momentum for the moving solid wall equ. 8.17 book Mohamad
                    }
                }
                else {
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }
            }
            else {                                          //C sets the middle of the domain (bulk phase) to type bulk
                lbm.flag[i] = CellType::bulk;
            }
        }
        else if(Couette){

            if (iX == 0 || iX == dim.nx-1 ) {
                if(x_periodic){
                    lbm.flag[i] = CellType::bulk;
                }
                else if(x_noslip){
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }
            }
            else if( iY ==0 || iY == dim.ny-1){
                if(y_periodic){
                    lbm.flag[i] = CellType::bulk;
                }
                else if(y_noslip){
                    if(iY == dim.ny-1){
                        lbm.flag[i] = CellType::bounce_back;
                        for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = - 6. * lbm.t[k] * ulb * (
                                lbm.c[k][0] * ulid[0] + lbm.c[k][1] * ulid[1] );     //C adjustment of the momentum for the moving solid wall equ. 8.17 book Mohamad
                        }
                    }
                    else{    
                        lbm.flag[i] = CellType::bounce_back;
                        for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                        }
                    }
                }
            }
            else {                                          //C sets the middle of the domain (bulk phase) to type bulk
                lbm.flag[i] = CellType::bulk;
            }
            

            if(y_noslip || x_noslip){
                if(iY ==0 ){
                    if(iX == 0 || iX == dim.nx-1){
                        lbm.flag[i] = CellType::bounce_back;
                        for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                        }
                    }
                }
                if( iY == dim.ny-1){
                    if(iX == 0 || iX == dim.nx-1){
                        lbm.flag[i] = CellType::bounce_back;
                        for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = - 6. * lbm.t[k] * ulb * (
                                lbm.c[k][0] * ulid[0] + lbm.c[k][1] * ulid[1] );     //C adjustment of the momentum for the moving solid wall equ. 8.17 book Mohamad
                        }
                    }
                }
            }
        }
        else if(Poiseuille){

            if (iX == 0 || iX == dim.nx-1 ) {
                if(x_periodic){
                    lbm.flag[i] = CellType::bulk;
                }
                else if(x_noslip){
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }
            }
            else if( iY ==0 || iY == dim.ny-1){
                if(y_periodic){
                    lbm.flag[i] = CellType::bulk;
                }
                else if(y_noslip){
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    } 
                }
            }
            else {                                          //C sets the middle of the domain (bulk phase) to type bulk
                lbm.flag[i] = CellType::bulk;
            }
            

            if(y_noslip || x_noslip){
                if(iY ==0 || iY == dim.ny-1){
                    if(iX == 0 || iX == dim.nx-1){
                        lbm.flag[i] = CellType::bounce_back;
                        for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                        }
                    }
                }
            }
        }
        else if(Inflow_Outflow){
            
            if (iX == 0 || iX == dim.nx-1 ) {
                if(x_periodic){
                    lbm.flag[i] = CellType::bulk;
                }
                else if(x_noslip){
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }
                else if(x_inflow_outflow){
                    if(iX==0){

                        //lbm.flag[i] = CellType::bulk;

                        lbm.flag[i] = CellType::bounce_back; 
                        for (int k = 0; k < q_value; ++k) {
                                lbm.f(i, k) = - 6. * lbm.t[k] * 1 * (
                                    lbm.c[k][0] * Inflow_vel[iY][0] + lbm.c[k][1] * Inflow_vel[iY][1] );     //TODO anpassen für inflow-bc

                        }

                    }
                    if(iX==dim.nx-1){
                        lbm.flag[i] = CellType::bulk;

                        //lbm.flag[i] = CellType::bounce_back;
                        /* for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                        } */
                    }
                }
                else if(x_freeslip_wall){
                lbm.flag[i] = CellType::specular_reflection;
                }
            }
            else if( iY ==0 || iY == dim.ny-1){
                if(y_periodic){
                    lbm.flag[i] = CellType::bulk;
                }
                else if(y_noslip){
                    lbm.flag[i] = CellType::bounce_back;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    } 
                }
                /* else if(y_freeslip_wall){
                lbm.flag[i] = CellType::specular_reflection;
                } */
                else {                                          //C sets the middle of the domain (bulk phase) to type bulk
                lbm.flag[i] = CellType::bulk;
                }
            }
            else {                                          //C sets the middle of the domain (bulk phase) to type bulk
                lbm.flag[i] = CellType::bulk;
            }
            

            if(y_noslip || x_noslip){
                if(iY ==0 || iY == dim.ny-1){
                    if(iX == 0 || iX == dim.nx-1){
                        lbm.flag[i] = CellType::bounce_back;
                        for (int k = 0; k < q_value; ++k) {
                            lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                        }
                    }
                }
            }

            /* if(y_freeslip_wall || x_freeslip_wall){
                if(iY ==0 || iY == dim.ny-1){
                    if(iX == 0 || iX == dim.nx-1){
                        lbm.flag[i] = CellType::specular_reflection;
                    }
                }
            } */


        }
        



        if(Obstacle1){      //C 1 obstacle 

            if(iY >= dim.ny*3./8. && iY < dim.ny*5./8.){

                if(iX> 49 && iX < 70){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

            }

            /* if(iY >0 && iY < dim.ny-1){

                if(iX> 49 && iX < 70){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

            } */

        }


        if(Obstacles){      //C 4 obstacles for the reaction test case from Xu 2018

            if(iY >= dim.ny*3./8. && iY < dim.ny*5./8.){

                if(iX> 99 && iX < 121){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

                if(iX> 140 && iX < 161){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

                if(iX> 180 && iX < 201){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

                if(iX> 220 && iX < 241){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

            }

        }

        if(ObstaclesTest){      //C 4 obstacles for the reaction test case from Xu 2018

            if(iY >= dim.ny*3./8. && iY < dim.ny*5./8.){

                if(iX>= dim.nx*5/21 && iX < dim.nx*6/21){ 
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

                if(iX>= dim.nx*7/21 && iX < dim.nx*8/21){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

                if(iX>= dim.nx*9/21 && iX < dim.nx*10/21){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

                if(iX>= dim.nx*11/21 && iX < dim.nx*12/21){
                    lbm.flag[i] = CellType::reactive_obstacle;
                    for (int k = 0; k < q_value; ++k) {
                        lbm.f(i, k) = 0.;           //C stationary solid wall -> momentum does not need to be adjusted
                    }
                }

            }

        }
        



    }    
}

// Runs a simulation of a flow in a lid-driven cavity using the two-population scheme.
void runCavityTwoPop(bool benchmark, double Re, double ulb, int N_x, int N_y,double max_t)                                //C inputs, which are given at the function call in the main() are declared as global Variables
{                                                                                                               //C at the top of the Code
    // CellData is either a double (structure-of-array) or an array<double, 19> (array-of-structure).
    using CellData = typename LBM::CellData;                                                                    //C is defined in struct LBM; the keyword "typename" is a placeholder for a type;
    vector<double> ulid{1. , 0.}; // Velocity on top lid in dimensionless units.                            //C LBM::CellData is synonymous with the type Double -> CellData in this function is now 
    vector<double> vlid;
    //double radius = 1.;      //C radius is used for initializing the geometry                                   //C double as well
    

    double E_now, E_prev;
    int Zaehler = 0;

    int out_freq  = 1000;      // Non-benchmark mode: Frequency in LU for output of terminal message and profiles (use 0 for no messages) 20 for compressible
    int data_freq = 0;        // Non-benchmark mode: Frequency in LU of full data dump (use 0 for no data dump)
    int vtk_freq = 1000;       //C was added by Claudius; frequency for the output in vtk files       50 for compressible
    int bench_ini_iter = 1000; // Benchmark mode: Number of warmup iterations
    int bench_max_iter = 2000; // Benchmark mode: Total number of iteration
    bool periodic = false;
    int q_value = 9;
    int d_value = 2;
    double omega_glob = 1.0;        //C war 1.8, führte aber zu Instabilitäten
    double Faktor = 1.;       //C Faktor for relaxation times in MRT of flow field
    double Faktor2= 1.;         //C Faktor for relaxation times in MRT of the Concentrations
    double Faktor3= 1.;         //C Faktor for relaxation times in MRT of the Temperature

    bool first_step = true;

    double CS2 = 1./3.;         //C lattice speed of sound squared

    //C parameters of the simulation

    double delta_t_lu = 1.;                 //C size of the time-step [t-lu]
    double delta_t_ph = 1./48771706.22;     //C size of the time-step [s]



    //C model parameters

    double D_n          = 1.;           //C diffusion coefficient of component n; sollte eigentlich ein Vektor sein, falls versch. Komponenten versch. Diff-koeff. haben
    double D_O2         = 0.1568739;    //C diffusion coefficient O2
    double D_CO2        = 0.1425984;    //C diffusion coefficient CO2
    double alpha_gas    = 0.2337421;    //C Wärmeleitfähigkeit für Gas
    double alpha_solid  = 0.0278153;    //C Temperaturleitfähigkeit a  =^= im Englischen thermal diffusivity alpha
    double ny_gas       = 0.1673593;    //C kinemat viscosity v
    double lambda_gas   = 83.251305;

    double Tau_flow     = 1.0;                //C (ny_gas   /CS2)+0.5;
    double Tau_O2       = 1.021;              //C (D_O2     /CS2)+0.5;
    double Tau_CO2      = 0.973;              //C (D_CO2    /CS2)+0.5;
    double Tau_T        = 1.198;              //C (alpha_gas/CS2)+0.5;
    double Tau_T_solid  = 0.583;              //C (alpha_solid/CS2)+0.5;

    double cp_gas   = 356.16739;    //C                     spezifische Wärmekapazität of the gas
    double cp_solid = 235.4266;     //C                     specific heat capacity of the coke
    double rho_gas  = 1.;           //C [m-lu/s-lu^3]       density of the gas
    double rho_solid = 556.694;     //C [m-lu/s-lu^3]       density of the coke
    double M_O2     = 32.;          //C [m-lu/mol-lu]       molar mass of Oxygen
    double M_CO2    = 44.;          //C [m-lu/mol-lu]       molar mass of carbon dioxyde

    double Sigma = (rho_solid*cp_solid)/(rho_gas*cp_gas);

    double R_id     = 2701.8026;    //C ideale Gaskonstante     [lu]

    //C parameters of the reaction

    int stoich_O2  = -1;
    int stoich_CO2 = 1;
    double Prae_exp_factor = 1.992343666625*pow(10,5);  // [lu]
    double E_akt   = 5.511041294*pow(10,4);                       // [lu]
    double delta_hr = 1.8132054257*pow(10,5);        //C1.633259243791*pow(10,5);                      // [lu]   1.633259243791*pow(10,5) pow(10,4) zu viel wurde nach 17% schon 1,8 warm
    //C     1.633259243791*pow(10,3) ein bissle zu wenig; bei t=0.006s war T 1.065 und Y_O2 noch zu hoch vor erstem obstacles

    //C Reaktionsparameter in SI-Einheiten

    double E_akt_SI = 131.09;           //C [kJ/mol]
    double A_SI     = 9.717*pow(10,6);  //C [m/s]
    double hr_SI    = 388.5;            //C [kJ/mol]
    double R_SI     = 8.314;            //C [J/(mol*K)]
    double dx_SI    = 1.002*pow(10,-6); //C [m]
    double dt_SI    = 2.05*pow(10,-8);  //C [s]
    double rho_SI   = 4.5;              //C [kg/m^3]
    double cp_SI    = 1.096;            //C [kJ/(kg*K)]
    double M_O2_SI  = 32.;               //C [g/mol]

    //C parameters of the inlet

    double u_lb         = 0.00011741168;              // [lu]  0.00011741168   //C laut Timan 0.0001166      //C bei 0.2 gibt es die Probleme mit T-divergenz am Rechten Rand des Obstacles
    double T_inlet      = 1.;                      // [lu]
    double Y_O2_inlet   = 0.22;                    // [lu]
    double Y_CO2_inlet  = 0.0;                     // [lu]
    double rho_inlet    = 1.;                      // [lu]

    //C Values at boundary conditions

    double T_links  = 0.0;
    double T_rechts = 0.0;

    //C parameters for initialization of the domain

    double T_ini    = 1.0;          //C Temperatur Gas
    double T_ini2   = 2.0;          //C temperatur solid, only in combination with obstacles_temp
    double rho_ini  = 1.;
    double Y_O2_ini = 0.;
    double Y_CO2_ini = 0.;
    double T_ref        = T_ini;


    bool Obstacles = true;
    bool ObstaclesTest = false;
    bool Obstacle1 = false;
    bool Konjug_Waermetransport = true;         //C toggles conjugated heat transfer in the stream_temp


    //C choice of konjugated heat transfer BC

    bool GUO2015 = false;
    // this works
    bool LI2014  = true;           //C bounce-back mit extratermL
    bool HUBER2015 = false;          //C normales Treatment mit force-term

    int NC = 2;                 //C number of components in the fluid flow


    //C choice if single relaxation time or multi relaxation time model
    bool MRT = true;

    //C choice of test cases
    bool Cavity             = false;
    bool Couette            = false;
    bool Poiseuille         = false;
    bool Inflow_Outflow     = true;             //C simulation of flow with inflow/outflow BC rather than force-driven
    bool Multi_Component    = true;            //C enables multiple components for the gas phase
    bool Temperature_coupling = false;      //C toggles beta in f_eq and C_dach in Kollisionsschritt für flowfield f
    bool Solid_reaction     = true;             //C toggles the reaction at the interface of solid/gas


    //C Vorkonfigurierte Testcases
    bool Reaktionstestcase  = true;
    bool Flowfield          = false;
    bool Testsimulation = false;

    bool Vorwaertsdiff  = false;
    bool Zentraldiff    = false;

    //C MRT Parameters for the flow field
    double w_q      = 1.0;            //C free parameter to tune
    double w_eps    = 1.;          //C free parameter to tune
    double w_e      = 1.0;            //C connected to bulk viscosity     //C in general, decrease           bei 0.5 und 1.2 explosion, zwischen 0.7 und 1 gehts; konvergiert schneller bei 0.7
    double w_ny     = 1./Tau_flow;       //1./Tau_flow;  //C connected to shear viscosity

    //C MRT Parameters for Component O2
    double w_qO2    = 1.;
    double w_epsO2  = 1.;
    double w_eO2    = 1.;
    double w_nyO2   = 1./Tau_O2;

    //C MRT Parameters for Component CO2
    double w_qCO2   = 1.;
    double w_epsCO2 = 1.;
    double w_eCO2   = 1.;
    double w_nyCO2  = 1./Tau_CO2;

    //C MRT Parameters for Temperature
    double w_qT = 1.;
    double w_epsT = 1.;
    double w_eT = 1.;
    double w_nyT = 1./Tau_T;

    //C MRT Parameters for Temperature solid
    double w_qT_solid = 1.;
    double w_epsT_solid = 1.;
    double w_eT_solid = 1.;
    double w_nyT_solid = 1./Tau_T_solid;

    //C configuration of boundary conditions
    bool x_noslip = false;
    bool y_noslip = true;
    bool x_periodic = false;
    bool y_periodic = false;
    bool x_inflow_outflow = true;
    bool x_freeslip_wall = false;                   //C toggles free-slip wall BC; implemented with specular reflection
    bool y_freeslip_wall = false;
    bool links_dirichlet_temp = false;
    bool rechts_dirichlet_temp = false;

    bool inlet_parabolic = true;
    bool inlet_constant  = false;

    double dpdx = 0.;           //C 0.00005;
    double dpdy = 0.;

    bool External_force = false;
    double efx = 0;
    double efy = 0;

    bool timestep_stop = false;
    bool print_temp    = true;     //C prints thermodynamical Mitteltemperatur


    //C Variables to configure reading in variables or saving them (-> for reading in those variables in the next simulation)
    bool Einleseoption_f = true;        //C true enables the reading in of the distribution function f for the flow field
    bool Ausgabeoption_f = false;       //C true enables printing the values of f into a file (for reading in the next simulation)




    //C u_lb auf 0 setzen und Konv auf true, um reines Diffusionsproblem zu haben
    bool Konv = true;          //C used to pass the konv information into the operator function
    double Konvergenzabweichung = 0.00001;          //C 0.00001 for u_lb = 0.0002

    //C definition of the flow profile at the inlet
    if(x_inflow_outflow){


        if(inlet_parabolic){
            for(int n=0; n<=N_y-1; n++){
            double y = (double)n/(N_y-1);
            Inflow_vel[n][0] = 4*(y)*(1-y)*u_lb;
            Inflow_vel[n][1] = 0.;
            }
        }
        if(inlet_constant){
            for(int n=0; n<=N_y-1; n++){
            double y = (double)n/(N_y-1);
            Inflow_vel[n][0] = 1.*u_lb;
            Inflow_vel[n][1] = 0.;
            }
        }
        
    }
    Dim dim {N_x, N_y};                                                                                      //C structure which defines the dimensions of the problem as well as the overall number of populations needed for the problem

    auto[nu, omega, dx, dt] = lbParameters(ulb, N - 2, Re);                                                 //C computes lattice unit variables as well as the time and space discretization step size and returns them as a tuple
    printParameters(benchmark, Re, omega, ulb, N, max_t, dt);                                                   //C prints out parameters to the console


    TC = 0;
    if(Temperature_coupling){
        TC = 1;
    }

    int length_lattice;
    if(Multi_Component){
        length_lattice = (1+NC+1)*LBM::sizeOfLattice(dim.nelem);             //C 1 because of the velocity field, 1 for the temperature field and N_Comp for the number of components in the fluid flow
    }
    else{
        length_lattice = LBM::sizeOfLattice(dim.nelem);
    }
    vector<CellData> lattice_vect(length_lattice);                                           //C std::vector<> is a sequence container that encapsulates dynamic size arrays; a vector with the name lattice_vect is created
    CellData *lattice = &lattice_vect[0];                                                                   //C set pointer "lattice" to the vector lattice_vect; alternate syntax: CellData* lattice = lattice_vect;

    // The "vector" is used as a convenient way to allocate the flag array on the heap.
    vector<CellType> flag_vect(dim.nelem);                                                                  //C allocates a vector called "flag_vect"
    CellType* flag = &flag_vect[0];                                                                         //C vector gets pointed onto pointer flag 

    vector<int> parity_vect {0};        //C creates vector "parity_vec" with entry 0; it is the variable which states in which of the two populations the pre-collision equilibrium density funcions are stored
    int* parity = &parity_vect[0];      //C writes address of vector "parity_vec" onto the pointer "parity"


    auto[c, opp, t, M, M_inv, S, S_GO2, S_GCO2, S_T, S_T_solid] = d2q9_constants();                                                                    //C the function "d3q19_constants" returns a tuple containing the velocity vectors, the information about their orientation (opposite vectors)
                                                                                                            //C and the weighting factors (t)


    // Instantiate the function object for the for_each call.
    LBM lbm{lattice, flag, parity, dx, &c[0], &opp[0], &S[0],&S_GO2[0],&S_GCO2[0],&S_T[0], &S_T_solid[0], &t[0], &M[0], &M_inv[0], omega, dim, dt};                                  //C lattice is the Celldata, flag contains ..., parity is the information in which set of populations the pre-coll stuff is 
                                                                                                        //C c are the velocity vectors, opp is a vector with the information about which velocity vectors are opposite of each other
                                                                                                        //C t is a vector with the weighting factors for the directions (velocity vectors)
    
    functions funct{};

    if(Einleseoption_f){
        funct.Einlesefunktion_F(dim);                                                        //C read in of the initial values from file in case it is chosen as an option
    }
    // Initialize the populations.
    for_each(lattice, lattice + dim.nelem, [&lbm](CellData& f0) { lbm.iniLattice(f0); });       ///for_each : Applies function fn to each of the elements in the range [first,last).    //C #TODO check if the implementation is like this because of
                                                                                                                                                                                        //C structure of arrays or array of structures
    iniCavity(lbm, ulb, ulid);                                                                          //C the result is a vector calles lbm.flag, which stores the information about the type of the node for every node of the lattice
    // Reset the clock, to be used when a benchmark simulation is executed.
    auto[start, clock_iter] = restartClock();
    // The average energy, dependent on time, can be used to monitor convergence, or statistical
    // convergence, of the simulation.
    ofstream energyfile("energy.dat");                                                                              //C data type represents output file stream; is used to create files and to write information to files
    ofstream enthalpyfile("thermal-energy.dat");     



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////// TIME LOOP ///////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Maximum number of time iterations depending on whether the simulation is in benchmark mode or production mode.
    int max_time_iter = benchmark ? bench_max_iter : static_cast<int>(max_t / dt);                                  //C if bool benchmark == true -> use bench_max_iter else static_cast<int>(max_t/dt)
    double t_ph, t_ph_prev=0.;
    double Abweichung_enthalpie=0.;
    double th_0=0.;

    for (int time_iter = 0; time_iter <= max_time_iter; ++time_iter) {                                               //C loops over all time steps
        t_ph = time_iter*delta_t_ph;
        //lbm.Outflow_BC();
        
        
        if (benchmark && time_iter == bench_ini_iter)  {                                                            //C if it is a benchmark run and the time step is exactly the last step of the warm up runs; so the last time step of the warmup runs
            cout << "Now running " << bench_max_iter - bench_ini_iter
                 << " benchmark iterations." << endl;                                                               //C console print the number of iterations left -> number of benchmark iterations = total number - warmup steps
            tie(start, clock_iter) = restartClock();                                                                //C timer gets reset
        }

        if (!benchmark && data_freq != 0 && time_iter % data_freq == 0 && time_iter > 0) {  //C save Slice
            saveSlice(lbm);                                                                                       
        }

        if (!benchmark && vtk_freq != 0 && time_iter % vtk_freq == 0 && time_iter >= 0) {   //C save VTK
            saveVtkFields(lbm, time_iter);
        }

        if (!benchmark && out_freq != 0 && time_iter % out_freq == 0 && time_iter > 0) {    //C check of convergency of the flow field by total kinetic energy change over time
            
            
        double Dev;
        if(Konv==false){            //C convergency check for the flow field
            E_now = computeEnergy(lbm) *dx*dx / (dt*dt);   
            double Deviation = abs((E_now-E_prev)/E_prev);
            if(Deviation<= Konvergenzabweichung){
                Zaehler = Zaehler + 1;

                if(Zaehler >= 4){
                    printf("Flowfield is converged with a deviation of %lf at timestep %d \n\n\n\n",Deviation*100,time_iter);
                    Konv = false;
                }
            }
            else{
                Zaehler = 0;
            }
            E_prev = E_now;
            Dev = Deviation;
        }
        
        
        cout << "Saving profiles at iteration " << time_iter                                                    //C AND time > 0        print the currently saved time step
                << ", t = " << setprecision(4) << time_iter * delta_t_ph << "s "<< setprecision(3)
                << " [" << time_iter * dt / max_t * 100. << "%]" 
                << "Abweichung ist " << Dev*100 << "%" << endl;
        saveProfiles(lbm, ulb);     //C saves the velocity profile in the centre line in x and y direction
        double energy = computeEnergy(lbm) *dx*dx / (dt*dt);    
                                                        //C returns 0.5*v² -> multiplication with dx²/dt² (making it dimensional?)
        cout << "Average kinetic energy: " << setprecision(8) << energy << endl;
        energyfile << setw(10) << time_iter * dt << setw(16) << setprecision(8) << energy << endl;
        }

        if(time_iter>0){

        }
        if(time_iter==0){
            E_prev = computeEnergy(lbm) *dx*dx / (dt*dt);
        }

        if(x_inflow_outflow){
            if( (t_ph > 0.002 && t_ph_prev < 0.002) || t_ph == 0.002){
                            funct.Middleline(lbm,t_ph);
            }
            if( (t_ph > 0.006 && t_ph_prev < 0.006) || t_ph == 0.006){
                            funct.Middleline(lbm,t_ph);
            }
            if( time_iter == max_time_iter){
                            funct.Middleline(lbm,t_ph);
            }
        }

        if (!benchmark && out_freq != 0 && time_iter % out_freq == 0 && time_iter >= 0) {  //C calculation of thermal energy
            
            auto [thermal_energy, t_average, t_middle] = funct.total_Energy(lbm);   
             
                                                            //C returns 0.5*v² -> multiplication with dx²/dt² (making it dimensional?)
            //cout << "Total enthalpy: " << setprecision(8) << thermal_energy << endl;
            //cout << "T_average: " << setprecision(8) << t_average << endl;
            //cout << "Mitteltemperatur: " << setprecision(8) << t_middle << endl;
            enthalpyfile << setw(10) << time_iter * dt_SI << setw(16) << setprecision(8) << thermal_energy << endl;
        }


        for_each(execution::par, lattice, lattice + dim.nelem, lbm);     //C applies function lbm to each element in the range lattice -> lattice+dim.nelem


        // After a collision-streaming cycle, swap the parity for the next iteration.
        *parity = 1 - *parity;     //C change parity to the jeweils other value
        ++clock_iter;

        if(time_iter==0){
            first_step = false;
        }

        int zaehler =0;

        char w;
        if(timestep_stop){
        printf("Press c to continue\n");
        scanf("%c", &w);
        while(getchar()!='c'){}
        }
        t_ph_prev = t_ph;
    }

    funct.Conservation_check_output(lbm);

    if(Konv){
        printf("Die Simulation ist konvergiert\n");
    }
    else{
        printf("Die Simulation ist nicht konvergiert\n");
    }


    if(Ausgabeoption_f){
    funct.save_converged_f_solution(lbm);       //C saves the converged solution for initializing a new simulation with it
    }


    //if (benchmark) {
        printMlups(start, clock_iter, dim.nelem);
    //}
}

int main() {

    double Re        = 200.;  // Reynolds number                                        //C the first 5 global variables, which are declared here, are the inputs for the function runCavitytwoPop, which is called
    double ulb       = 0.002;  // Velocity in lattice units                              //C in the main-function of the program
    int N            = 16;   // Number of nodes in x-direction //C used to be 128
    int N_X = 420;      //C 420         (davor 200)     120
    int N_Y = 80;       //C 80                          80
    bool benchmark   = false;  // Run in benchmark mode ?
    int out_freq  = 1000;      // Non-benchmark mode: Frequency in LU for output of terminal message and profiles (use 0 for no messages) 20 for compressible
    int data_freq = 0;        // Non-benchmark mode: Frequency in LU of full data dump (use 0 for no data dump)
    int vtk_freq = 1000;       //C was added by Claudius; frequency for the output in vtk files       50 for compressible
    int bench_ini_iter = 1000; // Benchmark mode: Number of warmup iterations
    int bench_max_iter = 2000; // Benchmark mode: Total number of iteration
    bool periodic = false;
    int q_value = 9;
    int d_value = 2;
    double omega_glob = 1.0;        //C war 1.8, führte aber zu Instabilitäten
    double Faktor = 1.;       //C Faktor for relaxation times in MRT of flow field
    double Faktor2= 1.;         //C Faktor for relaxation times in MRT of the Concentrations
    double Faktor3= 1.;         //C Faktor for relaxation times in MRT of the Temperature

    bool first_step = true;

    double CS2 = 1./3.;         //C lattice speed of sound squared

    //C parameters of the simulation

    double max_t     = 487717.0;              // number of time steps   //C Konvergenz bei ca. 90% von 30000 steps      //C value according to paper 487717.0
    double delta_t_lu = 1.;                 //C size of the time-step [t-lu]
    double delta_t_ph = 1./48771706.22;     //C size of the time-step [s]



    //C model parameters

    double D_n          = 1.;           //C diffusion coefficient of component n; sollte eigentlich ein Vektor sein, falls versch. Komponenten versch. Diff-koeff. haben
    double D_O2         = 0.1568739;    //C diffusion coefficient O2
    double D_CO2        = 0.1425984;    //C diffusion coefficient CO2
    double alpha_gas    = 0.2337421;    //C Wärmeleitfähigkeit für Gas
    double alpha_solid  = 0.0278153;    //C Temperaturleitfähigkeit a  =^= im Englischen thermal diffusivity alpha
    double ny_gas       = 0.1673593;    //C kinemat viscosity v
    double lambda_gas   = 83.251305;

    double Tau_flow     = 1.0;                //C (ny_gas   /CS2)+0.5;
    double Tau_O2       = 1.021;              //C (D_O2     /CS2)+0.5;
    double Tau_CO2      = 0.973;              //C (D_CO2    /CS2)+0.5;
    double Tau_T        = 1.198;              //C (alpha_gas/CS2)+0.5;
    double Tau_T_solid  = 0.583;              //C (alpha_solid/CS2)+0.5;

    double cp_gas   = 356.16739;    //C                     spezifische Wärmekapazität of the gas
    double cp_solid = 235.4266;     //C                     specific heat capacity of the coke
    double rho_gas  = 1.;           //C [m-lu/s-lu^3]       density of the gas
    double rho_solid = 556.694;     //C [m-lu/s-lu^3]       density of the coke
    double M_O2     = 32.;          //C [m-lu/mol-lu]       molar mass of Oxygen
    double M_CO2    = 44.;          //C [m-lu/mol-lu]       molar mass of carbon dioxyde

    double Sigma = (rho_solid*cp_solid)/(rho_gas*cp_gas);

    double R_id     = 2701.8026;    //C ideale Gaskonstante     [lu]

    //C parameters of the reaction

    int stoich_O2  = -1;
    int stoich_CO2 = 1;
    double Prae_exp_factor = 1.992343666625*pow(10,5);  // [lu]
    double E_akt   = 5.511041294*pow(10,4);                       // [lu]
    double delta_hr = 1.8132054257*pow(10,5);        //C1.633259243791*pow(10,5);                      // [lu]   1.633259243791*pow(10,5) pow(10,4) zu viel wurde nach 17% schon 1,8 warm
    //C     1.633259243791*pow(10,3) ein bissle zu wenig; bei t=0.006s war T 1.065 und Y_O2 noch zu hoch vor erstem obstacles

    //C Reaktionsparameter in SI-Einheiten

    double E_akt_SI = 131.09;           //C [kJ/mol]
    double A_SI     = 9.717*pow(10,6);  //C [m/s]
    double hr_SI    = 388.5;            //C [kJ/mol]
    double R_SI     = 8.314;            //C [J/(mol*K)]
    double dx_SI    = 1.002*pow(10,-6); //C [m]
    double dt_SI    = 2.05*pow(10,-8);  //C [s]
    double rho_SI   = 4.5;              //C [kg/m^3]
    double cp_SI    = 1.096;            //C [kJ/(kg*K)]
    double M_O2_SI  = 32.;               //C [g/mol]

    //C parameters of the inlet

    double u_lb         = 0.00011741168;              // [lu]  0.00011741168   //C laut Timan 0.0001166      //C bei 0.2 gibt es die Probleme mit T-divergenz am Rechten Rand des Obstacles
    double T_inlet      = 1.;                      // [lu]
    double Y_O2_inlet   = 0.22;                    // [lu]
    double Y_CO2_inlet  = 0.0;                     // [lu]
    double rho_inlet    = 1.;                      // [lu]

    //C Values at boundary conditions

    double T_links  = 0.0;
    double T_rechts = 0.0;

    //C parameters for initialization of the domain

    double T_ini    = 1.0;          //C Temperatur Gas
    double T_ini2   = 2.0;          //C temperatur solid, only in combination with obstacles_temp
    double rho_ini  = 1.;
    double Y_O2_ini = 0.;
    double Y_CO2_ini = 0.;
    double T_ref        = T_ini;


    bool Obstacles = true;
    bool ObstaclesTest = false;
    bool Obstacle1 = false;
    bool Konjug_Waermetransport = true;         //C toggles conjugated heat transfer in the stream_temp


    //C choice of konjugated heat transfer BC

    bool GUO2015 = false;
    // this works
    bool LI2014  = true;           //C bounce-back mit extratermL
    bool HUBER2015 = false;          //C normales Treatment mit force-term

    int NC = 2;                 //C number of components in the fluid flow


    //C choice if single relaxation time or multi relaxation time model
    bool MRT = true;

    //C choice of test cases
    bool Cavity             = false;
    bool Couette            = false;
    bool Poiseuille         = false;
    bool Inflow_Outflow     = true;             //C simulation of flow with inflow/outflow BC rather than force-driven
    bool Multi_Component    = true;            //C enables multiple components for the gas phase
    bool Temperature_coupling = false;      //C toggles beta in f_eq and C_dach in Kollisionsschritt für flowfield f
    bool Solid_reaction     = true;             //C toggles the reaction at the interface of solid/gas


    //C Vorkonfigurierte Testcases
    bool Reaktionstestcase  = true;
    bool Flowfield          = false;
    bool Testsimulation = false;

    bool Vorwaertsdiff  = false;
    bool Zentraldiff    = false;

    //C MRT Parameters for the flow field
    double w_q      = 1.0;            //C free parameter to tune
    double w_eps    = 1.;          //C free parameter to tune
    double w_e      = 1.0;            //C connected to bulk viscosity     //C in general, decrease           bei 0.5 und 1.2 explosion, zwischen 0.7 und 1 gehts; konvergiert schneller bei 0.7
    double w_ny     = 1./Tau_flow;       //1./Tau_flow;  //C connected to shear viscosity

    //C MRT Parameters for Component O2
    double w_qO2    = 1.;
    double w_epsO2  = 1.;
    double w_eO2    = 1.;
    double w_nyO2   = 1./Tau_O2;

    //C MRT Parameters for Component CO2
    double w_qCO2   = 1.;
    double w_epsCO2 = 1.;
    double w_eCO2   = 1.;
    double w_nyCO2  = 1./Tau_CO2;

    //C MRT Parameters for Temperature
    double w_qT = 1.;
    double w_epsT = 1.;
    double w_eT = 1.;
    double w_nyT = 1./Tau_T;

    //C MRT Parameters for Temperature solid
    double w_qT_solid = 1.;
    double w_epsT_solid = 1.;
    double w_eT_solid = 1.;
    double w_nyT_solid = 1./Tau_T_solid;

    //C configuration of boundary conditions
    bool x_noslip = false;
    bool y_noslip = true;
    bool x_periodic = false;
    bool y_periodic = false;
    bool x_inflow_outflow = true;
    bool x_freeslip_wall = false;                   //C toggles free-slip wall BC; implemented with specular reflection
    bool y_freeslip_wall = false;
    bool links_dirichlet_temp = false;
    bool rechts_dirichlet_temp = false;

    bool inlet_parabolic = true;
    bool inlet_constant  = false;

    double dpdx = 0.;           //C 0.00005;
    double dpdy = 0.;

    bool External_force = false;
    double efx = 0;
    double efy = 0;

    bool timestep_stop = false;
    bool print_temp    = true;     //C prints thermodynamical Mitteltemperatur


    //C Variables to configure reading in variables or saving them (-> for reading in those variables in the next simulation)
    bool Einleseoption_f = true;        //C true enables the reading in of the distribution function f for the flow field
    bool Ausgabeoption_f = false;       //C true enables printing the values of f into a file (for reading in the next simulation)




    //C u_lb auf 0 setzen und Konv auf true, um reines Diffusionsproblem zu haben
    bool Konv = true;          //C used to pass the konv information into the operator function
    double Konvergenzabweichung = 0.00001;          //C 0.00001 for u_lb = 0.0002

    runCavityTwoPop(benchmark, Re, ulb, N_X, N_Y, max_t);
    return 0;
}
