PulsatileBloodFlow2D (CooLBM-based, CPU-parallel)

This module simulates pulsatile blood flow in a deformable 2-D vessel using a D2Q9 Lattice Boltzmann Method (LBM) built on the CooLBM platform. The flow solver supports BGK or MRT collision, curved moving walls, and pulsatile inlet/outlet pressures. It is CPU-parallelized using C++17 parallel algorithms (std::for_each with std::execution::par_unseq).

What it solves

Problem: Pressure-driven, time-varying (pulsatile) flow through a compliant vessel whose top/bottom walls move in response to transmural pressure (P − p_tissue) with stiffness α. An option is_severed can drop outlet pressure to mimic a cut/severed vessel.

Physics/Model:

D2Q9 LBM with MRT (default) or BGK for the hydrodynamic population g.

Inlet/Outlet: Zou/He pressure boundaries with a sinusoidal inlet pressure and phase-shifted outlet pressure (wave propagation).

Curved moving walls: Bouzidi (quadratic) bounce-back on arbitrarily located boundaries; walls update each step from pressure, with clamped motion per time step for stability.

Deformability: Wall locations yr1(x), yr2(x) updated from local pressure; geometry is rebuilt, new fluid nodes are robustly seeded from neighbors.

Outputs: VTK files (sol_XXXXXXX.vtk) containing pressure P, velocities Ux, Uy, and wall mask for visualization in ParaView.

How it’s structured

LBM_PulsatileBloodFlow2D: main functor holding PDFs, macros, wall geometry, boundary nodes, and update logic.

Collision/Streaming: MRT/BGK collision → Bouzidi wall BCs → pull streaming.

Macros: pressure and velocity recovered each step; walls then moved and geometry updated.

Performance: reports runtime and MLUPS at the end.

Example setup (included in PulsatileBloodFlow2D()):

Grid: N = 64, domain nx = 1 + 10*(N-2), ny = N (long channel).

Parameters: tau = 0.75, α = 0.01, p0_in = 0.20, p0_out = 0.19, beat period t_beat = nx.

Run loop: collision → wall BCs → streaming → inlet/outlet → macros → deformable wall motion → VTK dump every ~1% of the run.

Usage notes:
Start with the provided driver PulsatileBloodFlow2D(); open the generated sol_*.vtk in ParaView to inspect pressure/velocity fields and the evolving lumen. Adjust compliance α, relaxation time τ, and pressure waveform to explore different Womersley/Reynolds regimes and severed vs. intact vessel scenarios.
