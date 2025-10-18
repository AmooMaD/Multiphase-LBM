# CooLBM — CPU‑Parallel LBM Modules (2D/3D)

CooLBM-based, **CPU‑parallel** lattice Boltzmann modules for fluids and multiphase flows in 2D (D2Q9) and 3D (D3Q19). This repository currently includes:

- **PulsatileBloodFlow2D** — pressure‑driven, time‑varying (pulsatile) flow in a **deformable vessel** with curved, moving walls.
- **Rayleigh–Taylor instability** test cases.
- **Droplet in gas** (static and advective) tests for surface tension / spurious‑current assessment.
- **Layered two‑phase Poiseuille/Couette** flow benchmarks.
- **Contact‑angle calibration** cases.
- **Multiphase modules:**
  - **Shan–Chen pseudopotential (single‑component)** for phase separation and wetting.
  - **He–Chen–Zhang (HCZ) phase‑field** (Cahn–Hilliard) for conservative multiphase dynamics.

All solvers are built on the **CooLBM** platform and are parallelized on CPU using **C++17 parallel algorithms** (`std::for_each` with `std::execution::par_unseq`).

---

## What it solves

**Core problem (PulsatileBloodFlow2D):** Pressure‑driven, pulsatile flow through a **compliant 2‑D vessel**. The top/bottom walls move in response to local transmural pressure, \(P - p_{\text{tissue}}\), with stiffness \(\alpha\). An optional flag `is_severed` can drop outlet pressure to mimic a cut vessel.

### Physics / Models
- **Hydrodynamics:** D2Q9 (2D) and D3Q19 (3D) LBM; **MRT** (default) or **BGK** collision for the hydrodynamic population.
- **Boundaries:** Zou/He **pressure** inlets/outlets with sinusoidal forcing and optional phase shift (wave propagation).
- **Curved moving walls:** **Bouzidi** (quadratic) bounce‑back on arbitrarily located boundaries; wall positions are updated each step from pressure with per‑step clamping for stability.
- **Deformability:** Wall locations \(y_{r1}(x), y_{r2}(x)\) updated from local pressure; geometry is rebuilt, and new fluid nodes are robustly seeded from neighbors.
- **Multiphase options:**
  - **Shan–Chen pseudopotential (single‑component):** density‑based forcing; supports contact angle via wall potential tuning.
  - **HCZ (Cahn–Hilliard) phase‑field:** conservative order‑parameter transport coupled to momentum; suitable for high‑fidelity interface dynamics.

### Outputs
VTK files `sol_XXXXXXX.vtk` containing fields:
- Pressure `P`
- Velocity components `Ux, Uy` (and `Uz` in 3D cases)
- Masks (e.g., wall/solid, phase indicator/order parameter)

Open the files in **ParaView** to visualize pressure/velocity fields, interfaces, and evolving lumen geometry.

---

## How it’s structured

- **`LBM_PulsatileBloodFlow2D`**: main functor/object holding PDFs, macroscopic fields, wall geometry, boundary nodes, and update logic.
- **Collision → BCs → Streaming**: MRT/BGK collision → Bouzidi wall BCs → pull‑streaming scheme.
- **Macro recovery**: pressure & velocity recovered each step; then walls are moved and geometry is updated.
- **Performance report**: runtime and **MLUPS** reported at the end of runs.

**Example setup (inside `PulsatileBloodFlow2D()`):**
- Grid: `N = 64`, domain `nx = 1 + 10*(N-2)`, `ny = N` (long channel)
- Parameters: `tau = 0.75`, `alpha = 0.01`, `p0_in = 0.20`, `p0_out = 0.19`, beat period `t_beat = nx`
- Loop order: collision → wall BCs → streaming → inlet/outlet → macros → deformable wall motion → VTK dump (≈ every 1% of the run)

> **Usage notes:** Start with the provided driver `PulsatileBloodFlow2D();` then open `sol_*.vtk` in ParaView. Vary compliance `alpha`, relaxation time `tau`, and pressure waveform to explore different **Womersley/Reynolds** regimes and severed vs intact vessel scenarios.

---

## Lattices & Options

- **Lattice sets:** D2Q9 (2D), D3Q19 (3D)
- **Collision:** MRT (default) or BGK
- **Wall BC:** Bouzidi bounce‑back (quadratic)
- **Inlet/Outlet:** Zou/He pressure
- **Parallelization:** `std::execution::par_unseq` (C++17), data‑parallel loops
- **Precision:** double (recommended) or float (depending on build flags/codepath)
- **Backends:** CPU (multicore). GPU versions exist in separate branches/modules.

---

## Build & Run (Linux)

From the project’s main folder, first make the compile script executable:
```bash
chmod +x compile.sh
```

Compile:
```bash
./compile.sh
```

Run the executable (generated in the `out/` folder):
```bash
cd out
./COOLBM
```

### Requirements
- A modern C++17 compiler (e.g., GCC 10+, Clang 12+)
- CMake / Make (as configured in `compile.sh`)
- Optional: OpenMP or TBB (if your local `compile.sh` enables them)
- ParaView (for VTK visualization)

> **Notes:** The code is CPU‑parallelized via C++17 `<execution>`. If your system toolchain lacks parallel STL, consider enabling TBB or using a compiler that provides it.

---

## Examples

### 1) Pulsatile blood flow (deformable walls)
- Set `tau`, `alpha`, `p0_in`, `p0_out`, `t_beat`
- Run the default driver and inspect `sol_*.vtk`

### 2) Rayleigh–Taylor (multiphase)
- Choose **Shan–Chen** or **HCZ** module
- Initialize a heavy‑over‑light stratification
- Monitor growth rate and interface roll‑up

### 3) Droplet in gas / spurious currents
- Initialize a stationary droplet
- Check pressure jump vs **Laplace law**
- Measure peak spurious currents vs resolution

### 4) Layered two‑phase flow
- Opposing density/viscosity layers
- Compare steady profiles vs analytical solutions

### 5) Contact‑angle calibration
- Impose wall affinity (Shan–Chen) or wetting BC (HCZ)
- Fit apparent contact angle vs prescribed value

---

## Visualization

- Load `sol_*.vtk` in **ParaView**
- Suggested filters: **Glyph** (velocity vectors), **Stream Tracer**, **Contour** (interface/pressure), **Clip** (3D)
- Save animations to inspect pulsatile cycles or interface dynamics

---

## Performance

At the end of each run, the solver reports total wall‑clock time and **MLUPS**. Performance depends on grid size, collision model (MRT vs BGK), and the cost of geometry updates for deformable walls.

---

## Roadmap (selected)

- 3D compliant vessel (D3Q19) with moving curved walls
- Temperature/thermal coupling (for thermo‑fluid problems)
- More robust wetting/adhesion laws for complex roughness
- Automated regression tests and benchmarking suite

---

## How to cite / Acknowledgment

If you use this code in academic work, please cite the **CooLBM** platform and reference the specific modules you used (PulsatileBloodFlow2D, Shan–Chen, HCZ). See `CITATION.cff` (if provided) or the project wiki for a suggested citation.

---

## License

The code is distributed under the license provided with the **CooLBM** platform (see `LICENSE` in this repository).

---

## Contact

- **Maintainer:** Mahdy Zadshakoyan  
  INSA Rouen Normandie — CORIA / Université de Sherbrooke

For questions about the CooLBM platform, contact your usual CORIA/CooLBM maintainers.

