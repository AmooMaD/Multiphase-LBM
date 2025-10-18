# CooLBM: A Collaborative Open-Source Reactive Multi-Phase/Component Simulation Code via Lattice Boltzmann Method

We present a novel multi-CPU/GPU code, so-called CooLBM (COllaborative Open-source Lattice Boltzmann Method), for the simulation of single and multi-component multi-phase flow problems, along with the capability to solve reactive interfaces with conjugate fluid-solid heat transfer using the lattice Boltzmann method. The code is developed using the STL library of C++, which provides powerful features for efficient implementation and execution on high-performance computing (HPC) systems. By leveraging the parallel processing capabilities of both CPUs and GPUs, CooLBM demonstrates enhanced computational performance and scalability, making it suitable for simulating large-scale and computationally demanding different problems.

# How to use the CooLBM library

## How to run a test case

Performing a simulation using the CooLBM library is straightforward. Follow these steps to simulate a three-dimensional Poiseuille flow, for instance:

1. Locate the Source File

   Navigate to the ``apps`` folder and select the source file ``poiseuille3D.h``. Modify or customize it if needed.

2. Edit the Input File

   The source file ``poiseuille3D.h`` references an input file ``config_poiseuille3D.txt``, which contains user-defined parameters. You can find and edit this file in the ``apps/Config_Files/`` directory.

3. Include the Source File in the Main Program

    Ensure that ``poiseuille3D.h`` is included in the main program ``COOLBM.cpp``, located in the ``apps`` folder.

4. Define the Problem in the Main Program

    In ``COOLBM.cpp``, set the problem name by defining: string problem = "poiseuille3D";

5. Add New Simulation Codes (If Needed)

    If you are adding a new simulation code to the CooLBM library, create a new source file for your LBM code. Then, follow steps 3 and 4 to integrate it.

6. Invoke the Appropriate Function

    In ``COOLBM.cpp``, ensure that the condition: if (problem == "poiseuille3D")

7. Compile the Code

    Compile the program using the Linux command console. Choose a suitable compiler from those listed in the ``compile.sh`` file. For instance, run: ./compile.sh

8. Run the Simulation

    After compilation, execute the program by running the following command from the ``out`` folder: ./COOLBM

9. Post-processing of Results

    Do the post-processing of results generated in the ``out`` folder using appropriate softwares.
