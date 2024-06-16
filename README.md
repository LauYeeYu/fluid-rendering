# CG project: A Complete High Performance Simulation and Rendering Pipeline for Fluid

In this project, we design and implement a complete simulation and rendering
pipeline for fluid. We leverage the computational power by means of
parallelization with [Taichi framework][taichi]. With PBF, screen-based
method and Layered Neighborhood Method optimization, we can achieve about
30 FPS with CUDA on a commercial laptop (RTX 3060 Laptop).

[taichi]: https://www.taichi-lang.org/

## How to Use

### Pre-requisites

To run the renderer, you must make sure you have python and taichi installed.
You can install taichi with pip. We recommend you to use taichi 1.7.1 or later.

We recommend using a virtual environment to avoid conflicts with other
packages. You can create a virtual environment with the following command:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, you can install taichi with the following command:

```bash
pip install taichi
```

### Running the Program

The main entry point of the program is `src/taichi/main.py`. You can run the
program with the following command:

```bash
python src/taichi/main.py
```

Just the `main.py` is enough to run the simulation and rendering pipeline.

### Parameters

We provide several parameters to control the simulation and rendering. You
can change them in `main.py`. Here are some of the parameters:

- architecture: the backend architecture to use. Default is `gpu`. If you wish
  to use CPU, you can uncomment the line `ti.init(arch=ti.cpu)` and comment
  the line `ti.init(arch=ti.gpu)`.
- `unsafe`: Avoid using time-consuming atomic operations. This will make the
  simulation faster but may make the result less accurate. In our experiment,
  no evident problem has been found. Default is `False`.
- `visualize_lnm`: Use the Layered Neighborhood Method to optimize the
  simulation. This will make the simulation faster. Default is `True`.

## Project File Layout

- [`src/taichi`](src/taichi): The main entry point of the program.
- [`paper/`](paper/): The source code of the final paper.
- [`paper/fluid-simulation-rendering.pdf`](paper/fluid-simulation-rendering.pdf): The final project paper.
- [`video/fluid.mp4`](video/fluid.mp4): The final project video.

## Acknowledgement

We sincerely thank TA Shulin Hong and Prof. Yang's help and suggestions!
