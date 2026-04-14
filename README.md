# leo-sat-flow
Author: Zhouyou Gu, Research Fellow at Singapore University of Technology and Design (SUTD), supervised by Prof. Jihong Park.

`leo-sat-flow` is a research codebase for simulating laser inter-satellite link (LISL) connectivity and traffic flow in LEO mega-constellations. It combines TLE-based orbit propagation, feasible optical-link generation under pointing constraints, traffic generation from population and gateway maps, optimization-based connectivity/routing solvers, and learned graph models used in the accompanying papers.

This repository accompanies:

- [Joint Laser Inter-Satellite Link Matching and Traffic Flow Routing in LEO Mega-Constellations via Lagrangian Duality](https://arxiv.org/abs/2601.21914)
- [Duality-Guided Graph Learning for Real-Time Joint Connectivity and Routing in LEO Mega-Constellations](https://arxiv.org/abs/2601.21921)

The code is organized around standalone experiment scripts rather than a packaged CLI. In practice, you run the repo from the root directory with `PYTHONPATH=.` and choose the script that matches the experiment you want.

## Repository Layout

- `sim_mld/`: active simulation core, TLE utilities, terrain and traffic models, optical channel model, solvers, visualization code, and learned models.
- `sim_src/`: shared utilities plus an older simulation stack still used by some exploratory scripts.
- `demo/`: interactive demos and figure-generation scripts.
- `sim_alg/`: early training and baseline experiments for link-pricing / dual updates.
- `sim_alg_j1_res/`: learned-scheduler training, evaluation, ablations, and pretrained checkpoints in `selected_nn/`.
- `sim_alg_v1_res/`: newer benchmark and paper-revision experiments, including shell-level evaluation scripts.
- `z_sim_alg/` and `zz_sim_alg/`: unused legacy folders kept in the repo but not part of the active workflow.
- `test_script/`: scratch scripts for environment checks, visualization experiments, and data preprocessing.

If you are new to the repo, start with `sim_mld/`, `demo/`, `sim_alg/`, `sim_alg_j1_res/`, and `sim_alg_v1_res/`. Directories prefixed with `z_` or `zz_` are unused.

## Data And Assets

The main scripts expect to be run from the repository root and load several assets via relative paths.

- Constellation snapshots: `starlink_16_jul_2025_1600.tle`, `oneweb_16_jul_2025_1600.tle`, `kuiper_20_jan_2026_1100.tle`
- Traffic and gateway data: `population_density_texture.npy`, `satnogs_locations.csv`
- Earth textures and related visuals: `population_density_texture.png`, `simple_earth_texture.png`, `land_sea_texture.png`, `land_sea_texture_bw.png`
- Auxiliary city and station files used by preprocessing and legacy scripts: `worldcities.csv`, `cities.csv`, `topkcity.csv`, `satnogs_stations.csv`

Most of the asset-generation helpers live under `test_script/test_earth_texture/`.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the base Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install highspy
```

`highspy` is required because the solver code uses the HiGHS backend in `cvxpy`.

### Extra Dependencies For Learned Schedulers

The learned experiments under `sim_mld/ml/`, `sim_alg/`, `sim_alg_j1_res/`, and parts of `sim_alg_v1_res/` also require:

- `torch`
- `torch_geometric`
- `plotext`

Install `torch` and `torch_geometric` for your CPU/CUDA platform using their official instructions, then install `plotext`:

```bash
python -m pip install plotext
python test_script/test_torch_cuda.py
```

Several evaluation scripts in `sim_alg_j1_res/` load pretrained checkpoints from `sim_alg_j1_res/selected_nn/` by default.

### Ubuntu Or WSL Visualization Dependencies

For `vispy` rendering and Numba's TBB threading layer on Ubuntu or WSL:

```bash
sudo apt-get update
sudo apt-get install qtbase5-dev qt5-qmake libtbb-dev
python -m pip install PyQt5 tbb
```

If you are inside a virtual environment, make sure TBB is discoverable:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
numba -s | grep TBB
```

You should see `TBB Threading Layer Available : True`.

### Headless Display

For remote 3D rendering over SSH, use a VNC-backed display:

```bash
vncserver :1
export DISPLAY=:1
vncserver -list
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

`tigervnc` and `novnc` are a practical combination for this setup.

## Running Experiments

Most scripts assume:

- your current working directory is the repository root
- `PYTHONPATH=.` is set
- the root-level data files are present

Representative entry points:

```bash
PYTHONPATH=. python demo/demo_4_jun_starlink.py
PYTHONPATH=. python demo/plot_for_conference_paper_22_jul_starlink_1000.py
PYTHONPATH=. python sim_alg/train_lpd.py
PYTHONPATH=. python sim_alg/test_lpd_lct4_dual_only.py
PYTHONPATH=. python sim_alg_v1_res/test_tcom_shell_time_baselines.py
```

Useful starting points by workflow:

- Interactive visualization: `demo/demo_4_jun_starlink.py`, `demo/demo_4_jun_regular.py`
- Paper-style figures: `demo/plot_for_conference_paper_*.py`, `sim_alg/figures/*.py`, `sim_alg_v1_res/figures/*.py`
- Initial dual / link-pricing experiments: `sim_alg/*.py`
- Learned duality-guided experiments: `sim_alg_j1_res/train_*.py`, `sim_alg_j1_res/test_*.py`
- Current shell-level benchmark suite: `sim_alg_v1_res/test_tcom_shell_time_baselines.py`, `sim_alg_v1_res/test_tcom_shell_time_load_scaling.py`

Most scripts create a timestamped output directory next to the script, for example:

```text
demo/plot_for_conference_paper_22_jul_starlink_1000/
sim_alg/train_lpd/
sim_alg_v1_res/test_tcom_shell_time_baselines/
```

Inside those directories you will usually find per-run folders ending in `-ail`, CSV logs, saved NumPy text dumps, or generated figures.

## Reproducing The Papers

There is no single top-level reproduction command. The repository is split by experiment family.

- The optimization-centric scripts for the dual baseline and matching/routing studies live mainly in `demo/`, `sim_alg/`, and `sim_alg_v1_res/`.
- The learned schedulers and ablations live mainly in `sim_alg_j1_res/`, backed by models in `sim_mld/ml/`.
- The newer shell-level evaluation scripts in `sim_alg_v1_res/` reuse the Starlink shell extraction utilities in `sim_mld/tle.py` and, when needed, checkpoints from `sim_alg_j1_res/selected_nn/`.

For the newer benchmark scripts, the source files themselves are the best reference for configurable environment variables such as `TCOM_REVISION_TIME_OFFSETS_MIN`, `TCOM_REVISION_BASE_ACTIVE_USER_PERCENTAGES`, and `TCOM_REVISION_DRL_MODEL_PATH`.

## Practical Notes

- This is research code. Many scripts hard-code experiment constants, TLE snapshots, and output behavior.
- Several modules use relative file paths such as `population_density_texture.npy` or `satnogs_locations.csv`, so running from the repo root matters.
- `cvxpy` plus `highspy` is required for the LP-based rate-allocation steps.
- Visualization scripts require a working OpenGL-capable desktop or a VNC-backed remote display.
- The script naming predates the final paper terminology in a few places. `LCT` refers to laser communication terminals, `LISL`/`ISL` to inter-satellite links, and many `ld*` / `lpd*` names correspond to link-pricing or duality-guided learning variants.

## Citation

If you use this code, please cite the relevant paper(s):

```bibtex
@article{gu2026lisl_duality,
  title={Joint Laser Inter-Satellite Link Matching and Traffic Flow Routing in LEO Mega-Constellations via Lagrangian Duality},
  author={Gu, Zhouyou and Park, Jihong and Choi, Jinho},
  journal={arXiv preprint arXiv:2601.21914},
  year={2026}
}

@article{gu2026duality_guided_graph_learning,
  title={Duality-Guided Graph Learning for Real-Time Joint Connectivity and Routing in LEO Mega-Constellations},
  author={Gu, Zhouyou and Choi, Jinho and Quek, Tony Q. S. and Park, Jihong},
  journal={arXiv preprint arXiv:2601.21921},
  year={2026}
}
```
