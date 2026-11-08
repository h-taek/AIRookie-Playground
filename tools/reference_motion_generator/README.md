![banner](https://github.com/user-attachments/assets/f445e373-74fc-413b-aa73-0f17c76b1171)


# Open Duck Reference Motion Generator

Open Duck project's reference motion generator for imitation learning, based on [Placo](https://github.com/Rhoban/placo).

The reference motions are used in two RL works, one using mujoco playground [here](https://github.com/SteveNguyen/openduckminiv2_playground) and another using Isaac Gym [here](https://github.com/rimim/AWD)

> This repo uses `git-lfs`. Install it with `sudo apt install git-lfs` before cloning the repo

## Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### Generate motions

```bash
uv run scripts/auto_waddle.py (-j?) --duck ["go_bdx", "open_duck_mini", "open_duck_mini_v2"] (--num <> / --sweep) --output_dir <>
```

> If you have an error like this : `Failed to determine STL storage representation for <...> Hint: the mesh directory may be wrong`, just run `git lfs pull`

Args : 
- `-j?` number of jobs. If `j` is not specified, will run sequentially. If `j` is specified without a number, your computer will crash :) (runs with all available cores). Use `j4` for example
- `--duck` selects the duck type
- `--sweep` generates all combinations of motion within the ranges specified in `Open_Duck_reference_motion_generator/open_duck_reference_motion_generator/robots/<duck>/auto_gait.json`
- `--num` generates <num> random motions
- `--output_dir` self explanatory

For example, to generate all the reference motion for Open Duck Mini, run :

```bash
uv run scripts/auto_waddle.py -j8 --duck open_duck_mini_v2 --sweep 
```

This will write in a directory called `recordings/`

### Fit polynomials

This will generate `polynomial_coefficients.pkl`
```bash
uv run scripts/fit_poly.py --ref_motion recordings/
```

To plot : 

```bash
uv run scripts/plot_poly_fit.py --coefficients polynomial_coefficients.pkl
```

### Replay

```bash
uv run scripts/replay_motion.py -f recordings/<file>.json
```

### Playground 

```bash
uv run open_duck_reference_motion_generator/gait_playground.py --duck ["go_bdx", "open_duck_mini", "open_duck_mini_v2"]
```

## TODO

- The robots descriptions should be in a separate repository and imported as a submodule. This would help having duplicates of the same files, or with different versions etc.
- Validate that we can train policies with these motions (might have broken something during the port...)
- Fix small bugs in gait_playground (refreshing, changing robots ...)
- Nicer visualization ? Directly visualize using meshcat maybe. 
- A document to specify the reference motion format, if someone wants to convert some mocap data to use for the imitation reward ?
- The repo duck themed, but it could be a generic motion generator based on placo for biped robots (will add sigmaban at some point)
  - Sub TODO : explain how to add a new robot
