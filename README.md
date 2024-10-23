# Embodied AI Seminar: Visual SB3 Starter Code

## Installation

### Create a conda environment

```
conda create -y -n vissb3 python=3.10
conda activate vissb3
```

### Install JAX

Make sure to run this ***BEFORE*** other dependencies.

```
pip install jax[cuda12]
```

### Install other dependencies

Then install the requirements.

```
pip install -r requirements.txt
```

Ensure that you can run the following command without any errors:

```
python enjoy_sb3.py
```

# Headless Training

In your assignment, you will need to train your agent headlessly. To do this, you can use the following command:

```
python train_sb3.py --run_name 000 --seed 0
```

You can use different values for `--run_name` and `--seed` to differentiate between different runs.
