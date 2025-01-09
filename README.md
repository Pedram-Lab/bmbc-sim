# ECM simulations
A collection of scripts for solving problems that arise in extracellular matrix simulations using [NGSolve](https://ngsolve.org/).

### Setup
Run the following line from the terminal to create a new environment called ecsim:
```bash
conda env create -f environment.yaml
```

Activate the environment:
```bash
conda activate ecsim
```

If not already installed, execute the following command (with active conda environment) to install the local package. This is necessary to run the examples with the `netgen` command (see below).
```bash
pip install -e .
```

Then, you should be able to run the examples using:
```bash
python <filename>  # just console output, no GUI
netgen <filename>  # console output and GUI
```

The `.py` scripts in the `scripts/` directory are text representations of notebook files courtesy of jupytext.
When you open them in jupyter (right-click and choose 'Open With > Jupytext notebook'), they will be automatically converted to interactive notebooks, which you can find in the `notebooks/` directory.
Saving a notebook will also automatically update the associated `.py` file.

### Examples
Currently, all examples are located in the `scripts/` directory. Apart from simulations, there are some files that showcase how to use some aspects of NGSolve in greater detail, since the documentation is sometimes lacking for what we need. These are located in the `tutorials/` subfolder.

