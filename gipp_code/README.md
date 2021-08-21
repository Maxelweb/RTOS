# GIPP Experiments Code

Please refer to `gipp-experiments/README.md` for details on this code base. Note that `gipp-experiments` relies on `schedcat` and `sched-experiments`. I recommend the use of a virtual environment when using these code bases. To use `schedcat` and `sched-experiments` as packages, I suggest the reader install them locally with pip as "editable packages".



## Basic Instructions

* Create a python2.7 virtual environment.
* Install `sched-experiments` as an editable package.
* Install the packages in `sched-experiments/requirements.txt`.
* Install `schedcat` as an editable package.
  * Either the GNU Linear Solver (GLPK) or the CPLEX Linear Solver need to be installed on your system with the appropriate development headers/libraries.
  * SWIG and the appropriate headers/libraries must be installed on your system.
* Run `make` in the `schedcat` directory.
* Install the packages in `gipp-experiments/requirements.txt`.
* Experiments can now be run with the scripts in `gipp-experiments`.
  * Use `gipp-experiments/run_experiments.py` and `gipp-experiments/README.md` as a starting point.
  * Unless `gipp-experiments` is turned into a git repo, warnings that git compares can't take place will appear during the running of experiments. However, the CSV files (ie, the experiment results) will still be generated, so turning the directory into a git repo is not strictly required.
