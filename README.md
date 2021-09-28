
![unipd-logo](unipd-logo.png)

# Analysis on "Nested, but Separate: Isolating Unrelated Critical Sections in Real-Time Nested Locking"

Real time Kernel and Systems course project - University of Padova (Italy) 2020-21 (Prof. Tullio Vardanega)

---

This project is based on J. Robb and B. B. Brandenburg work (see *[Nested, but Separate: Isolating Unrelated Critical Sections in Real-Time NestedLocking](https://pure.mpg.de/rest/items/item_3319125/component/file_3332456/content)*), where we analyze the GIPP protocol from the theoretical and practical viewpoint.
**In this repository the source code is entirely attributed to J. Robb and B. B. Brandenburg**, both for the GIPP LP analysis and schedcat library, respectively. The experiments reported are obtained through this exact source code, which was available from J. Robb's personal website (see [jamesrobb.ca](http://jamesrobb.ca)).

## File structure and branches

This repository is structured as follows: 
- `/experiments` contains all the experiments executed by the authors of this project in a form of PDFs and CSV data files. The PDFs are the graph generated from the CSV files, which can be imported in LaTeX if needed.
- `/gipp_code` contains all the source code of the analyzed experiment, made by J. Robb and B. B. Brandenburg. 
- `/` in the folder is possible to see the docker files made by the authors of the project, as well as this readme file.

The following branches are currently active and have been used to inspect the code and to edit part of the source code:

- `features/code-inspection` contains a few notes through the code (in italian and english language), useful to understand some functions and implementations.
- `features/c-omlp-comparison` contains edited code that replace the CA-RNLP protocol with C-OMLP, as well as other extra configuration inside the `/gipp_code/gipp-experiments/_extra` folder.

## Docker enviroment setup

We use a docker container to provide a single environment to use Python 2.7 along with different other tools, such as GLPK and SWIG. Make sure have a running Docker daemon before starting.

### Start the enviroment

1. Clone this Github repository.
2. Go to the repository folder and enter `docker-compose up -d --build` or `docker compose up -d --build` (depending on the docker / docker-compose version you have).
3. The container will start in a few minutes, after building.

### Stop the enviroment

1. Go to the repo folder and enter `docker-compose down -v`
2. The container will stop in a few seconds


## Experiment setup

### Execute the experiments
1. Once you attach the terminal to the container, go to `/app/schedcat/` and execute `make`.
    - This will compile all the C++ files needed for SWIG
    - Execute this part **only the first time you start the container**
2. Go to `/app/gipp-experiments` and execute `./generate_experiment_calls.sh`
    - If necessary, add execution permission with `chmod +x experiment_calls.sh`
3. Execute `./experiment_calls.sh` 
    - **Be careful!** you're going to execute a tons of Python commands and this might halt you system! Make sure you have enough resources to run the experiments or separate them in a different way.

### Graphs and plots realization

1. Make sure `/app/gipp-experiments/data_complete` has some csv files.
    - If not so, execute the experiment first.
2. Go to `/app/gipp-experiments` and execute `python2 produce_plots.py`




## Credits

- **Giuseppe Rossano**
- **Mariano Sciacco**

### Extra credits
- **James Robb** (`gipp_code`)
- **B. B. Brandenburg** (`schedcat` library, see [brandenburg/schedcat](https://github.com/brandenburg/schedcat))
