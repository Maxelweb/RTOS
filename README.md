# RTOS

Real time kernel and systems course project - University of Padova (Italy) 2021
Prof. Tullio Vardanega

## Docker enviroment setup

We use a docker container to provide a single common enviroment to use python 2.7 along with different other tools. Make sure to use Docker before starting.

### Start the enviroment

1. Clone this github repo
2. Go to the repo folder and enter `docker-compose up -d --build` or `docker compose up -d --build` (depending on the docker / docker-compose version you have)
3. The container will start in a few minutes

### Stop the enviroment

1. Go to the repo folder and enter `docker-compose down -v`
2. The container will stop in a few seconds


## Initial setup (first time)

1. Go to `/app/schedcat/` and execute `make`



## Execute experiments

1. Go to `/app/gipp-experiments` and execute `./generate_experiment_calls.sh`
2. If necessary, add execution permission with `chmod +x experiment_calls.sh`
3. Execute `./experiment_calls.sh`
4. Wait about 1000 seconds for each call, if in parallel around 20 min

## Graphs and plots realization

1. Make sure `/app/gipp-experiments/data_complete` has some csv files
2. Go to `/app/gipp-experiments` and execute `python2 produce_plots.py`




## Credits

- Giuseppe Rossano
- Mariano Sciacco
- (`gipp_code`) James Robb
