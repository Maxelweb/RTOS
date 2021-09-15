FROM python:2.7-buster
# COPY ./gipp_code /app
WORKDIR /app

RUN apt-get update && apt-get install glpk-utils libglpk-dev swig g++ make -y

RUN pip2 install numpy scipy matplotlib unionfind
RUN pip2 install -e sched-experiments
RUN pip2 install -e schedcat

# RUN cd schedcat && make clean && make

CMD tail -f /dev/null

# CMD python2 /app/app.py