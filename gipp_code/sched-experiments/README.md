# MPI-SWS Schedulability Experiments Collection

## Overview

The purpose of this repository is

1. to collect and **maintain in runnable form** the schedulability experiments carried out at MPI-SWS, and
2. to provide a framework  with **useful auxiliary code** (config reading, setup parallel experiments on cluster, etc.) and **established patterns** for schedulability experiments.

Maintaining this repository in a clean and up-to-date state adds a little effort up front, but helps greatly in the long term by significantly reducing the startup costs of setting up new experiments.

In contrast to SchedCAT itself, this repository is (for now) primarily an MPI-SWS-internal project. However, it will be made available to external researchers upon request. Thus, always design and code assuming your code will be publicly scrutinized one day.

## Quick Start

To get started with hacking on `sched-experiments`, first **fork the project on Gitlab**.

To clone the forked repository and to check out the included SchedCAT
submodule, execute:

    git clone --recursive git-rts@gitlab.rts.mpi-sws.org:${YOUR-GITLAB-USERNAME}/sched-experiments.git

This will also clone the right version of SchedCAT from the master branch of the [main SchedCAT repository](https://gitlab.rts.mpi-sws.org/bbb/schedcat) repository.

After cloning, you need to still compile the C++ part of SchedCAT. To do so, got to `lib/schedcat` and run `make`. It's good practice to run `make test` afterwards to make sure everything works as expected.

    cd sched-experiments/lib/schedcat
    make
    make test

If you want to update to a version of SchedCAT that is not part of the main SchedCAT repository (i.e., when developing experiments that use a new analysis that's not yet part of the master branch), *locally* edit `.gitmodules` to point to a different SchedCAT repository. Have a look at the [git submodule documentation](http://git-scm.com/docs/git-submodule) and the [chapter on submodules in the git book](http://git-scm.com/book/en/Git-Tools-Submodules) for details.

## Coding Guidelines

As always, it is our goal to maintain high coding standards: **keep your code simple and clean**, and free of dirty hacks as much as possible.

Documentation is less important. Rather, when in doubt, your code should be so simple and obvious that no documentation is required.

Do not worry too much about failure and/or exception handling. **We want all code to crash in case of failures.** Experiments that fail by crashing are easy to diagnose before publication; experiments that silently fail by producing wrong data are much more problematic.

## Merging Guidelines

During development, your branch(es) can be as messy as you'd like. However, when the time comes to merge the final experiments back into the master branch, please note the following rules.

1. Any changes to SchedCAT must be merged first.

1. Typically, the branch should consist of two or more commits:
    - *Exactly one* commit to update to a new version of SchedCAT, which *must* be a commit on the master branch in the [main SchedCAT repository](https://gitlab.rts.mpi-sws.org/bbb/schedcat) repository.
      Example commit: [ff617cce13ebc22db220a4004c16408e9dd68098](https://gitlab.rts.mpi-sws.org/bbb/sched-experiments/commit/ff617cce13ebc22db220a4004c16408e9dd68098)
    - *Exactly one* commit to add the finished experiments.
      Example commit: [6da2a0787310ec5eaf277accc5519193204aefdb](https://gitlab.rts.mpi-sws.org/bbb/sched-experiments/commit/6da2a0787310ec5eaf277accc5519193204aefdb)
    - If applicable, any number of commits to make any modifications to the support code in sched-experiments itself.
      Example commit: [b518ee5237219e98bb5f8385c7f9afacb6dffdb9](https://gitlab.rts.mpi-sws.org/bbb/sched-experiments/commit/b518ee5237219e98bb5f8385c7f9afacb6dffdb9)

1. When adding new functionality, make sure you don't break old experiments. In particular, make sure you don't change default behavior in incompatible ways. The idea of sched-experiments to maintain experiments such that they remain **always runnable**. If changes are strictly required, make sure you fix all the old experiments.

1. Make sure your ready-to-be-merged branch applies cleanly on top of master. This most likely means you need to rebase your branch prior to sending a merge request.

1. Make sure your ready-to-be-merged branch does not modify `.gitmodules` (i.e., discard any commits touching `.gitmodules` when rebasing).

