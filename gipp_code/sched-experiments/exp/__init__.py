from datetime import datetime

from toolbox.io import one_of
from toolbox.io import write_std_header, write_data, write_runtime, write_configuration


# To Bjorn:
# The inclusion of these modules/expriments breaks with the version of
# schedcat this was developed against. I'd suggest removing them all
# together.

# import exp.rtss16b
# import exp.binpack_heuristics
# import exp.count_tasks
# import exp.deadline_miss_heuristics
# import exp.omlp_comparison
# import exp.rtas13
# import exp.ecrts13
# import exp.rtas14
# import exp.spinlock_analysis
# import exp.global_analysis
# import exp.ecrts14

experiment_modules = [
    # exp.rtss16b,
    # exp.binpack_heuristics,
    # exp.count_tasks,
    # exp.deadline_miss_heuristics,
    # exp.omlp_comparison,
    # exp.rtas13,
    # exp.ecrts13,
    # exp.rtas14,
    # exp.spinlock_analysis,
    # exp.global_analysis,
    # exp.ecrts14,
]

# ecrts20 experiments automation is in a seperate git repository
# <insert repo link here>

EXPERIMENTS = {
    # name  =>  run_exp(config)
    "ping" : lambda conf: conf.output.write("pong\n"),
}

CONFIG_GENERATORS = {
    # name  =>  generate_configs()
    "ping" : lambda _: [],
}

for mod in experiment_modules:
    EXPERIMENTS.update(mod.EXPERIMENTS)
    CONFIG_GENERATORS.update(mod.CONFIG_GENERATORS)

# experiment interface:
#
#   run_XXX_experiment(configuration)
#
# Semantics: run the experiment described by input_config and store
# the results in the file-like object conf.output, which could be either
# an actual file, a socket, or a StringIO object. The actual schedulability
# experiment should not care.

def run_config(conf):
    conf.check('experiment',   type=one_of(EXPERIMENTS.keys()))

    experiment_driver = EXPERIMENTS[conf.experiment]

    write_std_header(conf.output)
    start  = datetime.now()
    experiment_driver(conf)
    end = datetime.now()

    write_configuration(conf.output, conf)
    write_runtime(conf.output, start, end)
    return end - start
