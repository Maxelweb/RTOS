from __future__ import division

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean, median
import toolbox.bootstrap as boot

import schedcat.generator.tasksets as gen

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.util.time import ms2us

def setup_stats(conf):
    stats  = [min, mean, median, max]
    titles = ["MIN", "AVG", "MED", "MAX"]
    return stats, titles

def run_range(conf, stats):

    for ucap in value_range(1, conf.num_cpus, conf.step_size):
        samples = [len(conf.generate(max_util=ucap, time_conversion=ms2us, squeeze=conf.squeeze))
                   for _ in xrange(conf.samples)]
        row = []
        for i in xrange(len(stats)):
            ci  = boot.confidence_interval(samples, stat=stats[i])
            row += [ci[0], stats[i](samples), ci[1]]
        yield [ucap] + row

def run_named_config(conf):
    conf.check('num_cpus',     type=int)
    conf.check('samples',      type=int, default=100)
    conf.check('step_size',    type=float, default=0.25)
    conf.check('utilizations', type=one_of(gen.NAMED_UTILIZATIONS))

    conf.generate = gen.DIST_BY_KEY['implicit']['uni-moderate'][conf.utilizations]

    stats, titles = setup_stats(conf)
    header = ['UCAP']
    for t in titles:
        header += ['%s CI-' % t, t, '%s CI+' % t]

    data = run_range(conf, stats)
    write_data(conf.output, data, header)

def generate_named_configs(options, cpus=64, squeeze=True):
    for util in gen.NAMED_UTILIZATIONS:
        name = 'tasks_exp=task-counts_util=%s_m=%02d_exact=%d' \
                    % (util, cpus, squeeze)
        c = Config()
        c.squeeze      = squeeze
        c.num_cpus     = cpus
        c.utilizations = util
        c.output_file  = name + '.csv'
        yield (name, c)

EXPERIMENTS = {
    'count/named' : run_named_config
}

CONFIG_GENERATORS = {
    'count/named' : generate_named_configs
}
