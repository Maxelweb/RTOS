from __future__ import division

import random
from functools import partial

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean
import toolbox.bootstrap as boot

import schedcat.generator.tasksets as gen
import schedcat.generator.generator_emstada as emstada
import schedcat.mapping.binpack as bp
import schedcat.sched.edf as edf

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.util.time import ms2us

def random_fit(items, bins, capacity=1.0, weight=bp.id, misfit=bp.ignore,
              empty_bin=list):
    sets = [empty_bin() for _ in xrange(0, bins)]
    sums = [0.0 for _ in xrange(0, bins)]
    for x in items:
        c = weight(x)
        # pick a random bin where the item  fits
        candidates = [i for i in xrange(len(sums)) if sums[i] + c <= capacity]
        if candidates:
            # fits somewhere
            i = random.choice(candidates)
            sets[i] += [x]
            sums[i] += c
        else:
            misfit(x)
    return sets

random_fit_decreasing = bp.decreasing(random_fit)

def true_worst_fit(items, bins, capacity=1.0, weight=bp.id, misfit=bp.ignore,
                      empty_bin=list):
    sets = [empty_bin() for _ in xrange(0, bins)]
    sums = [0.0 for _ in xrange(0, bins)]
    max_to_consider = 1
    for x in items:
        c = weight(x)
        # pick the bin where the item will leave the most space
        # after placing it, aka the bin with the least sum,
        # but only considering those that have been used before (only
        # use a new bin when the item won't fit anywhere else).
        candidates = None
        while not candidates and max_to_consider <= bins:
            candidates = [s for s in sums[:max_to_consider] if s + c <= capacity]
            if not candidates:
                # open up a new bin
                max_to_consider += 1
        if candidates:
            # fits somewhere
            i = sums.index(min(candidates))
            sets[i] += [x]
            sums[i] += c
        else:
            misfit(x)
    return sets

true_worst_fit_decreasing = bp.decreasing(true_worst_fit)

def partitioned_test(num_cpus, cluster_size,
                     sched_test, heuristic, taskset):
    num_clusters = num_cpus // cluster_size
    try:
        partitions = heuristic(taskset, num_clusters,
                               capacity=cluster_size,
                               weight=SporadicTask.utilization,
                               misfit=bp.report_failure,
                               empty_bin=TaskSystem)

        for part in partitions:
            if not sched_test(cluster_size, part):
                return False
        return True
    except bp.DidNotFit:
        # not schedulable
        return False

def setup_tests(conf):
    tests = [partial(partitioned_test,
                     conf.num_cpus, conf.cluster_size,
                     edf.is_schedulable, heuristic)
             for heuristic in (bp.next_fit, bp.first_fit,
                               bp.best_fit, bp.worst_fit,
                               bp.almost_worst_fit,
                               bp.any_fit, random_fit,
                               true_worst_fit,
                               bp.next_fit_decreasing,
                               bp.first_fit_decreasing,
                               bp.best_fit_decreasing,
                               bp.worst_fit_decreasing,
                               bp.almost_worst_fit_decreasing,
                               bp.any_fit_decreasing,
                               random_fit_decreasing,
                               true_worst_fit_decreasing)]
    titles = ['NF',  'FF',  'BF',  'WF',  'AWF' , 'AF',  'RF',  'TWF',
              'NFD', 'FFD', 'BFD', 'WFD', 'AWFD', 'AFD', 'RFD', 'TWFD']
    return tests, titles

def run_range(conf, tests):
    for ucap in value_range(conf.num_cpus / 2, conf.num_cpus - 0.005, conf.step_size, last = conf.num_cpus - 0.005):
        samples = [[] for _ in tests]
        for _ in xrange(conf.samples):
            taskset = conf.generate(max_util=ucap, time_conversion=ms2us, squeeze=True)
            for i in xrange(len(tests)):
                samples[i].append(tests[i](taskset))
        row = []
        for i in xrange(len(tests)):
            ci  = boot.confidence_interval(samples[i], stat=mean)
            row += [ci[0], mean(samples[i]), ci[1]]
        yield [ucap] + row

def common_checks(conf):
    conf.check('num_cpus',     type=int)
    conf.check('cluster_size', type=int, default=1)
    conf.check('samples',      type=int, default=100)
    conf.check('step_size',    type=float, default=conf.num_cpus / 2 / 40)

    if conf.cluster_size <= 0 or conf.num_cpus % conf.cluster_size != 0:
        raise ValueError, "num_cpus must be an integer multiple of cluster_size"


def run_pedf_config(conf):
    common_checks(conf)
    conf.check('utilizations', type=one_of(gen.NAMED_UTILIZATIONS))
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(gen.NAMED_PERIODS),   default='uni-moderate')

    conf.generate = gen.DIST_BY_KEY[conf.deadlines][conf.periods][conf.utilizations]

    tests, titles = setup_tests(conf)
    header = ['UCAP']
    for t in titles:
        header += ['%s CI-' % t, t, '%s CI+' % t]

    data = run_range(conf, tests)
    write_data(conf.output, data, header)


def generate_emstada(conf, max_util=1, time_conversion=ms2us, squeeze='ignored'):
    return emstada.gen_taskset(conf.periods, conf.periods_dist,
                               conf.task_count, max_util, period_granularity=1,
                               want_integral=True, scale=ms2us)

def run_emstada_pedf_config(conf):
    common_checks(conf)
    conf.check('periods',      type=one_of(emstada.NAMED_PERIODS),
                               default='uni-moderate')
    conf.check('periods_dist', type=one_of(emstada.NAMED_PERIOD_DISTRIBUTIONS),
                               default='logunif')
    conf.check('task_count',   type=int, default=10)

    conf.generate = partial(generate_emstada, conf)

    tests, titles = setup_tests(conf)
    header = ['UCAP']
    for t in titles:
        header += ['%s CI-' % t, t, '%s CI+' % t]

    data = run_range(conf, tests)
    write_data(conf.output, data, header)


def task_count_test(conf, sched_test, num_tasks, ucap):
    taskset = emstada.gen_taskset(conf.periods, conf.periods_dist,
                                  num_tasks, ucap, period_granularity=1,
                                  want_integral=True, scale=ms2us)
    return sched_test(taskset)

def setup_task_count_tests(conf):
    sched_test = partial(partitioned_test,
                         conf.num_cpus, conf.cluster_size,
                         edf.is_schedulable, bp.any_fit_decreasing)

    tests = []
    titles = []

    for n in [1, 1.5, 2, 3, 5, 7, 10]:
        tests.append(partial(task_count_test, conf, sched_test, int(n * conf.num_cpus)))
        titles.append('N=%s' % n)
    return tests, titles

def run_ucap_range(conf, tests):
    for ucap in value_range(conf.num_cpus / 2, conf.num_cpus - 0.005, conf.step_size, last = conf.num_cpus - 0.005):
        samples = [[] for _ in tests]
        for _ in xrange(conf.samples):
            for i in xrange(len(tests)):
                samples[i].append(tests[i](ucap))
        row = []
        for i in xrange(len(tests)):
            ci  = boot.confidence_interval(samples[i], stat=mean)
            row += [ci[0], mean(samples[i]), ci[1]]
        yield [ucap] + row

def run_task_count_config(conf):
    common_checks(conf)
    conf.check('periods',      type=one_of(emstada.NAMED_PERIODS),
                               default='uni-moderate')
    conf.check('periods_dist', type=one_of(emstada.NAMED_PERIOD_DISTRIBUTIONS),
                               default='logunif')

    tests, titles = setup_task_count_tests(conf)
    header = ['UCAP']
    for t in titles:
        header += ['[CI-', t, 'CI+]']

    data = run_ucap_range(conf, tests)
    write_data(conf.output, data, header)


def generate_pedf_configs(options):
    for cpus in [4, 8, 16, 24, 32, 64]:
        cluster_size = 1
        for util in gen.NAMED_UTILIZATIONS:
            name = 'sd_exp=bph_util=%s_m=%02d_c=%02d' \
                        % (util, cpus, cluster_size)
            c = Config()
            c.output_file  = name + '.csv'
            c.num_cpus     = cpus
            c.cluster_size = cluster_size
            c.utilizations = util
            yield (name, c)


def generate_emstada_pedf_configs(options):
    for cpus in [4, 8, 16, 24, 32, 64]:
        cluster_size = 1
        dist = 'logunif'
        for n in xrange(1, 21):
            name = 'sd_exp=ebph_dist=%s_npc=%02d_m=%02d_c=%02d' \
                        % (dist, n, cpus, cluster_size)
            c = Config()
            c.output_file  = name + '.csv'
            c.num_cpus     = cpus
            c.cluster_size = cluster_size
            c.task_count   = n * cpus
            c.periods_dist = dist
            yield (name, c)

def generate_task_count_configs(options):
    for cpus in xrange(2, 64, 2):
        cluster_size = 1
        dist = 'logunif'
        name = 'sd_exp=bph-tc_dist=%s_m=%02d_c=%02d' \
                    % (dist, cpus, cluster_size)
        c = Config()
        c.output_file  = name + '.csv'
        c.num_cpus     = cpus
        c.cluster_size = cluster_size
        c.periods_dist = dist
        yield (name, c)

EXPERIMENTS = {
    'binpack/pedf'         : run_pedf_config,
    'binpack/emstada-pedf' : run_emstada_pedf_config,
    'binpack/task-count-pedf' : run_task_count_config,
}

CONFIG_GENERATORS = {
    'binpack/pedf'         : generate_pedf_configs,
    'binpack/emstada-pedf' : generate_emstada_pedf_configs,
    'binpack/task-count-pedf' : generate_task_count_configs,
}
