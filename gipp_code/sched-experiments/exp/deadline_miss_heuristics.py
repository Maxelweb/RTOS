from __future__ import division

import random
from functools import partial
from math import floor

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean, Histogram
import toolbox.bootstrap as boot

import schedcat.generator.tasksets as gen
import schedcat.mapping.binpack as bp
import schedcat.sched.edf as edf
import schedcat.sim.edf as sim
import schedcat.model.serialize as serialize

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.util.time import ms2us

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

def simulation_test(num_cpus, phase_heuristic, length, taskset):
    phase_heuristic(num_cpus, taskset)
    return sim.no_counter_example(num_cpus, taskset, length)

seqno = 0

def simulation_miss_time(num_cpus, phase_heuristic, length, taskset, save_threshold=0):
    phase_heuristic(num_cpus, taskset)
    time = sim.time_of_first_miss(num_cpus, taskset, length)
    if save_threshold > 0  and time > save_threshold:
        global seqno
        seqno += 1
        serialize.write(taskset, 'late-miss-%03d.ts' % seqno)
    return time

def zero_phase(num_cpus, taskset):
    for t in taskset:
        t.phase = 0

def shared_deadline(num_cpus, taskset, offset=0):
    max_p = max(taskset.max_period(), taskset.max_deadline()) + offset

    for t in taskset:
        # release time of job that shares max_p as a deadline
        relt = max_p - t.deadline
        # how many jobs fit into max_p prior to relt?
        jobs = int(floor(relt / t.period))
        # set offset to have synchronous release after warm-up
        t.phase = max_p - (jobs + 1) * t.period
        assert t.phase >= 0

def stack_against_least_slack(num_cpus, taskset, offset=0):
    shared_deadline(num_cpus, taskset, offset)
    taskset.task_with_max_density().phase += 1

def random_phase(num_cpus, taskset):
    max_p = max(taskset.max_period(), taskset.max_deadline())
    for t in taskset:
        t.phase = random.randint(0, max_p - 1)

def setup_tests(conf):
    tests = [
        partial(simulation_test, conf.num_cpus, zero_phase, ms2us(200)),
        partial(simulation_test, conf.num_cpus, shared_deadline, ms2us(200)),
        partial(simulation_test, conf.num_cpus, stack_against_least_slack, ms2us(200)),
        partial(simulation_test, conf.num_cpus, random_phase, ms2us(200)),
        partial(simulation_test, conf.num_cpus, zero_phase, ms2us(20000)),
        partial(simulation_test, conf.num_cpus, shared_deadline, ms2us(20000)),
        partial(simulation_test, conf.num_cpus, stack_against_least_slack, ms2us(20000)),
        partial(simulation_test, conf.num_cpus, partial(stack_against_least_slack, offset=10000), ms2us(20000)),
        partial(simulation_test, conf.num_cpus, zero_phase, ms2us(60000)),
        partial(simulation_test, conf.num_cpus, random_phase, ms2us(60000)),
        partial(simulation_test, conf.num_cpus, partial(stack_against_least_slack, offset=10000), ms2us(60000)),
        partial(simulation_test, conf.num_cpus, zero_phase, ms2us(120000)),

        partial(edf.is_schedulable, conf.num_cpus, rta_min_step=1000),
        partial(partitioned_test, conf.num_cpus, 1, edf.is_schedulable,
                bp.any_fit_decreasing),
    ]
    titles = [
        'SIM (synch, 200ms)',
        'SIM (shared dl, 200ms)',
        'SIM (least-slack, 200ms)',
        'SIM (random, 200ms)',
        'SIM (synch, 20s)',
        'SIM (shared dl, 20s)',
        'SIM (least-slack, 20s)',
        'SIM (least-slack+10s, 20s)',
        'SIM (synch, 60s)',
        'SIM (random, 60s)',
        'SIM (least-slack+10s, 60s)',
        'SIM (synch, 120s)',
        'G-EDF',
        'P-EDF',
    ]
    return tests, titles

def run_range(conf, tests):
    conf.check('step_size',    type=float, default=0.25)
    conf.check('samples',      type=int, default=100)

    for ucap in value_range(1, conf.num_cpus, conf.step_size):
        samples = [[] for _ in tests]
        for _ in xrange(conf.samples):
            taskset = conf.generate(max_util=ucap, time_conversion=ms2us, squeeze=True)
            for i in xrange(len(tests)):
                samples[i].append(tests[i](taskset))
        row = []
        for i in xrange(len(tests)):
#            ci  = boot.confidence_interval(samples[i], stat=mean)
#            row += [ci[0], mean(samples[i]), ci[1]]
            row += [mean(samples[i])]
        yield [ucap] + row


def setup_heuristics(conf):
    tests = [
        partial(simulation_miss_time, conf.num_cpus, zero_phase), # , save_threshold=ms2us(100000)
        partial(simulation_miss_time, conf.num_cpus, shared_deadline),
        partial(simulation_miss_time, conf.num_cpus, stack_against_least_slack),
        partial(simulation_miss_time, conf.num_cpus, partial(stack_against_least_slack, offset=10000)),
        partial(simulation_miss_time, conf.num_cpus, random_phase),
    ]
    titles = [
        'SIM (synchronous)',
        'SIM (shared deadline)',
        'SIM (least-slack)',
        'SIM (least-slack+10s)',
        'SIM (random)',
    ]
    return tests, titles


def sample_miss_time(conf, heuristics):
    conf.check('ucap',        type=float, default=conf.num_cpus / 2)
    conf.check('sim_timeout', type=int, default=ms2us(120000))
    conf.check('samples',     type=int, default=1000)

    samples = [[] for _ in heuristics]
    for x in xrange(conf.samples):
        if x and x % 100 == 0:
            print 'sample %5d' % x
        taskset = conf.generate(max_util=conf.ucap, time_conversion=ms2us, squeeze=True)
        if edf.is_schedulable(conf.num_cpus, taskset):
            # skip expensive simulation if task is known to be schedulable
            for i in xrange(len(heuristics)):
                samples[i].append(0)
        else:
            for i in xrange(len(heuristics)):
                samples[i].append(heuristics[i](conf.sim_timeout, taskset))

    hists = []
    for i in xrange(len(heuristics)):
        h = Histogram(1, conf.sim_timeout + 1, conf.sim_timeout // ms2us(50))
        for x in samples[i]:
            h(x)
        hists.append(h)

    for i in xrange(len(h.bins)):
        yield [(i + 1) * h.bin_size] + [h.bins[i] for h in hists]

    yield ['NO-MISS'] + [h.underflow for h in hists]

def check_config(conf):
    conf.check('num_cpus',     type=int)
    conf.check('utilizations', type=one_of(gen.NAMED_UTILIZATIONS))
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(gen.NAMED_PERIODS),   default='uni-moderate')

    conf.generate = gen.DIST_BY_KEY[conf.deadlines][conf.periods][conf.utilizations]

def run_sched_config(conf):
    check_config(conf)
    tests, titles = setup_tests(conf)
    header = ['UCAP'] + titles
    data = run_range(conf, tests)
    write_data(conf.output, data, header)

def run_hist_config(conf):
    check_config(conf)
    tests, titles = setup_heuristics(conf)
    header = ['BIN'] + titles
    data = sample_miss_time(conf, tests)
    write_data(conf.output, data, header)

def generate_sched_configs(options):
    for cpus in [4, 8, 16, 24, 32, 64]:
        for util in gen.NAMED_UTILIZATIONS:
            name = 'sd_exp=dmh-sched_util=%s_m=%02d' \
                        % (util, cpus)
            c = Config()
            c.num_cpus     = cpus
            c.utilizations = util
            c.output_file  = name + '.csv'
            yield (name, c)

def generate_hist_configs(options):
    for cpus in [4, 8, 16, 24, 32, 64]:
        for ucap in value_range(cpus // 2, cpus, 0.25):
            for util in gen.NAMED_UTILIZATIONS:
                name = 'sd_exp=dmh-hist_util=%s_m=%02d_ucap=%.2f' \
                            % (util, cpus, ucap)
                c = Config()
                c.num_cpus     = cpus
                c.ucap         = ucap
                c.utilizations = util
                c.output_file  = name + '.csv'
                yield (name, c)

EXPERIMENTS = {
    'dmh/hist'  : run_hist_config,
    'dmh/sched' : run_sched_config,
}

CONFIG_GENERATORS = {
    'dmh/hist'  : generate_hist_configs,
    'dmh/sched' : generate_sched_configs,
}
