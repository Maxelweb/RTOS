from __future__ import division

import os
import sys

from functools import partial
from datetime import datetime

from toolbox.io import one_of, boolean_flag, Config, write_data
from toolbox.stats  import mean
from toolbox.sample import value_range

import schedcat.sched.edf as edf

import schedcat.generator.generator_emstada as emstada
import schedcat.generator.tasksets as gen
from  schedcat.generator.tasks import uniform_int, log_uniform_int, uniform_choice, uniform
from schedcat.model.tasks import SporadicTask, TaskSystem

from schedcat.overheads.jlfp import charge_scheduling_overheads as charge_scheduling_overheads_classic
from schedcat.overheads.jlfp import quantize_params, preemption_centric_irq_costs
from schedcat.overheads.model import Overheads, CacheDelay
from schedcat.util.math import const

from schedcat.util.time import ms2us

def charge_scheduling_overheads_message_passing(oheads, num_cpus, _unused, taskset):
    uscale, cpre = preemption_centric_irq_costs(oheads, True, taskset)

    if uscale <= 0:
        # interrupt overload from ticks alone !?
        return False

    n   = len(taskset)
    wss = taskset.max_wss()

    sched = 3 * (oheads.schedule(n) + oheads.ctx_switch(n)) \
            + oheads.cache_affinity_loss(wss) + oheads.master_scheduler(n)

    irq_latency = oheads.release_latency(n)

    unscaled = 2 * cpre + oheads.ipi_latency(n) + oheads.master_latency(n) + oheads.release(n)

    for ti in taskset:
        ti.period   -= irq_latency
        ti.deadline -= irq_latency
        ti.cost      = ((ti.cost + sched) / uscale) + unscaled
        if ti.density() > 1:
            return False

    return taskset


pick_wss = uniform_int(4, 512)

def test_with_overheads(test, oheads, account, dedicated_irq, num_cpus, taskset):
    ts, taskset = taskset.copy(), None

    for t in ts:
        t.wss = pick_wss()

    ts = account(oheads, num_cpus, dedicated_irq, ts)
    if ts:
        ts = quantize_params(ts)

    return ts and test(num_cpus, ts)


def load_overheads(implementation, num_cpus, soft):
    extra = [
        ('SCHED',   'sched'),
        ('SCHED2',  'sched2'),
        ('SEND-RESCHED', 'ipi_latency'),
        ('MASTER-HANDLER', 'master_scheduler'),
        ('CPU-REQUEST', 'master_latency'),
    ]

    fname = "data/rtas14/overheads_impl=%s_cpus=%d_average=%s.csv" \
            % (implementation, num_cpus, soft)
    try:
        oheads = Overheads.from_file(fname, custom_fields=extra,
            per_cpu_task_counts=True, num_cpus=num_cpus)
    except IOError:
        print "# Warning: loading dummy overhead values instead of %s!" % fname
        oheads = Overheads.from_file("data/rtas14/dummy.csv", custom_fields=extra,
            per_cpu_task_counts=True, num_cpus=num_cpus)

    # Convert LITMUS^RT terminology and measurements to SchedCAT terminology
    #  - Combine pre- and post-context-switch scheduling overhead.
    #  - LITMUS^RT SEND_RESCHED is mapped to ipi_latency.
    oheads.schedule = lambda x: oheads.sched(x) + oheads.sched2(x)

    fname = "data/rtas14/cpmd_host=nanping_cpus=%d_average=%s.csv" % (num_cpus, soft)
    try:
        cpmd   = CacheDelay.from_file(fname)
    except IOError:
        print "# Warning: loading dummy CPMD values instead of %s!" % fname
        cpmd = CacheDelay.from_file("data/rtas14/cpmd-dummy.csv")

    oheads.cache_affinity_loss = cpmd
    oheads.initial_cache_load  = cpmd
    return oheads

def setup_tests(conf):
    oheads_litmus               = load_overheads('G-EDF', conf.num_cpus, conf.soft)
    oheads_litmus_dedicated_irq = load_overheads('G-EDF-master', conf.num_cpus, conf.soft)
    oheads_message_passing      = load_overheads('G-EDF-dedicated', conf.num_cpus, conf.soft)
    oheads_sched_deadline       = load_overheads('DEADLINE', conf.num_cpus, conf.soft)

    edf_hrt = partial(edf.is_schedulable, rta_min_step=1000)
    edf_srt = edf.da.has_bounded_tardiness

    if not conf.soft:
        tests = [
            ("G-EDF (no overheads, m)", partial(edf_hrt, conf.num_cpus)),
            ("G-EDF (no overheads, m - 1)", partial(edf_hrt, conf.num_cpus - 1)),

            ("G-EDF (LITMUS^RT, m)",
             partial(test_with_overheads, edf_hrt, oheads_litmus,
                     charge_scheduling_overheads_classic, False, conf.num_cpus)),

            ("G-EDF (LITMUS^RT, m - 1)",
             partial(test_with_overheads, edf_hrt, oheads_litmus_dedicated_irq,
                     charge_scheduling_overheads_classic, True, conf.num_cpus - 1)),

            ("G-EDF (SCHED_DEADLINE, m)",
             partial(test_with_overheads, edf_hrt, oheads_sched_deadline,
                     charge_scheduling_overheads_classic, False, conf.num_cpus)),

            ("G-EDF (message passing, m - 1)",
             partial(test_with_overheads, edf_hrt, oheads_message_passing,
                     charge_scheduling_overheads_message_passing, True, conf.num_cpus - 1)),
        ]
    else:
        tests = [
            ("G-EDF (no overheads, m)", partial(edf_srt, conf.num_cpus)),

            ("G-EDF (no overheads, m - 1)", partial(edf_srt, conf.num_cpus - 1)),

            ("G-EDF (LITMUS^RT, m)",
             partial(test_with_overheads, edf_srt, oheads_litmus,
                     charge_scheduling_overheads_classic, False, conf.num_cpus)),

            ("G-EDF (LITMUS^RT, m - 1)",
             partial(test_with_overheads, edf_srt, oheads_litmus_dedicated_irq,
                     charge_scheduling_overheads_classic, True, conf.num_cpus - 1)),

            ("G-EDF (SCHED_DEADLINE, m)",
             partial(test_with_overheads, edf_srt, oheads_sched_deadline,
                     charge_scheduling_overheads_classic, False, conf.num_cpus)),

            ("G-EDF (message passing, m - 1)",
             partial(test_with_overheads, edf_srt, oheads_message_passing,
                     charge_scheduling_overheads_message_passing, True, conf.num_cpus - 1)),
        ]

    return ([t[0] for t in tests], [t[1] for t in tests])


def run_ucap(conf, tests):
    all_zeros_count = [0 for _ in tests]
    count = 0

    for ucap in value_range(1, conf.num_cpus, conf.step_size):
        print("[%d @ %s] u:\t%s" % (os.getpid(), datetime.now(), ucap))

        samples = [[] for _ in tests]

        for sample in xrange(conf.samples):
            taskset = conf.make_taskset(ucap)
            for i in xrange(len(tests)):
                if all_zeros_count[i] > 2:
                    samples[i].append(False)
                else:
                    samples[i].append(tests[i](taskset))

        row = []
        for i in xrange(len(tests)):
            avg = mean(samples[i])
            if avg > 0:
                all_zeros_count[i]  = 0
            else:
                all_zeros_count[i] += 1

            row.append(avg)

        yield [ucap] + ['%.2f' % x for x in row]


def make_emstada_taskset(conf, ucap):
    ts = emstada.gen_taskset((conf.period_min, conf.period_max),
                             'logunif' if conf.period_dist == 'log' else 'unif',
                             conf.num_tasks, ucap, scale=ms2us)
    return ts

def run_ucap_config(conf):
    # standard parameters
    conf.check('num_cpus', type=int, min=1)
    conf.check('samples',  type=int, min=1, default=DEFAULT_SAMPLES)

    conf.check('soft',      type=boolean_flag, default=False)
    conf.check('step_size', type=float, min=0.01, default=0.5 if conf.num_cpus > 8 else 0.25)

    conf.check('period_min', type=int, min=1, default=1)
    conf.check('period_max', type=int, min=conf.period_min, default=1000)

    conf.check('num_tasks', type=int, min=conf.num_cpus)

    conf.check('period_dist', type=one_of(['log', 'uni']), default='log')


    conf.make_taskset = partial(make_emstada_taskset, conf)

    (titles, tests) = setup_tests(conf)

    header = ['UTIL']
    header += titles

    data = run_ucap(conf, tests)
    write_data(conf.output, data, header)


def run_tcount(conf, tests):
    all_zeros_count = [0 for _ in tests]
    count = 0


    for n in range(conf.tasks_min, conf.tasks_max + 1, conf.step_size):
        print("[%d @ %s] n:\t%s" % (os.getpid(), datetime.now(), n))

        utils = []
        samples = [[] for _ in tests]

        for sample in xrange(conf.samples):
            taskset = conf.make_taskset(n)
            utils.append(taskset.utilization())

            for i in xrange(len(tests)):
                if all_zeros_count[i] > 2:
                    samples[i].append(False)
                else:
                    samples[i].append(tests[i](taskset))
        print '=> min-u:%.2f avg-u:%.2f max-u:%.2f' % (min(utils), sum(utils) / len(utils), max(utils))
        row = []
        for i in xrange(len(tests)):
            avg = mean(samples[i])
            if avg > 0:
                all_zeros_count[i]  = 0
            else:
                all_zeros_count[i] += 1

            row.append(avg)

        yield [n] + ['%.2f' % x for x in row]
        if min(utils) >= conf.num_cpus:
            break

PERIODS = {
    'uni-200-1000' : uniform_int(200, 1000),
    'uni-10-1000'  : uniform_int( 10, 1000),
    'uni-10-100'   : uniform_int( 10,  100),
    'uni-1-1000'   : uniform_int(  1, 1000),

    'log-uni-200-1000' : log_uniform_int(200, 1000),
    'log-uni-10-1000'  : log_uniform_int( 10, 1000),
    'log-uni-10-100'   : log_uniform_int( 10,  100),
    'log-uni-1-1000'   : log_uniform_int(  1, 1000),
}

UTILIZATIONS = {
}

# pull in standard periods
PERIODS.update(gen.NAMED_PERIODS)
UTILIZATIONS.update(gen.NAMED_UTILIZATIONS);

def run_tcount_config(conf):
    # standard parameters
    conf.check('num_cpus', type=int, min=1)
    conf.check('samples',  type=int, min=1, default=DEFAULT_SAMPLES)

    conf.check('utilizations', type=one_of(UTILIZATIONS), default='exp-light')
    conf.check('periods',      type=one_of(PERIODS),   default='uni-1-1000')

    conf.check('soft',      type=boolean_flag, default=False)

    conf.check('step_size', type=int, min=1, default=conf.num_cpus // 4)
    conf.check('tasks_min', type=int, min=1, default=conf.num_cpus)
    conf.check('tasks_max', type=int, min=1, default=conf.num_cpus * 10)

    conf.generate = gen.mkgen(UTILIZATIONS[conf.utilizations],
                              PERIODS[conf.periods])

    conf.make_taskset = lambda n: conf.generate(max_tasks=n, time_conversion=ms2us)

    (titles, tests) = setup_tests(conf)

    header = ['TASKS']
    header += titles

    data = run_tcount(conf, tests)
    write_data(conf.output, data, header)



###############################################


CPU_COUNTS = [8, 16, 24, 32, 48, 64]

UTILS = [
    'exp-light',
    'uni-light',
    'exp-medium',
    'uni-medium',
    'exp-heavy',
    'uni-heavy',
]

DEFAULT_SAMPLES = 100

def generate_ucap_configs(options):
    for cpus in CPU_COUNTS:
        for soft in [True, False]:
            for n in range(2, 11):
                for per in ['uni', 'log']:
                    name = 'sd_exp=rtas14-ucap_m=%02d_soft=%s_npc=%02d_per=%s' % (cpus, soft, n, per)

                    c = Config()
                    c.num_cpus = cpus
                    c.samples  = DEFAULT_SAMPLES

                    c.num_tasks    = n * cpus
                    c.period_dist  = per
                    c.soft         = soft

                    c.output_file  = name + '.csv'
                    yield (name, c)

def generate_tcount_configs(options):
    for cpus in CPU_COUNTS:
        for soft in [True, False]:
            for periods in PERIODS:
                for util in UTILS:
                    name = 'sd_exp=rtas14-tcount_m=%02d_soft=%s_util=%s_per=%s' \
                        % (cpus, soft, util, periods)

                    c = Config()
                    c.num_cpus = cpus
                    c.soft     = soft
                    c.samples  = DEFAULT_SAMPLES

                    c.utilizations = util
                    c.periods = periods

                    c.output_file  = name + '.csv'
                    yield (name, c)

###############################################

EXPERIMENTS = {
    'rtas14/ucap'    : run_ucap_config,
    'rtas14/tcount'  : run_tcount_config
}

CONFIG_GENERATORS = {
    'rtas14/ucap'    : generate_ucap_configs,
    'rtas14/tcount'  : generate_tcount_configs
}




###############################################

def print_gedf_table():
    print 'Exporting GSN-EDF scalability data...'
    data = []
    hdr  = [
        'm',
        'schedule (max)',
        'schedule (avg)',
        'release (max)',
        'release (avg)',
    ]
    for m in CPU_COUNTS:
        avg = load_overheads('G-EDF', m, True)
        wc  = load_overheads('G-EDF', m, False)

        tasks = [i * m for i in range(1, 11)]

        data += [
            [m,
             max([ wc.schedule(n) for n in tasks]),
             max([avg.schedule(n) for n in tasks]),
             max([ wc.release(n)  for n in tasks]),
             max([avg.release(n)  for n in tasks]),]
        ]
    write_data(sys.stdout, data, hdr)


def print_sd_table():
    print 'Exporting SCHED_DEADLINE scalability data...'
    data = []
    hdr  = [
        'm',
        'schedule (max)',
        'schedule (avg)',
        'release (max)',
        'release (avg)',
    ]
    for m in CPU_COUNTS:
        avg = load_overheads('DEADLINE', m, True)
        wc  = load_overheads('DEADLINE', m, False)

        tasks = [i * m for i in range(1, 11)]

        data += [
            [m,
             max([ wc.schedule(n) for n in tasks]),
             max([avg.schedule(n) for n in tasks]),
             max([ wc.release(n)  for n in tasks]),
             max([avg.release(n)  for n in tasks]),]
        ]
    write_data(sys.stdout, data, hdr)

def main():
    m = 32
    soft = False
    gedf = load_overheads('G-EDF-dedicated', m, soft)
    sdl  = load_overheads('DEADLINE', m, soft)

    def cost(oheads, n):
        return 2 * (oheads.schedule(n) + oheads.ctx_switch(n)) \
            + oheads.cache_affinity_loss(64)

    period = uniform_int(10, 100)
    tasks = TaskSystem([SporadicTask(1 * 1000, period() * 1000) for _ in range(1, 1000)])

    def cpre(oheads, n):
        (uscale, cpre) = preemption_centric_irq_costs(oheads, False, tasks[:n])
        return cpre

    def uscale(oheads, n):
        (uscale, cpre) = preemption_centric_irq_costs(oheads, False, tasks[:n])
        return 1 / uscale

    print 'Schedule'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = gedf.schedule(x)
        o_sdl  = sdl.schedule(x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'CXS'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = gedf.ctx_switch(x)
        o_sdl  = sdl.ctx_switch(x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'RELEASE'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = gedf.release(x)
        o_sdl  = sdl.release(x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'TICK'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = gedf.tick(x)
        o_sdl  = sdl.tick(x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'LATENCY'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = gedf.release_latency(x)
        o_sdl  = sdl.release_latency(x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'cost'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = cost(gedf, x)
        o_sdl  = cost(sdl, x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'CPRE'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = cpre(gedf, x)
        o_sdl  = cpre(sdl, x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'USCALE'
    for x in range(m * 1, m * 10 + 1, m):
        o_gedf = uscale(gedf, x)
        o_sdl  = uscale(sdl, x)
        print "%3d => %s (%.2f vs %.2f)" % (x, 'G-EDF' if o_gedf < o_sdl else 'DEADLINE', o_gedf, o_sdl)

    print 'Q', gedf.quantum_length, sdl.quantum_length

if __name__ == '__main__':
    print_gedf_table()
    print_sd_table()
