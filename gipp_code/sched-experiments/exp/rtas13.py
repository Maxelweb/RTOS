from __future__ import division

import random
from itertools import izip
from functools import partial
from collections import defaultdict
from datetime import datetime
import os

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats  import mean

from schedcat.model.serialize import write as write_to_xml, load as load_xml

# task set generation
from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.model.resources import initialize_resource_model, ResourceRequirement
import schedcat.generator.tasksets as gen
from schedcat.generator.tasks import uniform_int, uniform_choice, uniform
from schedcat.util.time import ms2us

# overheads
from schedcat.overheads.model import Overheads, CacheDelay
from schedcat.overheads.locking import charge_dpcp_overheads, charge_semaphore_overheads
from schedcat.overheads.fp import charge_scheduling_overheads, quantize_params
from schedcat.util.math import const, monotonic_pwlin

# schedulability testing
import schedcat.mapping.binpack as bp
import schedcat.sched.fp as fp
import schedcat.locking.bounds as locking

# Hard-coded overheads from Nanping-8 and Nanping-16

OVERHEADS = [       # Nanping-8             Nanping-16
    ('schedule',    const(15.67 + 4.90),    const(55.36 + 26.84)),
    ('ctx_switch',  const(8.78),            const(40.99)),
    ('release',     const(17.06),           const(36.37)),
    ('tick',        const(5.05),            const(30.18)),
    ('release_latency', const(23.71),       const(72.11)),
    ('ipi_latency', const(25.69),           const(47.68)),
    ('syscall_in',  const(5.16),            const(32.02)),
    ('syscall_out', const(4.85),            const(17.80))
]

LOCKS = [           # LOCK / UNLOCK 8c      # LOCK / UNLOCK 16c
    ('MPCP',        const(8.22),            const(34.15),
                    const(12.91),           const(37.51)),
    ('FMLP+',       const(9.81),            const(35.51),
                    const(11.90),           const(40.67)),
    ('DPCP',        const(5.95),            const(39.28),
                    const( 7.19),           const(37.76)),
    ('DFLP',        const(5.44),            const(35.96),
                    const(4.82),            const(33.25)),
    ('EXCL',        const(0),               const(0),
                    const(0),               const(0)),
]

cpmd = {}
cpmd[8] = CacheDelay()
cpmd[8].set_cpmd_cost(CacheDelay.L1, monotonic_pwlin([
    ( 1,	0.86),
    ( 2,	0.91),
    ( 4,	1.02),
    ( 8,	4.16),
    ( 16,   1.43),
    ( 32,   2.15),
    ( 64,   5.72),
    (128,  10.63),
    (256,  56.38),
    (512, 104.26)
]))

cpmd[16] = CacheDelay()
cpmd[16].set_cpmd_cost(CacheDelay.L1, monotonic_pwlin([
    (  1,	1.06),
    (  2,	2.57),
    (  4,	1.38),
    (  8,	1.65),
    ( 16,   2.20),
    ( 32,   3.58),
    ( 64,  22.99),
    (128,  41.07),
    (256,  74.34),
    (512, 206.60)
]))

def get_overhead_model(wss=1):
    overheads = {}
    overheads[8]  = {}
    overheads[16] = {}

    for (proto, l8, l16, u8, u16) in LOCKS:
        overheads[8][proto] = Overheads()
        overheads[16][proto] = Overheads()

        overheads[8][proto].lock    = l8
        overheads[8][proto].unlock  = u8
        overheads[16][proto].lock   = l16
        overheads[16][proto].unlock = u16

        for (key, val8, val16) in OVERHEADS:
            overheads[8][proto].__dict__[key] = val8
            overheads[16][proto].__dict__[key] = val16

        overheads[8][proto].cache_affinity_loss = const(cpmd[8](wss))
        overheads[16][proto].cache_affinity_loss = const(cpmd[16](wss))

    return overheads

def precompute_cs_costs(taskset):
    # Pre-compute how long agents execute on behalf of the task.
    for t in taskset:
        t.cost_without_requests = t.cost
        t.total_request_time = 0
        for res_id in t.resmodel:
            res_total = t.resmodel[res_id].max_length \
                        * t.resmodel[res_id].max_requests
            t.total_request_time += res_total

def prep_taskset_for_locking_analysis(taskset):
    # Deadline monotonic priorities.
    taskset.sort_by_deadline()

    # Assign IDs to match the order.
    taskset.assign_ids()

    # The locking code wants explicit priorities.
    locking.assign_fp_preemption_levels(taskset)

    precompute_cs_costs(taskset)

def pfp_test(apply_bounds, param_update, num_cpus, overheads, apply_lock_overheads,
             taskset, cluster_sizes, resource_locality, locality_override=None):
    # Assumes taskset has already been sorted and ID'd in priority order.
    original_taskset = taskset
    taskset = original_taskset.copy()

    if not locality_override is None:
        resource_locality = locality_override

    try:
        param_update(taskset)

        partitions = bp.worst_fit_decreasing(
                        taskset, num_cpus,
                        capacity=1,
                        weight=lambda t: t.utilization(),
                        misfit=bp.report_failure,
                        empty_bin=TaskSystem)

        for cpu, part in enumerate(partitions):
            # Restore deadline monotonic order.
            part.sort_by_deadline()
            for t in part:
                t.partition = cpu

        if overheads:
            # Inflate for scheduling overheads and add interrupt tasks.
            partitions = [charge_scheduling_overheads(overheads, num_cpus, False, part)
                          for part in partitions]

            # apply locking overheads to obtain effective critical section lengths
            if not apply_lock_overheads(taskset):
                # task set became infeasible
                return False
            for part in partitions:
                if quantize_params(part) is False:
                    # task set became infeasible
                    return False

        # Check schedulability without resource requirements before invoking
        # the more expensive analysis.
        for part in partitions:
            if not fp.is_schedulable(1, part):
                # Even without blocking, one partition is overloaded.
                # No point in going any further.
                return False

        # Keep track of the observed response times.
        response_times = [t.response_time for t in taskset]

        # Next, try deadlines before starting the more expensive iteration
        for t in taskset:
            t.response_time = t.deadline
        apply_bounds(num_cpus, taskset, cluster_sizes, resource_locality)
        schedulable = True
        for part in partitions:
            if not fp.is_schedulable(1, part):
                # Nope, we'll have to look more carefully.
                schedulable = False
                break

        if schedulable:
            # Great, each task's maximum response-time is bounded by its deadline.
            # With constrained deadlines, this is sufficient.
            return True

        # Schedulable without requests, but not when upper-bounding response-times
        # by implicit deadlines. We'll have to look more carefully.
        response_times_consistent = False
        schedulable               = True

        # To start with, restore the response-time bounds as observed without
        # resource sharing.
        for (resp, tsk) in izip(response_times, taskset):
            tsk.response_time = resp

        # iterate until the assumed response times match the
        # actual response times
        iteration = 0
        while schedulable and not response_times_consistent:
            iteration += 1
            # backup response time assumption
            for t in taskset:
                t.assumed_response_time = t.response_time

            apply_bounds(num_cpus, taskset, cluster_sizes, resource_locality)

            # Let's check if the task set is now schedulable and the observed
            # response time bounds did not invalidate our assumption.
            response_times_consistent = True

            for part in partitions:
                if not fp.is_schedulable(1, part):
                    # Nope. Since response times are monotonically increasing
                    # under this iterative approach, we can give up right away.
                    return False

            for t in taskset:
                # Check if anything changed.
                if t.response_time > t.assumed_response_time:
                    # Need another iteration to find fixpoint.
                    response_times_consistent = False
        # The response-time iteration finished. Either it converged or
        # some partition is not schedulable.
        return schedulable

    except bp.DidNotFit:
        # Could not even partition the task set.
        # Clearly not schedulable.
        return False

def shmem_cost_update(taskset):
    # account for CS length, which is part of the execution requirement
    for t in taskset:
        t.cost = t.cost_without_requests + t.total_request_time

def dist_jitter_update(taskset):
    # account for self suspension while agent executes
    for t in taskset:
        t.jitter = t.total_request_time

def dflp_bounds(num_cpus, taskset, cluster_sizes, resource_locality):
    locking.apply_lp_dflp_bounds(taskset, resource_locality, cluster_sizes)

def mpcp_bounds(num_cpus, taskset, cluster_sizes, resource_locality, use_linprog = True):
    if use_linprog:
        locking.apply_lp_mpcp_bounds(taskset)
    else:
        locking.apply_mpcp_bounds(taskset)

def fmlpp_bounds(num_cpus, taskset, cluster_sizes, resource_locality, use_linprog = True):
    if use_linprog:
        locking.apply_lp_part_fmlp_bounds(taskset)
    else:
        locking.apply_part_fmlp_bounds(taskset)

def dpcp_bounds(num_cpus, taskset, cluster_sizes, resource_locality,
                use_rta = True, use_linprog = True):
    if use_linprog:
        locking.apply_lp_dpcp_bounds(taskset, resource_locality, cluster_sizes, use_rta)
    else:
        locking.apply_dpcp_bounds(taskset, resource_locality, True)

def no_blocking_bounds(num_cpus, taskset, cluster_sizes, resource_locality):
    # Pretend that each task has private, local access to each resource.
    pass

def setup_theory_tests(conf):

    tests = [
        partial(pfp_test, no_blocking_bounds, shmem_cost_update, conf.num_cpus,
                None, None),
        partial(pfp_test, fmlpp_bounds, shmem_cost_update, conf.num_cpus,
                None, None),
        partial(pfp_test, partial(fmlpp_bounds, use_linprog=False), shmem_cost_update, conf.num_cpus,
                None, None),
        partial(pfp_test, mpcp_bounds, shmem_cost_update, conf.num_cpus,
                None, None),
        partial(pfp_test, partial(mpcp_bounds, use_linprog=False), shmem_cost_update, conf.num_cpus,
                None, None),
        partial(pfp_test, dflp_bounds, dist_jitter_update, conf.num_cpus - conf.nr_ctrl,
                None, None),
        partial(pfp_test, dpcp_bounds, dist_jitter_update, conf.num_cpus - conf.nr_ctrl,
                None, None),
        partial(pfp_test, partial(dpcp_bounds, use_rta=False), dist_jitter_update,
                conf.num_cpus - conf.nr_ctrl,
                None, None),
        partial(pfp_test, partial(dpcp_bounds, use_linprog=False), dist_jitter_update,
                conf.num_cpus - conf.nr_ctrl,
                None, None),
    ]
    titles = [
        'EXCL/FP',
        'FMLP+/FP',
        'FMLP+-OLD/FP',
        'MPCP/FP',
        'MPCP-OLD/FP',
        'DFLP/FP',
        'DPCP/FP',
        'DPCP-RTA/FP',
        'DPCP-OLD/FP'
    ]
    return tests, titles

def get_resource_locality(num_cpus, num_resources, nr_ctrl):
    # pre-compute resource assignment
    resource_locality = {}
    if nr_ctrl == 0:
        for i in range(num_resources):
            resource_locality[i] = i % num_cpus
    else:
        ctrl_offset = num_cpus - nr_ctrl
        for i in range(num_resources):
            resource_locality[i] = ctrl_offset + (i % nr_ctrl)


def setup_ctrl_tests(conf):

    tests = [
        partial(pfp_test, dflp_bounds, dist_jitter_update, conf.num_cpus,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 0)),
        partial(pfp_test, dflp_bounds, dist_jitter_update, conf.num_cpus - 1,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 1)),
        partial(pfp_test, dflp_bounds, dist_jitter_update, conf.num_cpus - 2,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 2)),
        partial(pfp_test, dflp_bounds, dist_jitter_update, conf.num_cpus - 4,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 4)),

        partial(pfp_test, dpcp_bounds, dist_jitter_update, conf.num_cpus,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 0)),
        partial(pfp_test, dpcp_bounds, dist_jitter_update, conf.num_cpus - 1,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 1)),
        partial(pfp_test, dpcp_bounds, dist_jitter_update, conf.num_cpus - 2,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 2)),
        partial(pfp_test, dpcp_bounds, dist_jitter_update, conf.num_cpus - 4,
                None, None, locality_override=get_resource_locality(conf.num_cpus, conf.num_resources, 4)),
    ]
    titles = [
        'DFLP-0/FP',
        'DFLP-1/FP',
        'DFLP-2/FP',
        'DFLP-4/FP',

        'DPCP-0/FP',
        'DPCP-1/FP',
        'DPCP-2/FP',
        'DPCP-4/FP',
    ]
    return tests, titles


def setup_tests_with_overheads(conf):

    overheads = get_overhead_model(conf.wss)[conf.num_cpus]
    locking_oheads = {
        'DPCP'  : partial(charge_dpcp_overheads, overheads['DPCP']),
        'MPCP'  : partial(charge_semaphore_overheads, overheads['MPCP'], True, True),
        'DFLP'  : partial(charge_dpcp_overheads, overheads['DFLP']),
        'FMLP+' : partial(charge_semaphore_overheads, overheads['FMLP+'], True, True),
        'EXCL'  : lambda _: True,
    }

    tests = [
           partial(pfp_test, no_blocking_bounds, shmem_cost_update, conf.num_cpus,
                overheads['FMLP+'], locking_oheads['EXCL']),
        partial(pfp_test, fmlpp_bounds, shmem_cost_update, conf.num_cpus,
                overheads['FMLP+'], locking_oheads['FMLP+']),
        partial(pfp_test, mpcp_bounds, shmem_cost_update, conf.num_cpus,
                overheads['MPCP'], locking_oheads['MPCP']),
        partial(pfp_test, dflp_bounds, dist_jitter_update, conf.num_cpus - conf.nr_ctrl,
                overheads['DFLP'], locking_oheads['DFLP']),
        partial(pfp_test, dpcp_bounds, dist_jitter_update, conf.num_cpus - conf.nr_ctrl,
                overheads['DPCP'], locking_oheads['DPCP']),
    ]
    titles = [
        'EXCL/FP',
        'FMLP+/FP',
        'MPCP/FP',
        'DFLP/FP',
        'DPCP/FP',
    ]
    return tests, titles

DEFAULT_SAMPLES = 10

CSLENGTH  = {
    'short'     : uniform_int(1,   15),
    'medium'    : uniform_int(1,  100),
    'midrange'  : uniform_int(100, 1000),
    'long'      : uniform_int(5, 1280),

    'uni-10-50'  : uniform_int(10, 50),
    'uni-50-100' : uniform_int(50, 100),
    'uni-50-150' : uniform_int(50, 150),
    'uni-100-150' : uniform_int(100, 150),
    'uni-10-500' : uniform_int(10, 500),
}

PERIODS = {
    'uni-100-200' : uniform_int(100, 200),
    'uni-50-100'  : uniform_int(50, 100),
    'uni-10-1000'  : uniform_int(10, 1000),
    'uni-10-100'  : uniform_int(10, 100),
    'disc-1-100'  : uniform_choice([1, 5, 25, 50, 100]),
    'disc-10-120'  : uniform_choice([10, 20, 60, 120]),
}

UTILIZATIONS = {
    'uni-low'  : uniform(0.1, 0.2),
}

# pull in standard periods
PERIODS.update(gen.NAMED_PERIODS)
UTILIZATIONS.update(gen.NAMED_UTILIZATIONS);

def draw_cs_count(conf):
    if random.random() <= conf.access_prob:
        return random.randint(1, conf.max_requests)
    else:
        return 0

def generate_requests(conf, task, res_id_offset=0):
    length = CSLENGTH[conf.cslength]
    for idx in xrange(conf.num_resources):
        res_id = idx + res_id_offset
        num_accesses = draw_cs_count(conf)
        if num_accesses:
            # add writes
            task.resmodel[res_id] = ResourceRequirement(res_id, num_accesses, length(), 0, 0)

def check_config(conf):
    # standard parameters
    conf.check('num_cpus',     type=int, min=1)
    conf.check('samples',      type=int, default=DEFAULT_SAMPLES, min=1)
    conf.check('step_size',    type=float, default=0.25, min=0.1)
    conf.check('utilizations', type=one_of(UTILIZATIONS), default='exp-light')
    conf.check('access_prob',  type=float, min=0)
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(PERIODS),   default='uni-moderate')
    conf.check('min_ucap',     type=float, min=1, max=conf.num_cpus, default=conf.num_cpus/4)
    conf.check('nr_ctrl',       type=int,   min=0, default=0)
    conf.check('max_ucap',     type=float, min=1, max=conf.num_cpus, default=conf.num_cpus)
    conf.check('cluster_size',  type=int, default=1, min=1, max=1)
    conf.check('num_resources', type=int,   min=1, max=128)
    conf.check('max_requests',  type=int,   min=1, default=8)
    conf.check('cslength',      type=one_of(CSLENGTH), default='short')

    # working set size in KB
    conf.check('wss',           type=int, min=1, default=128)

    conf.generate = gen.mkgen(UTILIZATIONS[conf.utilizations],
                              PERIODS[conf.periods],
                              gen.NAMED_DEADLINES[conf.deadlines])

def ensure_min_task_count(num_cpus, taskset, periods='uni-moderate'):
    # Make sure the scheduling problem isn't entirely trivial: have more
    # tasks than processors. The function works by scaling all tasks rather
    # than splitting only a few.
    if len(taskset) < num_cpus + 1:
        uavg    = taskset.utilization() / len(taskset)
        utarget = uavg * len(taskset) / (num_cpus + 1)
        scale   = utarget / uavg
        # adjust existing tasks to make room for additional tasks
        for t in taskset:
            t.cost = max(1, int(t.cost * scale))
        # add new tasks
        while len(taskset) < num_cpus + 1:
            u = utarget
            p = ms2us(gen.NAMED_PERIODS[periods]())
            c = max(int(p * u), 1)
            taskset.append(SporadicTask(c, p))
    return taskset

def gen_taskset(conf, ucap):
    ts = conf.generate(max_util=ucap, time_conversion=ms2us, squeeze=True)
    ts = ensure_min_task_count(conf.num_cpus, ts)
    initialize_resource_model(ts)

    # generate resource requirements
    for t in ts:
        generate_requests(conf, t)

    # pre-compute parameters relevant to the locking analysis
    prep_taskset_for_locking_analysis(ts)
    return ts

def gen_taskset_with_count(conf, count):
    ts = conf.generate(max_tasks=count, time_conversion=ms2us, squeeze=True)
    initialize_resource_model(ts)

    # generate resource requirements
    for t in ts:
        generate_requests(conf, t)

    # pre-compute parameters relevant to the locking analysis
    prep_taskset_for_locking_analysis(ts)
    return ts

def run_range(conf, tests):
    all_zeros_count = [0 for _ in tests]
    count = 0

    # only consider partitioned systems
    cluster_sizes = {}
    for i in range(conf.num_cpus):
        cluster_sizes[i] = 1

    # pre-compute resource assignment
    resource_locality = get_resource_locality(conf.num_cpus, conf.num_resources,
                                              conf.nr_ctrl)

    for ucap in value_range(conf.min_ucap, conf.max_ucap, conf.step_size):
        print("[run_range] ucap:\t%s" % ucap)


        samples = [[] for _ in tests]

        for sample in xrange(conf.samples):
            taskset = conf.make_taskset(ucap)

            for i in xrange(len(tests)):
                if all_zeros_count[i] > 2:
                    samples[i].append(False)
                else:
                    samples[i].append(tests[i](taskset, cluster_sizes, resource_locality))
        row = []
        for i in xrange(len(tests)):
            avg = mean(samples[i])
            if avg > 0:
                all_zeros_count[i]  = 0
            else:
                all_zeros_count[i] += 1

            row.append(avg)

        yield [ucap] + row

def run_task_count(conf, tests):
    all_zeros_count = [0 for _ in tests]
    count = 0

    # only consider partitioned systems
    cluster_sizes = {}
    for i in range(conf.num_cpus):
        cluster_sizes[i] = 1

    # pre-compute resource assignment
    resource_locality = {}
    if conf.nr_ctrl == 0:
        for i in range(conf.num_resources):
            resource_locality[i] = i % conf.num_cpus
    else:
        ctrl_offset = conf.num_cpus - conf.nr_ctrl
        for i in range(conf.num_resources):
            resource_locality[i] = ctrl_offset + (i % conf.nr_ctrl)

    for task_count in xrange(conf.num_cpus, conf.num_cpus * 10 + 1):
        print("[%d @ %s] n:\t%s" % (os.getpid(), datetime.now(), task_count))

        utils = []
        samples = [[] for _ in tests]

        for sample in xrange(conf.samples):
            taskset = conf.make_taskset(task_count)
            utils.append(taskset.utilization())

            for i in xrange(len(tests)):
                if all_zeros_count[i] > 2:
                    samples[i].append(False)
                else:
                    samples[i].append(tests[i](taskset, cluster_sizes, resource_locality))
        print '=> min-u:%.2f avg-u:%.2f max-u:%.2f' % (min(utils), sum(utils) / len(utils), max(utils))
        row = []
        for i in xrange(len(tests)):
            avg = mean(samples[i])
            if avg > 0:
                all_zeros_count[i]  = 0
            else:
                all_zeros_count[i] += 1

            row.append(avg)

        yield [task_count] + ['%.2f' % x for x in row]
        if min(utils) >= conf.num_cpus:
            break


def run_exp(conf, tests, titles):
    header = ['UCAP']
    header += titles

    data = run_range(conf, tests)
    write_data(conf.output, data, header)

def run_lp_config(setup_tests, conf):
    check_config(conf)
    tests, titles = setup_tests(conf)
    conf.make_taskset = partial(gen_taskset, conf)
    run_exp(conf, tests, titles)

def run_task_count_exp(conf, tests, titles):
    header = ['TASKS']
    header += titles

    data = run_task_count(conf, tests)
    write_data(conf.output, data, header)

def run_task_count_config(setup_tests, conf):
    check_config(conf)
    tests, titles = setup_tests(conf)
    conf.make_taskset = partial(gen_taskset_with_count, conf)
    run_task_count_exp(conf, tests, titles)


def generate_lp_configs(what, options):
    utilizations_to_use = [
        'exp-medium',
        'uni-medium',
        'exp-light',
        'uni-low',
        ]
    periods_to_use = [
        'uni-10-100',
        'uni-100-200',
        'uni-10-1000',
    ]
    max_requests = [1, 3, 5]

    for cpus in [8, 16]:
        for nr_ctrl in [0]:
            if nr_ctrl <= cpus / 3:
                for nr in set([1, 8, 16, 24]): #1, 4
                    for cs in ['uni-10-50', 'uni-50-150']:
                        for mr in max_requests:
                            for pacc in [0.1, 0.2, 0.3]:
                                for util in utilizations_to_use:
                                    for per in periods_to_use:
                                        name = 'sd_exp=rtas13-%s_util=%s_per=%s_m=%02d_nr=%02d_mr=%02d_csl=%s_pacc=%.2f' \
                                                    % (what, util, per, cpus, nr, mr, cs, pacc)
                                        c = Config()
                                        c.samples       = 100
                                        c.num_cpus      = cpus
                                        c.utilizations  = util
                                        c.periods       = per
                                        c.num_resources = nr
                                        c.cslength      = cs
                                        c.output_file   = name + '.csv'
                                        c.min_ucap      = 1
                                        c.max_requests  = mr
                                        c.access_prob   = pacc
                                        c.nr_ctrl       = nr_ctrl
                                        yield (name, c)

EXPERIMENTS = {
    'rtas13/ucap/theory-tests'   : partial(run_lp_config, setup_theory_tests),
    'rtas13/ucap/with-oheads'    : partial(run_lp_config, setup_tests_with_overheads),

    'rtas13/tasks/theory-tests'  : partial(run_task_count_config, setup_theory_tests),
    'rtas13/tasks/with-oheads'   : partial(run_task_count_config, setup_tests_with_overheads),
    'rtas13/tasks/ctrl' : partial(run_task_count_config, setup_ctrl_tests),
}

CONFIG_GENERATORS = {
    'rtas13/ucap/theory-tests'   : partial(generate_lp_configs, "theory"),
    'rtas13/ucap/with-oheads'    : partial(generate_lp_configs, "oheads"),

    'rtas13/tasks/theory-tests'  : partial(generate_lp_configs, "theory"),
    'rtas13/tasks/with-oheads'   : partial(generate_lp_configs, "oheads"),
    'rtas13/tasks/ctrl' : partial(generate_lp_configs, "ctrl"),
}

def example():
    overheads = get_overhead_model()
    for m in [8, 16]:
        print ' %d cores:' % m
        for proto in overheads[m]:
            print '\t', overheads[m][proto], 'CPMD:', overheads[m][proto].cache_affinity_loss
    print '16 cores:'
    for proto in overheads[16]:
        print '\t', overheads[16][proto]
    for x in (2**k for k in xrange(12)):
        print x, cpmd[8](x), cpmd[16](x)


def run_file(fname):
    taskset = load_xml(fname)
    prep_taskset_for_locking_analysis(taskset)

    print taskset, "%x" % hash(str(taskset))

    num_cpus = 8
    overheads = None
    apply_lock = None

    # pre-compute resource assignment
    resource_locality = {}
    for i in range(32):
        resource_locality[i] = i % num_cpus

    cluster_sizes = {}
    for i in range(16):
        cluster_sizes[i] = 1

    result = pfp_test(dpcp_bounds, dist_jitter_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'DPCP', result

    result = pfp_test(partial(dpcp_bounds, use_rta=False), dist_jitter_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'DPCP-RTA', result

    result = pfp_test(partial(dpcp_bounds, use_linprog=False), dist_jitter_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'DPCP-OLD', result

    result = pfp_test(dflp_bounds, dist_jitter_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'DFLP', result

    result = pfp_test(mpcp_bounds, shmem_cost_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'MPCP', result

    result = pfp_test(fmlpp_bounds, shmem_cost_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'FMLP+', result

    print '=' * 10

    overheads = get_overhead_model(128)[num_cpus]
    locking_oheads = {
        'DPCP'  : partial(charge_dpcp_overheads, overheads['DPCP']),
        'MPCP'  : partial(charge_semaphore_overheads, overheads['MPCP'], True, True),
        'DFLP'  : partial(charge_dpcp_overheads, overheads['DFLP']),
        'FMLP+' : partial(charge_semaphore_overheads, overheads['FMLP+'], True, True),
        'EXCL'  : lambda _: True,
    }


    result = pfp_test(dpcp_bounds, dist_jitter_update, num_cpus, overheads['DPCP'], locking_oheads['DPCP'],
                      taskset, cluster_sizes, resource_locality)
    print 'DPCP', result

    result = pfp_test(partial(dpcp_bounds, use_rta=False), dist_jitter_update, num_cpus, overheads['DPCP'], locking_oheads['DPCP'],
                      taskset, cluster_sizes, resource_locality)
    print 'DPCP-RTA', result

    result = pfp_test(partial(dpcp_bounds, use_linprog=False), dist_jitter_update, num_cpus,  overheads['DPCP'], locking_oheads['DPCP'],
                      taskset, cluster_sizes, resource_locality)
    print 'DPCP-OLD', result

    result = pfp_test(dflp_bounds, dist_jitter_update, num_cpus,  overheads['DFLP'], locking_oheads['DFLP'],
                      taskset, cluster_sizes, resource_locality)
    print 'DFLP', result

    result = pfp_test(mpcp_bounds, shmem_cost_update, num_cpus,  overheads['MPCP'], locking_oheads['MPCP'],
                      taskset, cluster_sizes, resource_locality)
    print 'MPCP', result

    result = pfp_test(fmlpp_bounds, shmem_cost_update, num_cpus,  overheads['FMLP+'], locking_oheads['FMLP+'],
                      taskset, cluster_sizes, resource_locality)
    print 'FMLP+', result


def mpcp_compare(fname):
    taskset = load_xml(fname)
    prep_taskset_for_locking_analysis(taskset)

    num_cpus = 8
    overheads = None
    apply_lock = None

    # pre-compute resource assignment
    resource_locality = {}
    for i in range(32):
        resource_locality[i] = i % num_cpus

    cluster_sizes = {}
    for i in range(16):
        cluster_sizes[i] = 1

    result = pfp_test(mpcp_bounds, shmem_cost_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'MPCP/new', result

    result = pfp_test(partial(mpcp_bounds, use_linprog=False), shmem_cost_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'MPCP/old', result


def fmlp_compare(fname):
    taskset = load_xml(fname)
    prep_taskset_for_locking_analysis(taskset)

    num_cpus = 16
    overheads = None
    apply_lock = None

    # pre-compute resource assignment
    resource_locality = {}
    for i in range(32):
        resource_locality[i] = i % num_cpus

    cluster_sizes = {}
    for i in range(16):
        cluster_sizes[i] = 1

    result = pfp_test(fmlpp_bounds, shmem_cost_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'FMLP+/new', result

    result = pfp_test(partial(fmlpp_bounds, use_linprog=False), shmem_cost_update, num_cpus, overheads, apply_lock,
                      taskset, cluster_sizes, resource_locality)
    print 'FMLP+/old', result


if __name__ == '__main__':
    import sys
    run_file(sys.argv[1])
