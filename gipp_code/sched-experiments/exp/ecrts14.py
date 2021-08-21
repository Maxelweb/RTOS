import os
import sys

import random
from itertools import izip
from functools import partial
from datetime import datetime

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.model.resources import initialize_resource_model, ResourceRequirement
import schedcat.generator.tasksets as gen

import schedcat.locking.bounds as locking
import schedcat.sched.edf as edf
import schedcat.sched.fp as fp
import schedcat.mapping.binpack as bp

from schedcat.sched import get_native_taskset
from schedcat.sched.native import LAGedf

from  schedcat.generator.tasks import uniform_int, log_uniform_int, uniform_choice, uniform

from schedcat.util.time import ms2us


# ############### ANALYSIS ###############

def comlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_clustered_omlp_bounds(taskset, cluster_size)

def omip_bounds(num_cpus, cluster_size, taskset):
    locking.apply_omip_bounds(taskset, num_cpus, cluster_size)

def edf_fmlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_generalized_fmlp_bounds(taskset, cluster_size, True)

def fp_fmlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_generalized_fmlp_bounds(taskset, cluster_size, False)

def sob_fmlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_generalized_fmlp_bounds(taskset, cluster_size, True)
    for t in taskset:
        t.cost += t.suspended
        t.suspended = 0

def pretend_no_blocking(num_cpus, cluster_size, taskset):
    for t in taskset:
        t.suspended = 0

def check_sob(cluster_size, part):
    sob = part.copy()
    for t in sob:
        t.cost += t.suspended
        t.suspended = 0
    if edf.is_schedulable(cluster_size, sob, rta_min_step=1000):
        print "Failed LA test:"
        for i, t in enumerate(part):
            print "T%d = (%5d, %5d, %5d)" % (i + 1, t.cost, t.period, t.suspended)
        print "S-Oblivious task set found schedulable:"
        for i, t in enumerate(sob):
            print "T%d = (%5d, %5d, %5d)" % (i + 1, t.cost, t.period, t.suspended)

    la = LAGedf(cluster_size)
    npart = get_native_taskset(part, with_suspensions=False)
    if la.is_schedulable(npart):
        print "LA test claims original task set schedulable if suspensions are omitted."

    npart = get_native_taskset(sob, with_suspensions=False)
    if la.is_schedulable(npart):
        print "LA test claims s-obliviously inflated task set (w/o suspensions) schedulable."


def edf_test(num_cpus, cluster_size, apply_bounds, suspension_aware, taskset):
    num_clusters = num_cpus // cluster_size
    original_taskset = taskset
    taskset = taskset.copy()

    locking.assign_edf_preemption_levels(taskset)
    for t in taskset:
        t.response_time = t.deadline

    try:
        partitions = bp.worst_fit_decreasing(taskset, num_clusters,
                               capacity=cluster_size,
                               weight=lambda t: t.utilization(),
                               misfit=bp.report_failure,
                               empty_bin=TaskSystem)

        for cpu, part in enumerate(partitions):
            for t in part:
                t.partition = cpu

        apply_bounds(num_cpus, cluster_size, taskset) # modifies task parameters

        if suspension_aware:
            la = LAGedf(cluster_size)
            for part in partitions:
                npart = get_native_taskset(part, with_suspensions=True)
                if not la.is_schedulable(npart):
#                    check_sob(cluster_size, part)
                    return False
        else:
            for part in partitions:
                if not edf.is_schedulable(cluster_size, part, rta_min_step=1000):
                    return False
        return True

    except bp.DidNotFit:
        # fell through; partitioning failed
        return False


def global_edf_test(num_cpus, suspension_aware, taskset):
    taskset = taskset.copy()

    if suspension_aware:
        la = LAGedf(num_cpus)
        nts = get_native_taskset(taskset, with_suspensions=True)
        return la.is_schedulable(nts)
    else:
        for t in taskset:
            t.cost += t.suspended
            t.suspended = 0
        return edf.is_schedulable(num_cpus, taskset, rta_min_step=1000)

def pfp_test(num_cpus, apply_bounds, taskset):
    # Assumes taskset has already been sorted and ID'd in priority order.
    original_taskset = taskset
    taskset = original_taskset.copy()

    cluster_size = 1 # partitioned scheduling

    locking.assign_fp_preemption_levels(taskset)

    def backup_params():
        for t in taskset:
            t.uninflated_cost = t.cost

    def restore_params():
        for t in taskset:
            t.cost = t.uninflated_cost
            t.blocked = 0
            t.suspended = 0
            t.locally_blocked = 0

    try:
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
        backup_params()
        apply_bounds(num_cpus, cluster_size, taskset) # modifies task parameters
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

            restore_params()
            apply_bounds(num_cpus, cluster_size, taskset) # modifies exec. cost

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
        return False

# ############### SCHEDULABILITY TESTS ###############

def setup_tests(conf):
    return [
        ("C-EDF (no blocking, all tests)",
         partial(edf_test, conf.num_cpus, conf.cluster_size,
                 pretend_no_blocking, False)),

        ("C-EDF (no blocking, LA test)",
         partial(edf_test, conf.num_cpus, conf.cluster_size,
                 pretend_no_blocking, True)),

        ("C-EDF (FMLP+, LA test)",
         partial(edf_test, conf.num_cpus, conf.cluster_size,
                 edf_fmlp_bounds, True)),

        ("C-EDF (FMLP+, SOB +  all tests)",
         partial(edf_test, conf.num_cpus, conf.cluster_size,
                 sob_fmlp_bounds, False)),

        ("C-EDF (OMLP, all tests)",
         partial(edf_test, conf.num_cpus, conf.cluster_size,
                 comlp_bounds, False)),

        ("C-EDF (OMIP, all tests)",
         partial(edf_test, conf.num_cpus, conf.cluster_size,
                 omip_bounds, False)),

    ]

def setup_suspension_tests(conf):
    return [
        ("G-EDF, s-oblivious",
         partial(global_edf_test, conf.num_cpus, False)),
        ("G-EDF, s-aware",
         partial(global_edf_test, conf.num_cpus, True)),
    ]

def setup_pfp_tests(conf):
    return [
        ("P-FP (no blocking)",
         partial(pfp_test, conf.num_cpus, pretend_no_blocking)),

        ("P-FP (FMLP+)",
         partial(pfp_test, conf.num_cpus, fp_fmlp_bounds)),

        ("P-FP (OMLP)",
         partial(pfp_test, conf.num_cpus, comlp_bounds)),

        ("P-FP (OMIP)",
         partial(pfp_test, conf.num_cpus, omip_bounds)),
    ]

# ############### EXPERIMENTS ###############

def run_tcount(conf, tests):
    all_zeros_count = [0 for _ in tests]
    count = 0

    for n in range(conf.tasks_min, conf.tasks_max + 1, conf.step_size):
        print("[%d @ %s] n:\t%s" % (os.getpid(), datetime.now(), n))

        utils = []
        samples = [[] for _ in tests]

        for sample in xrange(conf.samples):
            taskset = conf.make_taskset(n)
            print '.', conf.samples
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

# ############### TASK SET GENERATION ###############


def draw_cs_count(conf):
    if random.random() <= conf.access_prob:
        # access resource
        return random.randint(1, conf.max_requests)
    else:
        return 0

CSLENGTH  = {
    'short'   : lambda: random.randint(1,   15),
    'medium'  : lambda: random.randint(1,  100),
    'long'    : lambda: random.randint(5, 1280),
    'extreme' : lambda: random.randint(1000, 2000),
}

def generate_requests(conf, task):
    length = CSLENGTH[conf.cslength]
    for res_id in xrange(conf.num_resources):
        num_accesses = draw_cs_count(conf)
        if num_accesses:
            # add writes
            task.resmodel[res_id] = ResourceRequirement(res_id, num_accesses, length(), 0, 0)


def generate_task_set(conf, n):
    ts = conf.generate(max_tasks=n, time_conversion=ms2us)
    initialize_resource_model(ts)
    for t in ts:
        generate_requests(conf, t)
    return ts

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


SUSPENSIONS = {
    'short'    : 0.5,
    'moderate' : 1.0,
    'long'     : 1.5,
}

def generate_la_task_set(conf, n):
    ts = conf.generate(max_tasks=n, time_conversion=ms2us)

    sfactor = SUSPENSIONS[conf.suspension_length]
    for t in ts:
        t.suspended = int(t.cost * sfactor)

    return ts

UTILIZATIONS = {
    'uni-cong' : uniform(0.01, 0.3)
}

DEADLINES = {
}

# pull in standard periods
PERIODS.update(gen.NAMED_PERIODS)
UTILIZATIONS.update(gen.NAMED_UTILIZATIONS);
DEADLINES.update(gen.NAMED_DEADLINES)

# ############### CONFIG LOADING ###############

DEFAULT_SAMPLES = 10

def check_std_params(conf):
    # standard parameters
    conf.check('num_cpus',     type=int, min=1)
    conf.check('samples',      type=int, default=DEFAULT_SAMPLES, min=10)

    conf.check('deadlines',    type=one_of(DEADLINES), default='implicit')
    conf.check('utilizations', type=one_of(UTILIZATIONS), default='exp-light')
    conf.check('periods',      type=one_of(PERIODS),   default='uni-1-1000')

    conf.check('step_size', type=int, min=1, default=conf.num_cpus // 4)
    conf.check('tasks_min', type=int, min=1, default=conf.num_cpus)
    conf.check('tasks_max', type=int, min=1, default=conf.num_cpus * 10)


def check_tcount_config(conf):
    check_std_params(conf)
    conf.check('cluster_size',   type=int, default=1, min=1, max=conf.num_cpus)
    assert conf.num_cpus % conf.cluster_size == 0

    conf.check('access_prob',    type=float, min=0)
    conf.check('num_resources',  type=int,   min=1)
    conf.check('max_requests',   type=int,   min=1, default=5)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')

def run_tcount_config(edf, conf):
    check_tcount_config(conf)

    conf.generate = gen.mkgen(UTILIZATIONS[conf.utilizations],
                              PERIODS[conf.periods],
                              gen.NAMED_DEADLINES[conf.deadlines])

    conf.make_taskset = partial(generate_task_set, conf)

    if edf:
        (titles, tests) = zip(*setup_tests(conf))
    else:
        (titles, tests) = zip(*setup_pfp_tests(conf))

    header = ['TASKS']
    header += titles

    data = run_tcount(conf, tests)
    write_data(conf.output, data, header)

def check_suspend_config(conf):
    check_std_params(conf)
    conf.check('suspension_length', type=one_of(SUSPENSIONS), default='short')

def run_suspend_config(conf):
    check_suspend_config(conf)

    conf.generate = gen.mkgen(UTILIZATIONS[conf.utilizations],
                              PERIODS[conf.periods],
                              gen.NAMED_DEADLINES[conf.deadlines])

    conf.make_taskset = partial(generate_la_task_set, conf)

    (titles, tests) = zip(*setup_suspension_tests(conf))

    header = ['TASKS']
    header += titles

    data = run_tcount(conf, tests)
    write_data(conf.output, data, header)


# ############### CONFIG GENERATION ###############


#CPU_COUNTS = [8, 16, 24, 32, 48, 64]
CPU_COUNTS = [4, 8, 16, 32]

CLUSTER_SIZES = [4, 2]

UTILS = [
    'exp-light',
    'uni-light',
    'exp-medium',
    'uni-medium',
#     'exp-heavy',
#     'uni-heavy',
]

def generate_tcount_configs(edf, options):
    for cpus in CPU_COUNTS:
        for periods in ['log-uni-10-100', 'uni-10-100']:
            for util in UTILS:
                for cluster in CLUSTER_SIZES if edf else [1]:
                    for nr in [cpus // 4, cpus // 2, cpus, 2 * cpus]:
                        for pacc in [0.1, 0.25, 0.4]:
                            for cs in ['short', 'medium', 'long']:
                                name = 'sd_exp=ecrts14-%s_m=%02d_c=%d_util=%s_per=%s_nr=%d_pacc=%d_cs=%s' \
                                    % ('cedf' if edf else 'pfp', cpus, cluster, util, periods, nr, pacc * 100, cs)

                                c = Config()
                                c.num_cpus = cpus
                                c.cluster_size = cluster

                                c.samples = 100

                                c.utilizations = util
                                c.periods = periods

                                c.access_prob = pacc
                                c.num_resources = nr
                                c.cslength = cs

                                c.output_file  = name + '.csv'
                                yield (name, c)




EXPERIMENTS = {
     'ecrts14/tcount'  : partial(run_tcount_config, True),
     'ecrts14/pfp'     : partial(run_tcount_config, False),
     'ecrts14/suspend' : run_suspend_config,
}

CONFIG_GENERATORS = {
     'ecrts14/tcount'  : partial(generate_tcount_configs, True),
     'ecrts14/pfp'     : partial(generate_tcount_configs, False),
}

def example1():
# Failed LA test:
# T1 = ( 4484, 12000,  1852)
# T2 = ( 6776, 32000,  1120)
# T3 = (10246, 83000,  1448)
# S-Oblivious task set found schedulable:
# T1 = ( 6336, 12000,     0)
# T2 = ( 7896, 32000,     0)
# T3 = (11694, 83000,     0)
# LA test claims original task set schedulable if suspensions are omitted.
# LA test claims s-obliviously inflated task set (w/o suspensions) schedulable.

    ts = TaskSystem([
            SporadicTask(4484, 12000),
            SporadicTask(6776, 32000),
            SporadicTask(10246, 83000),
        ])

    ts[0].suspended = 1852
    ts[1].suspended = 1120
    ts[2].suspended = 1448

#     ts[0].suspended = 1800
#     ts[1].suspended = 1100
#     ts[2].suspended = 1400

    la = LAGedf(2)

    npart = get_native_taskset(ts, with_suspensions=False)
    print "LA, w/o suspensions: ", la.is_schedulable(npart)

    npart = get_native_taskset(ts, with_suspensions=True)
    print "LA, w/  suspensions: ", la.is_schedulable(npart)

    for t in ts:
        t.cost += t.suspended
        t.suspended = 0

    npart = get_native_taskset(ts, with_suspensions=False)
    print "LA, s-obvlious inf.: ", la.is_schedulable(npart)


if __name__ == '__main__':
    example1()
