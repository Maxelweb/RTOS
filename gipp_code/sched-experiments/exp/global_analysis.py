import os
import sys
import time

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
import schedcat.sched.fp as fp
import schedcat.sched.fp.bertogna as bert
import schedcat.sched.fp.guan as guan

from  schedcat.generator.tasks import uniform_int, log_uniform_int, uniform_choice, uniform

from schedcat.util.time import ms2us

import schedcat.model.serialize as ser

# directory and file name prefix for taskset files
current_taskset_dir    = "/tmp/"
current_taskset_prefix = "global_analysis_current_TS"


# ############### ANALYSIS ###############

def pip_bounds(num_cpus, taskset):
    locking.apply_pip_bounds(taskset, num_cpus)

def ppcp_bounds(num_cpus, taskset):
    locking.apply_ppcp_bounds(taskset, num_cpus)

def fmlp_bounds(num_cpus, taskset):
    locking.apply_sa_gfmlp_bounds(taskset, num_cpus)

def fmlpp_bounds(num_cpus, taskset):
    locking.apply_global_fmlpp_bounds(taskset, num_cpus)

def prsb_bounds(num_cpus, taskset):
    locking.apply_prsb_bounds(taskset, num_cpus)

def sob_fmlp_bounds(num_cpus, taskset):
    locking.apply_global_fmlp_sob_bounds(taskset)

def prior_fmlpp_bounds(num_cpus, taskset):
    locking.apply_generalized_fmlp_bounds(taskset, num_cpus, False)

def sob_omlp_bounds(num_cpus, taskset):
    locking.apply_global_omlp_bounds(taskset, num_cpus)

def classic_pip_bounds(num_cpus, taskset):
    locking.apply_classic_pip_bounds(taskset, num_cpus)

def classic_ppcp_bounds(num_cpus, taskset):
    locking.apply_classic_ppcp_bounds(taskset, num_cpus)

def no_blocking(num_cpus, taskset):
    for t in taskset:
        t.blocked = 0

def no_progress_fifo_bounds(num_cpus, taskset):
    locking.apply_no_progress_fifo_bounds(taskset, num_cpus)

def no_progress_priority_bounds(num_cpus, taskset):
    locking.apply_no_progress_priority_bounds(taskset, num_cpus)

def task_rta_schedulable_lp(i, taskset, num_cpus):
    task = taskset[i]
    return task.response_time <= task.deadline

def gfp_schedulable(num_cpus, rta, taskset):
    for i in xrange(len(taskset)):
        if not rta(i, taskset, num_cpus):
            return False
    return True

def gfp_test(num_cpus, apply_bounds, rta, taskset):
    # Assumes taskset has already been sorted and ID'd in priority order.
    locking.assign_fp_preemption_levels(taskset)

    test_start = time.time();

    # Setup the initial value for the iteration
    for t in taskset:
        t.response_time = t.cost
        t.blocked = 0
        # Backup uninflated WCET for s-oblivious analysis
        t.uninflated_cost = t.cost

    # Keep track of the observed response times.
    response_times = [t.response_time for t in taskset]

    # Try deadlines before starting the more expensive iteration
    for t in taskset:
        t.response_time = t.deadline
        if t.cost > t.response_time:
            return False
    apply_bounds(num_cpus, taskset) # modifies task parameters

    if gfp_schedulable(num_cpus, rta, taskset):
    # Great, each task's maximum response-time is bounded by its deadline.
    # With constrained deadlines, this is sufficient.
        return True

    # Schedulable without requests, but not when upper-bounding response-times
    # by implicit deadlines. We'll have to look more carefully.
    response_times_consistent = False
    schedulable               = True

    # To start with, restore the response-time bounds as observed before.
    for (resp, tsk) in izip(response_times, taskset):
        tsk.response_time = resp

    # iterate until the assumed response times match the
    # actual response times
    iterations = 0
    TS_written = False
    while not response_times_consistent:
        iterations = iterations + 1
        # backup and restore
        for t in taskset:
        	   # backup the response time
            t.assumed_response_time = t.response_time
            # restore total blocking and higher-priority blocking workload
            # restore uninflated WCET (only used under s-oblivious analysis)
            t.cost = t.uninflated_cost

        apply_bounds(num_cpus, taskset) # modifies exec. cost

        # Let's check if the task set is now schedulable and the observed
        # response time bounds did not invalidate our assumption.
        response_times_consistent = True
        if not gfp_schedulable(num_cpus, rta, taskset):
            # Nope. Since response times are monotonically increasing
            # under this iterative approach, we can give up right away.
            return False

        for t in taskset:
            # Check if anything changed.
            if t.response_time > t.assumed_response_time:
                # Need another iteration to find fixpoint.
                response_times_consistent = False
            if not t.response_time >= t.assumed_response_time:
                TS_f_name = "%s/%s_pid-%d_time-%d.ts" % (current_taskset_dir, current_taskset_prefix, os.getpid(), time.time())
                print "[gfp_test] Response times not monotonic! PID=%d" % os.getpid()
                print "[gfp_test] bounds:\t%s" % apply_bounds
                print "[gfp_test] rta:\t%s" % rta
                print "[gfp_test] taskset written to %s" % TS_f_name
                ser.write(taskset, TS_f_name)
                assert(False)

        if iterations >= 100 and not TS_written:
            TS_written = True
            current_time = time.time()
            TS_f_name = "%s/%s_pid-%d_time-%d.ts" % (current_taskset_dir, current_taskset_prefix, os.getpid(), current_time)
            print "[gfp_test] Processing current taskset for %d iterations, writing out to file %s ." % (iterations, TS_f_name)
            ser.write(taskset, TS_f_name)

    test_end = time.time()
    print "[gfp_test] Processing current taskset finished after %d iterations, PID=%d" % (iterations, os.getpid())
    return True

def gfp_multi_prio_order_test(num_cpus, apply_bounds, rta, taskset):
    original_taskset = taskset

    # First, try DkC priorities (already assigned)
    taskset = original_taskset.copy()
    if gfp_test(num_cpus, apply_bounds, rta, taskset):
        return True

    # Try RM-US(x) priorities
    taskset = original_taskset.copy()
    taskset.sort_by_RM_US(num_cpus)
    taskset.assign_ids()
    if gfp_test(num_cpus, apply_bounds, rta, taskset):
        return True

    # Try DM-US(x) priorities
    taskset = original_taskset.copy()
    taskset.sort_by_DM_US(num_cpus)
    taskset.assign_ids()
    if gfp_test(num_cpus, apply_bounds, rta, taskset):
        return True

    # Try deadline-monotonic priorities
    taskset = original_taskset.copy()
    taskset.sort_by_deadline()
    taskset.assign_ids()
    if gfp_test(num_cpus, apply_bounds, rta, taskset):
        return True

    return False

# ############### SCHEDULABILITY TESTS ###############

def setup_tests(conf):
    the_test = gfp_multi_prio_order_test
    return [
        ("no blocking",
         partial(the_test, conf.num_cpus, no_blocking, guan.rta_schedulable_guan)),

        ("PIP",
         partial(the_test, conf.num_cpus, pip_bounds, task_rta_schedulable_lp)),

        ("PPCP",
         partial(the_test, conf.num_cpus, ppcp_bounds, task_rta_schedulable_lp)),

        ("FMLP",
         partial(the_test, conf.num_cpus, fmlp_bounds, task_rta_schedulable_lp)),

        ("FMLP+",
         partial(the_test, conf.num_cpus, fmlpp_bounds, task_rta_schedulable_lp)),

        ("PRSB",
         partial(the_test, conf.num_cpus, prsb_bounds, task_rta_schedulable_lp)),

        ("Classic PIP",
         partial(the_test, conf.num_cpus, classic_pip_bounds, partial(bert.rta_schedulable, dont_use_slack=True))),

        ("Classic PPCP",
         partial(the_test, conf.num_cpus, classic_ppcp_bounds, partial(bert.rta_schedulable, dont_use_slack=True))),

        ("FMLP-sob",
         partial(the_test, conf.num_cpus, sob_fmlp_bounds, guan.rta_schedulable_guan)),

        ("OMLP",
         partial(the_test, conf.num_cpus, sob_omlp_bounds, guan.rta_schedulable_guan)),

        ("fifo only",
         partial(the_test, conf.num_cpus, no_progress_fifo_bounds, task_rta_schedulable_lp)),

        ("priority only",
         partial(the_test, conf.num_cpus, no_progress_priority_bounds, task_rta_schedulable_lp)),
    ]

# ############### EXPERIMENTS ###############

def run_tcount(conf, tests):
    count = 0

    for n in range(conf.tasks_min, conf.tasks_max + 1, conf.step_size):
        print("[%d @ %s] n:\t%s" % (os.getpid(), datetime.now(), n))
        print("tasks_max=%s", conf.tasks_max)

        utils = []
        samples = [[] for _ in tests]

        for sample in xrange(conf.samples):
            taskset = conf.make_taskset(n)

            # flush out taskset to disk for potential post-mortem analysis
            ser.write(taskset, "%s/%s_pid-%d.ts" % (current_taskset_dir, current_taskset_prefix, os.getpid()))

            print '.', conf.samples
            utils.append(taskset.utilization())

            for i in xrange(len(tests)):
                # This assumes that the no-blocking test is the first in tests.
                if i == 0: # This is the no-blocking test. We need to run this in any case.
                    samples[i].append(tests[i](taskset))
                else:
                    # This is one of the tests considering potential blocking.
                    # Did the no-blocking test succeed?
                    if samples[0][-1]:
                        # Yes, this taskset is schedulable without blocking, so let's run our
                        # analysis to see whether it's schedulable with blocking.
                        samples[i].append(tests[i](taskset))
                    else:
                        # No, the taskset is not even schedulable without blocking, so it cannot
                        # possibly be schedulable with blocking.
                         samples[i].append(False)

        print '=> min-u:%.2f avg-u:%.2f max-u:%.2f' % (min(utils), sum(utils) / len(utils), max(utils))
        row = []
        for i in xrange(len(tests)):
            avg = mean(samples[i])
            row.append(avg)

        yield [n] + ['%.2f' % x for x in row]



# ############### TASK SET GENERATION ###############

def draw_cs_count(conf):
    if random.random() <= conf.access_prob:
        # access resource
        return random.randint(1, conf.max_requests)
    else:
        return 0

CSLENGTH  = {
    'short'   : lambda: random.randint(1,   25),
    'medium'  : lambda: random.randint(25,  100),
    'long'    : lambda: random.randint(100, 500),
    'extreme' : lambda: random.randint(100, 1000),
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
    ts.sort_by_dkc(conf.num_cpus)
    ts.assign_ids()
    initialize_resource_model(ts)
    locking.assign_fp_preemption_levels(ts)

    for t in ts:
        generate_requests(conf, t)
        t.partition = 1
        cum_cs = sum([t.resmodel[res].max_length * t.resmodel[res].max_requests for res in t.resmodel])
        t.cost = max(t.cost, cum_cs)

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

UTILIZATIONS = {
}

DEADLINES = {
}

# pull in standard periods
PERIODS.update(gen.NAMED_PERIODS)
UTILIZATIONS.update(gen.NAMED_UTILIZATIONS);
DEADLINES.update(gen.NAMED_DEADLINES)

# ############### CONFIG LOADING ###############

DEFAULT_SAMPLES = 1

def check_std_params(conf):
    # standard parameters
    conf.check('num_cpus',     type=int, min=1)
    conf.check('samples',      type=int, default=DEFAULT_SAMPLES, min=1)
    conf.check('min_ucap',     type=float, min=1, max=conf.num_cpus, default=conf.num_cpus/4)
    conf.check('max_ucap',     type=float, min=1, max=conf.num_cpus, default=conf.num_cpus)
    conf.check('deadlines',    type=one_of(DEADLINES), default='implicit')
    conf.check('utilizations', type=one_of(UTILIZATIONS), default='exp-light')
    conf.check('periods',      type=one_of(PERIODS),   default='uni-1-1000')

    conf.check('step_size', type=int, min=1, default=conf.num_cpus // 2)
    conf.check('tasks_min', type=int, min=1, default=conf.num_cpus)
    conf.check('tasks_max', type=int, min=1, default=conf.num_cpus * 10)


def check_tcount_config(conf):
    check_std_params(conf)

    conf.check('access_prob',    type=float, min=0)
    conf.check('num_resources',  type=int,   min=1)
    conf.check('max_requests',   type=int,   min=1, default=5)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')

def run_tcount_config(ignor, conf):
    check_tcount_config(conf)

    conf.generate = gen.mkgen(UTILIZATIONS[conf.utilizations],
                              PERIODS[conf.periods],
                              gen.NAMED_DEADLINES[conf.deadlines])

    conf.make_taskset = partial(generate_task_set, conf)
    (titles, tests) = zip(*setup_tests(conf))

    header = ['TASKS']
    header += titles

    data = run_tcount(conf, tests)
    write_data(conf.output, data, header)

# ############### CONFIG GENERATION ###############

CPU_COUNTS = [4, 8]

UTILS = [
    'exp-light',
]

def generate_tcount_configs(fp, options):
    for cpus in CPU_COUNTS:
        for periods in ['log-uni-10-100', 'log-uni-1-1000']:
            for util in UTILS:
                for nr in [cpus/4, cpus/2, cpus, 2*cpus]:
               	    for pacc in [0.1, 0.25, 0.5]:
                        for cs in ['short', 'medium', 'long']:
                            for nreq in [1, 3, 5, 7, 10]:
                               name = 'sd_exp=global-analysis_mutex_m=%02d_util=%s_per=%s_nr=%d_pacc=%d_cs=%s_nreq=%d' \
                                      % (cpus, util, periods, nr, pacc * 100, cs, nreq)

                               c = Config()
                               c.num_cpus = cpus

                               c.samples = 100
                               c.tasks_min = 10
                               c.utilizations = util
                               c.periods = periods
                               c.max_requests = nreq
                               c.access_prob = pacc
                               c.num_resources = nr
                               c.cslength = cs

                               c.output_file  = name + '.csv'
                               yield (name, c)


EXPERIMENTS = {
     'global_analysis'  : partial(run_tcount_config, True),
}

CONFIG_GENERATORS = {
     'global_analysis'  : partial(generate_tcount_configs, True),
}
