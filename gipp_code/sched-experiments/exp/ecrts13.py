from __future__ import division

import random
from functools import partial

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean


import schedcat.generator.tasksets as gen
import schedcat.generator.generator_emstada as emstada
import schedcat.mapping.binpack as bp
import schedcat.sched.edf as edf
import schedcat.sched.fp as fp

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.model.resources import initialize_resource_model, ResourceRequirement, ResourceRequirements
import schedcat.locking.bounds as locking
from schedcat.locking.partition import find_independent_tasksubsets
from schedcat.util.time import ms2us

MAX_RESOURCES = 32

DEFAULT_SAMPLES = 200

# reference to global option object
options = None

def comlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_clustered_omlp_bounds(taskset, cluster_size)

def omip_bounds(num_cpus, cluster_size, taskset):
    locking.apply_omip_bounds(taskset, num_cpus, cluster_size)

def pretend_no_blocking(num_cpus, cluster_size, taskset):
    pass

# used when m = c
def no_partitioning_required(taskset, *args, **kargs):
    return [taskset]

def heuristics(num_cpus, cluster_size):
    if num_cpus > cluster_size:
        return MAPPINGS
    else:
        return [no_partitioning_required]

def edf_test(num_cpus, cluster_size, apply_bounds, taskset):
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

        for part in partitions:
            if not edf.is_schedulable(cluster_size, part, rta_min_step=1000):
                return False
        return True

    except bp.DidNotFit:
        # fell through; partitioning failed
        return False

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
}

def generate_requests(conf, task):
    length = CSLENGTH[conf.cslength]
    for idx in xrange(conf.num_resources):
        res_id = idx
        num_accesses = draw_cs_count(conf)
        if num_accesses:
            # add writes
            task.resmodel[res_id] = ResourceRequirement(res_id, num_accesses, length(), 0, 0)

def ensure_min_task_count_by_splitting(num_cpus, taskset, periods='uni-moderate'):
    # Make sure scheduling problem isn't entirely trivial: have more
    # tasks than processors.
    while len(taskset) < num_cpus + 1:
        t = random.choice(taskset)
        old_c = t.cost
        t.cost = t.cost // 2
        u = t.utilization()
        p = ms2us(gen.NAMED_PERIODS[periods]())
        c = max(int(p * u), 1)
        taskset.append(SporadicTask(c, p))
    return taskset

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

def ensure_plausible_request_lengths(taskset):
    for t in taskset:
        total = sum([t.resmodel[r].max_length * t.resmodel[r].max_requests
                     for r in t.resmodel])
        if total > t.cost:
            # need to scale down
            frac =  t.cost / total
            for r in t.resmodel:
                req = t.resmodel[r]
                if req.max_writes:
                    req.max_write_length = max(1, int(req.max_write_length * frac))
                if req.max_reads:
                    req.max_read_length = max(1, int(req.max_read_length * frac))
    return taskset

def setup_mutex_tests(conf):
    conf.check('cluster_size',   type=int, default=1, min=1, max=1)
    conf.check('access_prob',    type=float, min=0)
    conf.check('num_resources',  type=int,   min=1, max=MAX_RESOURCES)
    conf.check('max_requests',   type=int,   min=1, default=5)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')

    tests = [
        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds),
        partial(edf_test, conf.num_cpus, conf.cluster_size, omip_bounds),
    ]
    titles = ['C-OMLP/EDF', 'OMIP/EDF']
    return tests, titles

def gen_taskset(conf, ucap):
    ts = conf.generate(max_util=ucap, time_conversion=ms2us, squeeze=True)
    ts = ensure_min_task_count(conf.num_cpus, ts)
    initialize_resource_model(ts)

    # generate resource requirements
    for t in ts:
        generate_requests(conf, t)

#    ts[0].deadline = ts[0].cost + 100

    ts = ensure_plausible_request_lengths(ts)



    return ts

def gen_mutex_taskset(conf, ucap):
    return gen_taskset(conf, ucap)

def run_range(conf, tests):
    all_zeros_count = [0 for _ in tests]
    for ucap in value_range(conf.min_ucap, conf.num_cpus, conf.step_size):

        samples = [[] for _ in tests]
        for _ in xrange(conf.samples):
            taskset = conf.make_taskset(ucap)
            for i in xrange(len(tests)):
                if all_zeros_count[i] > 2:
                    samples[i].append(False)
                else:
                    samples[i].append(tests[i](taskset))

        row = [mean(x) for x in samples]
        yield [ucap] + row

def check_config(conf):
    # standard parameters
    conf.check('num_cpus',     type=int, min=1)
    conf.check('samples',      type=int, default=DEFAULT_SAMPLES, min=10)
    conf.check('step_size',    type=float, default=0.25, min=0.1)
    conf.check('utilizations', type=one_of(gen.NAMED_UTILIZATIONS))
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(gen.NAMED_PERIODS),   default='uni-moderate')
    conf.check('min_ucap',     type=float, min=1, max=conf.num_cpus, default=conf.num_cpus/4)

    conf.generate = gen.DIST_BY_KEY[conf.deadlines][conf.periods][conf.utilizations]

def run_exp(conf, tests, titles):
    header = ['UCAP'] + titles
    data = run_range(conf, tests)
    write_data(conf.output, data, header)

def run_mutex_config(conf):
    check_config(conf)
    tests, titles = setup_mutex_tests(conf)
    conf.make_taskset = partial(gen_mutex_taskset, conf)
    run_exp(conf, tests, titles)

def generate_mutex_configs(options):
    for cpus in [4, 8, 16]:
        cluster_size = 1
        for nr in [cpus // 4, cpus // 2, cpus, 2 * cpus]:
            for pacc in [0.1, 0.25, 0.4]:
                for cs in CSLENGTH.keys():
                    for util in gen.NAMED_UTILIZATIONS:
                        name = 'sd_exp=omlp-mutex_util=%s_m=%02d_nr=%d_pacc=%.2f_csl=%s' \
                                    % (util, cpus, nr, pacc, cs)
                        c = Config()
                        c.num_cpus     = cpus
                        c.utilizations = util
                        c.access_prob  = pacc
                        c.num_resources = nr
                        c.cslength      = cs
                        c.output_file  = name + '.csv'
                        yield (name, c)

##  ###########################################

def generate_emstada_taskset(
        num_tasks,
        num_latency_sensitive,
        util,
        num_cs = 2,
        num_resources = 12,
        num_ls_res = 3,
        csl_range=(1, 100)):

    n = num_tasks - num_latency_sensitive
    n_ls = num_latency_sensitive
    periods    = (10, 1000)
    periods_ls = (0.5, 2.5)

#     csl_ls = lambda t: min(t.cost // num_cs, random.randint(1, 15))
#     csl    = lambda t: min(t.cost // num_cs, random.randint(csl_range[0], csl_range[1]))
    csl_ls = lambda t: random.randint(1, 15)
    csl    = lambda t: random.randint(csl_range[0], csl_range[1])

    lsu = min(n_ls * 0.5, util * 0.5)

    if n_ls:
        ts_ls = emstada.gen_taskset(periods_ls, 'logunif', n_ls, lsu)
    else:
        ts_ls = []

    ts = emstada.gen_taskset(periods, 'logunif', n - n_ls, util - lsu, period_granularity=0.5)

    # generate resource accesses

    initialize_resource_model(ts_ls)
    initialize_resource_model(ts)

    for t in ts_ls:
        for q in xrange(num_ls_res):
            t.resmodel[q].add_request(csl_ls(t))

    for t in ts:
        for _ in xrange(num_cs):
            q = random.randint(num_ls_res + 1, num_ls_res + num_resources)
            cs_length = csl(t)
            t.resmodel[q].add_request(cs_length)
            t.cost += cs_length

    ts.extend(ts_ls)


    return ts

def run_csl_range(conf, tests):
    all_zeros_count = [0 for _ in tests]
    for csl in xrange(conf.step_size, conf.max_csl, conf.step_size):
        samples = [[] for _ in tests]
        for _ in xrange(conf.samples):
            taskset = conf.generate(csl)
            for i in xrange(len(tests)):
                if all_zeros_count[i] > 2:
                    samples[i].append(False)
                else:
                    samples[i].append(tests[i](taskset))

        row = [mean(x) for x in samples]
        for i in xrange(len(tests)):
            if row[i] > 0:
                all_zeros_count[i]  = 0
            else:
                all_zeros_count[i] += 1

        yield [csl] + row

def run_csl_config(conf):
    # standard parameters
    conf.check('num_cpus',     type=int, min=1)
    conf.check('cluster_size',   type=int, default=1, min=1, max=conf.num_cpus)

    conf.check('load',         type=float, default=0.75 * conf.num_cpus)
    conf.check('num_tasks',    type=int, min=1, default=conf.num_cpus * 5)
    conf.check('num_latency',  type=int, min=0, max=conf.num_tasks - 1)

    conf.check('num_resources', type=int, min=1, default=int(conf.num_cpus * 1.5))
    conf.check('num_cs',       type=int, min=1, default=2)
    conf.check('max_csl',      type=int, min=1, default=1000)

    conf.check('samples',      type=int, default=DEFAULT_SAMPLES)
    conf.check('step_size',    type=int, default=5, min=1)

    conf.generate = lambda csl: generate_emstada_taskset(
                        conf.num_tasks,
                        conf.num_latency,
                        conf.load,
                        num_cs = conf.num_cs,
                        num_resources= conf.num_resources,
                        csl_range = (1, csl))

    tests = [
        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds),
        partial(edf_test, conf.num_cpus, conf.cluster_size, omip_bounds),
    ]
    titles = ['MAX-CSL',
        'C-OMLP/EDF',
        'OMIP/EDF'
    ]

    data = run_csl_range(conf, tests)
    write_data(conf.output, data, titles)


def generate_csl_configs(options):
    for cpus in [2, 4, 8, 16]:
        cluster_size = 1
        num_res = 12
        for n in [20, 30, 40]:
            for l in set(range(min(8, cpus)) + [cpus]):
                for load in [0.4, 0.5, 0.7]:
                    for num_cs in [1, 2, 3]:
                        name = 'sd_exp=omip-csl_m=%02d_n=%02d_l=%d_load=%.2f_ncs=%d' \
                                    % (cpus, n, l, load, num_cs)
                        c = Config()
                        c.num_cpus      = cpus
                        c.cluster_size  = cluster_size
                        c.num_resources = num_res
                        c.num_tasks     = n
                        c.num_latency   = l
                        c.load          = load * cpus
                        c.num_cs        = num_cs
                        c.samples       = DEFAULT_SAMPLES

                        c.output_file  = name + '.csv'
                        yield (name, c)


def emstada_example():
    num_cpus = 16
    cluster_size = 1
    ts = generate_emstada_taskset(50, 8, 8)

    partitions = bp.worst_fit_decreasing(ts, num_cpus,
                           capacity=cluster_size,
                           weight=lambda t: t.utilization(),
                           misfit=bp.report_failure,
                           empty_bin=TaskSystem)

    for cpu, part in enumerate(partitions):
        for t in part:
            t.partition = cpu

    locking.assign_edf_preemption_levels(ts)
    for t in ts:
        t.response_time = t.deadline

    ts1 = ts.copy()
    ts2 = ts.copy()

    omip_bounds(num_cpus, cluster_size, ts1) # modifies task parameters

    comlp_bounds(num_cpus, cluster_size, ts2) # modifies task parameters

    for (t, t1, t2) in zip(ts, ts1, ts2):
        print t, 'on', t.partition
        print '\t blocking OMIP=%d OMLP=%d' % (t1.blocked, t2.blocked)
        for q in t.resmodel:
            print '\t\t', "%d for l%02d with L=%d" % (t.resmodel[q].max_requests, q, t.resmodel[q].max_length)
    print ts.utilization(), ts1.utilization(), ts2.utilization()

    print 'NONE'
    for x in xrange(num_cpus):
        print x, '->', sum((t.utilization() for t in ts if t.partition == x))

    print 'OMIP'
    for x in xrange(num_cpus):
        print x, '->', sum((t.utilization() for t in ts1 if t.partition == x))

    print 'OMLP'
    for x in xrange(num_cpus):
        print x, '->', sum((t.utilization() for t in ts2 if t.partition == x))

if __name__ == '__main__':
    emstada_example()

###############################################

EXPERIMENTS = {
    'ecrts13/ucap' : run_mutex_config,
    'ecrts13/csl'  : run_csl_config
}

CONFIG_GENERATORS = {
    'ecrts13/ucap' : generate_mutex_configs,
    'ecrts13/csl'  : generate_csl_configs
}


