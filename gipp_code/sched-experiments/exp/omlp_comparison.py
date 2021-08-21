from __future__ import division

import random
from functools import partial

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean
from toolbox.sci_cache import confidence_interval

import schedcat.generator.tasksets as gen
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

MAPPINGS = [bp.worst_fit_decreasing,
#            bp.best_fit_decreasing,
#            bp.first_fit_decreasing,
#            bp.next_fit_decreasing
]

def backup_params(taskset):
    return [(t.cost, t.period, t.deadline) for t in taskset]

def restore_params(taskset, backup):
    for (wcet, per, dl), t in zip(backup, taskset):
        t.cost     = wcet
        t.period   = per
        t.deadline = dl

def comlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_clustered_omlp_bounds(taskset, cluster_size)

def comlp_rw_bounds(num_cpus, cluster_size, taskset):
    locking.apply_clustered_rw_omlp_bounds(taskset, cluster_size)

def gomlp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_global_omlp_bounds(taskset, num_cpus)

def mpcpvs_bounds(num_cpus, cluster_size, taskset):
    locking.apply_mpcp_bounds(taskset, use_virtual_spin=True)

def mpcp_bounds(num_cpus, cluster_size, taskset):
    locking.apply_mpcp_bounds(taskset, use_virtual_spin=False)

def pretend_no_blocking(num_cpus, cluster_size, taskset):
    pass

def comlp_kx_bounds_replicate(num_replicas):
    replicas = {}
    for res_id in xrange(MAX_RESOURCES):
        replicas[res_id] = num_replicas

    def _comlp_kx_bounds(num_cpus, cluster_size, taskset):
        locking.apply_clustered_kx_omlp_bounds(taskset, cluster_size,
                                               replication_degrees=replicas)

    return _comlp_kx_bounds

def partition_resource_requests(num_replicas):

    def _partition_requests(taskset):
        cur_replica = [0 for _ in xrange(MAX_RESOURCES)]

        for t in taskset:
            old = t.resmodel
            new = ResourceRequirements()

            for res_id in old:
                # map to new ID
                newres = res_id * num_replicas + cur_replica[res_id]
                cur_replica[res_id] = (cur_replica[res_id] + 1) % num_replicas
                new[newres].max_writes        = old[res_id].max_writes
                new[newres].max_reads         = old[res_id].max_reads
                new[newres].max_write_length  = old[res_id].max_write_length
                new[newres].max_read_length   = old[res_id].max_read_length
            t.resmodel = new

    return _partition_requests

# used when m = c
def no_partitioning_required(taskset, *args, **kargs):
    return [taskset]

def heuristics(num_cpus, cluster_size):
    if num_cpus > cluster_size:
        return MAPPINGS
    else:
        return [no_partitioning_required]

def edf_test(num_cpus, cluster_size, apply_bounds, taskset,
             modify_taskset=None):
    num_clusters = num_cpus // cluster_size
    original_taskset = taskset
    taskset = taskset.copy()

    if modify_taskset:
        modify_taskset(taskset)

    locking.assign_edf_preemption_levels(taskset)
    for t in taskset:
        t.response_time = t.deadline

    params = backup_params(taskset)

    for heuristic in heuristics(num_cpus, cluster_size):
        try:
            partitions = heuristic(taskset, num_clusters,
                                   capacity=cluster_size,
                                   weight=lambda t: t.utilization(),
                                   misfit=bp.report_failure,
                                   empty_bin=TaskSystem)

            for cpu, part in enumerate(partitions):
                for t in part:
                    t.partition = cpu

            apply_bounds(num_cpus, cluster_size, taskset) # modifies task parameters

            ok = True
            for part in partitions:
                if not edf.is_schedulable(cluster_size, part, rta_min_step=1000):
                    ok = False
                    break

            restore_params(taskset, params)

            if ok:
                # schedulable
                return True
            # else -> try next heuristic
        except bp.DidNotFit:
            pass
            # try next heuristic
    # fell through; none of the partitioning heuristics worked
    return False

def fp_test(num_cpus, apply_bounds, taskset):
    original_taskset = taskset
    taskset = taskset.copy()

    taskset.sort_by_deadline()
    locking.assign_fp_preemption_levels(taskset)

    params = backup_params(taskset)

#    print '---'
    for heuristic in MAPPINGS:
        try:
            partitions = heuristic(taskset, num_cpus,
                                   capacity=1,
                                   weight=lambda t: t.utilization(),
                                   misfit=bp.report_failure,
                                   empty_bin=TaskSystem)

            for cpu, part in enumerate(partitions):
                part.sort_by_deadline()
                for t in part:
                    t.partition = cpu


            for t in taskset:
                t.response_time = t.cost
            response_times_consistent = False
            schedulable = True

            # iterate until the assumed response times match the
            # actual response times
            iteration = 0
            while schedulable and not response_times_consistent:
                iteration += 1
#                print MAPPINGS.index(heuristic), iteration, taskset.utilization()
                for t in taskset:
                    t.original_response_time = t.response_time
                apply_bounds(num_cpus, 1, taskset) # modifies task parameters

                response_times_consistent = True
                for p, part in enumerate(partitions):
                    if not fp.is_schedulable(1, part):
                        schedulable = False
#                        print 'part failed:', p, part, type(part), part.utilization()
                        break
                    for t in part:
                        if t.response_time != t.original_response_time:
                            response_times_consistent = False

                restore_params(taskset, params)

            if schedulable:
                return True
#            print 'fail', len(taskset)
#            for t in sorted(taskset, key=lambda t: t.partition):
#                print  t.partition, t, t.cost, t.period, t.response_time #, t.suspended, t.blocked,  t.utilization()
            # else -> try next heuristic
        except bp.DidNotFit:
            pass
            # try next heuristic
    # fell through; none of the partitioning heuristics worked
    return False

def gomlp_component_test(num_cpus, cluster_size, taskset):
    num_clusters = num_cpus // cluster_size
    original_taskset = taskset
    taskset = taskset.copy()

    locking.assign_edf_preemption_levels(taskset)
    for t in taskset:
        t.response_time = t.deadline

    params = backup_params(taskset)

    # find connected components and partition accordingly
    subsets = find_independent_tasksubsets(taskset)

    for heuristic in heuristics(num_cpus, cluster_size):
        try:
            # partition connected components instead of tasks
            partitions = heuristic(subsets, num_clusters,
                                   capacity=cluster_size,
                                   weight=lambda t: t.utilization(),
                                   misfit=bp.report_failure,
                                   empty_bin=list)
            flat_parts = []
            for cpu, part in enumerate(partitions):
                fpart = TaskSystem()
                for subset in part:
                    fpart.extend(subset)
                for t in fpart:
                    t.partition = cpu
                flat_parts.append(fpart)
            partitions = flat_parts

            ok = True
            for part in partitions:
                locking.apply_global_omlp_bounds(part, cluster_size)
                if not edf.is_schedulable(cluster_size, part, rta_min_step=1000):
                    ok = False
                    break

            restore_params(taskset, params)

            if ok:
                # schedulable
                return True
            # else -> try next heuristic
        except bp.DidNotFit:
            pass
            # try next heuristic
    # fell through; none of the partitioning heuristics worked
    return False

def draw_cs_count(conf):
    if random.random() <= conf.access_prob:
        # access resource
        return random.randint(1, conf.max_requests)
    else:
        return 0

def should_generate_writer(conf):
    return random.random() <= conf.writer_prob

CSLENGTH  = {
    'short'   : lambda: random.randint(1,   15),
    'medium'  : lambda: random.randint(1,  100),
    'long'    : lambda: random.randint(5, 1280),
}

def generate_requests(conf, task, want_reads=False, res_id_offset=0):
    length = CSLENGTH[conf.cslength]
    for idx in xrange(conf.num_resources):
        res_id = idx + res_id_offset
        num_accesses = draw_cs_count(conf)
        if num_accesses and (not want_reads or should_generate_writer(conf)):
            # add writes
            task.resmodel[res_id] = ResourceRequirement(res_id, num_accesses, length(), 0, 0)
        elif num_accesses:
            # add reads
            task.resmodel[res_id] = ResourceRequirement(res_id, 0, 0, num_accesses, length())

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
        partial(fp_test, conf.num_cpus, comlp_bounds),
        partial(fp_test, conf.num_cpus, mpcpvs_bounds),
        partial(fp_test, conf.num_cpus, mpcp_bounds),
    ]
    titles = ['C-OMLP/EDF', 'C-OMLP/FP', 'MPCP-VS/FP', 'MPCP/FP']
    return tests, titles

def gen_taskset(conf, ucap, want_reads):
    ts = conf.generate(max_util=ucap, time_conversion=ms2us, squeeze=True)
    ts = ensure_min_task_count(conf.num_cpus, ts)
    initialize_resource_model(ts)

    # generate resource requirements
    for t in ts:
        generate_requests(conf, t, want_reads)

    ts = ensure_plausible_request_lengths(ts)

    return ts

def gen_mutex_taskset(conf, ucap):
    return gen_taskset(conf, ucap, False)

def gen_rw_taskset(conf, ucap):
    return gen_taskset(conf, ucap, True)

def setup_component_tests(conf):
    conf.check('cluster_size',   type=int, default=1, min=1, max=conf.num_cpus)
    conf.check('access_prob',    type=float, min=0)
    conf.check('num_resources',  type=int,   min=1, max=MAX_RESOURCES)
    conf.check('max_requests',   type=int,   min=1, default=5)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')

    tests = [
        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds),
        partial(gomlp_component_test, conf.num_cpus, conf.cluster_size),
    ]
    titles = ['C-OMLP/EDF', 'G-OMLP/EDF']
    return tests, titles

def setup_ipp_tests(conf):
    conf.check('ufrac',      type=float, min=0, max=1, default=0.3)
    conf.check('cslength1',  type=one_of(CSLENGTH), default='long')
    conf.check('cslength2',  type=one_of(CSLENGTH), default='short')

    conf.check('utilizations2', type=one_of(gen.NAMED_UTILIZATIONS),
               default=conf.utilizations)
    conf.check('deadlines2',    type=one_of(gen.NAMED_DEADLINES),
               default=conf.deadlines)
    conf.check('periods2',      type=one_of(gen.NAMED_PERIODS),
               default='uni-short')

    conf.generate2 = gen.DIST_BY_KEY[conf.deadlines2][conf.periods2][conf.utilizations2]

    return setup_component_tests(conf)

def gen_ipp_taskset(conf, ucap):
    ts = TaskSystem()
    ts1 = conf.generate(max_util=ucap * conf.ufrac, time_conversion=ms2us, squeeze=True)
    ts2 = conf.generate2(max_util=ucap * (1 - conf.ufrac), time_conversion=ms2us, squeeze=True)
    ts.extend(ts1)
    ts.extend(ts2)
    ts = ensure_min_task_count(conf.num_cpus, ts)
    initialize_resource_model(ts)

    # generate resource requirements for "normal-period" tasks
    conf.cslength = conf.cslength1
    for t in ts1:
        generate_requests(conf, t)

    # Generate resource requirements for "short-period" tasks.
    # Note: the two sets do not share any resources.
    conf.cslength = conf.cslength2
    for t in ts2:
        generate_requests(conf, t, res_id_offset=conf.num_resources)

    ts = ensure_plausible_request_lengths(ts)

    return ts

def setup_kx_tests(conf):
    conf.check('cluster_size',   type=int, default=1, min=1, max=conf.num_cpus)
    conf.check('access_prob',    type=float, min=0)
    conf.check('num_resources',  type=int,   min=1, max=MAX_RESOURCES)
    conf.check('max_requests',   type=int,   min=1, default=5)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')

    tests = [
        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds),

        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds,
                modify_taskset=partition_resource_requests(2)),
        partial(edf_test, conf.num_cpus, conf.cluster_size,
                comlp_kx_bounds_replicate(2)),

        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds,
                modify_taskset=partition_resource_requests(3)),
        partial(edf_test, conf.num_cpus, conf.cluster_size,
                comlp_kx_bounds_replicate(3)),

        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds,
                modify_taskset=partition_resource_requests(4)),
        partial(edf_test, conf.num_cpus, conf.cluster_size,
                comlp_kx_bounds_replicate(4)),
    ]

    titles = ['MX-k1',
              'MX-k2', 'KX-k2',
              'MX-k3', 'KX-k3',
              'MX-k4', 'KX-k4']
    return tests, titles

def setup_rw_tests(conf):
    conf.check('cluster_size',   type=int, default=1, min=1, max=conf.num_cpus)
    conf.check('access_prob',    type=float, min=0, max=1)
    conf.check('writer_prob',    type=float, min=0, max=1, default=0.1)
    conf.check('num_resources',  type=int,   min=1, max=MAX_RESOURCES)
    conf.check('max_requests',   type=int,   min=1, default=5)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')

    tests = [
        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_bounds),
        partial(edf_test, conf.num_cpus, conf.cluster_size, comlp_rw_bounds),
#        partial(edf_test, conf.num_cpus, conf.cluster_size, pretend_no_blocking),
    ]
    titles = [
        'MX',
        'RW',
#        '0B'
    ]

    return tests, titles

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

        row = []
        for i in xrange(len(tests)):
            avg = mean(samples[i])
            ci  = confidence_interval(samples[i])
            if avg > 0:
                all_zeros_count[i]  = 0
            else:
                all_zeros_count[i] += 1
            row += [ci[0], avg, ci[1]]

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
    header = ['UCAP']
    for t in titles:
        header += ['%s CI-' % t, t, '%s CI+' % t]

    data = run_range(conf, tests)
    write_data(conf.output, data, header)

def run_mutex_config(conf):
    check_config(conf)
    tests, titles = setup_mutex_tests(conf)
    conf.make_taskset = partial(gen_mutex_taskset, conf)
    run_exp(conf, tests, titles)

def run_gvc_config(conf):
    check_config(conf)
    tests, titles = setup_component_tests(conf)
    conf.make_taskset = partial(gen_mutex_taskset, conf)
    run_exp(conf, tests, titles)

def run_ipp_config(conf):
    check_config(conf)
    tests, titles = setup_ipp_tests(conf)
    conf.make_taskset = partial(gen_ipp_taskset, conf)
    run_exp(conf, tests, titles)

def run_kx_config(conf):
    check_config(conf)
    tests, titles = setup_kx_tests(conf)
    conf.make_taskset = partial(gen_mutex_taskset, conf)
    run_exp(conf, tests, titles)

def run_rw_config(conf):
    check_config(conf)
    tests, titles = setup_rw_tests(conf)
    conf.make_taskset = partial(gen_rw_taskset, conf)
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

def generate_gvc_configs(options):
    for cpus in [4, 8, 16]:
        for cluster_size in set([1, cpus // 4, cpus // 2, cpus]):
            for nr in [cpus // 4, cpus // 2, cpus, 2 * cpus]:
                for pacc in [0.1, 0.25, 0.4]:
                    for cs in CSLENGTH.keys():
                        for util in gen.NAMED_UTILIZATIONS:
                            name = 'sd_exp=omlp-gvc_util=%s_m=%02d_c=%02d_nr=%d_pacc=%.2f_csl=%s' \
                                        % (util, cpus, cluster_size, nr, pacc, cs)
                            c = Config()
                            c.num_cpus     = cpus
                            c.cluster_size = cluster_size
                            c.utilizations = util
                            c.access_prob  = pacc
                            c.num_resources = nr
                            c.cslength     = cs
                            c.output_file  = name + '.csv'
                            yield (name, c)

def generate_ipp_configs(options):
    for cpus in [4, 8, 16]:
        for cluster_size in set([1, cpus // 4, cpus // 2, cpus]):
            for nr in [cpus // 4, cpus // 2, cpus, 2 * cpus]:
                for pacc in [0.1, 0.25, 0.4]:
                    for ufrac in [0.25, 0.5, 0.75]:
                        for util in gen.NAMED_UTILIZATIONS:
                            name = 'sd_exp=omlp-ipp_util=%s_m=%02d_c=%02d_nr=%d_pacc=%.2f_ufrac=%.2f' \
                                        % (util, cpus, cluster_size, nr, pacc, ufrac)
                            c = Config()
                            c.num_cpus     = cpus
                            c.cluster_size = cluster_size
                            c.utilizations = util
                            c.access_prob  = pacc
                            c.num_resources = nr
                            c.ufrac        = ufrac
                            c.cslength1    = 'long'
                            c.periods      = 'uni-long'
                            c.cslength2    = 'short'
                            c.periods2     = 'uni-short'
                            c.output_file  = name + '.csv'
                            yield (name, c)

def generate_kx_configs(options):
    for cpus in [4, 8, 16]:
        for cluster_size in set([1, cpus // 4, cpus // 2, cpus]):
            for nr in [cpus // 4, cpus // 2, cpus, 2 * cpus]:
                for pacc in [0.4, 0.55, 0.7]:
                    for cs in CSLENGTH.keys():
                        for util in gen.NAMED_UTILIZATIONS:
                            name = 'sd_exp=omlp-kx_util=%s_m=%02d_c=%d_nr=%d_pacc=%.2f_csl=%s' \
                                        % (util, cpus, cluster_size, nr, pacc, cs)
                            c = Config()
                            c.num_cpus     = cpus
                            c.cluster_size = cluster_size
                            c.utilizations = util
                            c.access_prob  = pacc
                            c.num_resources = nr
                            c.cslength      = cs
                            c.output_file  = name + '.csv'
                            yield (name, c)

def generate_rw_configs(options):
    for cpus in [4, 8, 16]:
        for cluster_size in set([1, cpus // 4, cpus // 2, cpus]):
            for nr in [cpus // 4, cpus // 2, cpus, 2 * cpus]:
                for pacc in [0.1, 0.25, 0.4]:
                    for wratio in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
                        for cs in CSLENGTH.keys():
                            for util in gen.NAMED_UTILIZATIONS:
                                name = 'sd_exp=omlp-rw_util=%s_m=%02d_c=%d_nr=%d_pacc=%.2f_wratio=%.2f_csl=%s' \
                                            % (util, cpus, cluster_size, nr, pacc, wratio, cs)
                                c = Config()
                                c.num_cpus     = cpus
                                c.cluster_size = cluster_size
                                c.utilizations = util
                                c.access_prob  = pacc
                                c.writer_prob  = wratio
                                c.num_resources = nr
                                c.cslength      = cs
                                c.output_file  = name + '.csv'
                                yield (name, c)

EXPERIMENTS = {
    'c-omlp/mutex' : run_mutex_config,
    'c-omlp/gvc'   : run_gvc_config,
    'c-omlp/ipp'   : run_ipp_config,
    'c-omlp/kx'    : run_kx_config,
    'c-omlp/rw'    : run_rw_config,
}


CONFIG_GENERATORS = {
    'c-omlp/mutex' : generate_mutex_configs,
    'c-omlp/gvc'   : generate_gvc_configs,
    'c-omlp/ipp'   : generate_ipp_configs,
    'c-omlp/kx'    : generate_kx_configs,
    'c-omlp/rw'    : generate_rw_configs,
}
