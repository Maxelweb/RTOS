from __future__ import division

import random
from math import ceil
from functools import partial

from .ecrts20_cs_gen import *

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean


import schedcat.generator.tasksets as gen
import schedcat.generator.generator_emstada as emstada
import schedcat.mapping.binpack as bp
import schedcat.sched.edf as edf
import schedcat.sched.fp as fp

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.model.resources import initialize_resource_model, initialize_nested_resource_model
from schedcat.model.resources import ResourceRequirement, ResourceRequirements
import schedcat.locking.bounds as locking
from schedcat.locking.partition import find_independent_tasksubsets
from schedcat.util.time import ms2us

DEFAULT_SAMPLES = 200
GROUP_WIDE = 0
GROUP_DEEP = 1

# ----- reference to global option object
options = None

# ----- Calls into CPP LP solver to blocking bounds
def gipp_bounds(num_cpus, cluster_size, taskset, force_single_group=False):
    return locking.apply_gipp_bounds(taskset, num_cpus, cluster_size, force_single_group)

def omip_bounds(num_cpus, cluster_size, taskset):
    return locking.apply_omip_bounds(taskset, num_cpus, cluster_size)

def pretend_no_blocking(num_cpus, cluster_size, taskset):
    pass

# ----- Schedulability tests
def edf_test_gipp_wrapper(num_cpus, cluster_size, tasksets):
    """
    This wrapper tackles the problem of making sure that the EDF
    test is applied on the the tasksets that only differ by how shared resources are modelled
    (i.e, fine-grained for GIPP, group locks for OMIP)

    num_cpus - int:
        number of cpus in the system
    cluster_size - int:
        number of cpus per cluster
    tasksets[0] - SporadicTask set (i.e. list) of tasks with nested resource model
    tasksets[1] - SporadicTAsk set of tasks with non-nested resource model

    taskset[1] should be identical to taskset[0] except that the former uses a single group
    lock for each group as defined by the GIPP
    """

    ts_rnlp = tasksets[0].copy()
    rnlp_bounds = partial(gipp_bounds, force_single_group=True)

    gipp_result = edf_test(num_cpus, cluster_size, gipp_bounds, tasksets[0])
    omip_result = edf_test(num_cpus, cluster_size, omip_bounds, tasksets[1])
    rnlp_result = edf_test(num_cpus, cluster_size, rnlp_bounds, ts_rnlp)

    # r = 0 indicates neither test succeeded
    r = '%d-%d-%d' % \
        (
            1 if gipp_result else 0,
            1 if omip_result else 0,
            1 if rnlp_result else 0
        )

    return r

def edf_test(num_cpus, cluster_size, apply_bounds, taskset):
    """
    Performs an EDF scheduability test on provided taskset

    num_cpus - int:
        number of cpus
    cluster_size - int:
        processors per cluster
    apply_bounds - func(num_cpus, cluster_size, taskset)
        calculates bounds for a taskset and then modifies each tasks
        parameters based on the calculated bounds
    taskset - list of SporadicTask
        the taskset to perform the EDF test on
    """

    num_clusters = num_cpus // cluster_size
    working_taskset = taskset.copy()

    locking.assign_edf_preemption_levels(working_taskset)
    for t in working_taskset:
        t.response_time = t.deadline

    try:
        partitions = bp.worst_fit_decreasing(
                        working_taskset, num_clusters,
                        capacity=cluster_size,
                        weight=lambda t: t.utilization(),
                        misfit=bp.report_failure,
                        empty_bin=TaskSystem)

        for cpu, part in enumerate(partitions):
            for t in part:
                t.partition = cpu

        _ = apply_bounds(num_cpus, cluster_size, working_taskset) # modifies task parameters
        

        for part in partitions:
            if not edf.is_schedulable(cluster_size, part, rta_min_step=1000):
                return False
        return True

    except bp.DidNotFit:
        # fell through; partitioning failed
        return False



# ----- Functions to automate experiments
class ImpossibleConfiguration(Exception):
    pass

def run_csl_config(conf):

    conf.generate = lambda csl_range: generate_emstada_taskset(
                        conf.num_tasks,
                        conf.num_tasks_ls,
                        conf.load,
                        num_res_ls=conf.num_resources_ls,
                        num_res_nls=conf.num_resources_nls,
                        group_size_nls=conf.group_size_nls,
                        group_type_nls=conf.group_type_nls,
                        period_range_ls=conf.period_range_ls,
                        period_range_nls=conf.period_range_nls,
                        acc_max_ls=conf.acc_max_ls,
                        acc_max_nls=conf.acc_max_nls,
                        csl_range_ls=conf.csl_range_ls,
                        csl_range_nls=csl_range,
                        top_probability=conf.top_probability,
                        asymmetric=conf.asymmetric
                        )

    # check to see if its possible to generate a taskset with such a configuraiton
    # return False if so to denote that its impossible.
    test_taskset = conf.generate((conf.csl_range_nls[0], conf.csl_range_nls[0]+5))
    if test_taskset == ([], []):
        return False

    tests = [
        partial(edf_test_gipp_wrapper, conf.num_cpus, conf.cluster_size),
    ]
    titles = ['GIPP-vs-OMIP-vs-RNLP/EDF']

    data = run_cls_tests(conf, tests)
    write_data(conf.output, data, titles)

    return True

def run_cls_tests(conf, tests):
    """
    Generate conf.samples number of sample points for a given
    configuration.

    conf - Config:
        configuration used for generation of task sets and sample number
    tests - []:
        list of test (which are functions) to be called with the taskset
    """

    # all_zeros makes the assumption that if a scheduability test fails three times
    # as the max critical section length increases, then it will continue to do so
    all_zeros_count = [0 for _ in tests]

    for csl in xrange(conf.csl_range_nls[0], conf.csl_range_nls[1] + conf.step_size, conf.step_size):

        samples = [[] for _ in tests]
        for _ in xrange(conf.samples):

            num_test_all_zeros = 0

            for i in xrange(0, len(tests)):

                # 20 is just a constant
                # if all locking protocols fail for 20 times in a row
                # its very unlikely more tests for the same configuration
                # will yield a different result.
                if (all_zeros_count[i] > 20):
                    samples[i].append('0-0-0')
                    num_test_all_zeros += 1
                    print "======== ALL ZEROS COUNT for test=%d and csl=%d" % (i, csl)

            if num_test_all_zeros == len(tests):
                continue

            taskset = conf.generate((conf.csl_range_nls[0], csl))

            if len(taskset) == 0:
                print "run_cls_test - couldn't generate emstada taskset for test"

            for i in xrange(len(tests)):

                if (len(taskset) == 0):
                    samples[i].append('0-0-0')
                    print "======== ALL ZEROS COUNT for test=%d and csl=%d" % (i, csl)
                else:
                    samples[i].append(tests[i](taskset))

        row = []
        for i in xrange(len(tests)):
            for x in samples:
                for y in x:
                    row.append(y)
                    test_result = [int(x) for x in y.split('-')]

                    if sum(test_result) == 0:
                        all_zeros_count[i] += 1
                    else:
                        all_zeros_count[i] = 0

        yield [csl] + row

# ----- Functions to automate generation of configuration files

def generate_csl_configs(
                        _cpus = [2, 4, 8, 16],
                        _n = [20, 40],
                        _n_ratio = False,
                        _ls = [0, 4, 8],
                        _ls_ratio = False,
                        _load = [0.4, 0.5, 0.6],
                        _acc_max_ls = [3],
                        _acc_max_nls = [0, 1, 2, 3],
                        _g_size_nls = [1, 2, 3, 4],
                        _g_type_nls = [GROUP_WIDE, GROUP_DEEP],
                        _tp = [0.5],
                        _asymmetric = [False],
                        _num_res_ls = 3,
                        _num_res_nls = 12,
                        _samples = 500
                        ):

    # one ls task is special case if there is a group of 3 resources

    cluster_size = 1
    samples = _samples
    nls_period_range = [10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000]
    ls_period_range = [1, 2, 4, 5, 8]

    # for cpus in [2, 4, 8, 16]:
    for cpus in _cpus:

        for n in _n:

            for ls in _ls:

                for num_res_nls in _num_res_nls:

                    for num_res_ls in _num_res_ls:

                        for load in _load:

                            for acc_max_ls in _acc_max_ls:

                                for acc_max_nls in _acc_max_nls:

                                    for gs_nls in _g_size_nls:

                                        for tp in _tp:

                                            for asymmentric in _asymmetric:

                                                for gt_nls in _g_type_nls:
                                                # t is for top_level resources
                                                # for tl in _tl:

                                                    if _n_ratio:
                                                        n = int(n * cpus)
                                                    else:
                                                        n = int(n)

                                                    if _ls_ratio:
                                                        ls = int(ls * cpus)
                                                    else:
                                                        ls = int(ls)

                                                    name = 'sd_exp=gipp'
                                                    name += '_m=%02d' % cpus
                                                    name += '_n=%02d' % n
                                                    name += '_ls=%02d' % ls
                                                    name += '_lo=%.2f' % load
                                                    name += '_res-ls=%02d' % num_res_ls
                                                    name += '_res-nls=%02d' % num_res_nls
                                                    name += '_acc-ls=%02d' % acc_max_ls
                                                    name += '_acc-nls=%02d' % acc_max_nls
                                                    name += '_gs-nls=%02d' % gs_nls
                                                    name += '_gt-nls=%02d' % gt_nls
                                                    name += '_asym=%02d' % (1 if asymmentric else 0)
                                                    name += '_tp=%.2f' % tp
                                                    # name += '_tl=%02d' % tl
                                                    name += '_s=%02d' % samples
                                                    

                                                    c = Config()
                                                    c.num_cpus              = cpus
                                                    c.cluster_size          = cluster_size
                                                    c.load                  = load * cpus
                                                    c.num_resources_ls      = num_res_ls
                                                    c.num_resources_nls     = num_res_nls
                                                    c.num_tasks             = n
                                                    c.num_tasks_ls          = ls
                                                    c.period_range_nls      = nls_period_range
                                                    c.period_range_ls       = ls_period_range
                                                    c.acc_max_ls            = acc_max_ls
                                                    c.acc_max_nls           = acc_max_nls
                                                    c.group_size_nls        = gs_nls
                                                    c.group_type_nls        = gt_nls
                                                    c.top_probability       = tp
                                                    c.asymmetric            = asymmentric
                                                    c.csl_range_ls          = (1, 15)
                                                    c.csl_range_nls         = (5, 1000)
                                                    c.step_size             = 5
                                                    # c.nesting_probability   = np
                                                    c.samples               = samples
                                                    c.experiment    = 'ecrts20/csl_experiment'

                                                    c.output_file  = name + '.csv'
                                                    #c.output_file  = name + '.handpicked'
                                                    yield (name, c)

# ----- Functions to generate a taskset with
def generate_emstada_taskset(
        num_tasks,
        num_latency_sensitive,
        util,
        num_res_ls = 3,
        num_res_nls = 12,
        group_size_nls = 3,
        group_type_nls = GROUP_WIDE,
        period_range_ls = [1, 2, 4, 5, 8],
        period_range_nls = [10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000],
        acc_max_ls = 1,
        acc_max_nls = 3,
        csl_range_ls = (1, 15),
        csl_range_nls = (5, 1000),
        asymmetric = False,
        top_probability = 0.5
        ):
    """
    Generates a pair of tasksets with the given parameters using the Paul Emberson, Roger Stafford, Robert Davis
    software/method. The task sets are identical expect where shared resources are concerned.
    The first taskset, ts, assumes fine-grained nested locking.
    The second taskset, ts_nn, assumes no nested locking. Therefore, the groups formed by nested locking
    in ts are reduced to a single shared resource in ts to emulate the use of group locks.

    **NOTE**: num_res_nls is expected to be a multiple of group_size

    num_tasks - int:
        number of total tasks to generate (including latency-sensitive)
    num_latency_sensitive - int:
        number of total tasks to be latency-sensitive
        **NOTE**: if this value is 1, then a group of size 1 is forced.
    util - float:
        total utilization the generated taskset should have
    num_res_ls - int:
        number of resources shared among latency-sensitive tasks
    num_res_nls - int:
        number of resources shared among non-latency-sensitive tasks
    group_size - int:
        the number of groups to split num_res_nls into
    group_type_nls - int (constant):
        specifices the group type for non-latency-sensitive tasks, currently either "wide" or "deep"
    num_top_level - int:
        for each group generate, there are num_top_level resources which are "greatest" in the
        partial ordering. The condition 2*num_top_level + 1 > num_resources must be held
        in order to generate a group with num_top_level top level resources. The value is
        clamped when necessary. This parameters only applies to non-latency-sensitive tasks.
    period_range_ls - (float, float):
        a list of values to draw latency-sensitive periods from in milliseconds
    period_range_nls - (float, float):
        a list of values to draw non-latentcy-sensitive periods from in milliseconds
    acc_max_ls - int:
        the amount of outermost requests each latency-sensitive task makes
    acc_max_nls - int:
        the amount of outermost requests each non-latency-sensitive task makes
    cs_range_ls - [int]:
        the range in microseconds to draw the length of a latency-sensitive's tasks outermost critical
        section.
    cs_range_nls - [int]:
        the range in microseconds to draw the length of a non-latency-sensitive's tasks outermost critical
        section.
    top_probability - float:
        when generating access patterns for a task, this is the probability fed to cs generation algorithm
        that determines how likely a "top-level" resource will be picked for the cs
    """

    n = num_tasks - num_latency_sensitive
    n_ls = num_latency_sensitive
    num_groups = num_res_nls // group_size_nls

    # num_res_nls is assumed to be a multiple of group_size_nls
    

    if num_groups > num_tasks:
        print "there must be at least as many tasks as there are groups"
        return [], []

    selected_group_nls = None
    if group_size_nls == 1:
        selected_group_nls = group_1
    
    if group_size_nls == 2:
        selected_group_nls = group_2
    
    if group_size_nls == 3:
        if group_type_nls == GROUP_WIDE:
            selected_group_nls = group_3_wide
        else:
            selected_group_nls = group_3_deep

    if group_size_nls == 4:
        if group_type_nls == GROUP_WIDE:
            selected_group_nls = group_4_wide
        else:
            selected_group_nls = group_4_deep

    if group_size_nls == 5:
        if group_type_nls == GROUP_WIDE:
            if random.random() < 0.5 or True:
                selected_group_nls = group_5_wide_1
            else:
                selected_group_nls = group_5_wide_2
        else:
            selected_group_nls = group_5_deep

    if selected_group_nls == None:
        print "no group selected."
        return [], []

    if n < num_groups * len(selected_group_nls["necessary"]):
        print "not possible to assign %d tasks to %d groups with min_tasks_per_group=%d" % (n, num_groups, len(selected_group_nls["necessary"]))
        return [], []

    # bound the total utilization of latench-sensitive tasks to not be more than half
    # of the total desired utilization
    lsu = min(n_ls * 0.5, util * 0.5)

    # generate tasks using with periods sampling from a log-uniform distribution
    # emstada.gen_taskset accets periods in milliseconds, but returns periods in
    # microseconds due to scale=ms2us.
    if n_ls > 0:
        ts_ls = emstada.gen_taskset(period_range_ls, period_range_ls, n_ls, lsu, scale=ms2us)
    else:
        ts_ls = TaskSystem()

    # in reality, not all tasks will be latency-sensitive, but this check is helpful
    # if we want to prented all tasks are latency-sensitive
    if n > 0:
        ts = emstada.gen_taskset(period_range_nls, period_range_nls, n, util - lsu, period_granularity=0.5, scale=ms2us)
    else:
        ts = TaskSystem()

    # tasksets for non-nested (denoted with _nn) resource model
    ts_ls_nn = ts_ls.copy()
    ts_nn = ts.copy()

    # initialize nested resource models
    initialize_nested_resource_model(ts_ls)
    initialize_nested_resource_model(ts)

    res_id_to_group = dict()
    group_counter = 0
    
    if n_ls > 0:

        # we assume latency-sensitive tasks touch one group
        # there must never be just one latency-sensitive task, there should be 0, two, or more

        selected_group_ls = None
        if n_ls == 1:
            selected_group_ls = group_1
            num_res_ls = 1
        else:
            selected_group_ls = group_3_wide

        acc_amounts = [random.randint(1, acc_max_ls) for _ in xrange(0, n_ls)]
        acc_ls = generate_group_cs_for_taskset(
                                    n_ls,
                                    selected_group_ls,
                                    acc_amounts,
                                    top_probability)

        for i in xrange(0, len(ts_ls)):
            t = ts_ls[i]
            t_acc = acc_ls[i]

            for a in t_acc:
                # Currently the GIPP only reasons about the length of outermost
                # critical sections, so having nested critical sections with
                # the same length suffices.
                x = [random.randint(csl_range_ls[0], csl_range_ls[1])]
                lengths = x * len(a)
                # lengths = random_less_len(len(a), csl_range_ls)

                cs = t.critical_sections.add_outermost(a[0], lengths[0])
                res_id_to_group[a[0]] = group_counter

                for k in xrange(1, len(a)):
                    cs = t.critical_sections.add_nested(cs, a[k], 0)
                    res_id_to_group[a[k]] = group_counter

        
        # any further resources belong to another group
        group_counter += 1
    else:
        # if no latency-sensitive tasks, then we set this to zero so the
        # offsets calculated for the non-latency-sensitive tasks are not
        # increased to compensate for resource_ids that don't existv
        num_res_ls = 0

    # generate a list of the groups that tasks accessses
    group_accesses = assign_tasks_to_groups(n, len(selected_group_nls["necessary"]), num_groups)

    for h in xrange(0, num_groups):  

        num_tasks_accessing_group = 0
        for nt in group_accesses:
            if nt == h:
                num_tasks_accessing_group += 1

        acc_amounts = [random.randint(1, acc_max_nls) for _ in xrange(0, num_tasks_accessing_group)]

        generate_cs_func = None
        if asymmetric:
            generate_cs_func = generate_asymmetric_group_cs_for_taskset
        else:
            generate_cs_func = generate_group_cs_for_taskset
        
        acc = generate_cs_func(
                                num_tasks_accessing_group,
                                selected_group_nls,
                                acc_amounts,
                                top_probability)

        offset = num_res_ls + (h * selected_group_nls["num_res"])

        for j in xrange(0, len(acc)):
            for k in xrange(0, len(acc[j])):
                for l in xrange(0, len(acc[j][k])):
                    acc[j][k][l] += offset

        t_acc_counter = 0
        for i in xrange(0, len(ts)):

            # if the task doesn't access the group, then move onto the next task
            if group_accesses[i] != h:
                continue

            t = ts[i]
            t_acc = acc[t_acc_counter]

            for a in t_acc:
                # Currently the GIPP only reasons about the length of outermost
                # critical sections, so having nested critical sections with
                # the same length suffices.

                x = [random.randint(csl_range_nls[0], csl_range_nls[1])]
                
                lengths = x * len(a)
                cs = t.critical_sections.add_outermost(a[0], lengths[0])
                res_id_to_group[a[0]] = group_counter

                for k in xrange(1, len(a)):
                    cs = t.critical_sections.add_nested(cs, a[k], lengths[k])
                    res_id_to_group[a[k]] = group_counter

            t_acc_counter += 1
        
        
        # any further resources belong to another group
        group_counter += 1

    # the idea from here on out is that for every group in the nested model, we create a single
    # group lock for the non-nested model.
    
    # initialize non-nested resource model
    initialize_resource_model(ts_ls_nn)
    initialize_resource_model(ts_nn)

    ts.extend(ts_ls)
    ts_nn.extend(ts_ls_nn)

    for i in xrange(0, len(ts)):

        t = ts[i]
        t_nn = ts_nn[i]

        for cs in t.critical_sections.all():

            # only interested in outermost critical sections
            if cs.outer == None:
                group_id = res_id_to_group[cs.res_id]
                t_nn.resmodel[group_id].add_request(cs.length)


    return (ts, ts_nn)

def emstada_example():
    """
    This almost certainly does not work anymore - hopefully it provides a good
    reference for a starting point, though. Place s.py content here later.
    """

    num_cpus = 1
    cluster_size = 1
    ts, ts_nn = generate_emstada_taskset(2, 2, 0.8)

    # assign tasks to clusters (partition) using worst_fit_decreasing algorithm
    partitions = bp.worst_fit_decreasing(
                    ts, 
                    num_cpus,
                    capacity=cluster_size,
                    weight=lambda t: t.utilization(),
                    misfit=bp.report_failure,
                    empty_bin=TaskSystem)

    for cpu, part in enumerate(partitions):
        for t in part:
            t.partition = cpu

    for i in xrange(0, len(ts)):
        ts_nn[i].partition = ts[i].partition

    # calculate response times for EDF
    locking.assign_edf_preemption_levels(ts)
    for t in ts:
        t.response_time = t.deadline

    locking.assign_edf_preemption_levels(ts_nn)
    for t in ts_nn:
        t.response_time = t.deadline

    ts_gipp = ts.copy()
    ts_omip = ts_nn.copy()

    

    print "--- the two tasksets should match"
    print "GIPP taskset before bounds calculation"
    for t in ts_gipp:
        print t, t.cost
    print "utilization: ", ts.utilization()
    print " "
    print "OMIP taskset before bounds calculation"
    for t in ts_gipp:
        print t, t.cost
    print "utilization: ", ts.utilization()
    print " "

    gipp_bounds(num_cpus, cluster_size, ts_gipp) # modifies task parameters
    omip_bounds(num_cpus, cluster_size, ts_omip) # modifies task parameters

    for (t, t1, t2) in zip(ts, ts_gipp, ts_omip):
        print t, 'on', t.partition
        print '\t blocking GIPP=%d OMIP=%d' % (t1.blocked, t2.blocked)
        # TODO: think about how to print out something similar given that the resource models differ
        # for q in t.resmodel:
        #     print '\t\t', "%d for l%02d with L=%d" % (t.resmodel[q].max_requests, q, t.resmodel[q].max_length)
    print ts.utilization(), ts_gipp.utilization(), ts_omip.utilization()

    print 'NONE'
    for x in xrange(num_cpus):
        print x, '->', sum((t.utilization() for t in ts if t.partition == x))

    print 'GIPP'
    for x in xrange(num_cpus):
        print x, '->', sum((t.utilization() for t in ts_gipp if t.partition == x))

    print 'OMIP'
    for x in xrange(num_cpus):
        print x, '->', sum((t.utilization() for t in ts_omip if t.partition == x))

    return

def dummy():
    pass


# ----- Configure EXPERIMENTS/CONFIG_GENERATORS
# csl is shorthand for 'critical section length'

EXPERIMENTS = {
    'ecrts20/csl_experiment' : run_csl_config
}

CONFIG_GENERATORS = {
    'ecrts20/csl_conf' : generate_csl_configs
}

# ----- Run basic example if module is main
if __name__ == '__main__':
    pass
    #emstada_example()
