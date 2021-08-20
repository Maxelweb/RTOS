from __future__ import division

import random
from functools import partial

from toolbox.io import one_of, Config, write_data
from toolbox.sample import value_range
from toolbox.stats import mean
from toolbox.sci_cache import confidence_interval

import schedcat.generator.tasksets as gen
import schedcat.mapping.binpack as bp
import schedcat.sched.fp as fp
import schedcat.sched.fp.rta as fp_rta
import schedcat.generator.generator_emstada as emstada

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.model.resources import initialize_resource_model, ResourceRequirement, ResourceRequirements
import schedcat.locking.bounds as locking
from schedcat.util.time import ms2us
import math
import copy
#import schedcat.locking.linprog.msrp as msrp
import schedcat.locking.linprog.native as cpp

MAX_RESOURCES = 32

DEFAULT_SAMPLES = 200

# reference to global option object
options = None

MAPPINGS = [bp.worst_fit_decreasing,
#            bp.best_fit_decreasing,
#            bp.first_fit_decreasing,
#            bp.next_fit_decreasing
]

# custom RTA for LP-based analysis

def check_for_suspension_parameters(taskset):
    "compatibility: add required parameters if they are not present"
    for t in taskset:
        if not 'jitter' in t.__dict__:
            # No arrival jitter (equivalent to an initial suspension).
            t.jitter = 0

def fp_demand(task, time):
    return task.cost * int(math.ceil((time + task.jitter) / task.period))

def bound_response_times(no_cpus, taskset):
    """Assumption: taskset is sorted in order of decreasing priority."""
    if not (no_cpus ==  1 and taskset.only_constrained_deadlines()):
        # This implements standard uniprocessor response-time analysis, which
        # does not handle arbitrary deadlines or multiprocessors.
        return False
    else:
        fp_rta.check_for_suspension_parameters(taskset)
        for i in xrange(len(taskset)):
            if not fp.rta.rta_schedulable(taskset, i):
                return i
        return len(taskset)+23 # return invalid index if taskset schedulable

def rta_schedulable_lp(taskset, i):
    task = taskset[i]
    higher_prio = taskset[:i]
    test_end = task.deadline
    
#     print "\n\n[rta_schedulable_lp] higher prio tasks: %s" % higher_prio

#    print "higher-prio tasks: %s" % higher_prio
#    print "taskset[i].blocked: %s" % taskset[i].blocked
    # pre-compute the additional terms for the processor demand bound
    own_demand = task.blocked + task.jitter + task.cost
#    print "[rta_schedula_lp] task: %s\town demand: %s" % (task, own_demand)

#     print "[rta_schedulable_lp] time calculation: higher prio costs:", [t.cost for t in higher_prio]
#     print "[rta_schedulable_lp] time calculation: task.blocked: %s" % task.blocked
#     print "[rta_schedulable_lp] time calculation: task.cost: %s" % task.cost
#     print "[rta_schedulable_lp] time calculation: task.jitter: %s" % task.jitter

    # see if we find a point where the demand is satisfied
    time = sum([t.cost for t in higher_prio]) + own_demand
#     print "\n[rta_schedulable_lp] task: %s" % task
#     print "[rta_schedulable_lp] own demand: %s" % own_demand
#     print "[rta_schedulable_lp] time: %s" % time
# #     print "[rta_schedulable_lp] task.blocked: %s" % task.blocked
#     print "[rta_schedulable_lp] higher-prio demand: %s" % sum([fp_demand(t, time) for t in higher_prio])
    
    while time <= test_end:
#        task.response_time = time # response time is used by locking analysis
#        blocked = locking.apply_cpp_lp_msrp_bounds_single(taskset, i)
#        bounds(taskset, i)
#        print "[rta_schedulable_lp] bounds: %s" % bounds
        blocked = taskset[i].blocked
#         print "[rta_schedulable_lp] blocked: %s" % blocked
#         print "[rta_schedulable_lp] taskset[i].blocked: %s" % task
        demand = sum([fp_demand(t, time) for t in higher_prio]) \
            + own_demand
#         print "[rta_schedulable_lp] demand: %s" % demand
        if demand == time:
            # yep, demand will be met by time
            task.response_time = time
#             print "[rta_schedulable_lp] task: %s\ttotal demand: %s\tslack: %s" % (taskset[i], demand, task.deadline - demand)
#             print "[rta_schedulable_lp] converged! response time: %s" % task.response_time
            
            return True
        else:
            # try again
            time = demand

    # if we get here, we didn't converge
    return False


def bound_response_times_lp(no_cpus, taskset):
    """Assumption: taskset is sorted in order of decreasing priority."""
    if not (no_cpus ==  1 and taskset.only_constrained_deadlines()):
        # This implements standard uniprocessor response-time analysis, which
        # does not handle arbitrary deadlines or multiprocessors.
        return False
    else:
        check_for_suspension_parameters(taskset)
        for i in xrange(len(taskset)):
            if not rta_schedulable_lp(taskset, i):
                return i # return index of failing task if not schedulable 
        return len(taskset)+23 # return invalid index if taskset schedulable

is_schedulable_lp = bound_response_times_lp

def backup_params(taskset):
    return [(t.cost, t.period, t.deadline) for t in taskset]

def restore_params(taskset, backup):
    for (wcet, per, dl), t in zip(backup, taskset):
        t.cost     = wcet
        t.period   = per
        t.deadline = dl

def msrp_bounds(num_cpus, cluster_size, taskset):
#     total_cost_original = sum([tx.cost for tx in taskset])
    locking.apply_msrp_bounds(taskset, num_cpus)
#     total_cost = sum([tx.cost for tx in taskset])
#     total_blocked = sum([tx.blocked for tx in taskset])
#     print "(MSRP)\ttotal blocked:\t%s \ttotal new cost:\t%s\ttotal original cost :\t%s" % (total_blocked, total_cost, total_cost_original)

def msrp_bounds_bbb(num_cpus, cluster_size, taskset):
#     total_cost_original = sum([tx.cost for tx in taskset])
    locking.apply_task_fair_mutex_bounds(taskset, 1)
#     total_cost = sum([tx.cost for tx in taskset])
#     total_blocked = sum([tx.blocked for tx in taskset])
#     print "(MSRP-BBB)\ttotal blocked:\t%s \ttotal cost:\t%s" % (total_blocked, total_cost)
#     print "(MSRP-BBB)\ttotal blocked:\t%s \ttotal new cost:\t%s\ttotal original cost :\t%s" % (total_blocked, total_cost, total_cost_original)

def msrp_bounds_holistic(num_cpus, cluster_size, taskset):
#     total_cost_original = sum([tx.cost for tx in taskset])
    res = locking.apply_msrp_bounds_holistic(taskset)
#     total_cost = sum([tx.cost for tx in taskset])
#     total_blocked = sum([tx.blocked for tx in taskset])

def apply_bounds(num_cpus, cluster_size, taskset, bounds):
    res = bounds(taskset)
    for i, _ in enumerate(taskset):
        taskset[i].blocked = res.get_blocking_term(i)

apply_msrp_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_msrp_bounds)
apply_preemptive_fifo_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_preemptive_fifo_bounds)

apply_unordered_spinlock_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_unordered_bounds)
apply_preemptive_unordered_spinlock_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_preemptive_unordered_bounds)

apply_prio_spinlock_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_prio_bounds)
apply_preemptive_prio_spinlock_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_preemptive_prio_bounds)

apply_prio_fifo_spinlock_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_prio_fifo_bounds)
apply_preemptive_prio_fifo_spinlock_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_preemptive_prio_fifo_bounds)

apply_baseline_bounds = partial(apply_bounds, bounds = locking.apply_pfp_lp_baseline_spinlock_bounds)

apply_dummy_bounds = partial(apply_bounds, bounds = locking.apply_dummy_bounds)

def fp_test(num_cpus, rta, apply_bounds, taskset, raise_locking_prio = False):
    original_taskset = taskset
    taskset = copy.deepcopy(taskset)

    for t in taskset:
        t.locking_prio = len(taskset)
    
    # assign the task's locking_prio to each of its requests
    for t in taskset:
        for r in t.resmodel:
            t.resmodel[r].priority = t.locking_prio;
        
    params = backup_params(taskset)   
     
    partitions = [ TaskSystem() for _ in range(num_cpus)]
    for ti in taskset:
        ti.response_time = ti.cost
        partitions[ti.partition] += [ti]
    for part in partitions:
        part.sort_by_deadline()

    response_times_consistent = False
    schedulable = True

    prio_increments = 0
    failing_task = None
    last_incremented_task = None
    prio_incremented_tasks = []
    max_locking_prio = len(taskset)
    while prio_increments < len(taskset):
        response_times_consistent = False
        schedulable = True
        # iterate until the assumed response times match the
        # actual response times
        iteration = 0
        while schedulable and not response_times_consistent:
            iteration += 1
            for t in taskset:
                t.original_response_time = t.response_time
                t.blocked = 0
                
            # first check schedulability without applying bounds
            # this can already rule out a couple of tasks
            if iteration == 0:
                for _, part in enumerate(partitions):
                    fail_task_idx = rta(1, part)
                    if fail_task_idx < len(part): # taskset not schedulable, even without blocking
                        return False
            
            for t in taskset:
                t.response_time = t.original_response_time
            
            if apply_bounds != None:
                apply_bounds(num_cpus, 1, taskset) # modifies task parameters
        
            response_times_consistent = True
            for _, part in enumerate(partitions):
                fail_task_idx = rta(1, part)
                if fail_task_idx < len(part): 
                    print "failing task: %s" % part[fail_task_idx]
                    schedulable = False
                    failing_partition = part[fail_task_idx].partition
                    break
                for t in part:
                    if t.response_time != t.original_response_time:
                        response_times_consistent = False
#            print "[fp_test] iterations:\t%s" % iteration
#            for t in taskset:
#                print "task %s\t blocked:\t%s" %(t.id, t.blocked)
            
            restore_params(taskset, params)
        
#        for ti in taskset:
#            print "[fp_test] task: %s\t slack: %s" % (ti, (ti.deadline - ti.response_time))
        
        if schedulable:
            if prio_increments > 0:
                print "taskset schedulable after %s locking priority increments!" % prio_increments
#             for t in taskset:
#                 print "task: %s,\tresponse time: %s" %(t, t.response_time)
            return True
        
#         print "failing task: %s" % partitions[failing_partition][fail_task_idx]
        if not raise_locking_prio:
            break;
#        print "priority increments: %s" % prio_increments
        
        # increment locking priority of failing task
        if prio_increments > 10 and len(set(prio_incremented_tasks[-5:])) == len(set(prio_incremented_tasks[-10:])): # incrementing the same task multiple times in a row doesn't help
            break
        if prio_increments >0 and partitions[failing_partition][fail_task_idx] == prio_incremented_tasks[-1]:
            break
        for ti in partitions[failing_partition][fail_task_idx:]:
            if ti.locking_prio > 0:
                ti.locking_prio = ti.locking_prio - 1
            else:
                break
            
        # assign the task's (increased) locking_prio to each of its requests
        for t in taskset:
            for r in t.resmodel:
                t.resmodel[r].priority = t.locking_prio;

        prio_incremented_tasks.append(partitions[failing_partition][fail_task_idx])
#         print "prio increments: %s" % prio_increments
        prio_increments = prio_increments + 1
        
    return False

def draw_cs_count(conf):
    if random.random() <= conf.access_prob and conf.max_requests > 0:
        # access resource
        return random.randint(1, conf.max_requests)
    else:
        return 0

def should_generate_writer(conf):
    return random.random() <= conf.writer_prob

CSLENGTH  = {
    'short'   : lambda: random.randint(1,   15),
    'medium'  : lambda: random.randint(1,  100),
}

def generate_requests(conf, taskset, want_reads=False, res_id_offset=0):
    max_locking_prio = len(taskset)
    length = CSLENGTH[conf.cslength]
    for idx in xrange(conf.num_r_resources):
        res_id = idx + res_id_offset
        accessing_tasks = random.sample(taskset, int(len(taskset) * conf.res_sharing_factor))
        for task in accessing_tasks:
            num_accesses = draw_cs_count(conf)
            if num_accesses and (not want_reads or should_generate_writer(conf)):
                # add writes
                task.resmodel[res_id] = ResourceRequirement(res_id, num_accesses, length(), 0, 0, max_locking_prio)
            elif num_accesses:
                # add reads
                task.resmodel[res_id] = ResourceRequirement(res_id, 0, 0, num_accesses, length(), max_locking_prio)

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
    conf.check('access_prob',    type=float, min=0, default=0.9)
    conf.check('access_prob_remote',    type=float, min=0, default=0)
    conf.check('num_l_resources',  type=int,   min=0, max=MAX_RESOURCES, default=0)
    conf.check('num_r_resources',  type=int,   min=0, max=MAX_RESOURCES, default=0)
    conf.check('max_requests',   type=int,   min=1, default=2)
    conf.check('cslength',       type=one_of(CSLENGTH), default='short')
    conf.check('num_cpus',     type=int, min=1)
    conf.check('samples',      type=int, default=DEFAULT_SAMPLES, min=1)
    conf.check('step_size',    type=float, default=0.05, min=0.025)
#    print "emstada.NAMED_PERIODS: ", emstada.NAMED_PERIODS
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(emstada.NAMED_PERIODS),   default='uni-moderate')
#    conf.check('period_distribution',   type=one_of(emstada.NAMED_PERIOD_DISTRIBUTIONS),    default='unif')
    conf.check('period_distribution',   type=one_of(emstada.NAMED_PERIOD_DISTRIBUTIONS),    default='logunif')
    conf.check('min_ucap',     type=float, min=0.1, max=1, default=0.05)
    conf.check('ucap',     type=float, min=0.1, max=1, default=0.5)
    conf.check('tasks_n',     type=int, min=1, default=5)
    conf.check('per_task_util', type=float, min=0, default=0)
    conf.check('only_partitionable', type=bool, default=False)
    conf.check('only_local_res', type=bool, default=False)
    conf.check('min_tasks', type=int, default=0)
    conf.check('res_sharing_factor', type=float, default=0.2)
    conf.check('cons_dl_fac', type=float, default=1.0)
    conf.generate = partial(emstada.gen_taskset, conf.periods, conf.period_distribution)
    
    tests = [
        partial(fp_test, conf.num_cpus, bound_response_times, msrp_bounds, raise_locking_prio = False),
        partial(fp_test, conf.num_cpus, bound_response_times, msrp_bounds_bbb),
        partial(fp_test, conf.num_cpus, bound_response_times, msrp_bounds_holistic),
#        partial(fp_test, conf.num_cpus, partial(is_schedulable_lp, bounds = locking.apply_cpp_lp_msrp_bounds_single), None),
#        partial(fp_test, conf.num_cpus, partial(is_schedulable_lp, bounds = locking.apply_cpp_lp_unordered_bounds_single), None),
#        partial(fp_test, conf.num_cpus, 
#                partial(is_schedulable_lp, bounds = locking.apply_cpp_lp_prio_bounds_single), 
#                None, 
#                raise_locking_prio = True),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_msrp_bounds, 
                raise_locking_prio = False),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_preemptive_fifo_bounds, 
                raise_locking_prio = False),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_unordered_spinlock_bounds, 
                raise_locking_prio = False),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_preemptive_unordered_spinlock_bounds, 
                raise_locking_prio = False),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_prio_spinlock_bounds, 
                raise_locking_prio = True),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp,
                apply_preemptive_prio_spinlock_bounds, 
                raise_locking_prio = True),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp,
                apply_prio_fifo_spinlock_bounds, 
                raise_locking_prio = True),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_preemptive_prio_fifo_spinlock_bounds, 
                raise_locking_prio = True),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_baseline_bounds, 
                raise_locking_prio = False),
        partial(fp_test, 
                conf.num_cpus, 
                bound_response_times_lp, 
                apply_dummy_bounds, 
                raise_locking_prio = False),
    ]
    titles = ['MSRP', 'MSRP-BBB', 'MSRP-hol', 'MSRP-LP', 'FIFO-LP-P', "unordered", "unordered-P", "prio", "prio-P", "prio-fifo", "prio-fifo-P", "base", "null"]
    return tests, titles

def gen_taskset(conf, ucap, tasks, want_reads):
    partitions = []
    for p in range(conf.num_cpus):
        n = int(math.floor(tasks/conf.num_cpus) + (1 if p < tasks%conf.num_cpus else 0))
        ts = conf.generate(tasks_n = n, utilization = (ucap/conf.num_cpus))
        initialize_resource_model(ts)
        for t in ts:
            t.partition = p
        partitions += [ts]
    # generate resource requirements
    ts = merge_tasksets(partitions)
    ts.sort_by_deadline()
    ts.assign_ids()
    generate_requests(conf, ts, want_reads)
    
    # make sure that the CSs don't exceed the execution cost
    for t in ts:
        cum_cs = sum([t.resmodel[res].max_length * t.resmodel[res].max_requests for res in t.resmodel])
        t.cost = max(t.cost, cum_cs)

    # Assign FP preemption levels.
    # Since the task set is sorted rate-monotonic, the preemption level 
    # of each task will be equal to the task ID (which is the scheduling priority).
    locking.assign_fp_preemption_levels(ts)

    return ts

def merge_tasksets(partitions):
    if partitions == None:
        return None
    merged = TaskSystem()
    for p in partitions:
        for t in p:
            merged.append(t)
    return merged

def gen_mutex_taskset(conf, ucap, tasks):
    return gen_taskset(conf, ucap, tasks, False)

def run_range(conf, tests, vary_CS = False):
    import schedcat.model.serialize as ser
    
    all_zeros_count = [0 for _ in tests]
    if vary_CS:
        task_num = int(math.ceil((conf.num_cpus / conf.per_task_util) *.5))
        max_num_CS = 40
        steps = 10
        cs_nr_range = sorted(list(set([int(0 + n * (max_num_CS/steps)) for n in range(steps)])))
        for cs_nr in cs_nr_range:
            conf.max_requests = cs_nr
            total_ucap = conf.per_task_util * task_num
            samples = [[] for _ in tests]
            for _ in xrange(conf.samples):
                #try to generate taskset at most 5 times
                taskset = conf.make_taskset(total_ucap, task_num)
#                ser.write(taskset, "normal_taskset_%s.ts" % task_num)
                for i in xrange(len(tests)):
                    if taskset == None:
                        samples[i].append(False)
                    elif all_zeros_count[i] > 2:
                        samples[i].append(False)
                    # If the holistic MSRP analysis succeeded, we don't run the ILP-based analysis for MSRP or prio-FIFO
                    # WARNING! Indices of holistic and ILP-based analysis hard-coded, need to be adapted if experiments change!
                    elif (i == 3 or i == 9) and samples[2][-1] == True:
                        samples[i].append(True)
                    else:
                        samples[i].append(tests[i](taskset))
                
                # dump taskset to disk if MSRP-hol performs better than MSRP-LP
                if samples[2][-1] > samples[3][-1]:
                    ser.write(taskset, "strange_taskset.ts")
                    assert(False)
            row = []
            for i in xrange(len(tests)):
                avg = mean(samples[i])
                if avg > 0:
                    all_zeros_count[i]  = 0
                else:
                    all_zeros_count[i] += 1
                row += [avg]
            print row
            yield [cs_nr] + row
                             
    else:
        max_task_num = int(math.ceil(conf.num_cpus / conf.per_task_util))
        min_task_num = max(conf.min_tasks, conf.num_cpus) 
        steps = 20
        task_nr_range = sorted(list(set([int(min_task_num + n * ((max_task_num-min_task_num)/steps)) for n in range(steps)])))
        for tasks in task_nr_range:
            total_ucap = conf.per_task_util * tasks
            samples = [[] for _ in tests]
            for _ in xrange(conf.samples):
                #try to generate taskset at most 5 times
                taskset = conf.make_taskset(total_ucap, tasks)
 #               ser.write(taskset, "normal_taskset_%s.ts" % tasks)
                for i in xrange(len(tests)):
                    if taskset == None:
                        samples[i].append(False)
                    elif all_zeros_count[i] > 2:
                        samples[i].append(False)
                    # If the holistic MSRP analysis succeeded, we don't run the ILP-based analysis for MSRP or prio-FIFO
                    # WARNING! Indices of holistic and ILP-based analysis hard-coded, need to be adapted if experiments change!
                    elif (i == 3 or i == 9) and samples[2][-1] == True:
                        samples[i].append(True)
                    else:
                        samples[i].append(tests[i](taskset))
#                    print "test: %s" % tests[i]
#                print "samples[2][-1]=%s\tsamples[5][-1]=%s" % (samples[2][-1], samples[5][-1])
#                if samples[2][-1] and not samples[5][-1]:
#                    ser.write(taskset, "strange_taskset.ts")
#                    assert(False)
#                 dump taskset to disk if MSRP-hol performs better than MSRP-LP
                if samples[2][-1] > samples[3][-1]:
                    ser.write(taskset, "strange_taskset.ts")
                    assert(False)
            row = []
            for i in xrange(len(tests)):
                avg = mean(samples[i])
                if avg > 0:
                    all_zeros_count[i]  = 0
                else:
                    all_zeros_count[i] += 1
                row += [avg]
            print row
            yield [tasks] + row

def check_config(conf):
    # standard parameters
    conf.check('num_cpus',     type=int, min=1)
    conf.check('step_size',    type=float, default=0.25, min=0.1)
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(emstada.NAMED_PERIODS),   default='uni-moderate')

#    conf.generate = gen.DIST_BY_KEY[conf.deadlines][conf.periods][conf.utilizations]

def run_exp(conf, tests, titles, vary_CS_ = False):
    if vary_CS_:
        header = ['num_CS']
    else:
        header = ['tasks']
    for t in titles:
        header += [t]
    data = run_range(conf, tests, vary_CS = vary_CS_)
    write_data(conf.output, data, header)

def run_mutex_config(conf, vary_CS = False):
    check_config(conf)
    tests, titles = setup_mutex_tests(conf)
    conf.make_taskset = partial(gen_mutex_taskset, conf)
    run_exp(conf, tests, titles, vary_CS_ = vary_CS)


def generate_mutex_configs(options, vary_CS = False):
    for cpus in [4, 8, 16]:
#    for cpus in [8, 16]:
        for num_r_resources in [cpus // 2, cpus, 2 * cpus]: #[1, cpus // 4, cpus // 2, cpus, 2 * cpus]:
            for res_sharing_factor in [0.1, 0.25, 0.4, 0.75]:
                for per_task_util in [0.1, 0.2, 0.3]:
#                for per_task_util in [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]:
                    for cons_dl_fac in [1.0]:
                        for cs in CSLENGTH.keys():
                            if not vary_CS:
                                num_req_range = [1, 2, 5, 10, 15]
                            else:
                                num_req_range = [1]
                            for num_reqs in num_req_range:
                                name = 'sd_exp=spinlock-lp-mutex_ptu=%s_m=%02d_nr_r=%d_nreqs=%d_rsf=%.2f_csl=%s_cdlf=%.2f' \
                                            % (per_task_util, cpus, num_r_resources, num_reqs, res_sharing_factor, cs, cons_dl_fac)
                                c = Config()
                                c.num_cpus          = cpus
                                c.per_task_util     = per_task_util
                                c.res_sharing_factor= res_sharing_factor
                                c.num_r_resources   = num_r_resources
                                c.cslength          = cs
                                c.output_file       = name + '.csv'
                                c.samples           = 25
                                c.cons_dl_fac       = cons_dl_fac
                                if not vary_CS:
                                    c.max_requests      = num_reqs
                                yield (name, c)

EXPERIMENTS = {
    'spinlock-lp/mutex' : run_mutex_config,
    'spinlock-lp/mutex-vcs' : partial(run_mutex_config, vary_CS = True),
}

CONFIG_GENERATORS = {
    'spinlock-lp/mutex' : generate_mutex_configs,
    'spinlock-lp/mutex-vcs' : partial(generate_mutex_configs, vary_CS = True),
}

def example():
    from schedcat.model.tasks import SporadicTask, TaskSystem
    from schedcat.model.resources import initialize_resource_model
    import schedcat.model.serialize as ser
    import schedcat.locking.bounds as bounds
    
    ts = ser.load("taskset.ts")
    for t in ts:
        t.response_time = t.cost
    ts.sort_by_period()
    ts.assign_ids()
        
    rsi = bounds.get_cpp_model(ts)
    result_py = dict()
    result_cpp = dict()
    
    print ""
    for i, _ in enumerate(ts):
        if result_py[i] == result_cpp[i]:
            marker = ""
        else:
            marker = "X "*20
        print "task:\t%s\tresult_py:\t%s\tresult_cpp:\t%s\t%s" % (ts[i], result_py[i], result_cpp[i], marker)

def example_profile():
    import schedcat.model.serialize as ser
    
    ts = ser.load("normal_taskset_64_.ts")
    for t in ts:
        t.response_time = t.cost
        t.deadline = int(t.deadline * 0.9)
    fp_test(8, bound_response_times, msrp_bounds, ts, raise_locking_prio = False)
    
    fp_test(8, bound_response_times, msrp_bounds_bbb, ts, raise_locking_prio = False)
    
    fp_test(8, bound_response_times_lp, apply_msrp_bounds, ts, raise_locking_prio = False)
    
    return
    
    fp_test(8, 
            bound_response_times_lp,
            apply_preemptive_fifo_bounds,
            ts, 
            raise_locking_prio = False)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_unordered_spinlock_bounds, 
            ts, 
            raise_locking_prio = False)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_preemptive_unordered_spinlock_bounds, 
            ts, 
            raise_locking_prio = False)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_prio_spinlock_bounds, 
            ts, 
            raise_locking_prio = True)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_preemptive_prio_spinlock_bounds,
            ts, 
            raise_locking_prio = True)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_prio_fifo_spinlock_bounds, 
            ts, 
            raise_locking_prio = True)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_preemptive_prio_fifo_spinlock_bounds, 
            ts, 
            raise_locking_prio = True)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_baseline_bounds, 
            ts, 
            raise_locking_prio = False)
    
    fp_test(8, 
            bound_response_times_lp, 
            apply_dummy_bounds, 
            ts, 
            raise_locking_prio = False)

def get_min_slack(ts):
    slack = False
    for t in ts:
        if not slack:
            slack = t.deadline- t.response_time
        else:
            slack = min(slack, t.deadline- t.response_time)
    return slack

def example_debug():
    from schedcat.model.tasks import SporadicTask, TaskSystem
    import schedcat.model.serialize as ser
    import schedcat.locking.bounds as bounds
    import copy
    
    ts = ser.load("strange_taskset.ts")
#    for t in ts:
#        t.response_time = t.cost
#    ts.sort_by_period()
#    ts.assign_ids()
    
    ts1 = copy.deepcopy(ts)
    
#     while 
    
    while True:
        removable = []
        for t in ts1:
            ts2 = copy.deepcopy(ts1)
            ts2 = [tx for tx in ts2 if tx.id != t.id]
            schedulable_holistic = fp_test(16, bound_response_times, msrp_bounds_holistic, ts2)
            ts3 = copy.deepcopy(ts1)
            ts3 = [tx for tx in ts3 if tx.id != t.id] 
            schedulable_msrp_lp = fp_test(16, 
                                           bound_response_times_lp, 
                                           apply_msrp_bounds,
                                           ts3, 
                                           False)
            if schedulable_holistic and not schedulable_msrp_lp:
                print "I could remove task %s " % t
                removable += [t]
        
        if len(removable) < 1:
    #         write out reduced task set
            print "shrunk TS to size %s" % len(ts1)
            ser.write(TaskSystem(ts1), "strange_taskset_shrunk.ts")
            break
        tx = removable[0]
        ts1 = [ty for ty in ts1 if ty.id != tx.id]
        


def example_debug2():
    from schedcat.model.tasks import SporadicTask, TaskSystem
    import schedcat.model.serialize as ser
    import schedcat.locking.bounds as bounds
    import copy
    
    ts = ser.load("strange_taskset_shrunk2.ts")
    
    print "\nholistic:"
    ts1 = copy.deepcopy(ts)
    schedulable_holistic = fp_test(16, bound_response_times, msrp_bounds_holistic, ts1)
    
    print "\n\n\nLP:"
    ts2 = copy.deepcopy(ts)
    schedulable_msrp_lp = fp_test(16, bound_response_times_lp, apply_msrp_bounds, ts2, False)
    
def example_debug3():
    from schedcat.model.tasks import SporadicTask, TaskSystem
    import schedcat.model.serialize as ser
    import schedcat.locking.bounds as bounds
    import copy
    
    ts = ser.load("strange_taskset_shrunk.ts")
    
    ts1 = copy.deepcopy(ts)
    for i, t in enumerate(ts1):
        print "current task: %s" % t
        removable = []
        for res in t.resmodel:    
            ts2 = copy.deepcopy(ts1)
            del(ts2[i].resmodel[res])
            schedulable_holistic = fp_test(16, bound_response_times, msrp_bounds_holistic, ts2)
            ts3 = copy.deepcopy(ts1)
            print "resreq ts3[i].resmodel[res].max_writes: ", ts3[i].resmodel[res].max_writes 
            del(ts3[i].resmodel[res])
            print "resreq ts3[i].resmodel[res].max_writes: ", ts3[i].resmodel[res].max_writes 
            schedulable_msrp_lp = fp_test(16, 
                                           bound_response_times_lp, 
                                           apply_msrp_bounds,
                                           ts3, 
                                           False)
            if schedulable_holistic and not schedulable_msrp_lp:
                print "I could remove res %s from task %s's resmodel " % (res, t)
                removable += [res]
        
        if len(removable) > 0:
            res_ = removable[0]
            del(ts1[i].resmodel[res_])
    ser.write(TaskSystem(ts1), "strange_taskset_shrunk2.ts")    



def example_prio_spinlocks():
    from schedcat.model.tasks import SporadicTask, TaskSystem
    from schedcat.model.resources import initialize_resource_model
    
    t1 = SporadicTask(1, 10, 10, 0)
    t1.partition = 0
    t2 = SporadicTask(10, 100, 100, 1)
    t2.partition = 1
    t3 = SporadicTask(10, 100, 100, 2)
    t3.partition = 2
    t4 = SporadicTask(10, 100, 100, 3)
    t4.partition = 3
    ts = TaskSystem([t1, t2, t3, t4])
    initialize_resource_model(ts)
    t1.resmodel[0] = ResourceRequirement(0, 1, 1, 0, 0, len(ts))
    t2.resmodel[0] = ResourceRequirement(0, 2, 4, 0, 0, len(ts))
    t3.resmodel[0] = ResourceRequirement(0, 2, 4, 0, 0, len(ts))
    t4.resmodel[0] = ResourceRequirement(0, 2, 4, 0, 0, len(ts))
    test_msrp = fp_test(4, 
                        partial(bound_response_times_lp, locking.apply_cpp_lp_msrp_bounds_single), 
                        apply_msrp_bounds, 
                        ts,
                        raise_locking_prio = False)
    print "\nschedulable with MSRP: %s\n" % test_msrp
    
    test_prio = fp_test(4, 
                        partial(bound_response_times_lp, locking.apply_cpp_lp_prio_bounds_single), 
                        apply_prio_spinlock_bounds, 
                        ts,
                        raise_locking_prio = True)
    print "\nschedulable with prio: %s\n" % test_prio

    test_prio_fifo = fp_test(4, 
                             partial(bound_response_times_lp, locking.apply_cpp_lp_prio_fifo_bounds_single), 
                             apply_prio_fifo_spinlock_bounds, 
                             ts,
                             raise_locking_prio = True)
    print "\nschedulable with prio-fifo: %s\n" % test_prio_fifo




if __name__ == '__main__':
    example_debug2()
