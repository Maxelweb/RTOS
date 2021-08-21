from __future__ import division

from fractions import Fraction

import random
import sys
import os
import time

import operator

import random
import numpy

from datetime import datetime

from functools import partial
from itertools import izip
from collections import defaultdict
from math import ceil, floor

from toolbox.datafile import parse_file_name
from toolbox.io import one_of, Config, write_data, boolean_flag, format_data
from toolbox.sample import value_range, round_to_next_multiple
from toolbox.stats import mean, median, stdev
from toolbox import select_min
import toolbox.bootstrap as boot

from schedcat.generator.tasks import uniform_choice
import schedcat.generator.tasksets as gen
import schedcat.generator.generator_emstada as emstada
import schedcat.mapping.binpack as bp
import schedcat.sched.edf as edf
import schedcat.sched as sched
import schedcat.sched.run as run

import schedcat.mapping.apa as apa

from schedcat.mapping.apa import double_wfd_split, edf_assign_wfd_wfd_split, \
    edf_assign_ffd_wfd_split, edf_assign_wfd_ffd_split, \
    edf_assign_ffd_ffd_split, meta_preassign_failures, is_feasible_pt

from schedcat.mapping.apa import meta_reduce_periods as _meta_reduce_periods

from schedcat.model.tasks import SporadicTask, TaskSystem
from schedcat.model.serialize import write as write_taskset, load as load_taskset
from schedcat.util.time import ms2us

WANT_CI = False

# some periods with manageable hyperperiod
PERIODS    = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000]
PERIODS_US = [ms2us(p) for p in PERIODS]

def edf_assign_unsplit_feas(taskset, sol):
    mapping = defaultdict(TaskSystem)
    unassigned = []

    for i, t in enumerate(taskset):
        allocs = [(cpu, sol.get_fraction(i, cpu))
                  for cpu in t.affinity
                  if sol.get_fraction(i, cpu) >= 1.0 / t.period]
        if len(allocs) == 1:
            (cpu, _) = allocs[0]
            mapping[cpu].append(t)
        else:
            unassigned.append(t)
    return (unassigned, mapping)


def edf_split_feas_wm(taskset, sol=None):
    if not sol:
        aff = sched.get_native_affinities(taskset)
        ts  = sched.get_native_taskset(taskset)
        sol = sched.native.apa_implicit_deadline_feasible(ts, aff)

    if not sol:
        return (set(taskset), {})

    (unassigned, mapping) = edf_assign_unsplit_feas(taskset, sol)

    failed = []
    for t in unassigned:
        i = taskset.index(t)
        allocs = [(cpu, sol.get_fraction(i, cpu))
                  for cpu in t.affinity
                  if sol.get_fraction(i, cpu) >= 1.0 / t.period]

        # split task, need to create subtasks
        # We apply a EDF-WM-like heuristic => create proportional subtasks
        total_e = 0
        total_d = 0
        while len(allocs) > 1:
            (cpu, frac) = allocs.pop()
            sub_deadline = int(min(t.deadline, t.period) * frac)
            sub_exec     = int(t.cost * frac)
            sub_tsk = SporadicTask(sub_exec, t.period, deadline=sub_deadline, id=t.id)
            sub_tsk.affinity = set(t.affinity)
            total_e += sub_exec
            total_d += sub_deadline
            mapping[cpu].append(sub_tsk)
            if not apa.qpa_it_fits(mapping[cpu]):
                # uh-oh, infeasible
                mapping[cpu].pop()
                failed.append(sub_tsk)

        # treat last one specially to catch any rounding issues
        (cpu, _) = allocs[0]
        sub_deadline = t.deadline - total_d
        sub_exec     = t.cost - total_e
        sub_tsk = SporadicTask(sub_exec, t.period, deadline=sub_deadline, id=t.id)
        sub_tsk.affinity = set(t.affinity)
        mapping[cpu].append(sub_tsk)
        if not apa.qpa_it_fits(mapping[cpu]):
            # uh-oh, infeasible
            mapping[cpu].pop()
            failed.append(sub_tsk)


    return (failed, mapping)

def edf_split_feas_c_equal_d(taskset, sol=None):
    if not sol:
        aff = sched.get_native_affinities(taskset)
        ts  = sched.get_native_taskset(taskset)
        sol = sched.native.apa_implicit_deadline_feasible(ts, aff)

    if not sol:
        return (set(taskset), {})

    (unassigned, mapping) = edf_assign_unsplit_feas(taskset, sol)

    if unassigned:
        # try C=D for the rest
        return double_wfd_split(
                    unassigned, with_splits=True, pre_assigned=mapping)
    else:
        return (unassigned, mapping)

def edf_prop_split_feas_c_equal_d(taskset):
    aff = sched.get_native_affinities(taskset)
    ts  = sched.get_native_taskset(taskset)

    sol = sched.native.apa_implicit_deadline_feasible(ts, aff)
    if sol:
        (unassigned, mapping) = edf_split_feas_c_equal_d(taskset, sol)
        return not unassigned
    else:
        return False

def edf_prop_split_feas_wm(taskset):
    aff = sched.get_native_affinities(taskset)
    ts  = sched.get_native_taskset(taskset)

    sol = sched.native.apa_implicit_deadline_feasible(ts, aff)
    if sol:
        (unassigned, mapping) = edf_split_feas_wm(taskset, sol)
        return not unassigned
    else:
        return False

def qps_ffd(num_cpus, taskset):
    bins = [TaskSystem() for _ in xrange(num_cpus)]

    # in order of decreasing utilization
    for t in sorted(taskset, key=lambda t: t.utilization(), reverse=True):
        # find first bin into which it fits
        part = None
        for b in bins:
            if b.utilization() + t.utilization() <= 1:
                part = b
                break
        if part is None:
            # Ok, did not fit anywhere, place in bin with largest remaining
            # capacity.
            part = select_min(bins, key=lambda b: b.utilization())
        part.append(t)

    return bins

def meta_period_transform(max_subtasks, heuristic, taskset):
    to_split_idx = set()
    max_feas = {}

    for num_subtasks in range(1, max_subtasks + 1):
        infeasible = True
        while infeasible:
            infeasible = False
            ts = taskset.copy()
            for i in to_split_idx:
                x = num_subtasks if not i in max_feas else max_feas[i]
                if x > 0:
#                    print 'Transforming %s/%d ->' % (ts[i], x),
                    ts[i].period_transform(x)
#                    print '%s, feas=%s' % (ts[i], ts[i].is_feasible())
                if not ts[i].is_feasible():
                    # this one cannot be split any longer, remove
                    max_feas[i] = x - 1
                    infeasible = True

        (unassigned, mapping) = heuristic(ts)

        if not unassigned:
#             print num_subtasks, len(to_split_idx)
#             for i in to_split_idx:
#                 print ts[i], 'split', num_subtasks if not i in max_feas else max_feas[i], taskset[i]
            return (unassigned, mapping)
        else:
            for t in unassigned:
                # assumes ID corresponds to index
                to_split_idx.add(t.id - 1)

    return (unassigned, mapping)

def max_transform_param(task, threshold):
    pt = task.period // threshold
    while pt > 1 and not is_feasible_pt(task, pt):
        pt -= 1
    return pt

def meta_transform_min_period(unused, heuristic, taskset, threshold=2000,
                              minimize_preemptions=False):
    ts = taskset.copy()
    (unassigned, mapping) = heuristic(ts)

    if not unassigned:
        return (unassigned, mapping)

    pt = {}

    def try_with_pt():
        ts = taskset.copy()
        for t in ts:
            if t.id in pt:
#                 print 'Transforming %s/%d ->' % (t, pt[t.id]),
                t.period_transform(pt[t.id])
#                 print '%s, feas=%s' % (t, t.is_feasible())
#         print 'Hyper', ts.hyperperiod()
        return heuristic(ts)

    # initial check --- transform maximally
    while True:
#         print '=' * 80
        new = False
        for t in unassigned:
            if not t.id in pt:
#                 print 'new', t
                pt[t.id] = max_transform_param(t, threshold)
                new = True

        if not new and unassigned:
            # give up
#             print 'giving up'
            return (unassigned, mapping)

        (unassigned, mapping) = try_with_pt()

        if not unassigned:
            break

    if not minimize_preemptions:
        return (unassigned, mapping)

    # found a solution, try to scale up

    to_try = set(pt.iterkeys())
    upper = {}
    for i in to_try:
        upper[i] = 0

    candidate = True
    while candidate:
#         print '#' * 80
        candidate = False
        for i in sorted(to_try, key=lambda i: pt[i], reverse=True):
#             print '>', i, '>' * 10, pt[i], upper[i],
            if pt[i] > 1 + upper[i]:
                x = pt[i]
                pt[i] = upper[i] + (x - upper[i]) // 2
                candidate = x != pt[i]
                (unassigned, mapping) = try_with_pt()
                if unassigned:
#                     print pt[i], 'failed.'
                    upper[i] = pt[i]
                    pt[i] = x

    return try_with_pt()


def meta_transform_min_period_orig(max_subtasks, heuristic, taskset, threshold=2000):
    to_split_idx = set()
    max_feas = {}
    num_subtasks = 0

    while not to_split_idx or len(max_feas) < len(to_split_idx):
        num_subtasks += 1
        infeasible = True
        while infeasible:
#             print '---' * 10
            infeasible = False
            ts = taskset.copy()
            for i in to_split_idx:
                x = num_subtasks if not i in max_feas else max_feas[i]
                if x > 0:
#                     print 'Transforming %s/%d ->' % (ts[i], x),
                    ts[i].period_transform(x)
#                     print '%s, feas=%s' % (ts[i], ts[i].is_feasible())
                    if not ts[i].is_feasible() or \
                        (ts[i].period < threshold and not i in max_feas):
                        # this one cannot be split any longer, remove
                        max_feas[i] = x - 1
                        infeasible = True

        (unassigned, mapping) = heuristic(ts)

        if not unassigned:
#             print num_subtasks, len(to_split_idx)
#             for i in to_split_idx:
#                 print 'MIN', ts[i], 'split', num_subtasks if not i in max_feas else max_feas[i], taskset[i]
            return (unassigned, mapping)
        else:
            for t in unassigned:
                # assumes ID corresponds to index
                to_split_idx.add(t.id - 1)

    return (unassigned, mapping)

def meta_slice_all_tasks(max_subtasks, heuristic, taskset):
    max_feas = {}
    for num_subtasks in range(1, max_subtasks + 1):
        infeasible = True
        while infeasible:
            infeasible = False
            ts = taskset.copy()
            for i in range(len(ts)):
                x = num_subtasks if not i in max_feas else max_feas[i]
                if x > 0:
#                    print 'Transforming %s/%d ->' % (ts[i], x),
                    ts[i].period_transform(x)
#                    print '%s, feas=%s' % (ts[i], ts[i].is_feasible())
                if not ts[i].is_feasible():
                    # this one cannot be split any longer, remove
                    max_feas[i] = x - 1
                    infeasible = True

        (unassigned, mapping) = heuristic(ts)

        if not unassigned:
            break

    return (unassigned, mapping)

def meta_reduce_periods(heuristic, taskset, threshold=2000, reduce_all=False):
    return _meta_reduce_periods(heuristic, taskset, threshold=threshold,
                candidate_periods=PERIODS_US, reduce_all=reduce_all)


def sanity_check(taskset, unassigned, mapping):
    by_id = defaultdict(list)
    for t in unassigned:
        by_id[t.id].append(t)
    for c in mapping:
        for t in mapping[c]:
            by_id[t.id].append(t)

    for t in taskset:
        # make sure we didn't lose any tasks
        assert t.id in by_id
        # make sure we have sufficient utilization
        usum = sum((x.utilization_q() for x in by_id[t.id]))
        assert t.utilization_q() <= usum
        # make sure periods match update
        first = by_id[t.id][0]
        assert t.period >= first.period
        assert t.period % first.period == 0
        for x in by_id[t.id][1:]:
            assert x.period == first.period
        # make sure relative deadlines are feasible
        for x in by_id[t.id]:
            assert x.deadline >= x.cost
        # make sure relative deadlines add up
        dls = sum((x.deadline for x in by_id[t.id]))
        assert dls <= first.period

def heuristic_wrapper(heuristic, taskset, *args, **kargs):
    (unassigned, assignment) = heuristic(taskset, *args, **kargs)
    sanity_check(taskset, unassigned, assignment)
    if unassigned:
        return False
    else:
        return True

def apa_feasibility_test(taskset):
    aff = sched.get_native_affinities(taskset)
    ts  = sched.get_native_taskset(taskset)

    sol = sched.native.apa_implicit_deadline_feasible(ts, aff)
    return not sol is None

def pfair_test(num_cpus, qsize, taskset):
    ts = taskset.copy()
    for t in ts:
        t.cost = int(ceil(t.cost / qsize) * qsize)
    return ts.utilization() <= num_cpus

def qps_test(num_cpus, taskset):
    return taskset.utilization() <= num_cpus


PARTITIONING_HEURISTICS = {
     'WFD' : apa.edf_worst_fit_decreasing_difficulty,
     'FFD' : apa.edf_first_fit_decreasing_difficulty,
}

SEMI_PARTITIONING_HEURISTICS = {
    'WFD-C=D' : partial(double_wfd_split, with_splits=True),
    'FFD-C=D' : partial(apa.edf_first_fit_decreasing_difficulty, with_splits=True),

    'WWFD' : edf_assign_wfd_wfd_split,
    'FWFD' : edf_assign_ffd_wfd_split,
    'WFFD' : edf_assign_wfd_ffd_split,
    'FFFD' : edf_assign_ffd_ffd_split,
}

PAF_SEMI_PARTITIONING_HEURISTICS = {}

def with_paf(h, taskset, *args, **kargs):
    return meta_preassign_failures(
                partial(h, *args, **kargs),
                partial(double_wfd_split, *args, with_splits=True, **kargs),
                taskset)

for label in SEMI_PARTITIONING_HEURISTICS:
    PAF_SEMI_PARTITIONING_HEURISTICS[label + '-PAF'] = \
        partial(with_paf, SEMI_PARTITIONING_HEURISTICS[label])

def with_pt(h, taskset, *args, **kargs):
    return meta_transform_min_period(0, partial(h, *args, **kargs), taskset)

PT_SEMI_PARTITIONING_HEURISTICS = {
    'WFD-C=D-PT' : partial(with_pt, SEMI_PARTITIONING_HEURISTICS['WFD-C=D']),
    'FWFD-PT'    : partial(with_pt, SEMI_PARTITIONING_HEURISTICS['FWFD']),
    'WWFD-PT'    : partial(with_pt, SEMI_PARTITIONING_HEURISTICS['WWFD']),
#     'FWFD-PT-PAS' : partial(with_pt, partial(edf_assign_ffd_wfd_split, pre_assign_small=True)),
#     'WWFD-PT-PAS' : partial(with_pt, partial(edf_assign_wfd_wfd_split, pre_assign_small=True)),
}

def with_rp(h, taskset, *args, **kargs):
    return meta_reduce_periods(partial(h, *args, **kargs), taskset)

RP_SEMI_PARTITIONING_HEURISTICS = {
    'WFD-C=D-RP' : partial(with_rp, SEMI_PARTITIONING_HEURISTICS['WFD-C=D']),
    'FWFD-RP'    : partial(with_rp, SEMI_PARTITIONING_HEURISTICS['FWFD']),
    'WWFD-RP'    : partial(with_rp, SEMI_PARTITIONING_HEURISTICS['WWFD']),
}

ALL_GROUPS = {
    'PART' :     PARTITIONING_HEURISTICS,
    'SEMI' :     SEMI_PARTITIONING_HEURISTICS,
    'SEMI/PAF' : PAF_SEMI_PARTITIONING_HEURISTICS,
#     'SEMI/PT'  : PT_SEMI_PARTITIONING_HEURISTICS,
    'SEMI/RP'  : RP_SEMI_PARTITIONING_HEURISTICS,
}

ORDERED_GROUPS = ['PART', 'SEMI', 'SEMI/PAF', 'SEMI/RP']

def apply_heuristics(heuristics, taskset, *args, **kargs):
    (best_u, best_m) = (taskset, {})
    for l in heuristics:
        h = heuristics[l]
        start = datetime.now()
#         print l, 'start',
        (unassigned, mapping) = h(taskset, *args, **kargs)
#         print l, 'end', datetime.now()
        end = datetime.now()
        if (end - start).total_seconds() > 10:
            print 'Slow!', l, end - start
#             write_taskset(taskset, 'slow-%s-%05d-%3d-%x.ts' % \
#                 (l, (end - start).total_seconds() * 100, taskset.utilization() * 100, abs(hash(str(taskset)))))
        # did we manage to assign all?
        if not unassigned:
            return (unassigned, mapping)
        # did we at least manage to assign more?
        if len(unassigned) < len(best_u):
            (best_u, best_m) = (unassigned, mapping)

    # return "closest" solution
    return (best_u, best_m)

def partition(taskset, **kargs):
    return apply_heuristics(PARTITIONING_HEURISTICS, taskset, **kargs)

def semi_partition(taskset, **kargs):
    return apply_heuristics(SEMI_PARTITIONING_HEURISTICS, taskset, **kargs)

def semi_partition_paf(taskset, **kargs):
    return apply_heuristics(PAF_SEMI_PARTITIONING_HEURISTICS, taskset, **kargs)

def semi_partition_pt(taskset, **kargs):
    return apply_heuristics(PT_SEMI_PARTITIONING_HEURISTICS, taskset, **kargs)

def semi_partition_rp(taskset, **kargs):
    return apply_heuristics(RP_SEMI_PARTITIONING_HEURISTICS, taskset, **kargs)

def setup_tests_common(conf, test_wrapper):
    tests = []
    titles = []

    for g in ORDERED_GROUPS:
        group = ALL_GROUPS[g]

        if conf.want_individual_results:
            for l in sorted(group.iterkeys()):
                h = group[l]
                titles += [l]
                tests  += [partial(test_wrapper,
                                   h, min_chunk_size=conf.min_slice_size)]
            titles += [g]
            tests  += [tuple(group.iterkeys())]
        else:
            titles += [g]
            tests  += [partial(test_wrapper, partial(apply_heuristics, group),
                               min_chunk_size=conf.min_slice_size)]

    titles += ['ANY']
    tests  += [tuple(ALL_GROUPS.iterkeys())]

    for i, test in enumerate(tests):
        if type(test) == tuple:
            # ANY test
            cols = [titles.index(col_name) for col_name in test]
            for c in cols:
                assert c < i
            tests[i] = tuple(cols)

    return tests, titles


def setup_tests(conf):
    tests, titles = setup_tests_common(conf, heuristic_wrapper)

    tests += [
        partial(pfair_test, conf.num_cpus, 1000),
        partial(pfair_test, conf.num_cpus, 500),
        partial(pfair_test, conf.num_cpus, 100),

        partial(qps_test, conf.num_cpus),
    ]
    titles += [
        'PFAIR-1000',
        'PFAIR-500',
        'PFAIR-100',
        'QPS',
    ]
    return tests, titles

def component_wise_max_diff(x, y):
    diff = 0
    for (a, b) in izip(x, y):
        diff = max(diff, abs(a - b))
    return diff



sample_worker   = None
sample_start    = None
sample_last_msg = None
sample_count    = 0

def init_worker():
    random.seed()
    numpy.random.seed()

def invoke_worker(n):
    return sample_worker(n)

from multiprocessing import Pool as process_pool, cpu_count

def run_tests_p(conf, tests, sample_point, want_raw=False, aggregate=any):
    global sample_start
    sample_start = time.time()
    sys.stdout.flush()

    input = range(conf.samples)

    threads = min(conf.samples, 2 * cpu_count())
    if conf.samples % threads:
        per_thread = conf.samples // cpu_count() + 1
    else:
        per_thread = conf.samples // cpu_count()

    def worker(n):
        global sample_last_msg
        global sample_count
        if sample_last_msg is None:
            sample_last_msg = sample_start
        if time.time() - sample_last_msg > 10:
            print '[%d] (%03d/%03d) at %s; cost=%.3fs/sample (%d/%d on this thread)' \
                % (os.getpid(), n, conf.samples, datetime.now(),
                (time.time() - sample_start) / sample_count, sample_count, per_thread)
            sys.stdout.flush()
            sample_last_msg = time.time()
        taskset = conf.make_taskset(sample_point)
        sample = []
        for i in xrange(len(tests)):
            if type(tests[i]) == tuple:
                # ANY test => column indices
                sample.append(aggregate((sample[col] for col in tests[i])))
            else:
                sample.append(tests[i](taskset))
        sample_count += 1
        return sample

    global sample_worker
    sample_worker = worker

    print sample_point, '[parallel] %d threads, %d samples per thread' % \
        (threads, per_thread)

    p = process_pool(threads, init_worker)
    samples = p.map(invoke_worker, input, 1)
    p.close()
    p.join()
    del p

    if want_raw:
        num_cols = len(tests)
        by_test = [[] for _ in xrange(num_cols)]
        for s in samples:
            for i in xrange(num_cols):
                by_test[i].append(s[i])
        return by_test
    else:
        sums = [0] * len(tests)
        for s in samples:
            for i in xrange(len(s)):
                sums[i] += s[i]
        row = [s/conf.samples for s in sums]

        print sample_point, row
        return row

def run_tests_s(conf, tests, sample_point, want_raw=False, aggregate=any):
    last_msg = time.time()
    start = last_msg
    print sample_point,
    sys.stdout.flush()
    samples = [[] for _ in tests]
    for n in xrange(conf.samples):
        if time.time() - last_msg > 10:
            print '(%d/%d) at %s; cost=%.3fs/sample' % (n, conf.samples,
                datetime.now(), (time.time() - start) / n)
            print sample_point,
            last_msg = time.time()
        taskset = conf.make_taskset(sample_point)
        for i in xrange(len(tests)):
            if type(tests[i]) == tuple:
                # ANY test => column indices
                samples[i].append(aggregate((samples[col][-1] for col in tests[i])))
            else:
                samples[i].append(tests[i](taskset))
        if conf.collect_feasible_failures and samples[-1][-1] != samples[-2][-1]:
            # Feasible, but all heuristics failed.
            write_taskset(taskset, 'failed-%3d-%x.ts' % (taskset.utilization() * 100, abs(hash(str(taskset)))))

    if want_raw:
        return samples
    else:
        if WANT_CI:
            row = []
            for i in xrange(len(tests)):
                val = mean(samples[i])
                if WANT_CI:
                    ci  = boot.confidence_interval(samples[i], stat=mean)
                    row += [ci[0], val, ci[1]]
                else:
                    row += [val]
        else:
            row = [mean(col) for col in samples]
        print row
        return row

def run_tests(conf, *args, **kargs):
    if conf.parallel_sampling:
        return run_tests_p(conf, *args, **kargs)
    else:
        return run_tests_s(conf, *args, **kargs)

def run_range(conf, tests, diff_end=-1, reverse=False, early_stop=False, adaptive=True):
    if adaptive:
        results = []
        results.append((1, run_tests(conf, tests, 1)))
        results.append((conf.num_cpus, run_tests(conf, tests, conf.num_cpus)))
        intervals = [(0, 1)]

        count = 0

        def ispace(ival):
            (x, y) = ival
            return results[y][0] - results[x][0]

        epsilon = 0.0001

        def too_close(x, y):
            (x_ucap, x_row) = results[x]
            (y_ucap, y_row) = results[y]
            return y_ucap - x_ucap < 2 * conf.adaptive_min_delta - epsilon

        def push(x, y):
            if not too_close(x, y):
                intervals.append((x, y))

        while intervals:
            count += 1
            print '#', count, '#' * 75
            intervals.sort(key=ispace)
            print 'I:', [(results[x][0], results[y][0]) for (x, y) in intervals]
            print 'Remaining space: %.3f' % sum((ispace(ival) for ival in intervals))
            (x, y) = intervals.pop()
            (x_ucap, x_row) = results[x]
            (y_ucap, y_row) = results[y]
            print '=> checking [%.3f, %.3f]' % (x_ucap, y_ucap)

            if too_close(x, y):
                print '=> skipped due to min delta'
            elif (y_ucap - x_ucap > conf.adaptive_max_delta
                  or component_wise_max_diff(x_row[:diff_end], y_row[:diff_end])
                     > conf.adaptive_threshold):

                print x_row, 'diff', y_row, '->', component_wise_max_diff(x_row[:diff_end], y_row[:diff_end])
                middle = (y_ucap - x_ucap) / 2.0 + x_ucap
                middle = round_to_next_multiple(middle, conf.adaptive_min_delta)
                results.append((middle, run_tests(conf, tests, middle)))
                z = len(results) - 1
#                if (y_ucap - x_ucap) / 2.0 > conf.adaptive_min_delta:
#                    print 'x =', x_ucap, 'middle =', middle, 'y = ', y_ucap
                push(x, z)
                push(z, y)
            else:
                print '=> ok'

        for (ucap, row) in sorted(results):
            yield [ucap, ucap/conf.num_cpus * 100.0] + row

    else:
        range = value_range(1, conf.num_cpus - 0.005, conf.step_size, last = conf.num_cpus - 0.005)

        if reverse:
            range = list(range)
            range.reverse()

        for ucap in range:
            row  = run_tests(conf, tests, ucap)
            yield [ucap, ucap/conf.num_cpus * 100.0] + row
            if early_stop:
                vals = set(row)
                if  1 in vals and len(vals) == 1:
                    break

# ##############################################################################

def normalized_job_release_rate(task):
    # jobs per second
    frequency = 1 / (task.period / 1E6)
    return frequency

def preemption_rate(task):
    # JLFP: two context switches per job
    return 2 * normalized_job_release_rate(task)

def pfair_rate(task, qsize):
    quanta = ceil(task.cost / qsize)
    # quantum-based: context switch at each quantum boundary
    return quanta / (task.period / 1E6)

def total_preemption_rate(assignment):
    all_freq = []
    for core in assignment:
        for t in assignment[core]:
            all_freq.append(preemption_rate(t))
    return sum(all_freq)

def total_pfair_rate(qsize, taskset):
    all_freq = []
    for t in taskset:
        all_freq.append(pfair_rate(t, qsize))
    return sum(all_freq)

def mp_apa_assignment(heuristic, taskset):
    (unassigned, assignment) = heuristic(taskset)
    if unassigned:
        return None
    else:
        return total_preemption_rate(assignment)

def mp_heuristic(heuristic, *args, **kargs):
    (unassigned, assignment) = heuristic(*args, **kargs)
    if unassigned:
        return None
    else:
        return total_preemption_rate(assignment)

def qps_preemption_rate(num_cpus, taskset):
    if taskset.utilization() > num_cpus:
        return None
    bins = qps_ffd(num_cpus, taskset)
    total_rate = 0
    for b in bins:
        if b.utilization() > 1:
            # major set => each task deadline triggers four extra server
            # activations, which in turn may cause context switches.
            # We multiply here by 5 to account for the task itself and
            # and the four server switches.
            for t in b:
                total_rate += preemption_rate(t) * 5
        else:
            # minor set => regular EDF
            for t in b:
                total_rate += preemption_rate(t)

    return total_rate

def run_preemption_rate(taskset):
    # First, determine the max number of preemptions per job release
    # based on the reduction tree.
    task_servers = [run.Server(Fraction(t.cost, t.period)) for t in taskset]
    final, levels = run.reduce(task_servers)
    preempt_per_release = run.max_number_of_preemptions_per_job_release(levels)

    # Next, sum up the job releases for all tasks.
    total_rate = 0
    for t in taskset:
        # two context switches per preemption
        total_rate += normalized_job_release_rate(t) * 2 * preempt_per_release

    return total_rate

def mp_setup_tests(conf):
    tests, titles = setup_tests_common(conf, mp_heuristic)

    tests += [
        partial(total_pfair_rate, 1000),
        partial(total_pfair_rate, 500),
        partial(total_pfair_rate, 100),

        partial(qps_preemption_rate, conf.num_cpus),
        run_preemption_rate,
    ]

    titles += [
        'PFAIR-1000',
        'PFAIR-500',
        'PFAIR-100',
        'QPS',
        'RUN',
    ]

    full_titles = []
    for t in titles:
        full_titles += ['%s (AVG/core)' % t, '%s (AVG)' % t,
                        '%s (MED)' % t, '%s (STD)' % t, '%s (MAX)' % t]

    return tests, full_titles

def min_ignoring_None(iterable):
    ok = [x for x in iterable if not x is None]
    if ok:
        return min(ok)
    else:
        return None

def measure_preemptions_for_ucap(conf, tests, ucap):


    samples = run_tests(conf, tests, ucap, want_raw=True,
                        aggregate=min_ignoring_None)

    row = []
    for i in xrange(len(tests)):
        valid = [x for x in samples[i] if not x is None]
        if valid:
            row += [mean(valid)/conf.num_cpus, mean(valid), median(valid), stdev(valid), max(valid)]
        else:
            row += [' ', ' ', ' ', ' ', ' ']
    return row

def measure_preemptions(conf, tests, reverse=False):
    vals = [x/100.0 * conf.num_cpus for x in range(75, 100)]
    vals.append(conf.num_cpus * 0.9995)

    if reverse:
        vals.reverse()

    for ucap in vals:
        row = measure_preemptions_for_ucap(conf, tests, ucap)
        yield [ucap, ucap/conf.num_cpus * 100.0] + row



def common_checks(conf):
    conf.check('num_cpus',     type=int)
    conf.all_cpus = frozenset(range(1, conf.num_cpus + 1))
    conf.check('affinity_prob', type=float, default=1.0)
    conf.check('max_job_slices', type=int, default=4)
    conf.check('min_slice_size', type=int, default=0)
    conf.check('min_transformed_period', type=int, default=4000)
    conf.check('samples',      type=int, default=256)
    conf.check('step_size',    type=float, default=0.25 if conf.num_cpus < 16 else 1)
    conf.check('want_adaptive', type=boolean_flag, default=True)
    conf.check('adaptive_threshold', type=float, default=0.05)
    conf.check('adaptive_min_delta', type=float, default=0.125)
    conf.check('adaptive_max_delta', type=float, default=conf.num_cpus)
    conf.check('collect_feasible_failures', type=boolean_flag, default=False)
    conf.check('want_feasibility_based_tests', type=boolean_flag, default=False)
    conf.check('want_individual_results', type=boolean_flag, default=False)
    conf.check('parallel_sampling', type=boolean_flag, default=True)

    # convert from percentage to actual utilization difference
    conf.adaptive_min_delta /= 100
    conf.adaptive_min_delta *= conf.num_cpus

def draw_random_affinity(conf):
    affinity = set()
    for cpu in xrange(1, conf.num_cpus + 1):
        if random.random() <= conf.affinity_prob:
            affinity.add(cpu)
    if not affinity:
        # at least one CPU
        affinity.add(random.randint(1, conf.num_cpus))
    return affinity

def assign_random_affinities(conf, taskset):
    for t in taskset:
        t.affinity = draw_random_affinity(conf)

def assign_global_affinities(conf, taskset):
    for t in taskset:
        t.affinity = conf.all_cpus

def assign_affinities(conf, taskset):
    if conf.affinity_prob == 1.0:
        assign_global_affinities(conf, taskset)
    else:
        assign_random_affinities(conf, taskset)

def generate_unc_style(conf, unc_gen, *args, **kargs):
    taskset = unc_gen(*args, **kargs)
    assign_affinities(conf, taskset)
    taskset.assign_ids()
    return taskset

UNC_PERIODS = {
    'hyper1000' : uniform_choice(PERIODS),
}

UNC_PERIODS.update(gen.NAMED_PERIODS)

def run_unc_config(conf):
    common_checks(conf)
    conf.check('utilizations', type=one_of(gen.NAMED_UTILIZATIONS))
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(UNC_PERIODS),   default='hyper1000')

    unc_gen = gen.mkgen(
        gen.NAMED_UTILIZATIONS[conf.utilizations],
        UNC_PERIODS[conf.periods],
        gen.NAMED_DEADLINES[conf.deadlines])
    conf.make_taskset = lambda ucap: generate_unc_style(conf, unc_gen,
                            max_util=ucap, time_conversion=ms2us, squeeze=True)

    tests, titles = setup_tests(conf)
    header = ['UCAP', 'R-UCAP']
    for t in titles:
        if WANT_CI:
            header += ['%s CI-' % t, t, '%s CI+' % t]
        else:
            header += [t]

    header += ['TCOUNT']

    data = run_range(conf, tests + [lambda ts: len(ts)], diff_end=-2)
    data = format_data(data)
    write_data(conf.output, data, header)

def run_task_range(conf, tests, should_stop, threshold=3):
    num_tasks = conf.num_cpus
    stop_count = 0

    while True:
        num_tasks += 1
        row = [num_tasks] + run_tests(conf, tests, num_tasks)
        yield row
        if should_stop(row):
            stop_count += 1
        else:
            stop_count = 0
        if stop_count >= threshold:
            break

def run_unc_tcount_config(conf):
    common_checks(conf)
    conf.check('utilizations', type=one_of(gen.NAMED_UTILIZATIONS))
    conf.check('deadlines',    type=one_of(gen.NAMED_DEADLINES), default='implicit')
    conf.check('periods',      type=one_of(UNC_PERIODS),   default='hyper1000')

    unc_gen = gen.mkgen(
        gen.NAMED_UTILIZATIONS[conf.utilizations],
        UNC_PERIODS[conf.periods],
        gen.NAMED_DEADLINES[conf.deadlines])
    conf.make_taskset = lambda tcount: generate_unc_style(conf, unc_gen,
                            max_tasks=tcount, time_conversion=ms2us, squeeze=True)

    tests, titles = setup_tests(conf)
    header = ['TCOUNT']
    for t in titles:
        if WANT_CI:
            header += ['%s CI-' % t, t, '%s CI+' % t]
        else:
            header += [t]

    header += ['UTIL', 'R-UTIL']

    tests += [
        lambda ts: ts.utilization(),
        lambda ts: ts.utilization() / conf.num_cpus,
    ]

    data = run_task_range(conf, tests, lambda row: row[-3] == 0, threshold=2)
    data = format_data(data)
    write_data(conf.output, data, header)


EMSTADA_PERIODS = {
    'hyper1000' : PERIODS
}

def generate_emstada(conf, max_util=1, time_conversion=ms2us, squeeze='ignored'):
    if conf.periods_dist in EMSTADA_PERIODS:
        per = EMSTADA_PERIODS[conf.periods_dist]
    else:
        per = conf.periods_dist

    taskset = emstada.gen_taskset(conf.periods, per,
                               conf.task_count, max_util, period_granularity=1,
                               want_integral=True, scale=ms2us)

    # randomly drop individual cycles until we are below the max_util
    # This is required to avoid over-utilizing the system due to rounding errors.
    # print taskset.utilization(), '->',
    while taskset.utilization() > max_util:
        t = random.choice(taskset)
        if t.cost > 1:
            t.cost -= 1
    # print taskset.utilization()

    assign_affinities(conf, taskset)
    taskset.assign_ids()
    return taskset

def run_emstada_config(conf):
    common_checks(conf)
    conf.check('periods',      type=one_of(emstada.NAMED_PERIODS),
                               default='uni-moderate')
    conf.check('periods_dist', type=one_of(EMSTADA_PERIODS.keys()
                                    + emstada.NAMED_PERIOD_DISTRIBUTIONS),
                               default='hyper1000')
    conf.check('task_count',   type=int, default=conf.num_cpus + 1)

    conf.make_taskset = partial(generate_emstada, conf)

    tests, titles = setup_tests(conf)
    header = ['UTIL', 'R-UTIL']
    for t in titles:
        if WANT_CI:
            header += ['%s CI-' % t, t, '%s CI+' % t]
        else:
            header += [t]

    data = run_range(conf, tests, adaptive=conf.want_adaptive)
    data = format_data(data)
    write_data(conf.output, data, header)

def run_emstada_measure_preemptions_config(conf):
    common_checks(conf)
    conf.check('periods',      type=one_of(emstada.NAMED_PERIODS),
                               default='uni-moderate')
    conf.check('periods_dist', type=one_of(EMSTADA_PERIODS.keys()
                                    + emstada.NAMED_PERIOD_DISTRIBUTIONS),
                               default='hyper1000')
    conf.check('task_count',   type=int, default=conf.num_cpus + 1)

    conf.make_taskset = partial(generate_emstada, conf)

    tests, titles = mp_setup_tests(conf)
    header = ['UTIL', 'R-UTIL']
    for t in titles:
        header += [t]

    data = measure_preemptions(conf, tests)
    data = format_data(data)
    write_data(conf.output, data, header)


SAMPLES = 48 * 2 * 12

def generate_unc_apa_configs(options):
    for cpus in [2, 4, 8, 16, 24, 32, 64]:
		for util in gen.NAMED_UTILIZATIONS:
			name = 'sd_exp=rtss16b-unc_m=%02d_util=%s' \
						% (cpus, util)
			c = Config()
			c.output_file  = name + '.csv'
			c.num_cpus     = cpus
			c.utilizations = util
			c.samples = SAMPLES
			yield (name, c)

def generate_unc_chunksize_configs(options):
    for cpus in [2, 4, 8, 16]:
		for chunk_size in [100, 200, 300, 400, 500, 750, 1000, 2000]:
			for util in gen.NAMED_UTILIZATIONS:
				name = 'sd_exp=rtss16b-unc-chunk_m=%02d_util=%s_chunk=%03d' \
							% (cpus, util, chunk_size)
				c = Config()
				c.output_file  = name + '.csv'
				c.num_cpus     = cpus
				c.utilizations = util
				c.min_slice_size = chunk_size
				c.samples = SAMPLES
				yield (name, c)

def generate_unc_tcount_configs(options):
    for cpus in [2, 4, 8, 16, 24, 32, 64]:
		for util in gen.NAMED_UTILIZATIONS:
			name = 'sd_exp=rtss16b-unc-tcount_m=%02d_util=%s' \
						% (cpus, util)
			c = Config()
			c.output_file  = name + '.csv'
			c.num_cpus     = cpus
			c.utilizations = util
			c.samples = SAMPLES
			yield (name, c)

def generate_emstada_apa_configs(options):
    for cpus in [2, 4, 8, 16, 24, 32, 64]:
		dist = 'hyper1000'
		cpu_step = (cpus * 3) // 12
		cpu_step = max(cpu_step, 1)
		task_counts = set([cpus + 1] + [n for n in xrange(cpus, cpus * 4  + 1, cpu_step)][1:])
		for n in range(cpus + 1, 2 * cpus)[:16]:
			task_counts.add(n)
		for n in task_counts:
			name = 'sd_exp=rtss16b-emstada_dist=%s_m=%02d_n=%02d' \
						% (dist, cpus, n)
			c = Config()
			c.output_file  = name + '.csv'
			c.num_cpus     = cpus
			c.task_count   = n
			c.periods_dist = dist
			c.samples = SAMPLES
			yield (name, c)

def generate_emstada_chuncksize_configs(options):
    for cpus in [2, 4, 8, 16]:
		for chunk_size in [100, 200, 300, 400, 500, 750, 1000, 2000]:
			dist = 'hyper1000'
			cpu_step = (cpus * 3) // 12
			cpu_step = max(cpu_step, 1)
			task_counts = set([n for n in xrange(cpus, cpus * 4  + 1, cpu_step)][1:])
			for n in range(cpus + 1, cpus + 5):
				task_counts.add(n)
			for n in task_counts:
				name = 'sd_exp=rtss16b-emstada-chunk_dist=%s_m=%02d_chunk=%04d_n=%02d' \
							% (dist, cpus, chunk_size, n)
				c = Config()
				c.output_file  = name + '.csv'
				c.num_cpus     = cpus
				c.task_count   = n
				c.periods_dist = dist
				c.min_slice_size = chunk_size
				c.samples = SAMPLES
				yield (name, c)

def generate_preempt_configs(options):
    for cpus in [2, 4, 8, 16, 24, 32, 64]:
		dist = 'hyper1000'
		cpu_step = (cpus * 3) // 12
		cpu_step = max(cpu_step, 1)
		task_counts = set([cpus + 1] + [n for n in xrange(cpus, cpus * 4  + 1, cpu_step)][1:])
		for n in range(cpus + 1, 2 * cpus)[:16]:
			task_counts.add(n)
		for n in task_counts:
			name = 'sd_exp=rtss16b-mp_dist=%s_m=%02d_n=%02d' \
						% (dist, cpus, n)
			c = Config()
			c.output_file  = name + '.csv'
			c.num_cpus     = cpus
			c.task_count   = n
			c.periods_dist = dist
			c.samples = SAMPLES
			# need individual results for the preemption configs
			c.want_individual_results = True
			yield (name, c)

EXPERIMENTS = {
    'rtss16b/unc'         : run_unc_config,
    'rtss16b/unc-tcount'  : run_unc_tcount_config,
    'rtss16b/unc-chunk'   : run_unc_config,
    'rtss16b/emstada'     : run_emstada_config,
    'rtss16b/emstada-chunk' : run_emstada_config,
    'rtss16b/mp'          : run_emstada_measure_preemptions_config,
}

CONFIG_GENERATORS = {
    'rtss16b/unc'         : generate_unc_apa_configs,
    'rtss16b/unc-tcount'  : generate_unc_tcount_configs,
    'rtss16b/unc-chunk'   : generate_unc_chunksize_configs,
    'rtss16b/emstada'     : generate_emstada_apa_configs,
    'rtss16b/emstada-chunk' : generate_emstada_chuncksize_configs,
    'rtss16b/mp'          : generate_preempt_configs,
}

from schedcat.model.serialize import load as load_taskset

def try_placement_heuristic(name, heuristic):
    print '\n* Trying %s:' % name
    (unassigned, mapping) = heuristic()

    for cpu in mapping:
        print '    CPU', cpu, 'util:', mapping[cpu].utilization(), 'dens:', mapping[cpu].density()
        for t in mapping[cpu]:
            print '     -', t.id, t

    print '    Unassigned:'
    for t in unassigned:
        print '     - ', t.id, t, 'util:', t.utilization(), 'alpha:', t.affinity


def main(args = sys.argv[1:]):
    for fname in args:
        try:
            print '*' * 80
            taskset = load_taskset(fname)
            taskset.assign_ids()

            aff = sched.get_native_affinities(taskset)
            ts  = sched.get_native_taskset(taskset)
            sol = sched.native.apa_implicit_deadline_feasible(ts, aff)

            if not sol:
                print 'APA Infeasible!'
                for t in enumerate(taskset):
                    print ' -', t, 'alpha =', t.affinity, allocs
            else:
                feas_map = defaultdict(list)
                for i, t in enumerate(taskset):
                    allocs = [(cpu, sol.get_fraction(i, cpu))
                              for cpu in t.affinity
                              if sol.get_fraction(i, cpu) >= 1.0 / t.period]

                    if len(allocs) == 1:
                        (cpu, frac) = allocs[0]
                        where = 'on CPU %d' % cpu
                        feas_map[cpu].append((t, 1))
                    else:
                        where = ', '.join(["%f on CPU %d" % (frac, cpu) for (cpu, frac) in allocs])
                        for a in allocs:
                            feas_map[a[0]].append((t, a[1]))

                    print ' -', t
                    print '   alpha =', t.affinity
                    print '   =>', where

                print '\n* Feasible solution:'
                for cpu in feas_map:
                    util = sum((t.utilization() * f for (t, f) in feas_map[cpu]))
                    print '    CPU', cpu, 'util:', util
                    for (t, f) in feas_map[cpu]:
                        print '     -', t.id, t, "" if f == 1 else " frac:%f" % f

            try_placement_heuristic('WFD C=D',
                partial(apa.edf_worst_fit_decreasing_difficulty,
                                        taskset.copy(), with_splits=True))

            try_placement_heuristic('WFD w/o splits, then WFD C=D',
                partial(edf_assign_wfd_wfd_split, taskset.copy()))

            try_placement_heuristic('FFD C=D',
                partial(apa.edf_first_fit_decreasing_difficulty,
                                        taskset.copy(), with_splits=True))

            try_placement_heuristic('FFD w/o splits, then WFD C=D',
                partial(edf_assign_ffd_wfd_split, taskset.copy()))

            try_placement_heuristic('use feasibility solution, then WFD C=D',
                partial(edf_split_feas_c_equal_d, taskset.copy(), sol))

            try_placement_heuristic('use feasibility solution, then WM',
                partial(edf_split_feas_wm, taskset.copy(), sol))

            max_slices = 100

            try_placement_heuristic('[WFD C=D] + meta: slicing-50',
                partial(meta_period_transform,
                    max_slices,
                    partial(apa.edf_worst_fit_decreasing_difficulty,
                            with_splits = True),
                    taskset.copy()))

            try_placement_heuristic('[use feas., then WFD C=D] + meta: slicing-10',
                partial(meta_period_transform,
                    max_slices,
                    edf_split_feas_c_equal_d,
                    taskset.copy()))

            try_placement_heuristic('[use feas., then WM] + meta: slicing-10',
                partial(meta_period_transform,
                    max_slices,
                    edf_split_feas_wm,
                    taskset.copy()))

            try_placement_heuristic('[WFD w/o splits, then WFD C=D] + meta: slicing-10',
                partial(meta_period_transform,
                    max_slices,
                    edf_assign_wfd_wfd_split,
                    taskset.copy()))

            try_placement_heuristic('[FFD w/o splits, then WFD C=D] + meta: slicing-10',
                partial(meta_period_transform,
                    max_slices,
                    edf_assign_ffd_wfd_split,
                    taskset.copy()))

            try_placement_heuristic('[use feas., then WFD C=D] + meta: slice-all',
                partial(meta_slice_all_tasks,
                    max_slices,
                    edf_split_feas_c_equal_d,
                    taskset.copy()))

        except IOError, err:
            print err


import json

def consolidate_mapping(taskset, mapping):
    for t in taskset:
        if len(t.slices) > 1:
            by_core = defaultdict(list)
            for sub in t.slices:
                by_core[sub.partition].append(sub)
            for c in by_core:
                if len(by_core[c]) > 1:
                    print 'consolidating slices on core', c
                    # aggregate cost and time window
                    by_core[c][-1].cost = sum((sub.cost for sub in by_core[c]))
                    by_core[c][-1].deadline = sum((sub.deadline for sub in by_core[c]))
                    # remove all but the last sub tasks
                    for sub in by_core[c][:-1]:
                        t.slices.remove(sub)
                        mapping[c].remove(sub)

    for c in mapping:
        assert apa.qpa_it_fits(mapping[c])

    for t in taskset:
        total_dl = sum((sub.deadline for sub in t.slices))
        for sub in t.slices:
            assert total_dl == sub.period
        multiple = t.period // total_dl
        assert t.period >= multiple * sub.period
        total_cost = sum((sub.cost for sub in t.slices))
        assert total_cost * multiple >= t.cost

def get_slices(task):
    slices = [(t.partition, t.rid, t.deadline) for t in task.slices]
    slices.sort(key = lambda x: x[2], reverse=True)
    return slices

def tasks_to_json(taskset, mapping):
    splits = set((t for t in taskset if len(t.slices) > 1))

    # record core assignment
    for core in mapping:
        for t in mapping[core]:
            t.partition = core

    # at most one slice per core
    consolidate_mapping(taskset, mapping)

    # make per-core reservations
    reservations = {}
    for core in mapping:
        rid = 1000 * (core + 1)
        for t in mapping[core]:
            rid += 1
            t.rid = rid
            t.task.rid = rid
            reservations[rid] = {
                'type'   : 'soft-polling',
                'budget' : t.cost,
                'period' : t.period,
                'deadline' : t.deadline,
                'partition' : core,
            }

    # make semi-part reservations
    rid = 100
    for t in splits:
        rid += 1
        t.rid = rid
        offset = 0
        slices = []
        cores = set()
        for (p, id, dl) in get_slices(t):
            cores.add(p)
            offset += dl
            slices += [{
                'partition' : p,
                'rid' : str(id),
                'offset' : offset,
            }]
        assert len(cores) == len(slices)
        reservations[rid] = {
            'type'   : 'semi-partitioned',
            'period' : t.slices[0].period,
            'slices' : slices,
            'partition' : slices[0]['partition'],
        }

    # task parameters from original task set (before period transformation)
    tsks = {}
    for t in taskset:
        tsks[t.id] = {
            'cost'   : t.cost,
            'period' : t.period
        }

    assignments = [{
        'rid' : str(t.rid),
        'tid' : str(t.id),
    } for t in taskset]


    data = {
        'tasks' : tsks,
        'reservations' : reservations,
        'espresso_mapping' : assignments,
    }

    return json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))


def find_assignment(ts):
    for g in ORDERED_GROUPS:
        group = ALL_GROUPS[g]
        (unassigned, mapping) = apply_heuristics(group, ts, min_chunk_size=750)
        if not unassigned:
            print g
            return (unassigned, mapping)

    return (unassigned, mapping)

def assign_taskset(ts, m):
    all_cores = frozenset(range(m))
    ts.assign_ids()
    for t in ts:
        t.affinity = all_cores
        t.slices = []
    (unassigned, mapping) = find_assignment(ts)
    if not unassigned:
        for c in mapping:
            for t in mapping[c]:
                t.task = ts[t.id - 1]
                t.task.slices.append(t)
        return ts, mapping
    else:
        return ts, None

def make_eval_taskset(n, u, m):
    while True:
        ts = emstada.gen_taskset((1, 1000), PERIODS, n, u * m,
                                 want_integral=True, scale=ms2us)
        for t in ts:
            # check for tasks with unrealistically small budgets
            if t.cost <= 400:
                larger_periods = [x for x in PERIODS_US
                                  if x > t.period and x % t.period == 0]
                for p in larger_periods:
                    scale = p // t.period
                    if t.cost * scale >= 400 or p == larger_periods[-1]:
                        print 'scaling %s %dx ->' % (t, scale),
                        t.cost     *= scale
                        t.period   *= scale
                        t.deadline *= scale
                        print t
                        break


        (ts, mapping) = assign_taskset(ts, m)
        if mapping:
            return ts, mapping
        else:
            print '.',

def generate_litmus_workload(m=8, prefix='tasksets/'):
    for n in set([m + 1, int(m * 1.5), m * 2, m * 4, m * 6, m * 8, m * 10]):
        for u in value_range(0.75, 0.99, 0.05):
            for x in xrange(10):
                ts, mapping = make_eval_taskset(n, u, m)
                fname = "%slitmus-workload_m=%02d_n=%02d_u=%2d_seq=%02d.json" % \
                        (prefix, m, n, int(100 * u), x + 1)
                print fname
                f  = open(fname, 'w')
                f.write(tasks_to_json(ts, mapping))
                f.close()

def adopt_qps_run_workload(xml_file, m, prefix):
    qps_run_ts = load_taskset(xml_file)

    n = len(qps_run_ts)
    u = qps_run_ts.utilization() / m
    params = parse_file_name(xml_file)
    id = params['id']
    fname = "%sadapted-workload_m=%02d_n=%02d_u=%2d_id=%s.json" % \
            (prefix, m, n, int(100 * u), id)

    if u > 1:
        print 'OVERUTILIZED', xml_file
        return

    if os.path.exists(fname):
        print 'SKIPPED', fname
        return

    print 'Processing', xml_file, '...',
    sys.stdout.flush()
    (ts, mapping) = assign_taskset(qps_run_ts, m)
    if not mapping:
        print 'FAILED'
        return
    else:
        print '->', fname

    f  = open(fname, 'w')
    f.write(tasks_to_json(ts, mapping))
    f.close()

def adopt_main(args=sys.argv[1:], m=44, prefix='qps-params/'):
    for fname in args:
        adopt_qps_run_workload(fname, m, prefix)

if __name__ == '__main__':
#     ts, mapping = make_eval_taskset(10, 0.95, 8)
#     print tasks_to_json(ts, mapping)
    main()
#     generate_litmus_workload(m=2)
#     generate_litmus_workload(m=4)
#     generate_litmus_workload(m=8)
#     generate_litmus_workload(m=16)
#     generate_litmus_workload(m=24)
#     generate_litmus_workload(m=32)
#     adopt_main(m=44)
