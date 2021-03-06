import schedcat.locking.native as cpp

# The blocking analysis needs to know which task can be preempted by which
# other task. This of course differs under EDF and FP scheduling. To simplify
# the interface, and to avoid having to propagate information about which
# scheduling policy is used to the blocking analysis, we instead determine for
# each task a *preemption level*, with the following interpretation: a task A
# can preempt a task B if and only if A.preemption_level < B.preemption_level.

def assign_edf_preemption_levels(all_tasks):
    all_deadlines = set([t.deadline for t in all_tasks])
    prio = {}
    for i, dl in enumerate(sorted(all_deadlines)):
        prio[int(dl)] = i

    for t in all_tasks:
        t.preemption_level = prio[int(t.deadline)]

def assign_prio_pt_preemption_levels(all_tasks):
    all_prio_pts = set([t.prio_pt for t in all_tasks])
    prio = {}
    for i, pp in enumerate(sorted(all_prio_pts)):
        prio[int(pp)] = i

    for t in all_tasks:
        t.preemption_level = prio[int(t.prio_pt)]

def assign_fp_preemption_levels(all_tasks):
    # prioritized in index order
    for i, t in enumerate(all_tasks):
        t.preemption_level = i



# assumes mutex constraints
def get_cpp_model(all_tasks, use_task_period=False, no_requests=False):
    rsi = cpp.ResourceSharingInfo(len(all_tasks))
    for t in all_tasks:
        rsi.add_task(t.period,
                     t.period if use_task_period else t.response_time,
                     t.partition,
                     t.preemption_level, # mapped to fixed priorities in the C++ code
                     t.cost,
                     t.deadline)
        if not no_requests:
            for req in t.resmodel:
                req = t.resmodel[req]
                if req.max_requests > 0:
                    rsi.add_request(req.res_id, req.max_requests, req.max_length,
                                    req.priority)
    return rsi

def get_cpp_model_rw(all_tasks, use_task_period=False):
    rsi = cpp.ResourceSharingInfo(len(all_tasks))
    for t in all_tasks:
        rsi.add_task(t.period,
                     t.period if use_task_period else t.response_time,
                     t.partition,
                     t.preemption_level, # mapped to fixed priorities in the C++ code
                     t.cost,
                     t.deadline)
        for req in t.resmodel:
            req = t.resmodel[req]
            if req.max_writes > 0:
                rsi.add_request_rw(req.res_id, req.max_writes, req.max_write_length, cpp.WRITE, req.priority)
            if req.max_reads > 0:
                rsi.add_request_rw(req.res_id, req.max_reads, req.max_read_length, cpp.READ, req.priority)
    return rsi

# S-aware bounds

def apply_mpcp_bounds(all_tasks, use_virtual_spin=False):
    model = get_cpp_model(all_tasks)
    res = cpp.mpcp_bounds(model, use_virtual_spin)

    if use_virtual_spin:
        for i,t in enumerate(all_tasks):
            # no suspension time
            t.suspended = 0
            # all blocking, including arrival blocking
            t.blocked = res.get_blocking_term(i)
            # remote blocking increases CPU demand (s-oblivious)
            t.cost += res.get_remote_blocking(i)
    else:
        for i,t in enumerate(all_tasks):
            # remote blocking <=> suspension time
            t.suspended = res.get_remote_blocking(i)
            # all blocking, including arrival blocking
            t.blocked   = res.get_blocking_term(i)
            t.locally_blocked = res.get_local_blocking(i)

    return res

def get_round_robin_resource_mapping(num_resources, num_cpus,
                                     dedicated_irq=cpp.NO_CPU):
    "Default resource assignment: just assign resources to CPUs in index order."
    loc   = {}
    for res_id in xrange(num_resources):
        cpu = res_id % num_cpus
        if cpu == dedicated_irq:
            cpu = (cpu + 1) % num_cpus
        loc[res_id] = cpu
    return loc

def get_cpp_topology(res_mapping):
    map = cpp.ResourceLocality()
    for res_id in res_mapping:
        map.assign_resource(res_id, res_mapping[res_id])
    return map

def apply_dpcp_bounds(all_tasks, resource_mapping,
                      use_text_book_definition=False):
    # The DPCP bounds are expressed in terms of task periods,
    # not response time, in the original definition.
    model = get_cpp_model(all_tasks, use_text_book_definition)
    topo  = get_cpp_topology(resource_mapping)
    res = cpp.dpcp_bounds(model, topo)

    for i,t in enumerate(all_tasks):
        # remote blocking <=> suspension time
        t.suspended = res.get_remote_blocking(i)
        # all blocking, including arrival blocking
        t.blocked   = res.get_blocking_term(i)

    return res

def apply_part_fmlp_bounds(all_tasks, preemptive=True):
    model = get_cpp_model(all_tasks)
    res = cpp.part_fmlp_bounds(model, preemptive)

    for i,t in enumerate(all_tasks):
        # remote blocking <=> suspension time
        t.suspended = res.get_remote_blocking(i)
        # all blocking, including local blocking
        t.blocked   = res.get_blocking_term(i)
        t.local_blocking_count = res.get_local_count(i)

    return res

# S-oblivious bounds

def apply_suspension_oblivious(all_tasks, res):
    for i,t in enumerate(all_tasks):
        # s-oblivious <=> no suspension
        t.suspended = 0
        # might be zero
        t.arrival_blocked = res.get_arrival_blocking(i)
        # all blocking, including arrival blocking
        t.blocked   = res.get_blocking_term(i)
        # s-oblivious: charge it as execution cost
        t.cost     += t.blocked

def apply_global_fmlp_sob_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = cpp.global_fmlp_bounds(model)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_global_omlp_bounds(all_tasks, num_cpus):
    model = get_cpp_model(all_tasks)
    res = cpp.global_omlp_bounds(model, num_cpus)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_clustered_omlp_bounds(all_tasks, procs_per_cluster,
                                dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model(all_tasks)
    res = cpp.clustered_omlp_bounds(model, procs_per_cluster, dedicated_irq)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_clustered_rw_omlp_bounds(all_tasks, procs_per_cluster,
                                   dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model_rw(all_tasks)
    res = cpp.clustered_rw_omlp_bounds(model, procs_per_cluster, dedicated_irq)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_clustered_kx_omlp_bounds(all_tasks, procs_per_cluster,
                                   replication_degrees = {},
                                   dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model(all_tasks)
    replica_info = cpp.ReplicaInfo()
    for res_id in replication_degrees:
        replica_info.set_replicas(res_id, replication_degrees[res_id])
    res = cpp.clustered_kx_omlp_bounds(model, replica_info, procs_per_cluster,
                                       dedicated_irq)
    apply_suspension_oblivious(all_tasks, res)
    return res

# spinlocks are charged similarly to s-oblivious analysis

def apply_task_fair_mutex_bounds(all_tasks, procs_per_cluster,
                                 dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model(all_tasks)
    res = cpp.task_fair_mutex_bounds(model, procs_per_cluster, dedicated_irq)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_task_fair_rw_bounds(all_tasks, procs_per_cluster,
                              dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model_rw(all_tasks)
    # mutex equivalent model
    model_mtx = get_cpp_model(all_tasks)
    res = cpp.task_fair_rw_bounds(model, model_mtx, procs_per_cluster, dedicated_irq)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_phase_fair_rw_bounds(all_tasks, procs_per_cluster,
                                 dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model_rw(all_tasks)
    res = cpp.phase_fair_rw_bounds(model, procs_per_cluster, dedicated_irq)
    apply_suspension_oblivious(all_tasks, res)
    return res

### spin lock analysis that assumes s-aware schedulability tests
#   (=> FP response-time analysis in schedcat.sched.fp.rta)

def apply_msrp_bounds_holistic(all_tasks, dedicated_irq=cpp.NO_CPU):
    model = get_cpp_model(all_tasks)
    res = cpp.msrp_bounds_holistic(model, dedicated_irq)

    for i, t in enumerate(all_tasks):
        # spin locks => no self-suspension (local blocking is not a self-suspension)
        t.suspended = 0
        # account for local arrival blocking (either local resources or NP section)
        t.blocked   = res.get_arrival_blocking(i)
        # charge only spinning (NOT arrival blocking) as execution time inflation
        t.cost     += res.get_remote_blocking(i)

    return res

### S-aware LP-based analysis of distributed locking protocols

try:
    import schedcat.locking.linprog.native as lp_cpp

    lp_cpp_available = True
except ImportError:
    lp_cpp_available = False

def get_cpp_nested_cs_model(all_tasks):
    model = lp_cpp.CriticalSectionsOfTaskset()

    for t in all_tasks:
        cs = model.new_task()
        for (id, len, outer, _) in t.critical_sections.all_flat():
            cs.add(id, len, outer)

    return model


import schedcat.locking.linprog.dflp as dflp
import schedcat.locking.linprog.dpcp as dpcp

def apply_py_lp_bounds(bounds, all_tasks, resource_mapping, *args):
    for t in all_tasks:
        lp = bounds.get_lp_for_task(resource_mapping,
                                    all_tasks, t, *args)

        lp.kill_non_positive_vars()
        solution = lp.solve()

        # total blocking
        t.blocked   = int(solution(lp.objective_function))
        t.suspended = int(solution(lp.remote_objective))

def apply_lp_dflp_bounds(all_tasks, resource_mapping,
                         use_py=False):
    if use_py or not lp_cpp_available:
        apply_py_lp_bounds(dflp, all_tasks, resource_mapping)
    else:
        model = get_cpp_model(all_tasks)
        topo  = get_cpp_topology(resource_mapping)
        res = lp_cpp.lp_dflp_bounds(model, topo)
        for i, t in enumerate(all_tasks):
            t.suspended = res.get_remote_blocking(i)
            t.blocked   = res.get_blocking_term(i)
        return res

def apply_lp_dpcp_bounds(all_tasks, resource_mapping,
                         use_rta = True, use_py=False):
    if use_py or not lp_cpp_available:
        apply_py_lp_bounds(dpcp, all_tasks, resource_mapping, use_rta)
    else:
        model = get_cpp_model(all_tasks)
        topo  = get_cpp_topology(resource_mapping)
        res = lp_cpp.lp_dpcp_bounds(model, topo, use_rta)
        for i, t in enumerate(all_tasks):
            t.suspended = res.get_remote_blocking(i)
            t.blocked   = res.get_blocking_term(i)
        return res


def apply_lp_mpcp_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_mpcp_bounds(model)

    for i,t in enumerate(all_tasks):
        # remote blocking <=> self-suspension time
        t.suspended = res.get_remote_blocking(i)
        # all blocking, including local blocking
        t.blocked   = res.get_blocking_term(i)
        t.locally_blocked = res.get_local_blocking(i)

    return res

def apply_lp_part_fmlp_bounds(all_tasks):
    # LP-based analysis of the partitioned, preemptive FMLP+
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_part_fmlp_bounds(model)

    for i,t in enumerate(all_tasks):
        # remote blocking <=> self-suspension time
        t.suspended = res.get_remote_blocking(i)
        # all blocking, including local blocking
        t.blocked   = res.get_blocking_term(i)
        t.locally_blocked = res.get_local_blocking(i)

    return res

def apply_generalized_fmlp_bounds(all_tasks, cluster_size, using_edf):
    # LP-based analysis of the generalized FMLP+
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_gfmlp_bounds(model, cluster_size, using_edf)

    if cluster_size == 1:
        # partitioned scheduling: charge uniproc-analysis compatible blocking
        for i,t in enumerate(all_tasks):
            # remote blocking <=> self-suspension time
            t.suspended = res.get_remote_blocking(i)
            # all blocking, including local blocking
            t.blocked   = res.get_blocking_term(i)
            t.locally_blocked = res.get_local_blocking(i)
    else:
        # global analysis => no special treatment of local blocking
        for i,t in enumerate(all_tasks):
            # remote blocking <=> self-suspension time
            t.suspended = res.get_blocking_term(i)
            # all blocking
            t.blocked   = res.get_blocking_term(i)

    return res


def apply_omip_bounds(all_tasks, num_cpus, procs_per_cluster):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_omip_bounds(model, num_cpus, procs_per_cluster)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_dummy_bounds(all_tasks):
    model = get_cpp_model(all_tasks, True)
    res = lp_cpp.dummy_bounds(model)
    for i,t in enumerate(all_tasks):
        t.blocked   = res.get_local_blocking(i)
        t.remote_blocking = res.get_remote_blocking(i)
    return res


def apply_msrp_bounds(all_tasks, num_cpus):
    for t in all_tasks:
        if t.partition == None:
            t.partition= num_cpus + 1
    model = get_cpp_model(all_tasks, True)
    res = cpp.msrp_bounds(model, num_cpus)

    for i,t in enumerate(all_tasks):
        t.blocked   = res.get_local_blocking(i)
        t.cost      += res.get_remote_blocking(i)
        t.remote_blocking = res.get_remote_blocking(i)

def apply_pfp_lp_preemptive_fifo_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_preemptive_fifo_spinlock_bounds(model)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_msrp_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_msrp_bounds(model)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_unordered_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_unordered_spinlock_bounds(model, False)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_preemptive_unordered_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_unordered_spinlock_bounds(model, True)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_prio_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_prio_spinlock_bounds(model, False)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_preemptive_prio_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_prio_spinlock_bounds(model, True)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_prio_fifo_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_prio_fifo_spinlock_bounds(model, False)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_preemptive_prio_fifo_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_prio_fifo_spinlock_bounds(model, True)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_pfp_lp_baseline_spinlock_bounds(all_tasks):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_pfp_baseline_spinlock_bounds(model)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_omip_bounds(all_tasks, num_cpus, procs_per_cluster):
    model = get_cpp_model(all_tasks)
    res = lp_cpp.lp_omip_bounds(model, num_cpus, procs_per_cluster)
    apply_suspension_oblivious(all_tasks, res)
    return res

def apply_pfp_nested_fifo_spinlock_bounds(all_tasks, rnlp=False,
                                          rnlp_nesting_aware=False):
    task_info    = get_cpp_model(all_tasks, no_requests=True)
    nested_model = get_cpp_nested_cs_model(all_tasks)
    res = lp_cpp.lp_nested_fifo_spinlock_bounds(task_info, nested_model,
                                                rnlp, rnlp_nesting_aware)
    for i, _ in enumerate(all_tasks):
        all_tasks[i].blocked = res.get_blocking_term(i)
    return res

def apply_gipp_bounds(all_tasks, num_cpus, procs_per_cluster, force_single_group=False):
    task_info = get_cpp_model(all_tasks, no_requests=True)
    nested_model = get_cpp_nested_cs_model(all_tasks)
    res = lp_cpp.lp_gipp_bounds(task_info, 
                                nested_model,
                                num_cpus, 
                                procs_per_cluster,
                                force_single_group)
    apply_suspension_oblivious(all_tasks, res)
    return res

