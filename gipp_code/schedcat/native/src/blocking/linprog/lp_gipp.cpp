#include <assert.h>
#include <limits.h>
#include <cmath>

#include <algorithm>
#include <iterator>

#include "linprog/model.h"
#include "linprog/varmapperbase.h"
#include "linprog/solver.h"

#include "sharedres_types.h"
#include "iter-helper.h"

#include "stl-hashmap.h"
#include "stl-helper.h"
#include "stl-io-helper.h"

#include "nested_cs.h"

#include <iostream>
#include <sstream>
#include "res_io.h"
#include "linprog/io.h"

#define G_ENABLE_DEBUG false
#define GIPP_LP_OUTPUT false

#define G_DEBUG(X)	if(G_ENABLE_DEBUG) { std::cout << X << std::endl << std::flush; }

/* Let us repurpose blocking types known to the VarMapper for our purposes. */
#define BLOCKING_RSM   BLOCKING_DIRECT
#define BLOCKING_TOKEN BLOCKING_INDIRECT


/*

	RTOS WARNING: analizzando la AI sappiamo che viene eseguito un priority boosting a seguito della migrazione di un Job in un altro cluster.
	Confrontando l'analisi fine-grained pi-blocking con il paper di Brandeburg ci siamo accorti che quanto riportato sopra (ossia l'associazione del blocking RSM e token) 
	potrebbe non essere del tutto accurata dal momento che manca un blocking: il preemption-delay!

	--> Non c'è preemption: guardando il funzionamento di AI e del CKIP non c'è preemption perché al momento della richiesta i job vengono accodati e immediatamente sospesi; ritornano ready solo quando saranno in testa alla coda, dunque non dovrebbe sussistere predizione.

*/

/* 
despite introducing new typedefs for for resourceIDs, groupIDs, etc.
the code currently *ABSOLUTELY REQUIRES* that all indicies used are
0-indexed. This includes, but is not limited to:
- taskIDs
- clusterIDs
- groupIDs
- resourceIDs

Notice that enumerate is used when iterating over tasks, and a task_id counter
is started at zero. The assumption is then that the provided taskset is sorted
in order of ascending task_ids. If this is not the case, everything *should*
still work, but then attention should be paid when debugging to make ones life
easier, as printing out task->get_id() may differ from the task's index in the 
taskset.
*/

typedef std::set<unsigned int> GroupSet;
typedef std::vector<GroupSet> ResourceGroups;
typedef std::vector<std::vector<unsigned int>> uint2D;

class GIPPLinearProgram : protected LinearProgram
{
	VarMapper vars;

	const int i;
	const TaskInfo& ti;
	const CriticalSectionsOfTask& csi;
	const TaskInfos& taskset;
	const CriticalSectionsOfTasks& taskset_cs;

	const unsigned int num_procs;
	const unsigned int cluster_size;

	const ResourceGroups& resource_groups;

	/*
	Index is resource_id, and elment at index is the resource's group.
	*/
	const std::vector<unsigned int>& resource_group_mapping;

	/*
	Index is resource_id, and element is vector of resource_ids that resource_id is "greater" than.
	*/
	const std::vector<std::set<unsigned int>> resource_greater_than;

	/* 
	struce: [task_id][group_id] = number_access (outermost)
	Number of outermost requests by a task for a resource in group. 
	Name derived from notation in paper - phi as in the greek letter. 
	*/
	const uint2D& task_phis;

	/*
	structure: [task_id][resource_id] = num_accesses
	Denotes the maximum number of outermost requests a job of the task makes
	for resource_id.
	*/
	const uint2D& task_outermost_accesses;

	/*
	structure: [task_id][resoure_id] = max cs length of task_id for resouce_id
	*/
	const uint2D& task_outermost_cs_max_lengths;

	/*
	structure: [cluster_id][group_id] = num_accesses
	Number of tasks per clusters that issue an outermost request for a resource in a group.
	Name derived from notation in paper.
	*/
	const uint2D& cluster_betas;

	/*
	structure: [group_id] = num_accesses
	Number of tasks issue an outermost request for a resource in a group.
	Name derived from notation in paper.
	*/
	const std::vector<unsigned int>& system_betas;

	/*
	structure: vec[task_id][resource_id] = num_accesses
	This is used in the context of the task actively being examined,
	hence it is 2D instead of 3D (i.e. vec[task_id][othertask_id][resource_id]). 
	It reflects the number of times reqeusts by task_id for resource_id overlap
	with a job of the task being examined. In the paper this is theta_{x,a}^i
	where x is the other task, i is the task being examined, and a is the resource.

	The values populates depend on the response time of other tasks, so it can't be 
	calculated once like the phi/beta/etc constants.
	*/
	uint2D task_thetas;

	/*
	structure: vec[group_id] = F_{i,g}
	Difficult to word succiently what F_{i,g} is. Its defined well in the paper.
	Briefly put though, its the number of times that an outermost request for a group
	will experience RSM blocking. i is assumed to be the task under analysis, so
	no need to state it explicitly.
	*/
	std::vector<unsigned int>  f_terms;

	/*
	structure: vec[group_id] = max_contentions
	This denotes the maximum number of times the task under observation needs to
	compete for a group's token. Index is group_id, element at index is the value 
	just described.
	*/
	std::vector<unsigned int> token_contention_max;

	void calculate_task_thetas();
	void calculate_f_terms();
	void calculate_token_contentions();
	void set_objective();
	void set_cluster_token_blocking_constraint();
	void set_global_token_blocking_constraint();
	void set_per_task_rsm_blocking_constraint();
	void set_per_cluster_rsm_blocking_constraint();
	unsigned int num_resource_set_conflicts(std::set<unsigned int> res_ids);
	void set_per_task_conflicting_rsm_blocking_constraint();
	void set_per_cluster_conflicting_rsm_blocking_constraint();
	void set_f_term_rsm_blocking_constraint();

	unsigned int num_jobs_to_consider(const TaskInfo& tx)
	{
		// standard formula
		// RTOS: calcolo del bound
		return tx.get_max_num_jobs(ti.get_response()); //Bound on the maximum number of Jobs of Tx
	}


public:

	GIPPLinearProgram(
		const ResourceSharingInfo& tsk,
		const CriticalSectionsOfTaskset& tsk_cs,
		const int task_under_analysis,
		unsigned int num_procs,
		unsigned int cluster_size,
		ResourceGroups _resource_groups,
		const std::vector<unsigned int>& _resource_group_mapping,
		const std::vector<std::set<unsigned int>>& generate_resource_greater_than,
		const uint2D& _task_phis,
		const uint2D& _task_outermost_accesses,
		const uint2D& _task_outermost_cs_max_lengths,
		const uint2D& _cluster_betas,
		const std::vector<unsigned int>& _system_betas
		);

	unsigned long solve();
};

GIPPLinearProgram::GIPPLinearProgram(
	const ResourceSharingInfo& tsk,
	const CriticalSectionsOfTaskset& tsk_cs,
	int task_under_analysis,
	unsigned int num_procs,
	unsigned int cluster_size,
	ResourceGroups _resource_groups,
	const std::vector<unsigned int>& _resource_group_mapping,
	const std::vector<std::set<unsigned int>>& _resource_greater_than,
	const uint2D& _task_phis,
	const uint2D& _task_outermost_accesses,
	const uint2D& _task_outermost_cs_max_lengths,
	const uint2D& _cluster_betas,
	const std::vector<unsigned int>& _system_betas
	)
  : i(task_under_analysis),
	ti(tsk.get_tasks()[i]),
	csi(tsk_cs.get_tasks()[i]),
	taskset(tsk.get_tasks()),
	taskset_cs(tsk_cs.get_tasks()),
	num_procs(num_procs),
	cluster_size(cluster_size),
	resource_groups(_resource_groups),
	resource_group_mapping(_resource_group_mapping),
	resource_greater_than(_resource_greater_than),
	task_phis(_task_phis),
	task_outermost_accesses(_task_outermost_accesses),
	task_outermost_cs_max_lengths(_task_outermost_cs_max_lengths),
	cluster_betas(_cluster_betas),
	system_betas(_system_betas)
{

	// calculate_task_thetas();
	// calculate_f_terms();
	calculate_token_contentions();
	set_objective();
	vars.seal();

	set_cluster_token_blocking_constraint();
	set_global_token_blocking_constraint();

	set_per_task_rsm_blocking_constraint();
	set_per_cluster_rsm_blocking_constraint();
	set_per_task_conflicting_rsm_blocking_constraint();
	set_per_cluster_conflicting_rsm_blocking_constraint();	
	// set_f_term_rsm_blocking_constraint();
}

unsigned long GIPPLinearProgram::solve()
{
	Solution *sol;
	double result;

	hashmap<unsigned int, std::string> var_map;

	var_map = vars.get_translation_table();

	if (GIPP_LP_OUTPUT) {
		std::cout << "LP:" << std::endl;
		pretty_print_linear_program(std::cout, *this, var_map) << std::endl;
	}

	sol = linprog_solve(*this, vars.get_num_vars());
	result = ceil(sol->evaluate(*get_objective()));

	if (GIPP_LP_OUTPUT) {
		std::cout << "Solution: " << result << std::endl; // 42
	}

	// for (unsigned int x = 0; x < vars.get_num_vars(); x++)
	// {
	// 	std::cout << "X" << x << ": " << var_map[x] << " = " << sol->get_value(x)
	// 		  << (is_binary_variable(x) ? " [binary]" : "")
	// 		  << std::endl;
	// }

	delete sol;

	assert(result < ULONG_MAX);
	return (unsigned long) result;
}


typedef const unsigned int var_t;


// RTOS: ref. formula 5.2 msc james robb
void GIPPLinearProgram::set_objective()
{
	/*
	Sets the objective function to be solved.
	... and a bit more. It's not purely the objetive function.
	A very obvious constraint is added here as well because
	separating the constraint out into a separate function would
	essentially be a copy and paste. The constraint is:
	For all x,y,v  TOKEN_x,y,v + RSM_x,y,v <= 1
	*/

	LinearExpression *obj;

	obj = get_objective();

	unsigned int x;
	enumerate(taskset, ti, x)
	{

		/* ignore the task being analyzed, as it does not contrinbute
		to its own blocking */
		if (x == (unsigned int) i) {
			continue;
		}

		const CriticalSections& tx_cs = taskset_cs[x].get_cs();

		/* y is tracked/incremented separately from cs_index, as cs_index
		reflects all critical sections of the task, but we only want to
		count the outermost critical sections */
		unsigned int y = 0;
		unsigned int cs_index;
		enumerate(tx_cs, cs, cs_index) {

			if (cs->is_nested()) {
				continue;
			}

			unsigned int overlapping_jobs = num_jobs_to_consider(*ti);
			for (unsigned int v = 0; v < overlapping_jobs; v++) {


				var_t var_token = vars.lookup(x, y, v, BLOCKING_TOKEN);
				var_t var_rsm = vars.lookup(x, y, v, BLOCKING_RSM);
				
				// cs-length => L{^O}{_x,y}
				obj->add_term(cs->length, var_token); // X^T
				obj->add_term(cs->length, var_rsm); // X^R

			
				/*
				this could be broken out into a separate function,
				but seeing as the looping logic would be identical, it feels
				natural here.
				*/
				LinearExpression *exp = new LinearExpression();
				exp->add_var(var_token);
				exp->add_var(var_rsm);
				add_inequality(exp, 1);
			
			}

			y++;

		}

	}

}

void GIPPLinearProgram::set_cluster_token_blocking_constraint()
{
	/*
	This contraints the token blocking that a task incurs.
	For each time it needs to request a token (i.e. W_{i,g} or
	token_contention_max[cluster_id]), at most cluster_size
	many other tasks can contribute to the blocking from each cluster.
	If there are less than cluster_size tasks in each respective cluster, 
	then that is the number of tasks that can contribute to the token 
	blocking.
	*/

	/* for all groups */
	unsigned int group_id;
	enumerate(resource_groups, group, group_id) // RTOS: group = current obj, group_id = current id starting from 0
	// RTOS - ENUMERATE: 
	// for (typeof(resource_groups.begin()) group = ({group_id = 0; (resource_groups).begin();}); group != (resource_groups).end(); group++, group_id++)
	{

		/* for all clusters */
		for(unsigned int k = 0; k < num_procs / cluster_size; k++) {

			LinearExpression *exp = new LinearExpression();

			/* sum over tasks in each cluster */
			unsigned int x;
			enumerate(taskset, task, x) {  
				/* ignore task being analyzed, or tasks not in the current cluster */
				if(task->get_cluster() != k || x == (unsigned int)i) {
					// RTOS: In case its unclear, i is the index of the tasking being examined, and
					// task_id is the task being compared against (e.g. Ti vs Tx)
					continue;
				}

				const CriticalSections& tx_cs = taskset_cs[x].get_cs();

				unsigned int y = 0; // Numero di tutte le outermost critical sections di Tx
				unsigned int cs_index;

				// RTOS: cicla su ciascuna sezione critica del task Tx
				enumerate(tx_cs, cs, cs_index) {

					/* we continue without incrementing y as we only consider outermost
					critical sections */
					if (cs->is_nested()) {
						continue;
					}

					/* the resource_id associated with the outermost request is not part of
					the group. however, we increment y as y is used to denote the task's
					y-th outermost critical section. */
					if (!group->count(cs->resource_id)) { // Se la risorsa non fa parte del gruppo di risorse 
						y++;
						continue;
					} 

					unsigned int overlapping_jobs = num_jobs_to_consider(*task); // = 0{_x}{^i}
					for (unsigned int v = 0; v < overlapping_jobs; v++) { // v = {1,...,0{_x}{^i}}

						var_t var_token = vars.lookup(x, y, v, BLOCKING_TOKEN); // RTOS: genera una codifica del token blocking 
						exp->add_var(var_token); // Aggiunge var_token (codificato) come espressione lineare per il problema da risolvere
					
					}

					y++;

				}

			}

			/* sum of token variables can't exceed W_{i,g} * min(cluster_size, beta_{k,g}) */
			add_inequality(
				exp,
				(token_contention_max[group_id]) * std::min(cluster_size, cluster_betas[k][group_id])
			);

		}

	}

}

void GIPPLinearProgram::set_global_token_blocking_constraint()
{
	/*
	This contraints the token blocking that a task incurs.
	For each group, the number of times other tasks in the system
	can cause a given task T_i to incur token blocking can not exceed
	the number of times T_i contents for a token of that type.
	*/

	/* for all groups */
	unsigned int group_id;
	enumerate(resource_groups, group, group_id)
	{
		/* for all tasks in the system */
		unsigned int x;
		enumerate(taskset, task, x) {

			/* ignore task being analyzed */
			if(x == (unsigned int)i) {
				continue;
			}

			LinearExpression *exp = new LinearExpression();

			const CriticalSections& tx_cs = taskset_cs[x].get_cs();

			unsigned int y = 0;
			unsigned int cs_index;
			enumerate(tx_cs, cs, cs_index) {

				/* we continue without incrementing y as we only consider outermost
				critical sections */
				if (cs->is_nested()) {
					continue;
				}

				/* the resource_id associated with the outermost request is not part of
				the group. however, we increment y as y is used to denote the task's
				y-th outermost critical section. 
				
				This is the equivelant O'_{x,y,g} term in the paper.
				*/
				if (!group->count(cs->resource_id)) {
					y++;
					continue;
				} 

				unsigned int overlapping_jobs = num_jobs_to_consider(*task);
				for (unsigned int v = 0; v < overlapping_jobs; v++) {

					var_t var_token = vars.lookup(x, y, v, BLOCKING_TOKEN);
					exp->add_var(var_token);
				
				}

				y++;

			}

			/* sum of token variables can't exceed W_{i,g} */
			add_inequality(exp, token_contention_max[group_id]);

		}

	}

}

void GIPPLinearProgram::set_per_task_rsm_blocking_constraint()
{
	/*
	This contraints the RSM blocking that a task incurs on a per-task basis.
	For each time a task T_i requests a resource in a group g, it can be blocked
	in the RSM the minimum of the number of times it enter the RSM for g and
	the number of times T_x enters the RSM for g (while also considering overlapping
	jobs) - min(phi_i,g, phi_i,x * theta_x^i)
	*/

	unsigned int analyzed_task_id = (unsigned int) i;

	/* for all groups */
	unsigned int group_id;
	enumerate(resource_groups, group, group_id)
	{			

		/* sum over tasks in each cluster */
		unsigned int x;
		enumerate(taskset, task, x) {
			
			/* ignore task being analyzed */
			if(x == analyzed_task_id) {
				continue;
			}

			LinearExpression *exp = new LinearExpression();

			const CriticalSections& tx_cs = taskset_cs[x].get_cs();

			unsigned int y = 0;
			unsigned int cs_index;
			unsigned int overlapping_jobs = num_jobs_to_consider(*task);

			enumerate(tx_cs, cs, cs_index) {

				/* we continue without incrementing y as we only consider outermost
				critical sections */
				if (cs->is_nested()) {
					continue;
				}

				/* the resource_id associated with the outermost request is not part of
				the group. however, we increment y as y is used to denote the task's
				y-th outermost critical section. 
				
				This is the equivelant O'_{x,y,g} term in the paper.
				*/
				if (!group->count(cs->resource_id)) {
					y++;
					continue;
				} 

				
				for (unsigned int v = 0; v < overlapping_jobs; v++) {

					var_t var_rsm = vars.lookup(x, y, v, BLOCKING_RSM);
					exp->add_var(var_rsm);
				
				}

				y++;

			}

			/* sum of token variables can't exceed W_{i,g} * min(cluster_size, beta_{k,g}) */
			add_inequality(
				exp,
				std::min(task_phis[analyzed_task_id][group_id], task_phis[x][group_id] * overlapping_jobs)
			);

		}

	}

}

void GIPPLinearProgram::set_per_cluster_rsm_blocking_constraint()
{
	/*
	This contraints the RSM blocking that a task incurs on a per-cluster basis.
	For each time a task T_i requests a resource in a group g, it can be
	blocked by at most phi_{i,g} (i.e. task_phi[task_id][group_id]) times
	the min of the cluster size and the number of jobs in said cluster
	that request a resource in g (i.e. cluster_betas[cluster_id][group_id]).
	In the case of T_i's own cluster, the same reasoning holds, except
	we subtract one from the terms in the min, as a task does not block
	itself.
	*/

	unsigned int analyzed_task_id = (unsigned int) i;

	/* for all groups */
	unsigned int group_id;
	enumerate(resource_groups, group, group_id)
	{

		/* for all clusters */
		for(unsigned int k = 0; k < num_procs / cluster_size; k++) {

			LinearExpression *exp = new LinearExpression();

			/* sum over tasks in each cluster */
			unsigned int x;
			enumerate(taskset, task, x) {
				
				/* ignore task being analyzed, or tasks not in the current cluster */
				if(task->get_cluster() != k || x == analyzed_task_id) {
					continue;
				}

				const CriticalSections& tx_cs = taskset_cs[x].get_cs();

				unsigned int y = 0;
				unsigned int cs_index;
				unsigned int overlapping_jobs = num_jobs_to_consider(*task);

				enumerate(tx_cs, cs, cs_index) {

					/* we continue without incrementing y as we only consider outermost
					critical sections */
					if (cs->is_nested()) {
						continue;
					}

					/* the resource_id associated with the outermost request is not part of
					the group. however, we increment y as y is used to denote the task's
					y-th outermost critical section. 
					
					This is the equivelant O'_{x,y,g} term in the paper.
					*/
					if (!group->count(cs->resource_id)) {
						y++;
						continue;
					} 

					for (unsigned int v = 0; v < overlapping_jobs; v++) {

						var_t var_rsm = vars.lookup(x, y, v, BLOCKING_RSM);
						exp->add_var(var_rsm);
					
					}

					y++;

				}

			}

			unsigned int cluster_term = cluster_size;
			unsigned int beta_term = cluster_betas[k][group_id];

			/* decrease by own in examined task's own cluster as it cannot block itself */
			if (taskset[analyzed_task_id].get_cluster() == k) {
				
				if (cluster_term > 0) {
					cluster_term -= 1;
				}

				if (beta_term > 0) {
					beta_term -= 1;
				}

			}

			/* sum of token variables can't exceed W_{i,g} * min(cluster_size, beta_{k,g}) */
			add_inequality(
				exp,
				(task_phis[(unsigned int)i][group_id]) * std::min(cluster_term, beta_term)
			);

		}

	}

}

unsigned int GIPPLinearProgram::num_resource_set_conflicts(std::set<unsigned int> res_set)
{
	/* 
	Given a set of resource_ids, count the number of outermost critical
	secitons that T_i executes that 'possibly conflict' (see def in paper)
	with the set of resource_ids. This corresponds F(s) function in the paper.

	*/

	if (G_ENABLE_DEBUG) {
		G_DEBUG("calculating number of possibly conflicting outermost critical sections for task " << i)
	}

	unsigned int analyzed_task_id = i;
	unsigned int num_conflicting = 0;

	const CriticalSections& tx_cs = taskset_cs[analyzed_task_id].get_cs();

	unsigned int cs_index;
	enumerate(tx_cs, cs, cs_index) {

		/* we continue without incrementing y as we only consider outermost
		critical sections */
		if (cs->is_nested()) {
			continue;
		}

		/* o_res_ids stands as short-hand for outermost_res_ids */
		std::vector<unsigned int> o_res_ids = taskset_cs[analyzed_task_id].get_all_cs_resource_ids(cs_index);
		bool found_conflict = false;

		unsigned int id_index;
		enumerate(res_set, res_id, id_index) {

			std::set<unsigned int> greater_than_ids = resource_greater_than[(*res_id)];

			unsigned int o_id_index;
			enumerate(o_res_ids, o_res_id, o_id_index) {

				if (greater_than_ids.count(*o_res_id)) {
					found_conflict = true;
					break;
				}

			}

			if (found_conflict) {
				num_conflicting += 1;
				break;
			}

		}

	}

	if (G_ENABLE_DEBUG) {
		G_DEBUG("F_i(" << res_set << ") = " << num_conflicting)
	}

	return num_conflicting;

}

void GIPPLinearProgram::set_per_task_conflicting_rsm_blocking_constraint()
{

	unsigned int analyzed_task_id = i;

	/* first step is to calculate the set of all critical sections that overlap
	with T_i's execution
	*/

	std::set<std::set<unsigned int>> overlapping_cs;

	unsigned int x;
	enumerate(taskset, task, x) {


		const CriticalSections& tx_cs = taskset_cs[x].get_cs();

		unsigned int cs_index;
		enumerate(tx_cs, cs, cs_index) {

			/* we continue without incrementing y as we only consider outermost
			critical sections */
			if (cs->is_nested()) {
				continue;
			}

			std::vector<unsigned int> cs_res_ids = taskset_cs[x].get_all_cs_resource_ids(cs_index);
			std::set<unsigned int> set_res_ids;

			for(auto t = cs_res_ids.begin(); t != cs_res_ids.end(); ++t) {
				set_res_ids.insert(*t);
			}

			overlapping_cs.insert(set_res_ids);

		}

	}

	unsigned int s_index;
	enumerate(overlapping_cs, s, s_index) {

		/* all resources in s belong to the same group, so whatever happens to be
		the first element produced by the iterator is sufficient to lookup the
		group id
		*/
		// unsigned int s_group_id = resource_group_mapping[*(s->begin())];

		/* sum over tasks in each cluster */
		unsigned int x;
		enumerate(taskset, task, x) {

			LinearExpression *exp = new LinearExpression();
			
			/* ignore task being analyzed, or tasks not in the current cluster */
			if(x == analyzed_task_id) {
				continue;
			}

			const CriticalSections& tx_cs = taskset_cs[x].get_cs();

			unsigned int y = 0;
			unsigned int cs_index;
			enumerate(tx_cs, cs, cs_index) {

				/* we continue without incrementing y as we only consider outermost
				critical sections */
				if (cs->is_nested()) {
					continue;
				}

				std::vector<unsigned int> cs_res_ids = taskset_cs[x].get_all_cs_resource_ids(cs_index);

				bool all_in_s = true;
				for(auto t = cs_res_ids.begin(); t != cs_res_ids.end(); ++t) {
					if (!s->count(*t)) {
						all_in_s = false;
					}
				}

				if (!all_in_s) {
					y++;
					continue;
				}

				unsigned int overlapping_jobs = num_jobs_to_consider(*task);
				for (unsigned int v = 0; v < overlapping_jobs; v++) {

					var_t var_rsm = vars.lookup(x, y, v, BLOCKING_RSM);
					exp->add_var(var_rsm);
				
				}

				y++;

			}

			/* sum of rsm variables can't exceed F(s) */
			add_inequality(
				exp,
				num_resource_set_conflicts(*s)
			);

		}

	}


}

void GIPPLinearProgram::set_per_cluster_conflicting_rsm_blocking_constraint()
{

	unsigned int analyzed_task_id = i;

	/* first step is to calculate the set of all critical sections that overlap
	with T_i's execution
	*/

	std::set<std::set<unsigned int>> overlapping_cs;

	unsigned int x;
	enumerate(taskset, task, x) {


		const CriticalSections& tx_cs = taskset_cs[x].get_cs();

		unsigned int cs_index;
		enumerate(tx_cs, cs, cs_index) {

			/* we continue without incrementing y as we only consider outermost
			critical sections */
			if (cs->is_nested()) {
				continue;
			}

			std::vector<unsigned int> cs_res_ids = taskset_cs[x].get_all_cs_resource_ids(cs_index);
			std::set<unsigned int> set_res_ids;

			for(auto t = cs_res_ids.begin(); t != cs_res_ids.end(); ++t) {
				set_res_ids.insert(*t);
			}

			overlapping_cs.insert(set_res_ids);

		}

	}

	unsigned int s_index;
	enumerate(overlapping_cs, s, s_index) {

		/* all resources in s belong to the same group, so whatever happens to be
		the first element produced by the iterator is sufficient to lookup the
		group id
		*/
		unsigned int s_group_id = resource_group_mapping[*(s->begin())];

		/* for all clusters */
		for(unsigned int k = 0; k < num_procs / cluster_size; k++) {

			LinearExpression *exp = new LinearExpression();

			/* sum over tasks in each cluster */
			unsigned int x;
			enumerate(taskset, task, x) {
				
				/* ignore task being analyzed, or tasks not in the current cluster */
				if(task->get_cluster() != k || x == analyzed_task_id) {
					continue;
				}

				const CriticalSections& tx_cs = taskset_cs[x].get_cs();

				unsigned int y = 0;
				unsigned int cs_index;
				enumerate(tx_cs, cs, cs_index) {

					/* we continue without incrementing y as we only consider outermost
					critical sections */
					if (cs->is_nested()) {
						continue;
					}

					std::vector<unsigned int> cs_res_ids = taskset_cs[x].get_all_cs_resource_ids(cs_index);

					bool all_in_s = true;
					for(auto t = cs_res_ids.begin(); t != cs_res_ids.end(); ++t) {
						if (!s->count(*t)) {
							all_in_s = false;
						}
					}

					if (!all_in_s) {
						y++;
						continue;
					}

					unsigned int overlapping_jobs = num_jobs_to_consider(*task);
					for (unsigned int v = 0; v < overlapping_jobs; v++) {

						var_t var_rsm = vars.lookup(x, y, v, BLOCKING_RSM);
						exp->add_var(var_rsm);
					
					}

					y++;

				}

			}

			unsigned int cluster_term = cluster_size;
			unsigned int beta_term = cluster_betas[k][s_group_id];
			/* decrease by own in examined task's own cluster as it cannot block itself */
			if (taskset[analyzed_task_id].get_cluster() == k) {
				
				if (cluster_term > 0) {
					cluster_term -= 1;
				}

				if (beta_term > 0) {
					beta_term -= 1;
				}

			}

			/* sum of rsm variables can't exceed F(s) * min(cluster_size, beta_{k,g}) */
			add_inequality(
				exp,
				num_resource_set_conflicts(*s) * std::min(cluster_term, beta_term)
			);


		}

	}

}


void GIPPLinearProgram::set_f_term_rsm_blocking_constraint()
{
	/*
	This contraints the RSM blocking that a task incurs.

	The f_terms represent how many distinct "pairs" of overlapping critical sections between T_i
	and all other tasks exist. We say a pair overlaps when one cs contains a resource that is
	"greater" than a resource in the other cs. See GIPP paper for deeper explanation.
	*/

	unsigned int analyzed_task_id = (unsigned int) i;

	/* for all groups */
	unsigned int group_id;
	enumerate(resource_groups, group, group_id)
	{

		LinearExpression *exp = new LinearExpression();

		/* sum over tasks in each cluster */
		unsigned int x;
		enumerate(taskset, task, x) {
			
			/* ignore task being analyzed, or tasks not in the current cluster */
			if(x == analyzed_task_id) {
				continue;
			}

			/* sum over resources in the gruop */
			foreach((*group), a) {

				/* sum over all thetas */
				for(unsigned int v = 0; v < task_thetas[x][*a]; v++) {

					var_t var_rsm = vars.lookup(x, *a, v, BLOCKING_RSM);
					exp->add_var(var_rsm);

				}

			}

		}

		add_inequality(
			exp,
			(f_terms[group_id]) * std::min(num_procs-1, system_betas[group_id]-1)
		);

	}

}

void GIPPLinearProgram::calculate_task_thetas()
{
	
	if (G_ENABLE_DEBUG) {
		std::cout << "theta values for task_id: " << i << "--------" << std::endl;
	}

	unsigned int task_id;
	enumerate(taskset, task, task_id) {

		task_thetas.push_back(std::vector<unsigned int>(resource_group_mapping.size(), 0));
		
		/* as task does not "overlap" with itself */
		if(task_id == (unsigned int)i) {
			continue;
		}

		unsigned int overlapping_jobs = num_jobs_to_consider(*task);
		unsigned int resource_id;
		enumerate(task_outermost_accesses[task_id], resource_count, resource_id) {

			task_thetas[task_id][resource_id] += (*resource_count) * overlapping_jobs;

		}

		if(G_ENABLE_DEBUG) {
			std::cout << "task_id: " << task_id << std::endl;
			std::cout << "thetas: " << task_thetas[task_id] << std::endl;
		}
	}

	if (G_ENABLE_DEBUG) {
		std::cout << std::endl;
	}
}

void GIPPLinearProgram::calculate_f_terms()
{
	// if (G_ENABLE_DEBUG) {
	// 	G_DEBUG("f_terms for task_id: " << i << "--------")
	// }

	unsigned int group_id;
	enumerate(resource_groups, group, group_id)
	{
		
		f_terms.push_back(0);

		const CriticalSections& tx_cs = taskset_cs[i].get_cs();
		uint2D task_all_cs_res_ids = uint2D();

		/* for all outmost critical sections of the task */
		unsigned int cs_index;
		// enumerate(tx_cs, cs, cs_index)
		// {

		// 	G_DEBUG("cs_index=" << cs_index << " res_id=" << (*cs).resource_id << " outermost_index=" << (*cs).outer)

		// }
		
		enumerate(tx_cs, cs, cs_index)
		{

			/* we focus on one group at a time */
			if (resource_group_mapping[(*cs).resource_id] != group_id || cs->is_nested())
			{
				continue;
			}	

			// builds a 2D vector whered
			std::vector<unsigned int> z = taskset_cs[i].get_all_cs_resource_ids(cs_index);
			task_all_cs_res_ids.push_back(z);

		}

		unsigned int task_id_x;
		enumerate(taskset, task_x, task_id_x) 
		{

			if ((int) task_id_x == i)
			{
				continue;
			}
			

			// _x suffices denotes the task we are comparing against.
			const CriticalSections& tx_cs_x = taskset_cs[task_id_x].get_cs();
			uint2D task_x_all_cs_res_ids = uint2D();

			unsigned int overlapping_jobs = num_jobs_to_consider(*task_x);

			// if (G_ENABLE_DEBUG) {
			// 	std::cout << "task_id=" << task_id_x << ", overlapping_jobs=" << overlapping_jobs << ", group_id=" << group_id << " cs's to consider -------- " << std::endl;
			// }

			/* for all critical sections of the task */
			unsigned int cs_index_x;
			enumerate(tx_cs_x, cs_x, cs_index_x)
			{

				/* we focus on one group at a time */
				if (resource_group_mapping[(*cs_x).resource_id] != group_id || cs_x->is_nested())
				{
					continue;
				}

				for (unsigned int k = 0; k < overlapping_jobs; k++)
				{
					task_x_all_cs_res_ids.push_back(taskset_cs[task_id_x].get_all_cs_resource_ids(cs_index_x));
				}


			}

			// if (G_ENABLE_DEBUG) {
			// 	std::cout << std::endl << std::flush;
			// }

			unsigned int res_ids_index_x;
			enumerate(task_x_all_cs_res_ids, res_ids_x, res_ids_index_x)
			{

				unsigned int res_ids_index;
				enumerate(task_all_cs_res_ids, res_ids, res_ids_index)
				{

					bool found = false;

					// check to see if a critical section of task_i and of task_x
					// have a critical section that touch the same resources
					for (unsigned int k = 0; k < (*res_ids).size(); k++)
					{

						for (unsigned int l = 0; l < (*res_ids_x).size(); l++)
						{

							/* is it the case that a resource_id in task_i's critical section is
							"less" than a resource in task_x's critical section?
							*/
							unsigned int greater_than_id = (*res_ids_x)[l];
							if (resource_greater_than[greater_than_id].find((*res_ids)[k]) != resource_greater_than[greater_than_id].end())
							{
								f_terms[group_id]++;
								task_all_cs_res_ids.erase(task_all_cs_res_ids.begin() + res_ids_index);
								found = true;
								break;
							}

						}

						/* break out of res_ids inner for loop as the vector has been reduced in size */
						if (found)
						{
							break;
						}

					}

					/* break out of res_ids outer for loop as the criticail section associated with the res_ids
					has now been "paired" with another critical section of another task. */
					if (found)
					{
						break;
					}

				// end of task_i (analyzed task) critical sections enumeration
				}

			// end of task_x critical sections enumeration
			}

		// end of task set enumeration
		}

	// end of group enumeration
	}

	if (G_ENABLE_DEBUG) {
		std::cout << "f_terms for task_id " << i << " ---------" << std::endl;
		for (unsigned int j = 0; j < f_terms.size(); j++) {
			printf("F_[%d][%d] = %d\n", i, j, f_terms[j]);
		}
		std::cout << std::endl << std::flush;
	}

}

void GIPPLinearProgram::calculate_token_contentions()
{
	/*
	Calculating W_{i,g} as seen in the paper.
	In case its unclear, i is the index of the tasking being examined, and
	task_id is the task being compared against (e.g. Ti vs Tx)
	*/

	if (G_ENABLE_DEBUG) {
		std::cout << "W_i,g values for task_id: " << i << "--------" << std::endl;
	}

	token_contention_max = std::vector<unsigned int>(resource_groups.size());
	unsigned int task_cluster = taskset[i].get_cluster();

	unsigned int group_id;
	enumerate(resource_groups, group, group_id) {
		/*
		if less than or equal to cluster_size tasks in the cluster access a resource
		from a group, then a token is always free, so no contention
		*/
		if (cluster_betas[task_cluster][group_id] <= cluster_size) {
			token_contention_max[group_id] = 0;
		} else {

			unsigned int phi = task_phis[(unsigned int)i][group_id];
			int phi_alternate = 0;
			unsigned int task_id;
			enumerate(taskset, task, task_id) {
				
				if (task_id == (unsigned int)i || task->get_cluster() != task_cluster) {
					continue;
				}

				phi_alternate += task_phis[task_id][group_id] * num_jobs_to_consider(*task);
			}

			phi_alternate += 1;
			phi_alternate -= cluster_size;

			/* this shouldn't ever be less than 0 */
			assert(phi_alternate >= 0);

			if (phi < (unsigned int) phi_alternate) {
				token_contention_max[group_id] = phi;
			} else {
				token_contention_max[group_id] += (unsigned int) phi_alternate;
			}
		
		}

		if (G_ENABLE_DEBUG) {
			std::cout << "g: " << group_id << ", count: " << token_contention_max[group_id] << std::endl;
		}

	}

	if (G_ENABLE_DEBUG) {
		std::cout << std::endl;
	}
}

void generate_resource_greater_than(
	unsigned int num_resources,
	const CriticalSectionsOfTasks& taskset_cs,
	std::vector<std::set<unsigned int>>& resource_greater_than
)
{

	for (unsigned int j = 0; j < num_resources; j++)
	{
		resource_greater_than.push_back(std::set<unsigned int>());
	}

	for (unsigned int j = 0; j < taskset_cs.size(); j++)
	{

		const CriticalSections& task_cs = taskset_cs[j].get_cs();

		unsigned int cs_index;
		enumerate(task_cs, cs, cs_index)
		{

			unsigned int resource_id = cs->resource_id;
			resource_greater_than[resource_id].insert(resource_id);

			/* if not nested, then the logic below doesn't apply */
			if (!cs->is_nested())
			{
				continue;
			}

			unsigned int working_cs_index = cs_index;
			while (true) {
				unsigned int parent_cs_index = taskset_cs[j].get_enclosing_cs_index(working_cs_index);
				unsigned int parent_resource_id = task_cs[parent_cs_index].resource_id;

				resource_greater_than[parent_resource_id].insert(resource_id);

				if (!task_cs[parent_cs_index].is_nested())
				{
					break;
				}

				working_cs_index = parent_cs_index;
			}

		}
	}

	return;
}

void generate_single_resource_group(
	const TaskInfos& taskset,
	const CriticalSectionsOfTasks& taskset_cs,
	ResourceGroups& resource_groups)
{

	/*
	This is a variant of generate_resource_groups that is used to build a single group
	so that the GIPP analysis can be applied to the RNLP. */

	G_DEBUG("inside generate_single_resource_group")

	GroupSet cs_set;

	/* for all tasks */
	unsigned int x;
	enumerate(taskset, tx, x)
	{

		const CriticalSections& tx_cs = taskset_cs[x].get_cs();

		/* for all critical sections of the task */
		unsigned int cs_index;
		enumerate(tx_cs, cs, cs_index)
		{

			const auto outermost_index = taskset_cs[x].get_outermost(cs_index);
			const auto outermost_resource_id = tx_cs[outermost_index].resource_id;

			cs_set.insert(outermost_resource_id);
			cs_set.insert(cs->resource_id);

		}
	}

	resource_groups.push_back(cs_set);

	return;
}

void generate_resource_groups(
	const TaskInfos& taskset,
	const CriticalSectionsOfTasks& taskset_cs,
	ResourceGroups& resource_groups)
{

	/*
	populates the mapping from group_id to a set of resource_ids. These are equivelance classes,
	so if one task accesses resources A and C, and another task accesses B and C, then
	all three resources should have the same group ID */

	G_DEBUG("inside generate_resource_groups")

	ResourceGroups temp_groups;

	/* for all tasks */
	unsigned int x;
	enumerate(taskset, tx, x)
	{

		const CriticalSections& tx_cs = taskset_cs[x].get_cs();

		/* for all critical sections of the task */
		unsigned int cs_index;
		enumerate(tx_cs, cs, cs_index)
		{

			const auto outermost_index = taskset_cs[x].get_outermost(cs_index);

			const auto outermost_resource_id = tx_cs[outermost_index].resource_id;

			/*
			construct a max two element set consisting of the resource_id of the
			critical section, and the resource_id of the outermost critical section
			*/

			GroupSet cs_set;
			cs_set.insert(outermost_resource_id);
			cs_set.insert(cs->resource_id);

			/*
			at this point it suffices to see if its possible to merge with any
			exiting group as unions are commutative
			*/
			bool merged = false;
			for(unsigned int i = 0; i < temp_groups.size(); i++) {

				if (!is_disjoint(temp_groups[i], cs_set)) {
					temp_groups[i] = get_union(temp_groups[i], cs_set);
					merged = true;
				}

			}

			/* couldn't union with any existing group, so we add a new one */
			if (!merged) {
				temp_groups.push_back(cs_set);
			}

		}
	}

	/*
	the above code can produce groups which aren't disjoint.
	the following code performs the last set of unions and
	ensures the groups are disjoint
	*/
	for (unsigned int i = 0; i < temp_groups.size(); i++) {

		if (temp_groups[i].empty()) {
			continue;
		}

		for(unsigned int g = i+1; g < temp_groups.size(); g++) {

			if(!is_disjoint(temp_groups[i], temp_groups[g])) {
				temp_groups[i] = get_union(temp_groups[i], temp_groups[g]);
				temp_groups[g].clear();
			}

		}

		resource_groups.push_back(temp_groups[i]);
	}

	return;
}

void generate_resource_group_mapping(
	const ResourceGroups& resource_groups,
	std::vector<unsigned int>& resource_group_mapping)
{

	/* calculate number of resources */
	/* remember, groups are disjoint */
	unsigned int num_resources = 0;
	foreach(resource_groups, group) {
		num_resources += group->size();
	}

	/* allocate resource to group mapping */
	resource_group_mapping = std::vector<unsigned int>(num_resources, 0);

	unsigned int group_id = 0;
	enumerate(resource_groups, group, group_id) { // group_id = index dell'iteratore, group = oggetto group
	
	// for (typeof(resource_groups.begin()) group = ({group_id = 0; (resource_groups).begin();}); group != (resource_groups).end(); group++, group_id++)

		for (auto resource = group->begin(); resource != group->end(); ++resource) {
			
			/* sanity assertion to make sure resource_ids area really 0-indexed */
			assert((*resource) < num_resources);
			resource_group_mapping[*resource] = group_id;
		
		}

	}

	return;
}

void generate_constants(
	const TaskInfos& taskset,
	const CriticalSectionsOfTasks& taskset_cs,
	const std::vector<unsigned int>& resource_group_map,
	const unsigned int num_resource_groups,
	const unsigned int num_clusters,
	uint2D& task_phis,
	uint2D& cluster_betas,
	std::vector<unsigned int>& system_betas,
	uint2D& task_outermost_accesses,
	uint2D& task_outermost_cs_max_lengths
)
{
	/* 
	large function that isn't split up into smaller functions to avoid
	looping over the task set again and again.
	Calculates:
		- phi values
		- beta values
		- # of times a task issues an outermost request for a resource
	*/

	/* intialize datastructure for per cluster resource counts */
	for (unsigned int i = 0; i < num_clusters; i++) {
		cluster_betas.push_back(std::vector<unsigned int>(num_resource_groups, 0));
	}

	/* initialize datastructure for system cluster resource counts */
	for (unsigned int i = 0; i < num_resource_groups; i++)
	{
		system_betas.push_back(0);
	}

	/* for all tasks */
	unsigned int tx_index;
	enumerate(taskset, tx, tx_index)
	{
		/* 
		num clusters = num_procs / cluster_size
		we assume clusters are 0-indexed, and that there are no "skipped" indicies
		i.e., there isn't a task with cluster=0, another with cluster=2,
		but no task with cluster=1
		*/
		assert(tx->get_cluster() < num_clusters);

		/* initialize the outermost access counts to 0 for each group */
		task_phis.push_back(std::vector<unsigned int>(num_resource_groups, 0));

		/* intialize temporary vector to store which resource groups
		the task touchs */
		std::vector<bool> temp_betas(num_resource_groups, false);

		/* populate outermost access structure for tx */
		task_outermost_accesses.push_back(
			std::vector<unsigned int>(resource_group_map.size(), 0)
			);

		task_outermost_cs_max_lengths.push_back(
			std::vector<unsigned int>(resource_group_map.size(), 0)
			);

		const CriticalSections& tx_cs = taskset_cs[tx_index].get_cs();

		/* for all critical sections of the task */
		unsigned int cs_index;
		enumerate(tx_cs, cs, cs_index)
		{

			if (cs->is_nested()) {
				continue;
			}

			unsigned int group_id = resource_group_map[cs->resource_id];

			task_phis[tx_index][group_id] += 1;
			temp_betas[group_id] = true;
			task_outermost_accesses[tx_index][cs->resource_id] += 1;

			if(cs->length > task_outermost_cs_max_lengths[tx_index][cs->resource_id]) {
				task_outermost_cs_max_lengths[tx_index][cs->resource_id] = cs->length;
			}

		}

		/*
		if the task accesses a group, increment the per cluster and system group
		accesses count by one
		*/
		unsigned int group_index;
		enumerate(temp_betas, is_access, group_index) {

			if (*is_access) {
				cluster_betas[tx->get_cluster()][group_index] += 1;
				system_betas[group_index] += 1;
			}
			
		}
	}

	return;
}

BlockingBounds* lp_gipp_bounds(
	const ResourceSharingInfo& info,
	const CriticalSectionsOfTaskset& tsk_cs,
	unsigned int num_procs,
	unsigned int cluster_size,
	bool force_single_group)
{
	assert(num_procs >= cluster_size);
	assert(num_procs % cluster_size == 0);

	if (G_ENABLE_DEBUG) {
		std::cout << "GIPP lp_gipp_bounds start --------" << std::endl << std::flush;

		std::cout << "Generating Resource Groups --------" << std::endl << std::flush;
	}
	
	ResourceGroups resource_groups;
	if (force_single_group) { // RTOS: valore impostato a TRUE dalla chiamata
		generate_single_resource_group(info.get_tasks(), tsk_cs.get_tasks(), resource_groups);
	} else {
		generate_resource_groups(info.get_tasks(), tsk_cs.get_tasks(), resource_groups);
	}

	if (G_ENABLE_DEBUG) {
		std::cout << "Generating Resource Group Mapping --------" << std::endl << std::flush;
	}

	std::vector<unsigned int> resource_group_mapping;
	generate_resource_group_mapping(resource_groups, resource_group_mapping); // RTOS: genera una mappa con resource_group_mapping

	if (G_ENABLE_DEBUG) {
		std::cout << "Generating Constants --------" << std::endl << std::flush;
	}

	/* calculate number of resources */
	/* remember, groups are disjoint */
	unsigned int num_resources = 0;
	foreach(resource_groups, group) {
		num_resources += group->size();
	}

	/* calculate partial ordering */
	std::vector<std::set<unsigned int>> resource_greater_than;
	generate_resource_greater_than(num_resources, tsk_cs.get_tasks(), resource_greater_than);

	uint2D task_outermost_accesses; // vector<std::vector<unsigned int>>

	uint2D task_phis;
	uint2D cluster_betas;
	std::vector<unsigned int> system_betas;
	uint2D task_outermost_cs_max_lengths;
	generate_constants(				// Restituisce l'oggetto relativo per riferimento
		info.get_tasks(),
		tsk_cs.get_tasks(),
		resource_group_mapping,
		resource_groups.size(),
		num_procs / cluster_size,
		task_phis,
		cluster_betas,
		system_betas,
		task_outermost_accesses,
		task_outermost_cs_max_lengths
	);

	/* debug output */
	if (G_ENABLE_DEBUG) {
		std::cout << "Groups --------" << std::endl;
		for (unsigned int i= 0; i < resource_groups.size(); i++) {
			std::cout << i << ": " << resource_groups[i] << std::endl;
		}

		std::cout << std::endl;
		std::cout << "Resource To Group Mapping ----" << std::endl;
		unsigned int resource_id;
		enumerate(resource_group_mapping, group_id, resource_id) {
			std::cout << "resource_id: " << resource_id << ", group_id: " << *group_id << std::endl;
		}

		std::cout << std::endl;
		G_DEBUG("Resource Partial Ordering --------")
		for (unsigned int j = 0; j < resource_greater_than.size(); j++) {
			G_DEBUG(resource_greater_than[j])
		}
		std::cout << std::endl;	

		std::cout << std::endl;
		std::cout << "Task Phis --------" << std::endl;
		unsigned tp_index;
		enumerate(task_phis, tp, tp_index) {

			std::cout << "task_index: " << tp_index << std::endl;

			unsigned group_id;
			enumerate((*tp), count, group_id) {
				std::cout << "group_id: " << group_id << ", number_of_accesses: " << *count << std::endl;
			}

			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << "Cluster Betas -------" << std::endl;
		unsigned cb_index;
		enumerate(cluster_betas, cb, cb_index) {

			std::cout << "cluster_index: " << cb_index << std::endl;

			unsigned group_id;
			enumerate((*cb), group_count, group_id) {
				std::cout << "groud_id: " << group_id << ", count: " << *group_count << std::endl;
			}

			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << "System Betas -------" << std::endl;
		for (unsigned int j = 0; j < system_betas.size(); j++)
		{

			G_DEBUG("group_id: " << j << ", count: " << system_betas[j])

		}

		std::cout << std::endl;
		std::cout << "Task Outermost Accesses (i.e. N_{i,a} values) --------" << std::endl;
		unsigned int task_index;
		enumerate(task_outermost_accesses, toa, task_index) {

			std::cout << "task_index: " << task_index << std::endl;
			unsigned int resource_index;
			enumerate((*toa), resource_count, resource_index) {
				std::cout << "resource_id: " << resource_index << ", count:" << *resource_count << std::endl;
			}
		}

		std::cout << std::endl << std::endl;

	}

	BlockingBounds* results = new BlockingBounds(info);

	for (unsigned int i = 0; i < info.get_tasks().size(); i++)
	{
		GIPPLinearProgram lp(
				info,
				tsk_cs, 
				i, 
				num_procs, 
				cluster_size,
				resource_groups,
				resource_group_mapping,
				resource_greater_than,
				task_phis,
				task_outermost_accesses,
				task_outermost_cs_max_lengths,
				cluster_betas,
				system_betas);

		(*results)[i] = lp.solve();
	}

	return results;
}
