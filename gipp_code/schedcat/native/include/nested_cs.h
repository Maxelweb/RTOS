#ifndef NESTED_CS_H
#define NESTED_CS_H

#ifndef SWIG
#include <vector>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>
#include "stl-hashmap.h"
#endif

class CriticalSectionsOfTask;

typedef std::set<unsigned int> LockSet;

struct CriticalSection
{
	unsigned int resource_id;
	unsigned int length; /* excluding nested requests, if any */
	int outer; /* index of containing critical section, -1 if outermost */

	enum {
		NO_PARENT = -1,
	};

	CriticalSection(unsigned int res_id, unsigned int len,
	                int outer_cs = NO_PARENT)
		: resource_id(res_id), length(len), outer(outer_cs) {}

	// return the set of resources already held when this resource is requested
	LockSet get_outer_locks(const CriticalSectionsOfTask &task) const;

	bool is_nested() const
	{
		return outer != NO_PARENT;
	}

	bool is_outermost() const
	{
		return outer == NO_PARENT;
	}

	bool has_common_outer(
		const CriticalSectionsOfTask &this_task,
		const LockSet &already_held_by_other) const;

	bool has_common_outer(
		const CriticalSectionsOfTask &this_task,
		const CriticalSection &other_cs,
		const CriticalSectionsOfTask &other_task) const
	{
		/* first check that neither is outermost */
		if (is_outermost() || other_cs.is_outermost())
			return false;
		else
			return other_cs.has_common_outer(
				this_task, other_cs.get_outer_locks(other_task));
	}
};


typedef std::vector<CriticalSection> CriticalSections;

class CriticalSectionsOfTask
{
	CriticalSections cs;

public:

	const CriticalSections& get_cs() const
	{
		return cs;
	}

	operator const CriticalSections&() const
	{
		return cs;
	}

	void add(unsigned int res_id, unsigned int len,
	         int outer_cs = CriticalSection::NO_PARENT)
	{
		// resource_ids don't necessarily come in in ascending order
		// so we reserve enough space for the maximum id we see.
		cs.reserve(std::max((int) res_id, outer_cs) + 1);
		assert( outer_cs == CriticalSection::NO_PARENT
		        || outer_cs < (int) cs.capacity() );
		
		cs.push_back(CriticalSection(res_id, len, outer_cs));
	}

	bool has_nested_requests(unsigned int cs_index) const
	{
		// not implemented correctly, but not used for GIPP either
		unsigned int res = cs[cs_index].resource_id;

		for (int i = 0; i < (int) cs.size(); i++) {
			if (cs[i].outer == (int) res) {
				return true;
			}
		}
		
		return false;
	}

	unsigned int get_outermost(unsigned int cs_index) const
	{
		unsigned int cur = cs_index;

		while (cs[cur].is_nested()) {
			
			cur = get_enclosing_cs_index(cur);

		}

		return cur;
	}

	std::vector<unsigned int> get_all_cs_resource_ids(unsigned int cs_index) const
	{
		/* for a given cs_index, return the resource_id of the cs
		and the resource_ids of all nested critical sectons of the cs */
		std::vector<unsigned int> ret = std::vector<unsigned int>();

		for (unsigned int i = 0; i < cs.size(); i++)
		{			

			unsigned int outermost = get_outermost(i);
			if (outermost == cs_index) {
				ret.push_back(cs[i].resource_id);
			}

		}

		return ret;
	}

	unsigned int get_enclosing_cs_index(unsigned int cs_index) const
	{

		if (!cs[cs_index].is_nested()) {
			return cs_index;
		}

		for (unsigned int i = 0; i < cs.size(); i++) {

				if (i == cs_index) {
					continue;
				}

				if ((int) i == cs[cs_index].outer) {
					return i;
				}
		}

		return 0;
	}

	bool is_nested_in(unsigned int cs_index, unsigned int outer) const
	{
		unsigned int cur = cs_index;

		while (cs[cur].is_nested()) {
			cur = get_enclosing_cs_index(cur);
			if (cur == outer)
				return true;
		}

		return false;
	}


	unsigned long get_total_length(unsigned int outer) const
	{
		unsigned long total = cs[outer].length;

		for (unsigned int i = 0; i < cs.size(); i++) {
			if (is_nested_in(i, outer))
				total += cs[i].length;
		}

		return total;
	}

};



typedef std::vector<CriticalSectionsOfTask> CriticalSectionsOfTasks;

class CriticalSectionsOfTaskset
{
	CriticalSectionsOfTasks tsks;

public:

	const CriticalSectionsOfTasks& get_tasks() const
	{
		return tsks;
	}

	operator const CriticalSectionsOfTasks&() const
	{
		return tsks;
	}

	CriticalSectionsOfTask& new_task()
	{
		tsks.push_back(CriticalSectionsOfTask());
		return tsks.back();
	}

	/* Compute for each resource 'q' the set of resources that could be
	 * transitively requested while holding 'q'. */
	hashmap<unsigned int, hashset<unsigned int> >
	get_transitive_nesting_relationship() const;
};


void dump(const CriticalSectionsOfTaskset &x);

BlockingBounds* lp_nested_fifo_spinlock_bounds(
	const ResourceSharingInfo& info,
	const CriticalSectionsOfTaskset& tsk_cs,
	bool rnlp = false,
	bool rnlp_nesting_aware = false);

BlockingBounds* lp_gipp_bounds(
	const ResourceSharingInfo& info,
	const CriticalSectionsOfTaskset& tsk_cs,
	unsigned int num_procs,
	unsigned int cluster_size,
	bool force_single_group);

#endif
