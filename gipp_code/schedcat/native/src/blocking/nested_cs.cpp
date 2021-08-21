#include "stl-hashmap.h"
#include "stl-helper.h"

#include "sharedres_types.h"
#include "nested_cs.h"


/* Compute for each resource 'q' the set of resources that could be
	 * transitively requested while holding 'q'. */
hashmap<unsigned int, hashset<unsigned int> >
CriticalSectionsOfTaskset::get_transitive_nesting_relationship() const
{
	hashmap<unsigned int, hashset<unsigned int> > nested;

	foreach(tsks, t)
	{
		foreach(t->get_cs(), cs)
		{
			if (nested.find(cs->resource_id) == nested.end())
				nested[cs->resource_id] = hashset<unsigned int>();

			int outer = cs->outer;
			unsigned int nested_res = cs->resource_id;

			while (outer != CriticalSection::NO_PARENT)
			{
				unsigned int parent = t->get_cs()[outer].resource_id;
				nested[parent].insert(nested_res);
				outer = t->get_cs()[outer].outer;
			}
		}
	}

	return nested;
}


LockSet CriticalSection::get_outer_locks(const CriticalSectionsOfTask &task) const
{
	LockSet already_held;

	int held = outer;
	while (held != NO_PARENT)
	{
		unsigned int parent = task.get_cs()[held].resource_id;
		already_held.insert(parent);
		held = task.get_cs()[held].outer;
	}

	return already_held;
}

bool CriticalSection::has_common_outer(
	const CriticalSectionsOfTask &this_task,
	const LockSet &already_held_by_other) const
{
	int held = outer;
	while (held != NO_PARENT)
	{
		unsigned int parent = this_task.get_cs()[held].resource_id;
		if (already_held_by_other.find(parent) != already_held_by_other.end())
			return true;
		held = this_task.get_cs()[held].outer;
	}

	return false;
}
