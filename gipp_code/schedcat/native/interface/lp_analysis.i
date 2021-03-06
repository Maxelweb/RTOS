%module lp_analysis
%{
#define SWIG_FILE_WITH_INIT
#include "lp_analysis.h"
#include "nested_cs.h"
%}

%newobject lp_dpcp_bounds;
%newobject lp_dflp_bounds;

%newobject lp_msrp_bounds;
%newobject lp_pfp_preemptive_fifo_spinlock_bounds;

%newobject lp_pfp_unordered_spinlock_bounds;

%newobject lp_pfp_prio_spinlock_bounds;

%newobject lp_pfp_prio_fifo_spinlock_bounds;

%newobject lp_pfp_baseline_spinlock_bounds;

%newobject dummy_bounds;

%include "sharedres_types.i"

%include "lp_analysis.h"

%ignore CriticalSectionsOfTaskset::get_transitive_nesting_relationship;

%include "nested_cs.h"
