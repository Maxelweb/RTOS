"""Implements the BAK G-EDF schedulability test.
"""

from __future__ import division

from schedcat.util.quantor import forall

from fractions import Fraction

ONE = Fraction(1) # 1/1

def beta(t_i, t_k, l):
    # assumes integral time
    u_i = t_i.utilization_q() # Fraction(self.cost, self.period)
    part1 = u_i * (ONE + Fraction(t_i.period - t_i.deadline, t_k.deadline))
    # (t_i.cost/t_i.period)*((t_k.deadline + t_i.period - t_i.deadline)/t_k.deadline)
    if l < u_i:
        part2 = (t_i.cost - l * t_i.period) / Fraction(t_k.deadline)
        return part1 + part2
    else:
        return part1

def task_schedulable(T, t_k, m):
    l = t_k.density_q() # lambda -> confronto tra frazioni   Frazione(WCET, min(Periodo, Deadline))
    if l > ONE:
        return False
    beta_sum = sum([min(ONE, beta(t_i, t_k, l)) for t_i in T])
    return beta_sum <= m - (m - 1) * l


# RTOS: 
# def forall(lst):
#     def predicate(p):
#         for x in lst:
#             if not p(x):
#                 return False
#         return True
#     return predicate

# RTOS: forall restituisce una funzione
def is_schedulable(no_cpus, tasks):
    return forall(tasks)(lambda t_k: task_schedulable(tasks, t_k, no_cpus))


# RTOS: 28/08/2021  --> vedere rta.py forse più adeguato