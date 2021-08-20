from __future__ import division

from math import floor


def value_range(start, max, step, last=None):
    x  =  start
    while x <= max:
        yield x
        x += step
    if not last is None:
        yield last

def round_to_next_multiple(x, quantum_size=1):
    quanta = floor(x / quantum_size)
    rounded = quanta * quantum_size
    if x - rounded > quantum_size / 2:
        rounded += quantum_size
    return rounded

