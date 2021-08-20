#!/usr/bin/env python

import re
import os.path

def is_comment(l):
    return len(l) > 0 and l[0] == '#'

def is_data(l):
    return len(l) > 0 and l[0] != '#'

def value(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def parse_file_name(fname):
    params = {}
    for kv in fname.split('_'):
        kvl = kv.split('=', 1)
        params[kvl[0]] = kvl[1] if len(kvl) > 1 else True
    return params

class DataFile(object):
    def __init__(self, fname):
        f = open(fname, 'r')
        lines = f.readlines()

        self.name = fname

        # find column headers
        headers = []
        for (prev, next) in zip(lines, lines[1:]):
            # convention: last line before data contains the headers
            if is_comment(prev) and is_data(next):
                # convention: headers are separated by at least two spaces
                headers = [h.strip() for h in re.split('  +', prev[1:].strip())]
                break

        # store column headers in lookup table
        self.headers = {}
        for (i, h) in enumerate(headers):
            self.headers[i] = h
            self.headers[h] = i

        # store comments
        self.comments = [l for l in lines if is_comment(l)]

        # parse data
        rows = [l for l in lines if is_data(l)]
        self.data = [[value(c.strip()) for c in row.split(',')] for row in rows]

        # split file name into components
        self.params = parse_file_name(os.path.basename(fname).replace('.csv', ''))

    def idx(self, key):
        if type(key) != int:
            return self.headers[key]
        else:
            return key

    def column(self, key):
        i = self.idx(key)
        return [row[i] for row in self.data]

    def columns(self, *keys):
        idx = [self.idx(k) for k in keys]
        return [[row[i] for i in idx] for row in self.data]

    def lookup(self, xaxis, xval, yaxis):
        try:
            i = self.column(xaxis).index(xval)
            return self.column(yaxis)[i]
        except ValueError:
            return None

def combine_files_by_param(xaxis, yaxis, param, files):
    xvals = set()
    for f in files:
        for x in f.column(xaxis):
            xvals.add(x)

    sorted_files = sorted(files, key=lambda f: f.params[param])

    header = [xaxis] + \
             ["%s=%s" % (param, f.params[param]) for f in sorted_files]

    data = [[f.lookup(xaxis, x, yaxis) for f in sorted_files]
             for x in sorted(xvals)]

    return (header, data)


