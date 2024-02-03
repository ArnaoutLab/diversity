"""
Mock versions of ray functions for unit testing
"""

from collections import namedtuple
import math

counter = 0
results_store = {}


def get_next_token():
    global counter
    counter += 1
    return counter


def put(data):
    return data


RemoteWrap = namedtuple("RemoteWrap", ["remote"])


def remote(func):
    def remote(*args, **kwargs):
        token = get_next_token()
        result = func(*args, **kwargs)
        results_store[token] = result
        return token

    return RemoteWrap(remote=remote)


def wait(tokens):
    # Choose index of job to report as "ready".
    # Generally, don't choose the first one; clients
    # should not count on jobs finishing in order.
    ready_index = math.floor(len(tokens) / 3)
    ready_refs = [tokens[ready_index]]
    futures = tokens[:ready_index] + tokens[(ready_index + 1) :]
    return (ready_refs, futures)


def get(tokens):
    results = []
    if not hasattr(tokens, "__iter__"):
        tokens = [tokens]
    for token in tokens:
        results.append(results_store[token])
        del results_store[token]
    return results
