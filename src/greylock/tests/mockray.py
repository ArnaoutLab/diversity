"""
Mock versions of ray functions for unit testing
"""

from collections import namedtuple

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


def get(tokens):
    results = []
    if not hasattr(tokens, "__iter__"):
        tokens = [tokens]
    for token in tokens:
        results.append(results_store[token])
        del results_store[token]
    return results
