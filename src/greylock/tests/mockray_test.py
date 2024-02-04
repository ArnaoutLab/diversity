"""
Testing the testing infrastructure!
"""

from greylock.tests.mockray import put, remote, wait, get


def test_mock():
    @remote
    def interesting(n):
        return [(n**k) for k in range(5)]

    thing = (5, "foobar")
    assert put(thing) == thing

    future = interesting.remote(3)
    one_result = get(future)
    assert len(one_result) == 5

    futures = []
    results = []
    for n in range(10):
        if len(futures) > 4:
            ready_refs, futures = wait(futures)
            results += get(ready_refs)
        futures.append(interesting.remote(n))
    results += get(futures)
    assert len(results) == 10
    results.sort()
    for n, result in enumerate(results):
        assert result[1] == n
