def pytest_collectstart(collector):
    # We need to ignore UserWarning: Trying to unpickle estimator when loading a pickle from an earlier
    # version of sklearn. The best solution would be to use a regex in nbval_sanitize.cfg to replace the
    # warnings by a constant string, but I haven't been able to reproduce them outside the test environment.
    # This is a little coarse-grained, since it will disregard all stderr output.
    if hasattr(collector, 'skip_compare'):
        collector.skip_compare += 'text/html', 'application/javascript', 'stderr',
