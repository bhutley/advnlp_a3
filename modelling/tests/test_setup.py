def test_imports():
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import scipy  # noqa: F401
    import sklearn

    major, minor = map(int, sklearn.__version__.split(".")[:2])
    assert (major, minor) >= (1, 4)
