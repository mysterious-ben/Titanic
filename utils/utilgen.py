import numpy as np
import contextlib


def viewdf(df):
    import os
    import webbrowser

    dir = os.getcwd()
    name = 'data\\temp\\frame.html'
    path = os.path.join(dir, name)
    df.to_html(os.path.join(dir, name))
    # 'http://localhost:8888/'
    url = path
    webbrowser.open(url, new=2)


@contextlib.contextmanager
def printOptions(precisionNp=2, suppressNp=False):
    original = np.get_printoptions()
    np.set_printoptions(precision=precisionNp, suppress=suppressNp)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class prettyFloat(float):
    def __repr__(self):
        return "%0.2f" % self