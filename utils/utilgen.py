import numpy as np
import pandas as pd
import contextlib
import os
import sys


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
def printOptions(precision=2, suppressNp=False):
    npOriginal = np.get_printoptions()
    pdOriginal = pd.options.display.float_format
    np.set_printoptions(precision=precision, suppress=suppressNp)
    pd.options.display.float_format = '{:,.2f}'.format
    try:
        yield
    finally:
        np.set_printoptions(**npOriginal)
        pd.options.display.float_format = pdOriginal


class prettyFloat(float):
    def __repr__(self):
        return "%0.2f" % self

@contextlib.contextmanager
def suppressStdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout