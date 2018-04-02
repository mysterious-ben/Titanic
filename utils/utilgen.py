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
