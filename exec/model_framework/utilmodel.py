def dataframeToXy(data):
    if 'Survived' in data.columns:
        X = data.drop(columns=['Survived'])
        y = data['Survived'].values
    else:
        X = data.copy()
        y = None
    return X, y
