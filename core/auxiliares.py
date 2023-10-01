import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def listar(titulo, lista):
    print()
    print(titulo)
    for i, v in enumerate(lista):
        print(f"# {i} -> {v}")
    print()


def generate_models(vetorizadores, classificadores):
    models = [
        [v, c]
        for c in classificadores
        for v in [v for v in vetorizadores]
    ]
    return models


def add_reducer(models, reducers):
    new_models = []
    for reducer in reducers:
        for model in models:
            new_model = model.copy()
            new_model.insert(1, reducer)
            new_models.append(new_model)
    return new_models


def oversample(X, y):
    def limitar_sample(grupo, n=5000):
        if len(grupo) > n:
            return grupo.sample(n, random_state=42)
        else:
            return grupo.sample(len(grupo))

    df_ = pd.DataFrame(data={'texto': X, 'classes': y})
    df_ = df_.astype(dtype={'classes': 'category'})
    df_ = df_.groupby(by='classes')
    df_ = df_.apply(lambda x: limitar_sample(x)).reset_index(drop=True)

    X_new = np.array(df_['texto']).reshape(-1, 1)
    y_new = np.array(df_['classes'].cat.codes).reshape(-1, 1)

    ros = RandomOverSampler(random_state=42)
    X_scaled, y_scaled = ros.fit_resample(X_new, y_new)
    return X_scaled, y_scaled
