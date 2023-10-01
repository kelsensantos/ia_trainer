import re
import os
import spacy

from datetime import datetime
from nltk.corpus import stopwords
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from joblib import load
from urllib.request import urlretrieve, urlcleanup

from juspln.auxiliares import *


# recursos
words_to_stop = set(stopwords.words('portuguese'))
lemmatizador = spacy.load('pt_core_news_md')


# recurso
TRAINED_MODELS = {
    'kelsensantos/bow_peticoes_classificador_tipo': 'https://huggingface.co/kelsensantos'
                                                    '/bow_peticoes_classificador_tipo/resolve/main/bow_model.pkl',
    'kelsensantos/bow_peticoes_classificador_assunto': 'https://huggingface.co/kelsensantos'
                                                       '/bow_peticoes_classificador_assunto/resolve/main/bow_model.pkl'
}


def pre_processar(texto):
    """"Realiza pre porcessamento do texto"""
    # remove caixa alta
    texto = texto.lower()
    # remove corte de linha
    texto = texto.replace('\n', ' ')
    # transforma símbolos em espaços
    # isso evita juntar palavras ao remover os símbolos
    regex = re.compile('[^a-zA-Z1-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]')
    texto = re.sub(regex, ' ', texto)
    # retira símbolos e caracteres numéricos
    regex = re.compile('[^a-záéíóúâêîôãõçÇ ]')
    texto = re.sub(regex, '', texto)
    # tokenizacao
    tokens = re.sub("[^\w]", " ",  texto).split()
    # removendo stopwords
    tokens_filtrados = [
        token for token in tokens if
        token not in words_to_stop
        and len(token) > 1
    ]
    # lemmatizando
    doc = lemmatizador(' '.join(tokens_filtrados))
    tokens_lemmatizados = [token.lemma_ for token in doc]
    # retorna uma string preprocessada
    return ' '.join(tokens_lemmatizados)


def pre_processar_df(df: pd.DataFrame):
    # separa o dataframe em chunks processar, evitando sobrepeso de memória
    chunk_num = (len(df) // 150)
    chunks = np.array_split(df, chunk_num)
    # aplica função para pre processar texto em cada chunk
    for i, chunk in enumerate(chunks):
        now = datetime.now().strftime("%H:%M")
        print(f"Chunk {i} iniciado às {now}...")
        chunk['conteudo'] = chunk['conteudo'].apply(lambda x: pre_processar(x))
        # exporta cada chunk para controle
        chunk.to_csv(f'data/chunks/preproc_chunk_{i}.csv')
        # ponto de controle
        print(f"Chunk {i} finalizado...")
    # concatena os chunks novamente em um único dataframe
    df = pd.concat(chunks)
    return df


def processar_bow(
        vetorizadores,    # lista de vetorizadores
        classificadores,  # lista de classificadores
        X,
        y,
        identificador,
        n_folds=10,       # número de folds,
        sep='-' * 50,     # variável impressa para separação
        how='base',
        reducers=None,
        pasta_resultados=os.getcwd(),
):
    # id único da rodada
    _id = datetime.now().strftime("%Y%m%d%H%M")
    print(sep)
    print(f"ID: {_id}")

    # Listas
    listar('Vetorizadores', vetorizadores)
    listar('Classificadores', classificadores)
    print('Redução de dimentsionalidade: ', reducers)

    # Stratifier
    skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)

    # Lista de modelos a serem executados
    models = generate_models(vetorizadores, classificadores)

    # Adiciona redução de dimensionalidade, se for o caso
    if reducers is not None:
        models = add_reducer(models, reducers)

    # cria pipelines
    models = [make_pipeline(*lista, verbose=True) for lista in models]

    # etapa de redução de dimensionalidade, se definida
    if how == 'oversampled':
        X, y = oversample(X, y)
    if how == 'undersampled':
        rus = RandomUnderSampler(random_state=42)
        X, y, y_scaled = rus.fit_resample(X.reshape(-1, 1), y.reshape(-1, 1))
    # corrige o shape para processar
    X_scaled = X.reshape(-1, )
    y_scaled = y.reshape(-1, )

    print('')

    # variável para retornar resultados
    resultados = []

    # Itera sobre os modelos vetorizador/classificador
    for i, model in enumerate(models):

        # Verbose pontos de controle
        print(sep)
        print(f'Executando modelo {i}...')

        vetorizador = model[-2]
        redutor = None
        classificador = model[-1]

        if reducers is not None:
            vetorizador = model[-3]
            redutor = model[-2]

        print(f'Vetorizador: {vetorizador}')
        print(f'Redutor: {redutor}')
        print(f'Classificador: {classificador}')

        # Métricas a serem colhidas
        metrics = [
            'accuracy',
            'precision_macro', 'precision_weighted',
            'recall_macro', 'recall_weighted',
            'f1_macro', 'f1_weighted'
        ]

        # Cross validation
        scores = cross_validate(
            model,
            X_scaled,
            y_scaled,
            scoring=metrics,
            cv=skf,
            error_score='raise',
            verbose=1
        )
        scores_medios = {k: round(np.mean(v), 2) for k, v in scores.items()}

        # metricas e resultados consolidados
        dados = {
            'method': how,
            'vetorizador': vetorizador,
            'classificador': classificador,
            'redutor_dimensionalidade': redutor
        }
        dados.update(scores_medios)
        resultados.append(dados)

        # export
        pd.DataFrame(resultados).to_csv(
            f'{pasta_resultados}/metricas_{identificador}_{_id}.csv'
        )
        # ponto de controle
        print(dados)
        print(f"Modelo {i} finalizado.")
        print(sep)

    return pd.DataFrame(resultados)


def make_grid_searcher(
        vetorizador,
        classificador,
        clf_params: dict,
        vec_params: dict,
        n_jobs: int = 20,
        pre_dispatch: int = 20
):
    params = {}
    clf_new_params = {'clf__' + key: clf_params[key] for key in clf_params}
    params.update(clf_new_params)
    vec_new_params = {'vec__' + key: vec_params[key] for key in vec_params}
    params.update(vec_new_params)

    pipeline = make_pipeline(vetorizador, classificador)

    grid = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
        scoring=['f1_macro', 'f1_weighted'],
        refit='f1_macro',
        return_train_score=True,
        verbose=1,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch
    )
    return grid


def load_model(
    model_path: str,
    trained_models_paths: None | dict = None,
    filename: None | str = None,
):
    # carrega a URL onde o modelo está depositado, se existir no dicionário
    if trained_models_paths is None:
        trained_models_paths = TRAINED_MODELS
    if model_path in trained_models_paths.keys():
        url = trained_models_paths[model_path]
    else:
        url = model_path
    # carrega o modelo
    response = urlretrieve(url, filename)
    model = load(response[0])
    # limpa arquivos temporários
    urlcleanup()
    # retorna o modelo
    return model
