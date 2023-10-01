
# Introdução

Esse repositório é inicialmente fruto de Monografia apresentada ao Departamento de
Ciências de Computação do Instituto de Ciências Matemáticas e de Computação 
da Universidade de São Paulo - ICMC/USP -, como parte dos requisitos para
obtenção do título de Especialista m Inteligência Artificial e Big Data. 

O estudo completo pode ser acessado
[neste link](https://drive.google.com/file/d/1BXzXhQVzZWz0B4ROfOmDHfm1KuLw__jp/view?usp=sharing).

Caso o utilize, por favor citar a seguinte fonte:

> Santos, Kelsen Henrique Rolim dos. Classificação de textos de petições jurídicas
com base em técnicas de Processamento de Línguas Naturais / 
Kelsen Henrique Rolim dos Santos; orientador Thiago Alexandre Salgueiro Pardo. -
São Carlos, 2023. 112 p. : il. (algumas color.)

```
@misc{SANTOS_2023, 
    title={Classificação de textos de petições jurídicas com base em
    técnicas de Processamento de Línguas Naturais},
    author={SANTOS, KELSEN HENRIQUE ROLIM DOS},
    year={2023}
} 
```

Me contate em caso de dúvidas ou sugestões  
[kelsensantos@gmail.com](mailto:kelsensantos@gmail.com)


# Resumo

O interesse na aplicação de inteligência artificial no domínio jurídico tem sido
crescente, ante o potencial para agilizar o trabalho de profissionais e melhorar
serviços públicos, sendo o Processamento de Línguas Naturais (PLN uma das áreas
de maior utilidade. Este trabalho explora a aplicação de PLN para classificação
de petições jurídicas destinadas a processos judiciais, utilizando um conjunto
de dados produzidos pela Defensoria Pública do Estado de Rondônia. São
experimentadas técnicas de Bag of Words e Bidirectional Encoder Representations
for Transformers (BERT) para realizar classificações por Assunto e por Tipo
das petições, tendo esta se destacado segundo as métricas calculadas. Os resultados
demonstram a viabilidade de aplicação de modelos de linguagem para a auxiliar
a rotina profissionais com automatização da classificação de documentos.

# Instalação

Instale como um pacote Python.

```
# instalation code exemple

from IPython.display import clear_output
! python -m pip install git+https://github.com/kelsensantos/juspln
clear_output()

from juspln import bow, bert

```

# Bag of Words

## Para treinar novos modelos

Para melhor otimizar o processo, realize o pré processamento do texto utilizando a 
função ```pre_processar_df```. Ela retorna um DataFrame Pandas que pode ser guardado 
para uso posterior na etapa de validação cruzada. Considere que o vetorizador criado 
com ```TfidfVectorizer``` deve transformar o texto já pré-processado.

A validação cruzada é realizada pela função ```processar_bow```, sobre uma lista 
de vetorizadores e classificadores. Podem ser utilizadas os métodos implementados pela 
biblioteca [Scikit Learn](https://scikit-learn.org/stable/), como fizemos em nosso
experimento. Por favor, consulta o texto do trabalho para maiores detalhes acerca dos 
classificadores utilizados.

```
from juspln.bow import processar_bow

# exemplo de lista de classificadores
classificadores = [
    RidgeClassifier(tol=1e-2, solver="sparse_cg"),
    LinearSVC(C=1),
    RandomForestClassifier(),
    LogisticRegression(C=5, max_iter=1000)
]

# exemplo de lista de vetorizadores
[TfidfVectorizer(max_df=0.5, min_df=10,sublinear_tf=True)]

# exemplo de processamento de validação cruzada
scores = processar_bow(
    vetorizadores=vetorizadores,
    classificadores=classificadores,
    X=X_train,
    y=y_train,
    n_folds=10,
    identificador='base',
    how='base',
)
```

A validação cruzada retorna um DataFrame com os resultados, que também são gravados 
em formato ```csv``` na pasta local (controlado pelo parâmetro ```pasta_resultados```). Compare os resultados e então utilize o método 
```fit``` de um objeto GridSearchCV (que pode ser criado pela função 
```make_grid_searcher```) para identificar os melhores hiperparâmetros para
o vetorizador e classificador de melhor desempenho.  

````
# exemplo de grid search

from core.bow import make_grid_searcher

vetorizador = TfidfVectorizer()
classificador = LinearSVC()

clf_params = {
    'C': [.5, 5, 10],
    'max_iter': [100, 1000, 10000],
}

vec_params = {
    'ngram_range': [(1,1), (1,2)],
    'max_df': [.7, .5, .3],
    'min_df': [10, 35, 70, 100],
    'max_features': [None, 15000, 8000],
}

grid = make_grid_searcher(
        vetorizador=vetorizador,
        classificador=classificador,
        clf_params=clf_params,
        vec_params=vec_params,
)

grid.fit()

import joblib
joblib.dump(grip, f'{path}.pkl')
````

## Para utilizar o modelo treinado

O modelo que realiza classificações por tipo de petição está disponível na comunidade
[Hugging Face](https://huggingface.co/), no repositório 
[kelsensantos/bow_peticoes_classificador_tipo](https://huggingface.co/kelsensantos/bow_peticoes_classificador_tipo).
Para utilizá-lo baixe o modelo e carregue utilizando a função ```load``` da biblioteca 
```joblib```, ou a função ```load_model``` implementada neste pacote:

````
from juspln import bow

model = bow.load_model('kelsensantos/bow_peticoes_classificador_tipo')
````

Essa função pode ser utilizada para carrgar outros modelos, basta direcioná-la para
o caminho com o arquivo ```pkl``` correspondente com o parâmetro ```model_path``` 
(pode ser uma URL ou um caminho de arquivo).

# BERT

## Para treinar um novo modelo

A função ```prepare_bert_objects``` no arquivo bert.py prepara os objetos necessários 
para realizar um treinamento com um modelo BERT. A função recebe dois arrays numpy, um com as _features_ (X) e outros com as _classes_ (y).
O parâmetro ```model_name``` recebe o caminho do repositório de um modelo da comunidade 
[Hugging Face](https://huggingface.co/). 

A função devolve, em ordem, os objetos:
* Modelo
* Módulo de dados
* Trainer (Pytorch Lightning)

```
from core.bert import prepare_bert_objects

model, data_module, trainer =  prepare_bert_objects(
    X=X_array, 
    y=y_array, 
    model_name='model/path'
)
```

Outros parâmetros podem ser customizados ou
configurados como variáveis de ambiente (veja o aquivo .env modelo).

Para treinar um novo modelo, basta usar o mótodo ```fit``` do objeto [```Trainer```](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer),
que recebe o modelo e o módulo de dados como entrada.  Para maiores informações sobre 
como utilizá-lo, por favor, consulte a documentação completa fornecida pela 
biblioteca [Pytorch Lightning](https://lightning.ai/docs/pytorch/latest/)


```
# executing training
trainer.fit(model, datamodule=data_module)
```

Com a execução de ```Trainer```, _logs_ completos no formato
[TensorBoard](https://www.tensorflow.org/) serão guardados na pasta local.


## Para utilizar modelos treinados

Os modelos que treinamos estão disponíveis abertamente em:
* [kelsensantos/bert_peticoes_classificador_tipo](https://huggingface.co/kelsensantos/bert_peticoes_classificador_tipo)
* [kelsensantos/bert_peticoes_classificador_assunto](https://huggingface.co/kelsensantos/bert_peticoes_classificador_assunto)



