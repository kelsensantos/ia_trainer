# ia_trainer










# BERT

## Para treinar um novo modelo

A função "prepare_bert_objects" no arquivo bert.py prepara os objetos necessários 
para realizar um treinamento com um modelo BERT. 

A função recebe dois arrays numpy, um com as _features_ (X) e outros com as _classes_ (y).
O parâmetro "model_name" recebe o caminho do repositório de um modelo da comunidade 
[Hugging Face](https://huggingface.co/). Outros parâmetros podem ser customizados ou
configurados como variáveis de ambiente (veja o aquivo .env modelo).

A função devolve, em ordem, os objetos:
* Modelo
* Módulo de dados
* Trainer (Pytorch Lightning)

Os modelos que treinamos estão disponíveis abertamente em:
* [kelsensantos/bert_peticoes_classificador_tipo](https://huggingface.co/kelsensantos/bert_peticoes_classificador_tipo)
* [kelsensantos/bert_peticoes_classificador_assunto](https://huggingface.co/kelsensantos/bert_peticoes_classificador_assunto)

Para treinar um novo modelo, basta usar o mótodo _fit_ do objeto [Trainer](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer),
que recebe o modelo e o módulo de dados como entrada.  Para maiores informações sobre 
como utilizá-lo, por favor, consulte a documentação completa fornecida pela 
biblioteca [Pytorch Lightning](https://lightning.ai/docs/pytorch/latest/)

Com a execução de Trainer, _logs_ completos no formato
[TensorBoard](https://www.tensorflow.org/) serão guardados na pasta local.


## Para utilizar modelos treinados



Os modelos que treinamos estão disponíveis abertamente em:
* [kelsensantos/bert_peticoes_classificador_tipo](https://huggingface.co/kelsensantos/bert_peticoes_classificador_tipo)
* [kelsensantos/bert_peticoes_classificador_assunto](https://huggingface.co/kelsensantos/bert_peticoes_classificador_assunto)
