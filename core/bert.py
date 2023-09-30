import torch
import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from decouple import config
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import AdamW
from datetime import datetime
from typing import Any

# atribui valores padrão
DEFAULTS: dict[str, Any] = {
    'seed': config('SEED', cast=int, default=42),
    'padding': config('PADDING', default='max_length', cast=str),
    'truncation': config('TRUNCATION', default=True, cast=bool),
    'truncation_side': config('TRUNCATION_SIDE', default='right', cast=str),
    'return_tensors': config('RETURN_TENSORS', default='pt', cast=str),
    'max_tokens': config('MAX_TOKENS', default=512, cast=int),
    'batch_size': config('BATCH_SIZE', default=16, cast=int),
    'num_workers': config('MAX_TOKENS', default=512, cast=int),
    'split_size': config('SPLIT_SIZE_BERT', default=.2, cast=float),
    'learning_rate': config('LEARNING_RATE_BERT', default=2e-5, cast=float),
    'max_epochs': config('MAX_EPOCHS', default=5, cast=int),
    'enable_checkpointing': config('ENABLE_CHECKPOINTING', default=True, cast=bool),
}

# configure global seed
pl.seed_everything(DEFAULTS['seed'])


# helper functions
def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def check_possible_variable(variable, possible_values):
    if variable not in possible_values:
        raise ValueError(f"Only followed values are allowed for '{get_variable_name(variable)}': {possible_values}.")


def cria_logger(
        experimento: str,
        id,
):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        name=experimento,
        version=id
    )
    return logger


class CreateDataset(Dataset):
    """
    Class that handles dataset, tokenizing and batching for BERT.
    Inherits from torch.utils.data.Dataset
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            tokenizer,
            max_tokens: int,
            class_col: str,
            padding: str = DEFAULTS['padding'],
            truncation: bool = DEFAULTS['truncation'],
            truncation_side: str = DEFAULTS['truncation_side'],
            return_tensors: str = DEFAULTS['return_tensors'],
    ):

        check_possible_variable(
            truncation_side,
            ['left', 'right', 'combined']
        )

        self.dataframe = dataframe.copy()
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.padding = padding
        self.class_col = class_col
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.truncation_side = truncation_side

        if truncation_side == 'combined':
            self.truncation = False

        if type(tokenizer) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                truncation_side=truncation_side
            )
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):

        conteudo = self.dataframe.iloc[index]['conteudo']
        label = self.dataframe.iloc[index][self.class_col]

        # tokenização
        encoding = self.tokenizer.encode_plus(
            conteudo,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding=self.padding,
            truncation=self.truncation,
            return_attention_mask=True,
            return_tensors=self.return_tensors
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        if self.truncation_side == 'combined':
            a = input_ids[0:255]
            c = input_ids[-1:]
            b = input_ids[-256:]
            input_ids = torch.cat([a, c, b])
            a = attention_mask[0:255]
            c = attention_mask[-1:]
            b = attention_mask[-256:]
            attention_mask = torch.cat([a, c, b])

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=torch.tensor(label),
            conteudo=conteudo
        )


# noinspection PyPep8Naming
class DataModuler(pl.LightningDataModule):
    """Encapsulates all data loading logic and returns the necessary data loaders."""

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            tokenizer,
            split_size: float = DEFAULTS['split_size'],
            max_tokens: int = DEFAULTS['max_tokens'],
            batch_size: int = DEFAULTS['batch_size'],
            num_workers: int = DEFAULTS['num_workers'],
            padding: str = DEFAULTS['padding'],
            truncation: bool = DEFAULTS['truncation'],
            truncation_side: str = DEFAULTS['truncation_side']
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.split_size = split_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.truncation_side = truncation_side

        self.validation_dataframe = None
        self.train_dataframe = None
        self.train_dataset = None
        self.validation_dataset = None

    def prepare_data(self):
        print("Preparando dados...")
        print('...')

        # SPLIT
        # conteudo e classes
        X = self.X
        y = self.y

        print(f'Total de registros {len(X)}')
        print(f'Realizando split em {self.split_size * 100}%...')
        print('...')

        # split em conjunto de treinamento e conjunto de testes
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.split_size,
            random_state=DEFAULTS['seed'],
            shuffle=True,
            stratify=y
        )

        # verificando tamanhos
        print(f'A base de treinamento possui {len(X_train)} exemplos')
        print(f'A base de validação possui {len(X_test)} exemplos')
        print('...')

        self.train_dataframe = pd.DataFrame(data={
            'conteudo': X_train.reshape(-1),
            'classe': y_train.reshape(-1)
        }
        )

        self.validation_dataframe = pd.DataFrame(data={
            'conteudo': X_test.reshape(-1),
            'classe': y_test.reshape(-1)
        }
        )

    def setup(self, stage=None):
        self.prepare_data()

        self.train_dataset = CreateDataset(
            dataframe=self.train_dataframe,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            class_col='classe',
            padding=self.padding,
            truncation=self.truncation,
            truncation_side=self.truncation_side
        )

        self.validation_dataset = CreateDataset(
            dataframe=self.validation_dataframe,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            class_col='classe',
            padding=self.padding,
            truncation=self.truncation,
            truncation_side=self.truncation_side
        )

        print('Datasets prontos!')
        print('')
        print('')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


# noinspection PyTypeChecker
class BERTClassifier(pl.LightningModule):

    def __init__(
            self,
            model_name: str,
            num_classes: int,
            batch_size: int = DEFAULTS['batch_size'],
            freeze_bert: bool = False,
            lr: float = DEFAULTS['learning_rate'],
    ):
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            output_hidden_states=True,
            problem_type="single_label_classification"
        )

        self.batch_size = batch_size
        self.save_hyperparameters()
        self.lr = lr

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = output.loss

        log_dict = {
            'epoch': float(self.current_epoch) + 1,
            'train_loss': float(loss),
        }

        self.log_dict(log_dict, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # labels=labels
        )

        preds = torch.argmax(output.logits, dim=1)
        report = classification_report(
            labels.cpu(),
            preds.cpu(),
            output_dict=True,
            zero_division=0,
            digits=3
        )

        log_dict = {
            'epoch': float(self.current_epoch) + 1,
            'val_acc': report['accuracy'],
            'val_precision': report['macro avg']['precision'],
            'val_recall': report['macro avg']['recall'],
            'val_f1-score': report['macro avg']['f1-score'],
        }

        self.log_dict(log_dict, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr
        )
        return optimizer


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    verbose=True,
    monitor="val_f1-score",
    save_top_k=1,
    mode='max',
    save_on_train_epoch_end=False
)


# noinspection PyPep8Naming
def prepare_bert_objects(
        X,
        y,
        model_name: str,
        split_size: float = DEFAULTS['split_size'],
        max_tokens: int = DEFAULTS['max_tokens'],
        batch_size: int = DEFAULTS['batch_size'],
        lr: float = DEFAULTS['learning_rate'],
        id=None,
        experimento: str = "",
        tokenizer=None,
        fast_dev_run=False,  # recommended for debugging purposes
        num_workers=DEFAULTS['num_workers'],
        padding=DEFAULTS['padding'],
        truncation=DEFAULTS['truncation'],
        truncation_side=DEFAULTS['truncation_side'],
        devices="auto",
        max_epochs=DEFAULTS['max_epochs'],
        enable_checkpointing=DEFAULTS['enable_checkpointing'],
        callback=checkpoint_callback,
):

    num_classes = np.unique(y).shape[0]

    if id is None:
        now = datetime.now().strftime('%y-%m-%d_%H%M%S')
        id = f"{model_name.replace('/', '-')}__{max_tokens}__{truncation_side}__{now}"

    # modelo
    model = BERTClassifier(
        model_name=model_name,
        batch_size=batch_size,
        num_classes=num_classes,
        lr=lr
    )

    # data_moduler
    data_module = DataModuler(
        X,
        y,
        split_size=split_size,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        num_workers=num_workers,
        batch_size=batch_size,
        padding=padding,
        truncation=truncation,
        truncation_side=truncation_side
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # if not set is set to auto
        devices=devices,  # if not set is set to auto
        benchmark=True,  # can increase the speed of your system if your input sizes don’t change
        enable_checkpointing=enable_checkpointing,
        fast_dev_run=fast_dev_run,
        callbacks=[callback],
        logger=cria_logger(
            experimento=experimento,
            id=id
        ),
    )

    return model, data_module, trainer
