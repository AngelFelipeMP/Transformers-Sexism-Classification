import os
import dataset
# import engine
import engine_accelerator
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()
from accelerate import Accelerator

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def run(df_train, df_val, task, transformer, max_len, batch_size, lr, drop_out, language, df_results):
    
    train_dataset = dataset.TransformerDataset(
        text=df_train[language].values,
        target=df_train[task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        # num_workers = config.TRAIN_WORKERS
    )

    val_dataset = dataset.TransformerDataset(
        text=df_val[language].values,
        target=df_val[task].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        # num_workers=config.VAL_WORKERS
    )

    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained(transformer, num_labels=max(list(config.DATASET_CLASSES[task].values()))+1)
    # model = TransforomerModel(transformer, drop_out, number_of_classes=max(list(config.DATASET_CLASSES[task].values()))+1)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / batch_size * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    
    ### NEW LINE
    model, optimizer, train_data_loader, val_data_loader = accelerator.prepare(model, optimizer, train_data_loader, val_data_loader)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    
    for epoch in range(1, config.EPOCHS+1):
        model.train()

        
        if accelerator.is_main_process:
            print('OK!! accelerator.is_main_process')
        else:
            print('NOT accelerator.is_main_process')
            
        
        for batch in train_data_loader:
            targets = batch["targets"]
            del batch["targets"]
            
            optimizer.zero_grad()
            outputs = model(batch)
            
            with accelerator.autocast():
                with accelerator.autocast():
                    loss = loss_fn(outputs, targets)

            accelerator.backward(loss)
            print('*TRAIN*'*30)
            optimizer.step()
            print('*BBBB*'*30)
            scheduler.step()
            
            
        model.eval()
        with torch.no_grad():
            for batch in val_data_loader:
                targets = batch["targets"]
                del batch["targets"]
                
                
        f1_train = metrics.f1_score([0,0], [0,0], average='macro')
        acc_train = metrics.accuracy_score([0,0], [0,0])
        loss_train = 0.1

        f1_val = metrics.f1_score([0,0], [0,0], average='macro')
        acc_val = metrics.accuracy_score([0,0], [0,0])
        loss_val = 0.1
    
        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'language': language,
                            'accuracy_train':acc_train,
                            'f1-macro_train':f1_train,
                            'loss_train':loss_train,
                            'accuracy_val':acc_val,
                            'f1-macro_val':f1_val,
                            'loss_val':loss_val
                        }, index=[0]
        ) 
        
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-macro_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(epoch, config.EPOCHS, f1_train, acc_train, loss_train, f1_val, acc_val, loss_val))

    return df_results

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, nrows=config.N_ROWS).fillna("none")
    skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)

    df_results = pd.DataFrame(columns=['task',
                                        'epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        'language',
                                        'accuracy_train',
                                        'f1-macro_train',
                                        'loss_train',
                                        'accuracy_val',
                                        'f1-macro_val',
                                        'loss_val'
            ]
    )
    
    inter = len(config.LABELS) * sum([len(l) for l in config.TRANSFORMERS.values()]) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
    grid_search_bar = tqdm(total=inter, desc='GRID SEARCH', position=2)
    
    for task in tqdm(config.LABELS, desc='TASKS', position=1):
        df_grid_search = dfx.loc[dfx[task]>=0].reset_index(drop=True)
        
        for language in tqdm(config.TRANSFORMERS.keys(), desc='TRANSFOMERS', position=0):
            for transformer in config.TRANSFORMERS[language]:
                
                for max_len in config.MAX_LEN:
                    for batch_size in config.BATCH_SIZE:
                        for drop_out in config.DROPOUT:
                            for lr in config.LR:
                                
                                for fold, (train_index, val_index) in enumerate(skf.split(df_grid_search[language], df_grid_search[task])):
                                    df_train = df_grid_search.loc[train_index]
                                    df_val = df_grid_search.loc[val_index]
                                    
                                    tqdm.write(f'\nTask: {task} Transfomer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Language/Text: {language.upper()} Fold: {fold+1}/{config.SPLITS}')
                                    
                                    df_results = run(df_train,
                                                        df_val,
                                                        task, 
                                                        transformer,
                                                        max_len, 
                                                        batch_size,
                                                        lr,
                                                        drop_out,
                                                        language,
                                                        df_results
                                    )
                                
                                    grid_search_bar.update(1)
                                
                                df_results = df_results.groupby(['task',
                                                                'epoch',
                                                                'transformer',
                                                                'max_len',
                                                                'batch_size',
                                                                'lr',
                                                                'dropout',
                                                                'language'], as_index=False, sort=False)['accuracy_train',
                                                                                                    'f1-macro_train',
                                                                                                    'loss_train',
                                                                                                    'accuracy_val',
                                                                                                    'f1-macro_val',
                                                                                                    'loss_val'].mean()
                                
                                df_results.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.csv', index=False)