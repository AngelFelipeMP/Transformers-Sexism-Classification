import os
import dataset
import engine
import engine_2gpus
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
import argparse

from model import TransforomerModel
from model_2gpus import TransforomerModel as TransforomerModel_2gpus
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error() 

def join_results():
    list_of_results = []
    
    for file in os.listdir(config.LOGS_PATH):
        if config.DOMAIN_GRID_SEARCH in file and file != config.DOMAIN_GRID_SEARCH + '.csv':
            list_of_results.append(pd.read_csv(config.LOGS_PATH + '/' + file))
            
    if len(list_of_results) > 1:
        df = pd.concat(list_of_results, ignore_index=True)
        df.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_GRID_SEARCH + '.csv', index=False)
            

def best_parameters(task, transformer):
    join_results()
    all_grid_search = config.DOMAIN_GRID_SEARCH + '.csv'
    
    for file in os.listdir(config.LOGS_PATH):
        if all_grid_search in file:
            df = pd.read_csv(config.LOGS_PATH + '/' + file)
            parameters = df.loc[(df['transformer'] == transformer) & (df['task'] == task)].sort_values(by=[config.METRICS[task] + '_val'], ascending=False).iloc[0,:]
            break
            
    return int(parameters['epoch']), int(parameters['max_len']), int(parameters['batch_size']), float(parameters['lr']), float(parameters['dropout'])


def train(df_train, task, epochs, best_epoch, transformer, max_len, batch_size, lr, drop_out, language, df_results, data):
    
    train_dataset = dataset.TransformerDataset(
        text=df_train[language].values,
        target=df_train[task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )

    if any(transformer == m for m in config.GRID_2GPUS ):
        device = torch.device('cuda:0')
        model = TransforomerModel_2gpus(transformer, drop_out, number_of_classes=max(list(config.DATASET_CLASSES[task].values()))+1, device=device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransforomerModel(transformer, drop_out, number_of_classes=max(list(config.DATASET_CLASSES[task].values()))+1)
        model.to(device)
    
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

    num_train_steps = int(len(df_train) / batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    for epoch in range(1, best_epoch+1):
        if any(transformer == m for m in config.GRID_2GPUS ):
            pred_train, targ_train, loss_train = engine_2gpus.train_fn(train_data_loader, model, optimizer, device, scheduler)
            f1_train = metrics.f1_score(targ_train, pred_train, average='macro')
            acc_train = metrics.accuracy_score(targ_train, pred_train)
        else:
            pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
            f1_train = metrics.f1_score(targ_train, pred_train, average='macro')
            acc_train = metrics.accuracy_score(targ_train, pred_train)
        
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
                            'loss_train':loss_train
                        }, index=[0]
        )
        
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f}".format(epoch, best_epoch, f1_train, acc_train, loss_train))
    
    # add "language" as input
    torch.save(model.state_dict(), f'{config.LOGS_PATH}/{data}_task[{task}]_transformer[{transformer.split("/")[-1]}]_epoch[{epoch}]_maxlen[{max_len}]_batchsize[{batch_size}]_dropout[{drop_out}]_lr[{lr}]_language[{language}].model')

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Data/Datasets to train the models')
    args = parser.parse_args()
    
    if args.data == 'train':
        datasets = [config.DATASET_TRAIN]
        domain = config.DOMAIN_TRAIN
    
    elif args.data == 'all':
        datasets = [config.DATASET_TRAIN, config.DATASET_DEV]
        domain = config.DOMAIN_TRAIN_ALL_DATA
    
    elif not args.data:
        print('Specifying --data is required')
        exit(1)
    
    else:
        print('Specifying --data train or all')
        exit(1)
        
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    dataset_list = []
    for data in datasets:
        df = pd.read_csv(config.DATA_PATH + '/' + data, index_col=None)
        dataset_list.append(df)
    
    dfx = pd.concat(dataset_list, axis=0, ignore_index=True).iloc[:config.N_ROWS]
    print('Dataset shape: ', dfx.shape)
    
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
                                    'loss_train'
        ]
    )
    
    for task in tqdm(config.LABELS, desc='TRAIN', position=0):
        
        df_train = dfx.loc[dfx[task]>=0]
        
        for language in config.TRANSFORMERS.keys():
            for transformer in config.TRANSFORMERS[language]:
            
                best_epoch, max_len, batch_size, lr, drop_out = best_parameters(task, transformer)
                tqdm.write(f'\nTask: {task} Data: {domain} Transfomer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Language/Text: {language.upper()}')
                
                df_results = train(df_train,
                                    task,
                                    config.EPOCHS,
                                    best_epoch,
                                    transformer,
                                    max_len,
                                    batch_size,
                                    lr,
                                    drop_out,
                                    language,
                                    df_results,
                                    domain
                )
                
                df_results.to_csv(config.LOGS_PATH + '/' + domain + '.csv', index=False)
