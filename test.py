import os
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm

from scipy.special import softmax
from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from transformers import AdamW
from transformers import logging
logging.set_verbosity_error()
from validation import higher, majority, transformer_parameters

def join_predictions(task, domain):
    list_of_results = []
    
    for file in os.listdir(config.LOGS_PATH):
        if all(item in file for item in [domain, '_1_', 'maped', task]):
            list_of_results.append(pd.read_csv(config.LOGS_PATH + '/' + file))
            
    df = pd.concat(list_of_results, ignore_index=True).sort_values(by=['id'])
    df.to_csv(config.LOGS_PATH + '/' + domain + '_' + task + '_final_prediction' +'.csv', index=False)

def map_pred(pred, task):  
    for label, num in config.DATASET_CLASSES[task].items():
        if num == pred:
            return label
        
def test(df_test, task, transformer):
    parameters = transformer_parameters(task, transformer, config.DOMAIN_TRAIN_ALL_DATA)
    
    test_dataset = dataset.TransformerDataset_Test(
        text=df_test[config.ORIGINAL_TEXT].values,
        max_len=parameters['max_len'],
        transformer=transformer
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=parameters['batch_size'], 
        num_workers=config.VAL_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, parameters['dropout'], number_of_classes=max(list(config.DATASET_CLASSES[task].values()))+1)
    model.load_state_dict(torch.load(parameters['weights']))
    model.to(device)
    
    pred_test = engine.test_fn(test_data_loader, model, device)
    
    return pred_test

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    inter =  len([l for l in list(config.TRANSFORMERS.keys()) if l != config.ORIGINAL_TEXT]) * len(config.LABELS) * (sum([len(t) + len(config.TRANSFORMERS[config.ORIGINAL_TEXT]) for l,t in config.TRANSFORMERS.items() if l != config.ORIGINAL_TEXT ])/2)
    test_bar = tqdm(total=inter, desc='TEST', position=0)
    
    for language in [l for l in list(config.TRANSFORMERS.keys()) if l != config.ORIGINAL_TEXT]:
        df_test = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TEST)
        
        df_val_metrics = pd.read_csv(config.LOGS_PATH + '/' + config.DOMAIN_VALIDATION + '_metrics' + '_' + language + '.csv')
        models_dict = {}
        for task in config.LABELS:
            models_dict[task] = df_val_metrics.loc[df_val_metrics['model'].str.contains(task)].sort_values(by=[config.METRICS[task]], ascending=False).reset_index(drop=True).loc[[0],'model'].tolist()
        
        for task in config.LABELS:
            if task == 'task2':
                df_val = df_test.loc[df_test[models_dict['task1'][0]] >= 1]
            else:
                df_val = df_test
                
            for transformer in config.TRANSFORMERS[language] + config.TRANSFORMERS[config.ORIGINAL_TEXT]:
                
                tqdm.write(f'Task: {task} - Transfomer: {transformer.split("/")[-1]} - Language/Text: {language}')
                test_bar.update(1)
                
                predictions = test(df_val, 
                                    task,
                                    transformer
                )
                
                df_val[task + '_' + transformer.split("/")[-1] + '_outputs'] = [softmax(pred).tolist() for pred in predictions]
                df_val[task + '_' + transformer.split("/")[-1] + '_prediction'] = [pred.index(max(pred)) for pred in predictions]

            columns_higher_sum = [col for col in df_val if all(item in col for item in [task, '_outputs'])]
            columns_majority_vote = [col for col in df_val if all(item in col for item in [task, '_prediction'])]

            df_val[task + '_higher_sum'] = df_val.loc[:,columns_higher_sum].apply(lambda x: higher(x), axis=1)
            df_val[task + '_majority_vote'] = df_val.loc[:,columns_majority_vote].apply(lambda x: majority(x), axis=1)
            
            remove_col = [list(df_test.columns)[0]] + list(df_test.columns)[2:]
            df_test = pd.merge(df_test.loc[df_test['language']==language], df_val.loc[:, df_val.columns.difference(remove_col)], how='left', on='id')
            
        df_test = df_test.fillna(-1)
        
        df_test.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_all_model' + '_' + language + '.csv', index=False)

        for task in config.LABELS:
            for j, model in enumerate(models_dict[task]):
                df_test.loc[:, ['id'] + [models_dict[task][j]]].to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_task_' + task + '_rank_' + str(j+1) + '_model_' + model + '_' + language +  '.csv', index=False)

                df_original_labels = df_test.loc[:, ['id'] + [models_dict[task][j]]].copy()
                df_original_labels['maped_predictions'] = df_original_labels.loc[:, models_dict[task][j]].apply(lambda x: map_pred(x, task))
                df_original_labels.loc[:, ['id','maped_predictions']].to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_task_' + task + '_rank_' + str(j+1) + '_model_' + model + '_maped_labels_' + language +  '.csv', index=False)


    for task in config.LABELS:
        join_predictions(task, config.DOMAIN_TEST)