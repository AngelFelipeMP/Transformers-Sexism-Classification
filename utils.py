import os
import shutil
import pandas as pd
import gdown
import config

def download_data(data_path, data_urls):
    # create a data folder
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    for url in data_urls:
        #download data folders to current directory
        gdown.download_folder(url, quiet=True)
        sorce_folder = os.getcwd() + '/' + 'data'
        
        # move datasets to the data folder
        file_names = os.listdir(sorce_folder)
        for file_name in file_names:
            shutil.move(os.path.join(sorce_folder, file_name), data_path)
            
        # delete data folders from current directory
        shutil.rmtree(sorce_folder)
        

def process_EXIST2022_data(data_path, labels_col, index_col):
    files = [f for f in os.listdir(data_path) if 'processed' not in f]
    
    for file in files:
        df = pd.read_csv(data_path + '/' + file)
        print(df)
        
        if 'train' in file or 'dev' in file:
            df.replace(labels_col, inplace=True)
            print(df.head())

        dataset_name =  file[:-4] + '_processed' + '.csv'
        variable = 'DATASET' + ['_TRAIN' if 'train' in file else '_DEV' if 'dev' in file else '_TEST'][0]
        pass_value_config(variable, '\'' + dataset_name + '\'')
        
        df.to_csv(data_path + '/' + dataset_name, index=False,  index_label=index_col)


def pass_value_config(variable, value):
    with open(config.CODE_PATH + '/' + 'config.py', 'r') as conf:
        content = conf.read()
        new = content.replace(variable + ' = ' + "''", variable + ' = ' +  value )
        
    with open(config.CODE_PATH + '/' + 'config.py', 'w') as conf_new:
        conf_new.write(new)


def map_labels(df, labels_col):
    for col, labels in labels_col.items():
        df.replace({col:{number: string for string, number in labels.items()}}, inplace=True)
    return df