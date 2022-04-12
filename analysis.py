import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import config

if __name__ == "__main__":
    df_train = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN)
    df_dev = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV)
    df = pd.concat([df_train, df_dev]).reset_index(drop=True)

    for language in config.TRANSFORMERS.keys():
        for transformer in config.TRANSFORMERS[language]:
        
            print(f'Transformer: {transformer} \n')

            tokenizer = AutoTokenizer.from_pretrained(transformer)

            if language == 'en':
                df['tokens'] = df['English'].apply(lambda x: tokenizer.tokenize(x))
            elif language == 'es':
                df['tokens'] = df['Spanish'].apply(lambda x: tokenizer.tokenize(x))
            else:
                df['tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
            
            df['number_tokens'] = df['tokens'].apply(lambda x: len(x))

            ax = df['number_tokens'].plot.hist(bins=20, range=(0, 512))
            ax.plot()
            plt.show()

            print('Porcentage of comments under 64 tokens: ',((df.query('number_tokens < 64').size / df.size) * 100))
            print('Porcentage of comments under 128 tokens: ',((df.query('number_tokens < 128').size / df.size) * 100))
            print('Porcentage of comments under 256 tokens: ',((df.query('number_tokens < 256').size / df.size) * 100))
            print('Porcentage of comments under 512 tokens: ',((df.query('number_tokens < 512').size / df.size) * 100))