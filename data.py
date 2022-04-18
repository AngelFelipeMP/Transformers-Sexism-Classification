import config 
from utils import download_data, process_EXIST2022_data

if __name__ == "__main__":
    download_data(config.DATA_PATH,
                    config.DATA_URL
    )
    
    process_EXIST2022_data(config.DATA_PATH, 
                    config.DATASET_CLASSES, 
                    config.DATASET_INDEX
    )

