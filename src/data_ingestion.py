import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.info(f"Data loaded successfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error: Failed to parse the CSV file from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error while loading the data from {data_url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values and separating features and target."""
    try:
        df.columns = ['tweet_ID','entity','sentiment','tweet_content']
        df.drop(columns=['tweet_ID','entity'],inplace=True)
        final_df = df[df['sentiment'].isin(['Positive', 'Negative'])]
        final_df['sentiment'].replace({'Positive':1, 'Negative':0},inplace=True)
        logger.info("Data preprocessing completed successfully")
        return final_df
    except KeyError as e:
        logger.error(f"Error: Target column {e} missing in the DataFrame.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error during data preprocessing: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save the training and testing data to CSV files."""
    try:
        output_dir = os.path.join(output_dir, 'raw')
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        logger.info(f"Training and testing data saved successfully in {output_dir}")
    except Exception as e:
        logger.error(f"Error: An unexpected error while saving the data: {e}")
        raise

def main():
    """Main function to execute data ingestion process."""
    """
       # Load configuration
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        data_url = config['data_source']['data_url']
        test_size = config['data_split']['test_size']
        random_state = config['data_split']['random_state']
        output_dir = config['artifacts']['artifacts_dir'] """

    try:
        df = load_data("C:\\Users\\MeghaGuha\\Dropbox\\PC\\Desktop\\Megha\\MBA practice\\mlops\\MLPipelinesWithDVC\\MLPipelinesWithDVC\\twitter_sentiment.csv")
        # Preprocess data
        processed_df = preprocess_data(df)
        # Split data
        train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)
        # Save data
        save_data(train_df, test_df, output_dir='data')
        
    except Exception as e:
        print(f"Error: Failed to complete the data ingestion process: {e}")
        raise
if __name__ == "__main__":
    main()