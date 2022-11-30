from binary_classifier.read_write_data import read_data, write_data
from binary_classifier.split_dataframe import split_dataframe
from binary_classifier.cleantext import drop_empty_rows, rename_columns
from binary_classifier.cleantext import finalpreprocess
from binary_classifier.cleantext import find_initial_columns
from tqdm import tqdm
from pathlib import Path
import sys
import configparser

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_name = config_parse.get('ANALYSIS', 'folder_name')
    seed = int(config_parse.get('PREPROCESS', 'seed'))
    dataset_folder = Path('datasets') / analysis_name

    dfs_raw = read_data(dataset_folder)
    if len(dfs_raw) != 3:

        fractions =    (float(config_parse.get('PREPROCESS', 'train_fraction')),
                        float(config_parse.get('PREPROCESS', 'test_fraction')))
        dfs_raw = split_dataframe(dfs_raw, fractions, seed)

    text_col_name, label_col_name = find_initial_columns(analysis_name)

    dfs_processed = []
    for df in dfs_raw:
        df_cleaned = rename_columns(df, text_col_name, label_col_name)
        df_cleaned = drop_empty_rows(df_cleaned)

        df_cleaned['clean_text'] = df_cleaned['text']
        cleaned_text = [finalpreprocess(text_to_clean) for text_to_clean in tqdm(df_cleaned['clean_text'])]
        df_cleaned['clean_text'] = cleaned_text
        df_cleaned = drop_empty_rows(df_cleaned)
        dfs_processed.append(df_cleaned)
    

    output_folder = Path('preprocessed_datasets') / analysis_name
    output_folder.mkdir(parents=True, exist_ok=True)

    write_data(dfs_processed, output_folder = output_folder, analysis = analysis_name)

    df_train, df_valid, df_test = dfs_processed
    print(df_train['clean_text'].head())
    print("="*40)
    print(df_valid['clean_text'].head())
    print("="*40)

    print("Some random texts:\n" + "="*40)
    for index, row in df_train.sample(n = 3).iterrows():
        print("\nOriginal text:\n" + "="*40) 
        print(row['text'])
        print("\nCleaned text:\n" + "="*40)
        print(row['clean_text'])


if __name__ == '__main__':
    main()
