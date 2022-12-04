from text_analysis.read_write_data import read_data, write_data
from text_analysis.split_dataframe import split_dataframe
from text_analysis.cleantext import drop_empty_rows, rename_columns
from text_analysis.cleantext import finalpreprocess
from tqdm import tqdm
from pathlib import Path
import sys
import configparser


def find_initial_columns(analysis_name: str) -> tuple[str]:
    '''
    This function is used to get the original column names 
    for text and labels in the initial dataset.
    The possible analysis are three: "covid", "spam", "disaster".

    Parameters:
    ============
    analysis_name: str
                   Name of the analysis the user wants to do.
    
    Raise:
    ======
    ValueErorr: if the analysis_name is not "covid", "spam", "disaster"

    Returns:
    =========
    column_names: tuple[str]
                  Sequence that contains the column names (for text and labels)
                  for the dataset selected to analyze.
    '''
    if analysis_name not in ("covid", "spam", "disaster"):
        raise ValueError("Wrong analyis name."+
                        f' Passed {analysis_name}'+
                         ' but it has to be "covid", "spam" or "disaster".')
    column_names_list = [{'analysis': 'covid',
                          'text_column': 'tweet',
                          'label_column': 'label'},
                          {'analysis': 'spam',
                           'text_column': 'original_message',
                           'label_column': 'spam'},
                           {'analysis': 'disaster',
                            'text_column': 'text',
                            'label_column': 'target'}]
    analysis_index = next((index for (index, d) in enumerate(column_names_list)
                                             if d["analysis"] == analysis_name), None)
    column_names = (column_names_list[analysis_index]['text_column'],
                    column_names_list[analysis_index]['label_column'])
    return column_names

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
