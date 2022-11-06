from pathlib import Path
import pandas as pd

def read_data(name_folder = 'datasets'):
    datasets_path = Path(name_folder)
    csv_paths = list(datasets_path.glob('**/*.csv'))
    csv_paths_stem = [x.stem + x.suffix for x in csv_paths]
    csv_paths_stem_sorted = sorted(csv_paths_stem)
    train_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[0])
    val_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[1])
    test_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[2])
    return train_ds, val_ds, test_ds

def write_data(dataframes, name_folder = 'datasets_modified'):
    datasets_path = Path(name_folder)
    datasets_path.mkdir(parents=True, exist_ok=True)
    df_train, df_val, df_test = dataframes
    df_train.to_csv(path_or_buf = datasets_path / 'Constraint_English_Train_mod.csv')
    df_val.to_csv(path_or_buf = datasets_path / 'Constraint_English_Val_mod.csv')
    df_test.to_csv(path_or_buf = datasets_path / 'english_test_with_labels_mod.csv')