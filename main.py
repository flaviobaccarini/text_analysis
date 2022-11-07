import visualize_data
import configparser
import sys

def explore(input_folder):
    df_train, df_val, df_test = visualize_data.read_data(input_folder)
    visualize_data.info_data(df_train, df_val, df_test)
    visualize_data.plot_label_distribution(df_train, df_val, df_test)

    df_all_word_count = visualize_data.word_count_twitter(df_train, df_val, df_test)
    df_train_word_count, df_val_word_count, df_test_word_count = df_all_word_count
    visualize_data.word_count_printer(df_train_word_count, 
                        df_val_word_count,
                        df_test_word_count)

    visualize_data.plotting_word_count(df_train_word_count, 
                        df_val_word_count,
                        df_test_word_count)

    df_all_char_count = visualize_data.char_count_twitter(df_train_word_count, 
                                            df_val_word_count,
                                            df_test_word_count)
    df_train_char_count, df_val_char_count, df_test_char_count = df_all_char_count
    visualize_data.char_count_printer(df_train_char_count,
                       df_val_char_count,
                       df_test_char_count)
    visualize_data.plotting_char_count(df_train_char_count,
                       df_val_char_count,
                       df_test_char_count)     


def main():
    config_parse = configparser.ConfigParser()
    function_to_execute = sys.argv[1]
    configuration = config_parse.read(sys.argv[2])
    
    input_folder = str(config_parse.get('INPUT_OUTPUT', 'input_folder'))
    output_folder = str(config_parse.get('INPUT_OUTPUT', 'output_folder'))

    function_to_execute(input_folder = input_folder)

if __name__ == '__main__':
    main()