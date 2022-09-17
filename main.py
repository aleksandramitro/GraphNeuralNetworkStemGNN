import os
import data_parser
import auxilary 


if __name__ == '__main__':
    fs = 200
    training_dir = '/TRAINING/training/'
    file_list = data_parser.get_file_list(training_dir)

    for file_num in range(file_list.shape[0]):

        try:
            sample = data_parser.get_sample_data(file_list.iloc[file_num, :], istest=False)
        except:
            #TODO - debug
            continue

        record_name = os.path.basename(file_list.iloc[file_num, 0][:-4])
        PB = auxilary.PreprocessingBlock(60 * fs + 1)  # 60s*200Hz + 1
        print('Preprocessing: {}'.format(record_name))
        sample = PB.do(sample, record_name)
