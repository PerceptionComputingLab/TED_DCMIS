from mp.utils.load_restore import pkl_load
import os
from mp.visualization.plot_results import plot_results
import pandas as pd


def extracted_test(file_dir, data):
    metrics = ['Mean_ScoreDice', 'Mean_ScoreIoU', 'Mean_ScoreHausdorff', 'Std_ScoreDice', 'Std_ScoreIoU',
               'Std_ScoreHausdorff']

    for metric in metrics:
        result = pkl_load('results.pkl', file_dir)
        df = result.to_pandas()
        if 'mm' in data:
            df = df.loc[df['Metric'].isin(['{}[cardiac]'.format(metric)])]
        elif 'prostate' in data:
            df = df.loc[df['Metric'].isin(['{}[{}]'.format(metric, data)])]

        drop_row = []
        for index, row in df.iterrows():
            if 'test' in row[2]:
                continue
            else:
                drop_row.append(index)
        df = df.drop(drop_row)
        data_dict = {}
        for index, row in df.iterrows():
            if not data_dict.get(row[2]):
                data_dict[row[2]] = []
            data_dict[row[2]].append(row[3])

        dataframe = pd.DataFrame(data_dict)
        dataframe.to_csv(os.path.join(file_dir, 'result_{}.csv'.format(metric)), index=False, sep=',')
        plot_results(result=df, save_path=file_dir, save_name='result_{}.png'.format(metric))


if __name__ == '__main__':
    file_dirs = '/Share8/zhuzhanshi/TED_DCMIS/storage/exp'
    dataset = 'mm-i'  # mm-i or mm-o or mm-r or prostate
    approach_path_list = []
    approach_list = []
    for name in os.listdir(file_dirs):
        if dataset in name:
            approach_path_list.append(os.path.join(file_dirs, name, '0', 'results'))
            approach_list.append(name.split('-')[-2])

    for path in approach_path_list:
        print(path)
        try:
            extracted_test(path, dataset)
        except Exception as e:
            print('error')
