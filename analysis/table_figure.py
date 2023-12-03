import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


def extracted_test(file_path):
    data = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if 'test' in row[0]:
                continue
            else:
                r_float = []
                for r in row:
                    r_float.append(float(r))
                data.append(np.array(r_float))
    return np.array(data)


if __name__ == '__main__':
    file_dirs = '/Share8/zhuzhanshi/TED_DCMIS/storage/exp'
    out_dir = '/Share8/zhuzhanshi/TED_DCMIS/storage/figure'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataset = 'mm-i'  # mm-i or mm-o or mm-r or hippocampus or prostate
    metric = "Mean_ScoreDice"  # Mean_ScoreDice, Mean_ScoreIoU, Mean_ScoreHausdorff, Std_ScoreDice, Std_ScoreIoU, Std_ScoreHausdorff
    approach_path_list = []
    approach_list = []
    table_dict = {}
    up_bound = extracted_test(
        os.path.join(file_dirs, dataset + '-joint-unet', '0', 'results', 'result_{}.csv'.format(metric)))
    print(up_bound)
    print('AVG:', np.average(np.array(up_bound[-1, :])))
    for name in os.listdir(file_dirs):
        if dataset in name:
            if 'joint' in name:
                continue
            table_dict[name.split('-')[-2]] = []
            if not os.path.exists(os.path.join(file_dirs, name, '0', 'results', 'result_{}.csv'.format(metric))):
                continue
            result = extracted_test(os.path.join(file_dirs, name, '0', 'results', 'result_{}.csv'.format(metric)))

            avg = np.average(np.array(result[-1, :]))  # avg
            avg_curve = []
            for i in range(len(up_bound[0])):
                avg_curve.append(np.average(np.array(result[i, :i + 1])))

            ravg = np.average(np.array(result[-1, :] / up_bound[0]))  # ravg
            ravg_curve = []
            for i in range(len(up_bound[0])):
                ravg_curve.append(np.average(np.array(result[i, :i + 1] / up_bound[0, :i + 1])))

            bwt_curve = []
            rbwt_curve = []
            for j in range(1, len(up_bound[0])):
                bwt = 0
                rbwt = 0
                for i in range(j):
                    bwt = result[j, i] - result[i, i] + bwt
                    rbwt = (result[j, i] - result[i, i]) / result[i, i] + rbwt
                bwt = bwt / j
                rbwt = rbwt / j
                bwt_curve.append(bwt)
                rbwt_curve.append(rbwt)

            bwt = 0
            rbwt = 0
            for i in range(len(up_bound[0]) - 1):
                bwt = result[-1, i] - result[i, i] + bwt
                rbwt = (result[-1, i] - result[i, i]) / result[i, i] + rbwt
            bwt = bwt / (len(up_bound[0]) - 1)
            rbwt = rbwt / (len(up_bound[0]) - 1)

            print(name.split('-')[-2])
            print(result)
            print('avg:', avg)
            table_dict[name.split('-')[-2]].append(avg)
            print('ravg:', ravg)
            table_dict[name.split('-')[-2]].append(ravg)
            print('bwt:', bwt)
            table_dict[name.split('-')[-2]].append(bwt)
            print('rbwt:', rbwt)
            table_dict[name.split('-')[-2]].append(rbwt)
            print('avg_curve:', avg_curve)
            table_dict[name.split('-')[-2]].append(avg_curve)
            print('ravg_curve:', ravg_curve)
            table_dict[name.split('-')[-2]].append(ravg_curve)
            print('bwt_curve:', bwt_curve)
            table_dict[name.split('-')[-2]].append(bwt_curve)
            print('rbwt_curve:', rbwt_curve)
            table_dict[name.split('-')[-2]].append(rbwt_curve)

    # table_dict_sort = {'kd':None, 'ted': None}

    table_dict_sort = None

    if table_dict_sort:
        for key in table_dict_sort:
            table_dict_sort[key] = table_dict[key]
        table_dict = table_dict_sort
    print(table_dict)

    print('approach'.rjust(16), 'avg'.rjust(16), 'ravg'.rjust(16), 'bwt'.rjust(16), 'rbwt'.rjust(16))
    for key in table_dict:
        print(key.rjust(16), str(round(table_dict[key][0], 3)).rjust(16), str(round(table_dict[key][1], 3)).rjust(16),
              str(round(table_dict[key][2], 3)).rjust(16), str(round(table_dict[key][3], 3)).rjust(16))

    dataframe = pd.DataFrame(table_dict)
    dataframe.to_csv(os.path.join(out_dir, dataset + '_results.csv'), index=False, sep=',')

    plt.title('avg_curve')
    for key in table_dict:
        plt.plot([i for i in range(len(table_dict[key][4]))], table_dict[key][4], label=key)
    plt.legend()
    plt.savefig(os.path.join(out_dir, dataset + '_avg.png'))
    plt.show()

    plt.title('ravg_curve')
    for key in table_dict:
        plt.plot([i for i in range(len(table_dict[key][5]))], table_dict[key][5], label=key)
    plt.legend()
    plt.savefig(os.path.join(out_dir, dataset + '_ravg.png'))
    plt.show()

    plt.title('bwt_curve')
    for key in table_dict:
        plt.plot([i for i in range(len(table_dict[key][6]))], table_dict[key][6], label=key)
    plt.legend()
    plt.savefig(os.path.join(out_dir, dataset + '_bwt.png'))
    plt.show()

    plt.title('rbwt_curve')
    for key in table_dict:
        plt.plot([i for i in range(len(table_dict[key][7]))], table_dict[key][7], label=key)
    plt.legend()
    plt.savefig(os.path.join(out_dir, dataset + '_rbwt.png'))
    plt.show()
