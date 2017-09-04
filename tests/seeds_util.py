import csv
import math
import random


# For seeds data set
def tsv_to_input_format(tsv_in='seed.txt'):
    with open(tsv_in, 'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        inp = []
        res = []
        for row in tsvin:
            n_row = []
            res.append([float(row[-1])])
            for val in row[:-1]:
                if val == '':
                    continue
                else:
                    n_row.append(float(val))
            inp.append(n_row)
        return inp, res


def min_max_normalize(data):
    d = data[0]
    max = [0 for _ in range(len(d))]
    min = [math.inf for _ in range(len(d))]
    for i in range(len(d)):
        for j in range(len(data)):
            if max[i] < data[j][i]:
                max[i] = data[j][i]
            if min[i] > data[j][i]:
                min[i] = data[j][i]
    return min, max


def normalize_data(data, min_arr, max_arr, h_norm=1.0, l_norm=0.0):
    norm_data = []
    for i in range(len(data)):
        norm_row = []
        for j in range(len(data[0])):
            norm_row.append(((data[i][j]-min_arr[j])*(h_norm - l_norm)/(max_arr[j] - min_arr[j])) + l_norm)
        norm_data.append(norm_row)
    return norm_data


# Assuming that the data is ordered by class and it's got the same amount by class, this is just for the seeds set
def test_train_split(data, test_number_by_class, total_by_class):
    test = []
    train = []

    for i in range(int(len(data)/total_by_class)):
        test.extend(data[i*total_by_class:i*total_by_class+test_number_by_class])
        train.extend(data[test_number_by_class+i*total_by_class:(i+1)*total_by_class])

    random.shuffle(test)
    random.shuffle(train)

    train_data = []
    test_data = []
    test_expected = []
    train_expected = []
    for a in train:
        train_data.append(a[0])
        train_expected.append(a[1])
    for a in test:
        test_data.append(a[0])
        test_expected.append(a[1])

    return test_data, train_data, prepare_output_neuron(test_expected), prepare_output_neuron(train_expected)


def prepare_output_neuron(expected):
    neuron_output = []
    for a in expected:
        if a == [1]:
            neuron_output.append([1, 0, 0])
        elif a == [2]:
            neuron_output.append([0, 1, 0])
        else:
            neuron_output.append([0, 0, 1])
    return neuron_output


def get_ready_dataset():
    data, results = tsv_to_input_format()
    min, max = min_max_normalize(data)
    norm_data = normalize_data(data, min, max)
    data_set = [list(a) for a in zip(norm_data, results)]
    return data_set


def get_prepared_split():
    data_set = get_ready_dataset()
    test_data, train_data, test_expected, train_expected = test_train_split(data_set, test_number_by_class=35, total_by_class=70)
    return test_data, train_data, test_expected, train_expected
