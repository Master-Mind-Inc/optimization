import numpy as np
import pandas as pd
import json
import itertools
from scipy.special import expit

# import os

# print(os.listdir('../'))
PATH = '../'
config = json.loads(open(PATH + 'config.json').read())


def convert_start_points(points, params_amount, list_of_ranges):
    points = [elem[0] + elem[1] for elem in points]
    columns = ['idx' + str(val + 1) for val in range(params_amount)] + ['pnl', 'MDD']  # TODO: add other columns
    data = pd.DataFrame.from_records(points, columns=columns)
    data['opt'] = np.arange(len(points)).reshape((-1, 1))

    # Converter float variables to int[0, 100]
    for idx, border in enumerate(list_of_ranges):
        left, right, type_of_var = border
        if type_of_var == 0:
            step = round((right - left) / 100.0, 2)
            column = 'idx' + str(idx + 1)
            data[column] = list(map(int, ((data[column] - left) / step)))

    return data[['opt'] + columns]


def create_start_moment(N: int, parameters_amount: int) -> pd.DataFrame:
    numbers_of_optimizations = np.arange(N).reshape((-1, 1))
    point_values = np.zeros((N, parameters_amount))
    values = np.concatenate([numbers_of_optimizations, point_values], axis=1)
    columns = ['opt'] + ['idx' + str(val + 1) for val in range(parameters_amount)]

    return pd.DataFrame(values, columns=columns, dtype=np.int64)


def get_mask(k, ind_num):
    change = np.vstack(list(itertools.product(np.array([1, 0, -1]), np.array([1, 0, -1]))))
    indexes = np.unique(list(map(lambda x: list(set(x)), list(itertools.product(np.array([1, 2, 3, 4]) - 1,
                                                                                np.array([1, 2, 3, 4]) - 1)))))

    init_mask = np.array([0, 0, 0, 0])
    res = []
    for y in indexes:
        for x in change:
            if len(y) == 1:
                np.put(init_mask, y[0], 1)
                res.append(init_mask)
                init_mask = np.array([0, 0, 0, 0])
                np.put(init_mask, y[0], -1)
                res.append(init_mask)
                init_mask = np.array([0, 0, 0, 0])
                np.put(init_mask, y[0], 0)
                res.append(init_mask)
                init_mask = np.array([0, 0, 0, 0])
            if len(y) > 1:
                if k == 1:
                    continue
                np.put(init_mask, y, x)
                res.append(init_mask)
                init_mask = np.array([0, 0, 0, 0])
    res = np.unique(np.vstack(res), axis=0)
    return res


def get_all_neighbours(point, ind_columns):
/*
------------- UNDER NDA -----------------------
*/
        return data

    mask = get_mask(1, 1)
    kernels = point + mask
    data = get_neighs(kernels, mask)
    data.columns = ['ind'] + ind_columns

    return data, pd.DataFrame(kernels, columns=ind_columns)


def drop_for_api(data):
    data_to_api = data.drop_duplicates(subset=['idx1', 'idx2', 'idx3', 'idx4']).reset_index()[
        ['idx1', 'idx2', 'idx3', 'idx4']]
    return data_to_api


def fake_api(points, api_params):
    def drop_for_api(data):
        data_to_api = data.drop_duplicates(subset=['idx1', 'idx2', 'idx3']).reset_index()[
            ['idx1', 'idx2', 'idx3']]
        return data_to_api

    points = drop_for_api(points).values.tolist()

    def create_fake_api_answer(N, points):
        point_values = np.random.random(size=(N, 2))
        return {'res1': [points, point_values * 2],
                'res2': [points, point_values * (-2)],
                'res3': [points, point_values * 5]}

    data = create_fake_api_answer(len(points), points)

    data1, data2, data3 = data['res1'], data['res2'], data['res3']
    columns = ['idx' + str(i + 1) for i in range(3)] + ['pnl, maxd', 'hui1', 'hui2']

    ti1 = pd.DataFrame([row[0] + row[1] for row in data1], columns=columns)
    ti2 = pd.DataFrame([row[0] + row[1] for row in data2], columns=columns)
    ti3 = pd.DataFrame([row[0] + row[1] for row in data3], columns=columns)
    ti1['ti'] = 1
    ti2['ti'] = 2
    ti3['ti'] = 3

    return pd.concat([ti1, ti2, ti3])


def api(data, api_params):
    def create_fake_api_answer(N):
        point_values = np.random.random(size=(N, 2))
        # columns = ['pnlTotal', 'maxDrawdown']
        return [[point_values, point_values * 2],
                [point_values * (-1), point_values * (-2)],
                [point_values * 4, point_values * 5]]

    ans = create_fake_api_answer(len(data))

    return ans


def calculate_crit(all_points_data, distinct_data, answer, ind_columns):
    if distinct_data is not None:
        ti_1 = pd.concat(
            [distinct_data, pd.DataFrame(np.array(answer[0]).reshape([-1, 2]), columns=['pnlTotal', 'maxDrawdown'])],
            axis=1, join_axes=[distinct_data.index])
        ti_2 = pd.concat(
            [distinct_data, pd.DataFrame(np.array(answer[1]).reshape([-1, 2]), columns=['pnlTotal', 'maxDrawdown'])],
            axis=1, join_axes=[distinct_data.index])
        ti_3 = pd.concat(
            [distinct_data, pd.DataFrame(np.array(answer[2]).reshape([-1, 2]), columns=['pnlTotal', 'maxDrawdown'])],
            axis=1, join_axes=[distinct_data.index])

        ti_1_data = all_points_data.merge(ti_1, how='left', on=ind_columns)
        ti_2_data = all_points_data.merge(ti_2, how='left', on=ind_columns)
        ti_3_data = all_points_data.merge(ti_3, how='left', on=ind_columns)
    else:
        ti_1_data = pd.concat(
            [all_points_data, pd.DataFrame(np.array(answer[0]).reshape([-1, 2]), columns=['pnlTotal', 'maxDrawdown'])],
            axis=1, join_axes=[all_points_data.index])
        ti_2_data = pd.concat(
            [all_points_data, pd.DataFrame(np.array(answer[1]).reshape([-1, 2]), columns=['pnlTotal', 'maxDrawdown'])],
            axis=1, join_axes=[all_points_data.index])
        ti_3_data = pd.concat(
            [all_points_data, pd.DataFrame(np.array(answer[2]).reshape([-1, 2]), columns=['pnlTotal', 'maxDrawdown'])],
            axis=1, join_axes=[all_points_data.index])

    ti_1_data['k_pl'] = ti_1_data['pnlTotal'] / (np.abs(ti_1_data['maxDrawdown']) + 1)
    ti_2_data['k_pl'] = ti_2_data['pnlTotal'] / (np.abs(ti_2_data['maxDrawdown']) + 1)
    ti_3_data['k_pl'] = ti_3_data['pnlTotal'] / (np.abs(ti_3_data['maxDrawdown']) + 1)

    if distinct_data is not None:
        grouped_by_ind_1 = ti_1_data[['ind', 'k_pl', 'opt']].groupby(['opt', 'ind'])
        grouped_by_ind_2 = ti_2_data[['ind', 'k_pl', 'opt']].groupby(['opt', 'ind'])
        grouped_by_ind_3 = ti_3_data[['ind', 'k_pl', 'opt']].groupby(['opt', 'ind'])
    else:
        grouped_by_ind_1 = ti_1_data['k_pl']
        grouped_by_ind_2 = ti_2_data['k_pl']
        grouped_by_ind_3 = ti_3_data['k_pl']

    crit_1 = grouped_by_ind_1.mean() - (2 * grouped_by_ind_1.std())
    crit_2 = grouped_by_ind_2.mean() - (2 * grouped_by_ind_2.std())
    crit_3 = grouped_by_ind_3.mean() - (2 * grouped_by_ind_3.std())
    if distinct_data is not None:
        crit_1.columns = ['crit_1']
        crit_2.columns = ['crit_2']
        crit_3.columns = ['crit_3']
    else:
        crit_1 = pd.DataFrame(crit_1, columns=['crit_1'])
        crit_2 = pd.DataFrame(crit_2, columns=['crit_2'])
        crit_3 = pd.DataFrame(crit_3, columns=['crit_3'])

    crit = pd.concat([crit_1, crit_2, crit_3], axis=1)
    print(crit)
    crit['crit'] = crit['crit_1'].apply(expit) * crit['crit_2'].apply(expit) * crit['crit_3'].apply(expit)

    return crit['crit']


def add_par(x, point):
    par = 1 if np.array_equal(x, np.array(point)) else 0
    return par


def prepare_next_points_for_mult_opt(neighbours_with_crit, params_amount) -> pd.DataFrame:
/*
------------- UNDER NDA -----------------------
*/
    return next_points.astype(np.int64, copy=False)


def calculate_gradient(next_points, current_points, alpha, moment):
/*
------------- UNDER NDA -----------------------
*/
    return gradient


def generate_candidates(current_points, gradient, steps, params_amount):
/*
------------- UNDER NDA -----------------------
*/
    return candidates


def optimizer(points, api_params, list_of_ranges):
/*
------------- UNDER NDA -----------------------
*/
        gradient = calculate_gradient(best_points, current_points, config['alpha'], moment)
        candidates = generate_candidates(current_points, gradient, config['steps'], params_amount)

        print(candidates)


if __name__ == '__main__':
    def create_start_points(N: int, parameters_amount: int) -> pd.DataFrame:
        numbers_of_optimizations = np.arange(N).reshape((-1, 1))
        point_values = np.random.randint(low=1, high=100, size=(N, parameters_amount))
        # point_zeros = np.zeros((N, parameters_amount))
        # ti = np.ones(count, dtype=np.int32).reshape((count, 1))
        values = np.concatenate([numbers_of_optimizations, point_values], axis=1)
        columns = ['opt'] + ['idx' + str(val + 1) for val in range(parameters_amount)]

        return pd.DataFrame(values, columns=columns, dtype=np.int64)


    points = create_start_points(2, 4)
    moment = create_start_moment(len(points), 4)
    all_points_data = pd.DataFrame()
    all_points_kernels = pd.DataFrame()
    ind_columns = ['idx' + str(val + 1) for val in range(4)]
    for row in points.values.tolist():
        opt = row[0]
        point = row[1:]
        data, kernels = get_all_neighbours(point, ind_columns)
        data['opt'] = opt
        kernels['opt'] = opt
        kernels['par'] = kernels[ind_columns].apply(lambda x: add_par(np.array(x), point), axis=1)
        all_points_data = all_points_data.append(data)
        all_points_kernels = all_points_kernels.append(kernels)
    distinct_data = drop_for_api(all_points_data)
    answer = api(distinct_data, None)
    crit = calculate_crit(all_points_data.reset_index(drop=True), distinct_data, answer, ind_columns)
    data_with_crit = pd.concat([all_points_kernels.reset_index(drop=True), crit.reset_index()['crit']], axis=1)
    best_points = prepare_next_points_for_mult_opt(data_with_crit, 4)

    gradient = calculate_gradient(best_points, points, config['alpha'], moment)
    candidates = generate_candidates(points, gradient, config['steps'], 4)

    all_candidates_kernels = pd.DataFrame()
    for row in candidates.values.tolist():
        opt = row[0]
        point = row[1:]
        mask = get_mask(1, 1)
        kernels = pd.DataFrame(point + mask, columns=ind_columns)
        kernels['opt'] = opt
        kernels['par'] = kernels[ind_columns].apply(lambda x: add_par(np.array(x), point), axis=1)
        all_candidates_kernels = all_candidates_kernels.append(kernels)

    answer = api(all_candidates_kernels, None)
    crit = calculate_crit(all_candidates_kernels.reset_index(drop=True), None, answer, ind_columns)
    candidates_with_crit = pd.concat([all_points_kernels.reset_index(drop=True), crit.reset_index()['crit']], axis=1)
    print(candidates)
    print(crit)
