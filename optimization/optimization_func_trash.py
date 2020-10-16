import numpy as np
import pandas as pd
import json
import itertools
import requests
from scipy.special import expit

PATH = 'app/'
config = json.loads(open(PATH + 'config.json').read())


def create_start_points(N: int, parameters_amount: int) -> pd.DataFrame:
    numbers_of_optimizations = np.arange(N).reshape((-1, 1))
    point_values = np.random.randint(low=1, high=100, size=(N, parameters_amount))
    # point_zeros = np.zeros((N, parameters_amount))
    # ti = np.ones(count, dtype=np.int32).reshape((count, 1))
    values = np.concatenate([numbers_of_optimizations, point_values], axis=1)
    columns = ['opt'] + ['idx' + str(val + 1) for val in range(parameters_amount)]

    return pd.DataFrame(values, columns=columns, dtype=np.int64)


def get_mask(k, ind_num):
    change = np.vstack(list(itertools.product(np.array([1, 0, -1]), np.array([1, 0, -1]))))
    indexes = np.unique(list(map(lambda x: list(set(x)), list(itertools.product(np.array([1, 2, 3, 4]) - 1, \
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


def get_kernel(dot, mask):
    return dot + mask


def get_neighs(kernel, mask):
    neighs = np.vstack([x + mask for x in kernel])
    neighs[neighs < 0] = 0
    labels = np.repeat(np.arange(len(kernel)), len(kernel)).reshape(len(kernel) ** 2, 1)
    data = pd.DataFrame((np.concatenate((labels, neighs), axis=1)))
    return data

def drop_for_api(data):
    data_to_api = data.drop_duplicates(subset=['idx1', 'idx2', 'idx3', 'idx4']).reset_index()[
        ['idx1', 'idx2', 'idx3', 'idx4']]
    return data_to_api


def get_k_pl(x):
    x['k_pl'] = x['pnlTotal'] / (np.abs(x['maxDrawdown']) + 1)
    return x


def calc_res(x):
    return x['k_pl'].mean() - (2 * x['k_pl'].std())


def generate_data(point, params_amount):
    print(kernels.shape)
    data = get_neighs(kernels, mask)
    data = pd.DataFrame(data)
    data.columns = ['ind'] + ['idx' + str(val + 1) for val in range(params_amount)]
    return data

def fake_api(data):
    def create_fake_api_answer(N):
        point_values = np.random.random(size=(N, 2))
        columns = ['pnlTotal', 'maxDrawdown']
        return pd.DataFrame(point_values, columns=columns)

    ans = create_fake_api_answer(len(data))
    return ans


def calculate_crit(data):
    data['k_pl'] = data['pnlTotal'] / (np.abs(data['maxDrawdown']) + 1)
    grouped_by_ind = data[['ind', 'k_pl']].groupby(['ind'])
    crit = grouped_by_ind.mean() - (2 * grouped_by_ind.std())
    crit.columns = ['crit']
    return crit


def prepare_next_points_for_mult_opt(neighbours_with_crit, params_amount) -> pd.DataFrame:
/*
------------- UNDER NDA -----------------------
*/
    return next_points.astype(np.int64, copy=False)


def select_best_neighbours(neighbours: pd.DataFrame, parameters_amount: int) -> pd.DataFrame:
/*
------------- UNDER NDA -----------------------
*/
    return best_neighbours[columns]


def save_points(saved, neighbours_with_crit, params_amount, counter):
/*
------------- UNDER NDA -----------------------
*/

    best_neighbours = select_best_neighbours(neighbours_with_crit, params_amount)
    data = best_neighbours.loc[best_neighbours['par'] == 1, ['opt', 'crit']]
    for opt in map(int, list(data['opt'])):
        value = best_neighbours[(best_neighbours['par'] == 1) &
                                (best_neighbours['opt'] == opt)].drop(['par'], axis=1)
        saved = saved.append(pd.DataFrame(data=value.values))
    return saved


def add_par(x, point):
    par = 1 if np.array_equal(x, np.array(point)) else 0
    return par



def create_start_moment(N: int, parameters_amount: int) -> pd.DataFrame:
    numbers_of_optimizations = np.arange(N).reshape((-1, 1))
    point_values = np.zeros((N, parameters_amount))
    values = np.concatenate([numbers_of_optimizations, point_values], axis=1)
    columns = ['opt'] + ['idx' + str(val + 1) for val in range(parameters_amount)]

    return pd.DataFrame(values, columns=columns, dtype=np.int64)


def check_borders(current_points, params_amount, list_of_range):
    # Need testing
    columns = ['idx' + str(val + 1) for val in range(params_amount)]
    for current_range, column_name in zip(list_of_range, columns):
        current_points.loc[current_points[column_name] < current_range[0], column_name] = current_range[0]
        current_points.loc[current_points[column_name] > current_range[1], column_name] = current_range[1]

    return current_points.astype(np.int64)

def get_best_points(points):
/*
------------- UNDER NDA -----------------------
*/


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


def update_moment(current_points, best_points, gradient, moment, coef_forget, params_amount):
    coordinate_names = ['idx' + str(val + 1) for val in range(params_amount)]
    moment = moment[coordinate_names] * coef_forget + (1. - coef_forget) * (best_points - current_points)[
        coordinate_names]
    moment['opt'] = gradient['opt']
    moment = moment.dropna()
    return moment


def calculate_crit(points_with_api, params_amount):
/*
------------- UNDER NDA -----------------------
*/
    return points_with_api


def call_API_1(point, api_params, list_of_ranges):
    json = {"api_params": api_params,
            "opt": point}

    resp = None  # should be array? what columns
    return resp


def call_API_2(points, api_params, list_of_ranges):
    json = {"api_params": api_params,
            "opt_list": points}

    resp = None  # should be array? what columns
    return resp


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


def convert_points_from_our_format_to_normal(points, list_of_ranges):
    for idx, border in enumerate(list_of_ranges):
        left, right, type_of_var = border
        if type_of_var == 0:
            step = round((right - left) / 100.0, 2)
            column = 'idx' + str(idx + 1)
            points[column] = left + points[column] * step

    return points


def optimizer(points, api_params, list_of_ranges):
    params_amount = len(list_of_ranges)
    current_points = convert_start_points(points, params_amount)
    counter = 1

    # current_points = create_start_points_with_ranges(config['n_opts'], params_amount, list_of_ranges)
    moment = create_start_moment(len(current_points), params_amount)

    while not current_points.empty:
        best_points = []
        current_points = check_borders(current_points, params_amount, list_of_ranges)
        all_points = pd.DataFrame()
        # current_points = create_start_points(2, params_amount)
        ind_columns = ['idx' + str(val + 1) for val in range(params_amount)]
        for row in current_points.values.tolist():
            # data = generate_data(point, params_amount)
            opt = row[0]
            point = row[1:]

            mask = get_mask(1, 1)
            kernels = get_kernel(point, mask)
            data = get_neighs(kernels, mask)
            data.columns = ['ind'] + ind_columns

            data_for_api = drop_for_api(data)
            answer = fake_api(data_for_api)
            #
            result = pd.concat([data_for_api, answer], axis=1, join_axes=[data_for_api.index])
            merged = data.merge(result, how='left', on=ind_columns)
            crit = calculate_crit(merged)

            neighs_with_crit = pd.concat([pd.DataFrame(kernels, columns=ind_columns), crit], axis=1,
                                         join_axes=[crit.index])

            neighs_with_crit['par'] = neighs_with_crit[ind_columns].apply(lambda x: add_par(np.array(x), point), axis=1)
            neighs_with_crit['opt'] = opt

            all_points = all_points.append(neighs_with_crit)

        best_points = prepare_next_points_for_mult_opt(all_points, params_amount)

        gradient = calculate_gradient(best_points, current_points, config['alpha'], moment)
        candidates = generate_candidates(current_points, gradient, config['steps'], params_amount)

        all_candidates = pd.DataFrame()
        for idx, point in enumerate(candidates.values.tolist()):
            mask = get_mask(1, 1)
            data_for_api = get_kernel(point, mask)
            answer = call_API_2(data_for_api, api_params)
            result = pd.concat([data_for_api, answer], axis=1, join_axes=[data_for_api.index])
            merged = data_for_api.merge(result, how='left', on=ind_columns)
            crit = calculate_crit(merged)
            candidates_with_crit = pd.concat([pd.DataFrame(data_for_api, columns=ind_columns), crit], axis=1,
                                             join_axes=[crit.index])
            all_candidates = all_candidates.append(candidates_with_crit)

        best_candidates = get_best_points(all_candidates)

        all_points_with_crit = pd.concat([best_points, best_candidates], axis=0).reset_index(drop=True)
        best = prepare_next_points_for_mult_opt(all_points_with_crit, params_amount)
        moment = update_moment(current_points, best, gradient, moment, config['coef_forget'], params_amount)

        current_points = best_points

        print('iteration:', counter)
        counter += 1

    print('DEAD END')
    return None


def api(points, api_params):
    assert type(api_params) == dict
    api_params["points"] = points.values.astype(np.int64)

    url = ""
    status = requests.post(url, json={
        "strategyItemDNF": api_params,
        "base": None,
        "quote": None,
        "interval": None,
        "exchange": None,
        "dateStart": None,
        "dateFinish": None
    })
    data = status.json()
    assert type(data) == list

    columns = ['pnl', 'mdd']  # TODO: add columns name
    res = [pd.concat([points, pd.DataFrame(columns=columns, data=data[i])], axis=1) for i in [0, 1, 2]]
    return res


def generate_data(dot):
    data = get_kernel(dot=dot, mask=get_mask(1, 1))
    data = pd.DataFrame(data)
    data.columns = ['idx1', 'idx2', 'idx3', 'idx4']
    return data


def crit(point, api_params):
    calc = pd.DataFrame(np.array(point))
    calc = calc.T
    calc.columns = ['idx1', 'idx2', 'idx3', 'idx4']

    calc = calc.apply(lambda x: generate_data(np.array(x)), axis=1)

    data_to_api = calc.apply(drop_for_api)
    answer = api(data_to_api, api_params)

    merged = list(map(lambda a, b: a.merge(b, how='left', on=['idx1', 'idx2', 'idx3', 'idx4']), calc, answer))
    res = list(map(get_k_pl, merged))[0]
    return res.groupby('ti').apply(calc_res).apply(expit).prod()


params_amount = 4
all_points = pd.DataFrame()
all_candidates = pd.DataFrame()
api_params = {}
current_points = create_start_points(1, params_amount)
ind_columns = ['idx' + str(val + 1) for val in range(params_amount)]
for row in current_points.values.tolist():
    # data = generate_data(point, params_amount)
    opt = row[0]
    point = row[1:]

    neighbours_with_crit = crit(point, api_params)
    all_points = all_points.append(neighbours_with_crit)

best_points = prepare_next_points_for_mult_opt(all_points, params_amount)

gradient = calculate_gradient(best_points, current_points, config['alpha'], moment)
candidates = generate_candidates(current_points, gradient, config['steps'], params_amount)

for idx, point in enumerate(candidates.values.tolist()):
    mask = get_mask(1, 1)
    data_for_api = get_kernel(point, mask)
    answer = call_API_2(data_for_api, api_params)
    result = pd.concat([data_for_api, answer], axis=1, join_axes=[data_for_api.index])
    merged = data_for_api.merge(result, how='left', on=ind_columns)
    crit = calculate_crit(merged)
    candidates_with_crit = pd.concat([pd.DataFrame(data_for_api, columns=ind_columns), crit], axis=1,
                                     join_axes=[crit.index])
    all_candidates = all_candidates.append(candidates_with_crit)

best_candidates = get_best_points(all_candidates)

all_points_with_crit = pd.concat([best_points, best_candidates], axis=0).reset_index(drop=True)
best = prepare_next_points_for_mult_opt(all_points_with_crit, params_amount)
moment = update_moment(current_points, best, gradient, moment, config['coef_forget'], params_amount)

current_points = best_points
