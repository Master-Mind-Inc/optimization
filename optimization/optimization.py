import itertools
import json
import time

import numpy as np
import pandas as pd
import requests
from scipy.special import expit


class Optimization:
    def __init__(self, logger, dict_of_ranges):
        self.api_time = 0
        # with open('/app/config.json') as f:
        with open('config.json') as f:
            config = f.read()
        self.config = json.loads(config)
        self.logger = logger
        self.ind_columns = None
        list_of_ranges = np.array([item for sublist in dict_of_ranges.values() for item in sublist])
        self.lower_bounds = list_of_ranges.T[0]
        self.upper_bounds = list_of_ranges.T[1]
        self.steps = list_of_ranges.T[2]
        self.params_amount = self.get_params_amount(dict_of_ranges)
        self.ind_columns = ['idx' + str(val + 1) for val in range(self.params_amount)]

    @staticmethod
    def get_params_amount(dict_of_ranges: dict) -> int:
        params_amount = sum([len(item) for item in dict_of_ranges.values()])
        return params_amount

    def convert_start_point(self, point: list) -> pd.DataFrame:
        data = pd.DataFrame([point], columns=self.ind_columns)
        return data[self.ind_columns].astype(np.float64).round(1)

    def create_start_moment(self, point: list) -> pd.DataFrame:
        return pd.DataFrame([[0.] * len(point)], columns=self.ind_columns)

    def create_random_points(self, N) -> list:
        point_values = [np.random.randint(lower_bound, upper_bound, size=(N, 1)) for lower_bound, upper_bound in
                        zip(self.lower_bounds, self.upper_bounds)]
        point_values = np.array(point_values).T[0]

        return point_values.tolist()

    @staticmethod
    def get_mask(k, ind_num):
        change = np.vstack(list(itertools.product(np.array([1, 0, -1]), np.array([1, 0, -1]))))
        indexes = np.unique(
            list(map(lambda x: list(set(x)), list(itertools.product(np.arange(1, ind_num + 1) - 1,
                                                                    np.arange(1, ind_num + 1) - 1)))))

        init_mask = np.zeros(ind_num)
        res = []
        for y in indexes:
            for x in change:
                if len(y) == 1:
                    np.put(init_mask, y[0], 1)
                    res.append(init_mask)
                    init_mask = np.zeros(ind_num)
                    np.put(init_mask, y[0], -1)
                    res.append(init_mask)
                    init_mask = np.zeros(ind_num)
                    np.put(init_mask, y[0], 0)
                    res.append(init_mask)
                    init_mask = np.zeros(ind_num)
                if len(y) > 1:
                    if k == 1:
                        continue
                    np.put(init_mask, y, x)
                    res.append(init_mask)
                    init_mask = np.zeros(ind_num)
        res = np.unique(np.vstack(res), axis=0)
        return res

    def get_all_neighbours(self, point):
        point = point.values.tolist()

        def get_neighs(kernel, mask):
            neighs = np.vstack([x + mask for x in kernel])
            labels = np.repeat(np.arange(len(kernel)), len(kernel)).reshape(len(kernel) ** 2, 1)
            data = pd.DataFrame((np.concatenate((labels, neighs), axis=1)))
            return data

        mask = self.get_mask(1, self.params_amount)
        mask = mask * self.steps

        kernels = point + mask

        data = get_neighs(kernels, mask)
        data.columns = ['ind'] + self.ind_columns

        return data

    def get_kernels(self, point):
        point = point.values.tolist()
        mask = self.get_mask(1, self.params_amount)
        mask = mask * self.steps
        kernels = point + mask

        ind = np.arange(len(kernels)).reshape((-1, 1))

        return pd.DataFrame(np.concatenate([ind, kernels], axis=1), columns=['ind'] + self.ind_columns)

    @staticmethod
    def add_par(x, point):
        par = 1 if np.array_equal(x.reshape(-1), point.reshape(-1)) else 0
        return par

    @staticmethod
    def get_k_pl(x):
        x['k_pl'] = x['pnlTotal'] / (np.abs(x['maxDrawdown']) + 1)
        return x

    @staticmethod
    def calc_res(x):
        return x['k_pl'].mean() - (2 * x['k_pl'].std())

    def clean_neighbours_by_ranges(self, data):
        mask = np.array(
            (self.lower_bounds <= data[self.ind_columns]) & (data[self.ind_columns] <= self.upper_bounds)).all(axis=1)
        return data[mask].reset_index(drop=True)

    def api(self, points: pd.DataFrame, api_params: dict, points_only: bool) -> pd.DataFrame:
        count_tries = 20
        points = points[:].values.tolist()
        self.logger.info(f'Points to api: {points}')
        self.logger.info(f'Points amount: {len(points)}')

        assert type(api_params) == dict

        api_params['points'] = points
        # api_params['pointsOnly'] = points_only

        url = self.config["URL"]

        while count_tries != 0:
            count_tries -= 1
            try:
                start_time = time.time()
                status = requests.post(url, json=api_params)
                finish_time = time.time()
                self.logger.info(f'API working time: {finish_time - start_time}')
                self.api_time += finish_time - start_time
            except Exception as e:
                self.logger.error(f'API Error: {e}')
                self.logger.error(f'Request: {api_params}')
                continue
            break
        try:
            data = status.json()
        except Exception as e:
            self.logger.error(f'Response status: {e}, {status}')
            raise

        assert type(data) == dict

        data = data['result']
        data1, data2, data3 = data[0], data[1], data[2]

        columns = self.ind_columns + ['pnlTotal', 'maxDrawdown', 'tradesNum', 'riskReward']

        ti1 = pd.DataFrame([row[0] + row[1] for row in data1], columns=columns)
        ti2 = pd.DataFrame([row[0] + row[1] for row in data2], columns=columns)
        ti3 = pd.DataFrame([row[0] + row[1] for row in data3], columns=columns)

        ti1['ti'] = 1
        ti2['ti'] = 2
        ti3['ti'] = 3

        response = pd.concat([ti1, ti2, ti3])
        response[self.ind_columns] = response[self.ind_columns].astype(np.float64).round(2)

        return response

    def approve_types(self, points: pd.DataFrame) -> pd.DataFrame:
        for step, column in zip(self.steps, self.ind_columns):
            if step == 1:
                points[column] = points[column].astype(np.int64)
            else:
                points[column] = points[column].astype(np.float64).round(1)

        return points

    def calculate_moment(self, moment, next_point, prev_point):
        forgetting = self.config['forgetting']
        up = (next_point[self.ind_columns] - prev_point[self.ind_columns]).reset_index(drop=True)
        moment = (forgetting * moment + (1. - forgetting) * up[self.ind_columns]).reset_index(drop=True)
        return moment

    def create_candidates(self, gradient, prev_point) -> pd.DataFrame:
        steps = self.config['steps']
        candidates = [(prev_point[self.ind_columns].reset_index(drop=True) + gradient * step) for step in steps]
        candidates = [self.clean_neighbours_by_ranges(self.approve_types(candidate)) for candidate in candidates]
        candidates = pd.concat(candidates).reset_index(drop=True)
        self.logger.info(f'Candidates: {candidates.values.tolist()}')
        return candidates

    def calculate_gradient(self, moment, next_point, prev_point):
        alpha = self.config['alpha']
        gradient = (next_point[self.ind_columns] - prev_point[self.ind_columns]).reset_index(drop=True)
        gradient = (alpha * gradient + (1. - alpha) * moment).reset_index(drop=True)
        self.logger.info(f'Gradient: {gradient.values.tolist()}')
        return gradient

    def get_crit_for_candidate(self, candidates: pd.DataFrame, api_params: dict) -> pd.DataFrame:
        data = candidates.to_numpy()
        mask = self.get_mask(1, len(data[0]))

        cols = self.ind_columns + ['crit']
        result = pd.DataFrame()
        for point in data:
            points = point + mask
            points = self.approve_types(pd.DataFrame(points, columns=self.ind_columns))
            points = self.clean_neighbours_by_ranges(points)
            points = self.api(points, api_params, points_only=True)
            points = self.get_k_pl(points)

            current_crit = points.groupby(['ti']).apply(self.calc_res).apply(expit).prod()
            current_raw = pd.DataFrame([point.tolist() + [current_crit]], columns=cols)
            result = result.append(current_raw)

        return result

    def make_result_df(self, prev_point, data_merged):
        mask = (data_merged[self.ind_columns] == prev_point[self.ind_columns].values)
        res_ind = data_merged[self.ind_columns][mask].dropna().index

        res = data_merged.iloc[res_ind].drop(['ind'], axis=1).drop_duplicates()
        res = res.set_index(self.ind_columns + ['ti']).unstack(level=-1)
        res['crit'] = prev_point['crit'].values
        res = res.reset_index()

        return res

    def get_expected_len(self, data):
        data_to_api = data.drop_duplicates(subset=self.ind_columns).reset_index()[self.ind_columns]
        return data_to_api.shape[0] * 3

    def one_point_optimizer(self, point: list, api_params: dict) -> pd.DataFrame:
        res = None
        self.logger.info(f'START POINT: {point}')

        current_point = self.convert_start_point(point)
        moment = self.create_start_moment(point)

        iteration = 1
        while not current_point.empty and iteration < 50:
            data = self.get_all_neighbours(current_point)
            kernels_with_ind = self.approve_types(data.groupby('ind').mean())

            self.logger.info(f'Neighbours shape before cleaning: {data.shape}')
            data = self.approve_types(data)
            data = self.clean_neighbours_by_ranges(data)
            self.logger.info(f'Neighbours shape after cleaning: {data.shape}')

            # response = self.api(current_point, api_params, points_only=False).reset_index(drop=True)
            # self.logger.info(f'Response shape: {response.shape}, expected len {self.get_expected_len(data)}')
            # response = self.approve_types(response)

            response_only_points = self.api(data[self.ind_columns], api_params, points_only=True).reset_index(drop=True)
            self.logger.info(f'Response shape: {response_only_points.shape}, expected len {data.shape[0] * 3}')
            response_only_points = self.approve_types(response_only_points)

            # data_merged = pd.merge(data, response, how='left', on=self.ind_columns)
            data_merged = pd.merge(data, response_only_points, how='left', on=self.ind_columns)

            self.logger.info(f'Merged data shape: {data_merged.shape}')
            data_merged = self.get_k_pl(data_merged)

            result = data_merged.groupby(['ti', 'ind']).apply(self.calc_res).apply(expit).reset_index().groupby(
                'ind').prod()
            result = result.reset_index().drop(['ti'], axis=1).rename({0: 'crit'}, axis=1)

            neigh_with_crit = kernels_with_ind.join(result, on='ind')
            self.logger.info(f'Neighbours with crit shape before cleaning: {neigh_with_crit.shape}')
            neigh_with_crit = self.approve_types(neigh_with_crit)
            neigh_with_crit = self.clean_neighbours_by_ranges(neigh_with_crit)
            self.logger.info(f'Neighbours with crit shape after cleaning: {neigh_with_crit.shape}')

            neigh_with_crit['par'] = neigh_with_crit[self.ind_columns].apply(
                lambda x: self.add_par(np.array(x), np.array(current_point)), axis=1)

            prev_point = neigh_with_crit.loc[neigh_with_crit['par'] == 1]
            next_point = neigh_with_crit.loc[neigh_with_crit['crit'].idxmax()]

            self.logger.info('CALCULATING MOMENT')

            gradient = self.calculate_gradient(moment, next_point, prev_point)
            candidates = self.create_candidates(gradient, prev_point)

            if not candidates.empty:
                candidates = self.get_crit_for_candidate(candidates, api_params).reset_index(drop=True)
                points = candidates.append(next_point[self.ind_columns + ['crit']]).reset_index(drop=True)
                next_point = points.loc[points['crit'].idxmax()]

                moment = self.calculate_moment(moment, next_point, prev_point)

            self.logger.info(f'Previous crit: {np.float64(prev_point["crit"])}, '
                             f'Current crit: {str(next_point["crit"])}')

            if prev_point['crit'].values >= next_point['crit']:
                res = self.make_result_df(prev_point, data_merged)
                self.logger.info(f'Result points: {prev_point[self.ind_columns + ["crit"]].values.tolist()}')
                break
            else:
                current_point = next_point[self.ind_columns].to_frame().T
                self.logger.info(f'Next points: {current_point.values.tolist()}')
                self.logger.info(f'----------------------Iteration {iteration} DONE------------------------')
                iteration += 1

        return res

    def one_point_optimizer_only_kernels(self, point: list, api_params: dict) -> pd.DataFrame:
        res = None
        self.logger.info(f'START POINT: {point}')

        current_point = self.convert_start_point(point)
        moment = self.create_start_moment(point)

        iteration = 1
        while not current_point.empty and iteration < 50:
            data = self.get_kernels(current_point)

            self.logger.info(f'Neighbours shape before cleaning: {data.shape}')
            data = self.approve_types(data)
            data = self.clean_neighbours_by_ranges(data)
            self.logger.info(f'Neighbours shape after cleaning: {data.shape}')

            response_only_points = self.api(data[self.ind_columns], api_params, points_only=True).reset_index(drop=True)
            self.logger.info(f'Response shape: {response_only_points.shape}, expected len {data.shape[0] * 3}')
            response_only_points = self.approve_types(response_only_points)

            data_merged = pd.merge(data, response_only_points, how='left', on=self.ind_columns)

            self.logger.info(f'Merged data shape: {data_merged.shape}')
            data_merged = self.get_k_pl(data_merged)
            data_merged['k_pl'] = data_merged[['k_pl']].apply(expit)
            result = data_merged[['ind', 'k_pl']].groupby(['ind']).prod().reset_index()

            neigh_with_crit = data.merge(result, on='ind').rename({'k_pl': 'crit'}, axis=1)

            self.logger.info(f'Neighbours with crit shape before cleaning: {neigh_with_crit.shape}')
            neigh_with_crit = self.approve_types(neigh_with_crit)
            neigh_with_crit = self.clean_neighbours_by_ranges(neigh_with_crit)
            self.logger.info(f'Neighbours with crit shape after cleaning: {neigh_with_crit.shape}')

            neigh_with_crit['par'] = neigh_with_crit[self.ind_columns].apply(
                lambda x: self.add_par(np.array(x), np.array(current_point)), axis=1)

            prev_point = neigh_with_crit.loc[neigh_with_crit['par'] == 1]
            next_point = neigh_with_crit.loc[neigh_with_crit['crit'].idxmax()]

            self.logger.info('CALCULATING MOMENT')

            gradient = self.calculate_gradient(moment, next_point, prev_point)
            candidates = self.create_candidates(gradient, prev_point)

            if not candidates.empty:
                candidates['ind'] = np.arange(len(candidates)).reshape((-1, 1))
                candidates = self.approve_types(candidates)
                candidates = self.clean_neighbours_by_ranges(candidates)
                response = self.api(candidates[self.ind_columns], api_params, points_only=True).reset_index(drop=True)
                response = self.approve_types(response)
                candidates_merged = pd.merge(candidates, response, how='left', on=self.ind_columns)
                candidates_merged = self.get_k_pl(candidates_merged)
                candidates_merged['k_pl'] = candidates_merged[['k_pl']].apply(expit)

                crit = candidates_merged[['ind', 'k_pl']].groupby(['ind']).prod().reset_index()
                candidates = candidates.merge(crit, on='ind').rename({'k_pl': 'crit'}, axis=1)

                points = candidates.append(next_point[self.ind_columns + ['crit']]).reset_index(drop=True)
                next_point = points.loc[points['crit'].idxmax()]

                moment = self.calculate_moment(moment, next_point, prev_point)

            self.logger.info(f'Previous crit: {np.float64(prev_point["crit"])}, '
                             f'Current crit: {str(next_point["crit"])}')

            if prev_point['crit'].values >= next_point['crit']:
                res = self.make_result_df(prev_point, data_merged)
                self.logger.info(f'Result points: {prev_point[self.ind_columns + ["crit"]].values.tolist()}')
                break
            else:
                current_point = next_point[self.ind_columns].to_frame().T
                self.logger.info(f'Next points: {current_point.values.tolist()}')
                self.logger.info(f'----------------------Iteration {iteration} DONE------------------------')
                iteration += 1

        return res

    def optimize(self, points, api_params):
        points_cnt = 1
        if (api_params['settings']['isAutoBuySell'] is True) and (api_params['settings']['isAutoEntryExit'] is True):
            points = self.create_random_points(50)
        best_points = pd.DataFrame()
        for point in points:
            res = self.one_point_optimizer_only_kernels(point, api_params)
            best_points = best_points.append(res)
            self.logger.info(f'===================== POINT {points_cnt} DONE =====================')
            points_cnt += 1
        best_points = best_points.reset_index(drop=True)
        self.logger.info(f'API time total: {self.api_time}')

        return best_points.loc[best_points['crit'].idxmax()].to_frame().T
