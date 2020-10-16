import collections
import json
import logging
import time

import numpy as np
from flask import Flask, request

from optimization.optimization import Optimization

app = Flask(__name__)


@app.route('/')
@app.route("/livetest")
def livetest():
    return ('', 204)


@app.route('/init', methods=["GET", "POST"])
def adder():
    dict_value = request.get_json()  # Значение json переданное POST - запросом, type=dict по умолчанию
    log_id = '1'
    if 'unique_database_id' in dict_value:
        log_id = str(dict_value['unique_database_id'])
        del dict_value['unique_database_id']

    logger = set_logger(log_id)

    logger.info(dict_value)

    indicators_limits = dict_value['indicatorsLimits']
    indicators_order = dict_value['indicatorsOrder']
    api_parameters = dict_value
    if 'points' in dict_value:
        points = dict_value['points']
        list_of_tuples = [(key, indicators_limits[key]) for key in indicators_order]
    else:
        points = None
        list_of_tuples = [(key, indicators_limits[key]) for key in [indicators_order[0]]]

    dict_of_ranges = collections.OrderedDict(list_of_tuples)

    opt_start = time.time()
    optimizer = Optimization(logger, dict_of_ranges)
    result = optimizer.optimize(points, api_parameters)
    logger.info(f'Full optimization time: {time.time() - opt_start}')

    columns = [name for name in result.columns.get_level_values(0) if 'idx' in name]
    answer = json.dumps({"point": result[columns].values.reshape(-1).tolist(),
                         "crit": np.float64(result['crit']),
                         "pnlTotal": [np.float64(result[('pnlTotal', 1)]),
                                      np.float64(result[('pnlTotal', 2)]),
                                      np.float64(result[('pnlTotal', 3)])],
                         "maxDrawdown": [np.float64(result[('maxDrawdown', 1)]),
                                         np.float64(result[('maxDrawdown', 2)]),
                                         np.float64(result[('maxDrawdown', 3)])],
                         "tradesNum": [np.float64(result[('tradesNum', 1)]),
                                       np.float64(result[('tradesNum', 2)]),
                                       np.float64(result[('tradesNum', 3)])],
                         "riskReward": [np.float64(result[('riskReward', 1)]),
                                        np.float64(result[('riskReward', 2)]),
                                        np.float64(result[('riskReward', 3)])]
                         })

    return answer


def set_logger(log_id):
    extra = {'log_id': str(log_id)}
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    formatter = logging.Formatter('- %(log_id)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger = logging.LoggerAdapter(logger, extra)

    return logger


if __name__ == '__main__':
    # log.set_file_handler('logs')
    app.run(debug=True, host='0.0.0.0', port=80, threaded=True)
