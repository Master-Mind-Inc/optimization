import collections
import json
import logging
import time

import mysql.connector
import numpy as np

from optimization.optimization import Optimization


def main():
    # config = json.loads(open('config.json').read())
    config = json.loads(open('/app/config.json').read())
    global mySQL_conn, cursor, dict_value, key
    try:
        mySQL_conn = mysql.connector.connect(host=config["DB_HOST"],
                                             database=config["DB_DATABASE"],
                                             user=config["DB_USER"],
                                             password=config["DB_PASSWORD"])
        cursor = mySQL_conn.cursor()
        cursor.callproc('db_Optimizer.udp_getTask')

        for result in cursor.stored_results():
            result = result.fetchall()
            key = json.dumps(json.loads(result[0][0].replace('\n', '')))
            dict_value = json.loads(result[0][1].replace('\n', ''))

        cursor.close()
        mySQL_conn.close()

        # optimize
        if (not key) or (not dict_value):
            return

        log_id = 1
        if 'unique_database_id' in dict_value:
            log_id = dict_value['unique_database_id']
            del dict_value['unique_database_id']

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

        logger = set_logger(log_id)

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

        mySQL_conn = mysql.connector.connect(host=config["DB_HOST"],
                                             database=config["DB_DATABASE"],
                                             user=config["DB_USER"],
                                             password=config["DB_PASSWORD"])
        cursor = mySQL_conn.cursor()

        sql = 'db_Optimizer.udp_updateResult'
        cursor.callproc(sql, (answer, key))
        mySQL_conn.commit()

    except mysql.connector.Error as error:
        logger.error("Failed to execute stored procedure: {}".format(error))
    finally:
        if mySQL_conn.is_connected():
            cursor.close()
            mySQL_conn.close()


def set_logger(log_id):
    extra = {'log_id': str(log_id)}
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('- %(log_id)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger = logging.LoggerAdapter(logger, extra)

    return logger


if __name__ == '__main__':
    main()
