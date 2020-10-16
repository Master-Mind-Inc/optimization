import pandas as pd
# какая-то лажа с загрузкой модуля на каталог выше.
# TODO: need fix
from ..optimization.optimization_func_trash import check_borders

print('-----TEST #1 -----')
print('-----BEFORE-------')
data = pd.DataFrame({'idx1': [9, 10, 11],
                     'idx2': [39, 30, 31],
                     'idx3': [78, 79, 80]})
print(data)
print(check_borders(data, 3, [[1, 100], [1, 32], [79, 100]]))
print('-----AFTER-------')

print('-----TEST #2 -----')
print('-----BEFORE-------')
data = pd.DataFrame({'idx1': [1, 2, 3, 4, 5, 6, 7]})
print(data)
print(check_borders(data, 1, [[3, 6]]))
print('-----AFTER-------')
