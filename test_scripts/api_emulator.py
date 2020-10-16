import requests
import json
import os

# print(os.listdir(os.curdir))

# 0 - float
# 1 - int

PATH = 'test_scripts/'

import os

print(os.listdir('.'))


init = json.loads(open(PATH + 'init.json').read())

status = requests.post("http://localhost/init", json=init)

# http://<host>:<post>/<comand>


print(status)
