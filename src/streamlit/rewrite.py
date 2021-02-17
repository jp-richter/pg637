import json
import numpy

with open('SAC-A-0205/LogsCache.json', 'r') as file:
    logs = json.load(file)

logs['actionAnkleDistributionTube'] = {
    'Framestamps': 'True',
    'Plot Type ': 'tube',
    'Values': []
}

a = 2
b = 2
c = 1

for i in range(1000):
    a += 1 - numpy.random.normal(1, 1)
    b += 1 - numpy.random.normal(1, 1)
    b = max(b, 1.0)

    logs['actionAnkleDistributionTube']['Values'].append((c, a, b))

    c += 1
    c = c % 30

for name, log in logs.items():
    log['Framestamps'] = 'True'


with open('SAC-A-0205/Logs.json', 'w') as file:
    json.dump(logs, file, indent=4)
