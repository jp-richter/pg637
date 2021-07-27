#!/usr/bin/python

import random

batchSizeRange = (4, 4096)
sampleUsageRange = (3, 30)

lrActorRange = (-3, -6)
lrCriticRange= (-3, -6)

startStdRange = (-0.4, -1.0)
epsilonRange = (0.1, 0.3)

batchSize = random.sample(range(batchSizeRange[0], batchSizeRange[1]), 1)[0]
sampleUsage = random.sample(range(sampleUsageRange[0], sampleUsageRange[1]), 1)[0]

lrActor = str(round(10**random.uniform(lrActorRange[0], lrActorRange[1]), 7)) + "f"
lrCritic = str(round(10**random.uniform(lrCriticRange[0], lrCriticRange[1]), 7)) + "f"


startStd = str(round(random.uniform(startStdRange[0], startStdRange[1]), 7)) + "f"
epsilon = str(round(random.uniform(epsilonRange[0], epsilonRange[1]), 7)) + "f"

outputStr = ""

with open("HParamsSingletonTemplate.h") as f:
    outputStr = f.read()

outputStr = outputStr.format(
    batchSize=batchSize,
    sampleUsage=sampleUsage,
    lrActor=lrActor,
    lrCritic=lrCritic,
    startStd=startStd,
    epsilon=epsilon
)

with open("HParamsSingleton.h", "w") as f:
    f.write(outputStr)
