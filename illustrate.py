import sys
import json
import matplotlib.pyplot as plt
import os

BLANK = (190, 219, 57)
QUEEN = (51, 105, 30)

if (len(sys.argv) < 2):
    raise Exception('please provide a file')
path = sys.argv[1]
try:
    with open(path) as f:
        c = json.load(f)
except:
    raise Exception('invalid file')

N = len(c)
m = [[BLANK for _ in range(N)] for _ in range(N)]

for i in range(N):
    m[i][c[i]] = QUEEN


plt.imshow(m, interpolation='none')
plt.axis('off')
plt.savefig(os.path.splitext(path)[0] + '.png')
