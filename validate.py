import sys
import json

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

__maindiagonals = {key: 0 for key in range(-N, N + 1)}
__antidiagonals = {key: 0 for key in range(2 * N - 1)}
cost = 0
for i in range(N):
    __maindiagonals[i - c[i]] += 1
    __antidiagonals[i + c[i]] += 1
diagonals = list(__maindiagonals.values()) + list(__antidiagonals.values())
for diagonal in diagonals:
    if (diagonal > 0):
        cost += diagonal - 1

print(cost)
