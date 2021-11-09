import math
import os
import random
import multiprocessing as mp
import numpy
import time

PRINT_SLICE_INFO = False
PRINT_ITERATION_NO = True
PRINT_ITERATION_BEST_ANSWER = True
PRINT_ITERATION_BEST_ANSWER_DETAILS = False
PRINT_ITERATION_ALL_ANSWERS = False
PRINT_TIME_INFO = False
PRINT_ALL_TIME_INFO = True

PARALLEL = False


class Chromosome:
    def __init__(self, data=None):
        global QUEENS

        if data is None:
            self.__data = list(range(QUEENS))
            random.shuffle(self.__data)
        else:
            self.__data = data

        self.__maindiagonals = {key: 0 for key in range(-QUEENS, QUEENS + 1)}
        self.__antidiagonals = {key: 0 for key in range(2 * QUEENS - 1)}
        self.cost = 0
        for i in range(QUEENS):
            self.__maindiagonals[i - self.__data[i]] += 1
            self.__antidiagonals[i + self.__data[i]] += 1
        diagonals = list(self.__maindiagonals.values()) + list(self.__antidiagonals.values())
        for diagonal in diagonals:
            if (diagonal > 0):
                self.cost += diagonal - 1

    def __str__(self):
        return self.__data.__str__() + ': ' + str(self.cost)

    def __lt__(self, other):
        return self.cost > other.cost

    def __mul__(self, other):
        global QUEENS

        (side1, side2) = random.sample(range(QUEENS + 1), 2)
        start = min(side1, side2)
        end = max(side1, side2)
        if PRINT_SLICE_INFO:
            print(start, end)
        first_child = Chromosome(self.__crossover(self.__data, other.__data, start, end))
        second_child = Chromosome(self.__crossover(other.__data, self.__data, start, end))
        return [first_child, second_child]

    def __invert__(self):
        return self.__swap(random.randint(0, MUTATION_DEGREE), False)

    def __pos__(self):
        return self.__swap(random.randint(LOCAL_SEARCH_DEGREE[0], LOCAL_SEARCH_DEGREE[1]), True)

    def __swap(self, count, should_be_better):
        global QUEENS

        result = Chromosome(self.__data)

        for _ in range(count):
            (q1, q2) = random.sample(range(QUEENS), 2)
            if PRINT_SLICE_INFO:
                print(q1, q2)
            new_cost = result.cost
            new_maindiagonals = result.__maindiagonals.copy()
            new_antidiagonals = result.__antidiagonals.copy()

            new_maindiagonals[q1 - result.__data[q1]] -= 1
            if (new_maindiagonals[q1 - result.__data[q1]] >= 1):
                new_cost -= 1
            new_maindiagonals[q2 - result.__data[q2]] -= 1
            if (new_maindiagonals[q2 - result.__data[q2]] >= 1):
                new_cost -= 1
            new_antidiagonals[q1 + result.__data[q1]] -= 1
            if (new_antidiagonals[q1 + result.__data[q1]] >= 1):
                new_cost -= 1
            new_antidiagonals[q2 + result.__data[q2]] -= 1
            if (new_antidiagonals[q2 + result.__data[q2]] >= 1):
                new_cost -= 1
            new_maindiagonals[q1 - result.__data[q2]] += 1
            if (new_maindiagonals[q1 - result.__data[q2]] > 1):
                new_cost += 1
            new_maindiagonals[q2 - result.__data[q1]] += 1
            if (new_maindiagonals[q2 - result.__data[q1]] > 1):
                new_cost += 1
            new_antidiagonals[q1 + result.__data[q2]] += 1
            if (new_antidiagonals[q1 + result.__data[q2]] > 1):
                new_cost += 1
            new_antidiagonals[q2 + result.__data[q1]] += 1
            if (new_antidiagonals[q2 + result.__data[q1]] > 1):
                new_cost += 1

            if new_cost <= result.cost or not should_be_better:
                result.__data[q1], result.__data[q2] = result.__data[q2], result.__data[q1]
                result.__maindiagonals = new_maindiagonals
                result.__antidiagonals = new_antidiagonals
                result.cost = new_cost

        return result

    @staticmethod
    def __crossover(mother_data: list, father_data: list, start: int, end: int) -> list:
        dimension = len(mother_data)
        data = [None] * dimension
        data[start:end] = mother_data[start:end]
        i = end
        for v in father_data[end:] + father_data[:end]:
            if v not in data:
                if i == start:
                    i = end
                if i == dimension:
                    i = 0
                data[i] = v
                i += 1
        return data

    def solved(self):
        return self.cost == 0


class Population:
    def __init__(self, countOrData):
        if type(countOrData) == int:
            self.__data = [Chromosome() for _ in range(countOrData)]
        elif type(countOrData) == list:
            self.__data = countOrData
        else:
            raise Exception()
        self.__data.sort()

    def iterate(self):
        t0 = time.time()

        children = self.__crossover()

        t1 = time.time()
        if PRINT_TIME_INFO:
            print(f'Crossover took {t1 - t0}')

        children.__mutate()

        t2 = time.time()
        if PRINT_TIME_INFO:
            print(f'Mutation took {t2 - t1}')

        self.__replacement(children)

        t3 = time.time()
        if PRINT_TIME_INFO:
            print(f'Replacement took {t3 - t2}')

        self.__local_search()

        t4 = time.time()
        if PRINT_TIME_INFO:
            print(f'Local Search took {t4 - t3}')

    def __choose(self):
        n = len(self.__data)
        roulette = sum([[i] * (i + 1) for i in range(n)], [])
        turning = random.randint(0, n)
        roulette = roulette[turning:] + roulette[:turning]
        pointers = range(0, len(roulette), math.ceil(len(roulette) / n))

        choices = []
        for pointer in pointers:
            choices.append(self.__data[roulette[pointer]])

        return choices

    def __crossover(self):
        global P_COUNT

        parents = self.__choose()
        random.shuffle(parents)

        if PARALLEL:
            def pair_chunk_calculator(i, pair_chunk, rd):
                rd[i] = sum([pair[0] * pair[1] for pair in pair_chunk], [])

            pair_chunks = numpy.array_split([[parents[i], parents[i + 1]]
                                            for i in range(0, len(parents) - 1, 2)], P_COUNT)
            manager = mp.Manager()
            rd = manager.dict()
            processes = [mp.Process(
                target=pair_chunk_calculator,
                args=(i, pair_chunks[i], rd)
            ) for i in range(P_COUNT)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            return Population(sum(rd.values(), []))
        else:
            return Population(sum([parents[i] * parents[i + 1] for i in range(0, len(parents) - 1, 2)], []))

    def __mutate(self):
        self.__data = [~c for c in self.__data]

    def __replacement(self, children):
        n = len(children.__data)
        best_children_count = math.floor(REPLACEMENT[0] * n)
        other_children_count = math.floor(REPLACEMENT[1] * n)
        other_parents_count = math.floor(REPLACEMENT[2] * n)
        best_parents_count = n - best_children_count - other_children_count - other_parents_count
        self.__data = (
            children.__data[-best_children_count:] +
            random.sample(children.__data[:(n - best_children_count)], other_children_count) +
            random.sample(self.__data[:(n - best_parents_count)], other_parents_count) +
            self.__data[-best_parents_count:]
        )
        self.__data.sort()

    def __local_search(self):
        self.__data = [+c for c in self.__data]

    def answer(self) -> Chromosome:
        return self.__data[-1]

    def answers(self) -> list:
        return list(map(lambda c: c.cost, self.__data))


t_start = time.time()

P_COUNT = os.cpu_count()

QUEENS = 5000
N = 10
MUTATION_DEGREE = 1
LOCAL_SEARCH_DEGREE = [150, 200]
REPLACEMENT = [.7, .1, .1]
ESCAPE_THRESHOLD_PROPORTION = .3
ESCAPE_PROPORTION = .5
population = Population(N)
i = 0
while True:
    if PRINT_ITERATION_NO:
        print(f"Iteration: {i}")
    if PRINT_ITERATION_BEST_ANSWER:
        print(f"Best Answer: {population.answer().cost}")
    if PRINT_ITERATION_BEST_ANSWER_DETAILS:
        print(population.answer())
    if PRINT_ITERATION_ALL_ANSWERS:
        print(f"All Answers: {population.answers()}")

    population.iterate()

    if population.answer().solved():
        break
    i += 1

print(population.answer())

t_end = time.time()
if PRINT_ALL_TIME_INFO:
    print(f'The whole process took {t_end - t_start}')
