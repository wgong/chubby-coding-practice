import multiprocessing as mp
import functools
import time

def square(x):
    return x*x

def map_multiprocessing(nums, n_processes=1):
    start = time.time()
    with mp.Pool(n_processes) as p:
        result = p.map(square, nums)
    print(f'multiprocessing: {time.time()-start}')
    # print(result)

def map_functools(nums):
    start = time.time()
    result = map(square, nums)
    print(f'functools: {time.time()-start}')
    # print(result)

if __name__ == '__main__':
    for n in range(1,8):
        size = 10**n
        print(size)

        nums = list(range(size))
        map_multiprocessing(nums)

        map_functools(nums)