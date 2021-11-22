Welcome to Facebook!

This is just a simple shared plaintext pad, with no execution capabilities.

When you know what language you would like to use for your interview,
simply choose it from the dropdown in the top bar.

Enjoy your interview!


# Create a function which takes two inputs: a sorted integer array and a target number N. This function should return the count of times the target appears in the array. The target number might appear zero or more times in the array. Example: for nums [1, 2, 3, 7, 7, 7, 9, 9] and a target of 7, this would return 3 since 7 appears 3 times in the array. If the target is 9, it would return 2, and if the target is 4, it would return 0. */

[1], 1

"""
searchLeft -> 0
0,1 -> m=0
0,0


searchNextRight -> 1
0,1 -> m=0
1,1
"""


def count(nums, target):
    counter = 0
    for x in nums:
        if x == target:
            counter += 1
    return counter


"""
0, 8 -> m = 4
0, 4 -> m = 2
3, 4 -> m = 3
3, 3
"""

def searchLeft(nums, target):
    l, r = 0, len(nums)
    
    while l < r:
        m = (l+r) // 2
        if nums[m] >= target:
            r = m
        else:
            l = m + 1
    return l


"""
0, 8 -> m = 4
5, 8 -> m = 6
5, 6 -> m = 5
6, 6
"""

def searchNextRight(nums, target):
    l, r = 0, len(nums)
    
    while l < r:
        m = (l+r) // 2
        if nums[m] > target:
            r = m
        else:
            l = m + 1
    return l

def count2(nums, target):
    left = searchLeft(nums, target)
    right = searchNextRight(nums, target)
    return right - left # + 1




# Given a list of fraction expressions, evaluate a target fraction based on the inputs. i.e inputs "a/b = 2", "b/d = 3", "d/c = 5", "f/h = 20", find "a/c" -> a/b * b/d * d/c = 2 * 3 * 5 = 30 */

from queue import Queue

def getNumerDenom(frac):
    # returns (numerator, denominator)
    pass

def getAllVariables(input_fracs)
    # return all variables as list
    pass

def getFracValue(numer, denom, input_values):
    pass

def evaluateFraction(input_fracs, input_values, target_frac):
    # 1. construct graph (adjacency matrix) -> O(n^2)
    variables = getAllVariables(input_fracs) # n
    
    adj_matrix = [[0 for _ in range(len(variables))] for _ in range(len(variables))]
    for frac in input_fracs:
        numer, denom = getNumerDenom(frac)
        adj_matrix[numer][denom] = 1
        # adj_matrix[denom][numer] = 1
        
    # 2. Find path from start -> end 
    # Use BFS -> O(n+m) ; O(n^2)
    start, end = getNumerDenom(target_frac)
    
    q = Queue()
    visited = [0 for _ in range(len(variables))]
    pred = [None for _ in range(len(variables))]
    
    q.put(start)
    visited[start] = 1
    
    while not q.empty():
        i = q.get()
        
        if i == end:
            break
        
        for j in range(len(variables)):
            if adj_matrix[i][j] and not visited[j]:
                q.put(j)
                pred[j] = i
                visited[j] = 1
    
    # 3. Reconstruct the shortest path -> O(n)
    path = []
    ptr = end
    while ptr is not None:
        path.append(ptr)
        ptr = pred[ptr]
    
    # 4. evaluate fractions
    # path = [ a , b, d , c ]
    result = 1
    for i in range(len(path)-1):
        result *= getFracValue(path[i], path[i+1], input_values)
        
    return result
    
            











