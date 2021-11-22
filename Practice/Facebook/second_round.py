Description
Given an integer array, find whether there is duplication inside subarray with length of K.
Expected to return true if there is a duplication inside any of the array, and false if there is no duplication.

Example:
Array: [1, 2, 3, 4, 2, 5, 6]
K = 3  -->  false // No duplication in any subarray with length of K
K = 4  -->  true  // Because of subarray of [2, 3, 4, 2]


"""
Approach 1: Brute Force:

isUnique:
1. construct hash map (key, value)=(number, freq [bool])
2. iterate through hash map, check for duplicates


for all subarray:
    if not isUnique on subarry:
        return false
return True


Approach 2: Sliding window + hash

1. Initialize global hash
2. for start = 0, n-k:
    update hash
    3. remove the left value/decrement freq in hash
    
    4. add the right value/increment freq in hash
        check to duplicates
"""

def existDuplicates(nums, k):
#     1. Initialize global hash

    present = set()
    for i in range(min(k, len(nums)):
        if nums[i] in present:
            return True
        present.add(nums[i])
                   
    # present = {1,2,3}
    
#     2. for start = 0, n-k:

    for start in range(1, len(nums)-k+1):
        
#     update hash
#     3. remove the left value/decrement freq in hash
        present.remove(nums[start-1])
    
#     4. add the right value/increment freq in hash
        if nums[start+k-1] in present:
            return True
#         check to duplicates
        present.add(nums[start+k-1])
    
    return False
    
                   
# Array: [1, 2, 3, 4, 2, 5, 6] len= 7
# K = 3  -->  false // No duplication in any subarray with length of K
# K = 4  -->  true  // Because of subarray of [2, 3, 4, 2]
                   
start = 1: present = {2,3,4}
start = 2: present = {3,4,2}
start = 3: present = {4,2,5}
                   
                   
Complexity: O(n), Space: O(k)

############
                   
Find kth Smallest Value Among m Sorted Arrays

You have m arrays of sorted integers. The sum of the array lengths is n. Find the kth smallest value of all the values.

For exmaple, if m = 3, n=8, and we have these lists:

list1 = [3,6,9]
list2 = [8,15]
list3 = [4, 7, 12]

if k = 1, then returned value should be 3
if k = 2, then returned value should be 4
if k = 3, then returned value should be 6
                   
                   
"""
1: Merge all, sort


2: "Mergesort"


m pointers
"""

import heapq
                   
def kthSmallest(arrs, m, k):
    heap = [(arrs[i][0], i, 0) for i in range(m)] # (value, index in arrs: i, index in arrs[i])
    
    
    # Find pointer that corresponds to minimum value among in current pointers
    # min_p = 0
    # for i in range(m):
    #     if arrs[i][ptrs[i]] < arrs[min_p][ptrs[min_p]]:
    #         min_p = i
                  
    for i in range(k):
        if len(heap) > 0:
            val, idx, ptr = heapq.heappop(heap)

            if i == k-1:
                return val

            if ptr+1 < len(arrs[idx]):
                heapq.heappush(heap, (arrs[idx][ptr+1], idx, ptr+1))
        else:
            return None
                   
    
"""
list1 = [3,6,9]
list2 = [8,15]
list3 = [4, 7, 12]

if k = 1, then returned value should be 3
(3,0,0), (8,1,0), (4,2,0)



if k = 2, then returned value should be 4
(3,0,0), (8,1,0), (4,2,0)

(6,0,1), (8,1,0), (4,2,0)


if k = 3, then returned value should be 6
                   
(3,0,0), (8,1,0), (4,2,0)

(6,0,1), (8,1,0), (4,2,0)

(6,0,1), (8,1,0), (7,2,1)
                   

Complexity:  O(m) + O(k logm) Space: O(m)


                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
