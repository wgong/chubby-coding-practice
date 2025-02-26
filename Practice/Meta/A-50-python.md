
# 1-10 

# Answers to Python Interview Questions 1-10

## 1. What are the key differences between Python 2 and Python 3?

Python 3 introduced several important changes that were not backward compatible with Python 2:

**Print Statement vs Function:**
- Python 2: `print "Hello"` (statement)
- Python 3: `print("Hello")` (function)

**Integer Division:**
- Python 2: `5 / 2` returns `2` (integer division)
- Python 3: `5 / 2` returns `2.5` (true division)
- Python 3: `5 // 2` returns `2` (floor division)

**Unicode Handling:**
- Python 2: Strings are ASCII by default, Unicode strings with `u""`
- Python 3: All strings are Unicode by default, bytes with `b""`

**Error Handling:**
- Python 2: `except Exception, e`
- Python 3: `except Exception as e`

**xrange vs range:**
- Python 2: `range()` returns a list, `xrange()` returns an iterator
- Python 3: `range()` returns an iterator for memory efficiency

**Other Important Differences:**
- Improved function annotations and type hinting in Python 3
- Dictionary views instead of lists for methods like `.keys()`
- Removal of `<>` in favor of `!=` for inequality
- Ordering comparisons between different types (like `1 < "2"`) raise TypeError in Python 3
- Changed handling of metaclasses
- Improved asynchronous capabilities with `async`/`await` (Python 3.5+)

## 2. Explain mutable vs immutable data types in Python with examples.

**Immutable Data Types:**
These cannot be changed after creation. When you "modify" them, you're actually creating a new object.

Examples:
- `int`: `x = 5` - integers cannot be modified in-place
- `float`: `y = 3.14` - floats cannot be modified in-place
- `str`: `s = "hello"` - strings create new objects when modified
- `tuple`: `t = (1, 2, 3)` - tuples cannot be modified after creation
- `frozenset`: `fs = frozenset([1, 2, 3])` - immutable version of set
- `bool`: `b = True` - booleans are immutable

Demonstration with strings:
```python
s = "hello"
id_before = id(s)
s += " world"
id_after = id(s)
print(id_before == id_after)  # False - a new string object was created
```

**Mutable Data Types:**
These can be modified in-place after creation.

Examples:
- `list`: `l = [1, 2, 3]` - you can modify elements or add/remove them
- `dict`: `d = {"a": 1, "b": 2}` - keys and values can be added/changed/removed
- `set`: `s = {1, 2, 3}` - elements can be added or removed
- Custom class instances (unless designed otherwise)

Demonstration with lists:
```python
l = [1, 2, 3]
id_before = id(l)
l.append(4)
id_after = id(l)
print(id_before == id_after)  # True - same object was modified
```

**Important Implications:**
- Immutable objects are hashable (can be dictionary keys or set members)
- Mutable objects cannot be used as dictionary keys
- When passing mutable objects to functions, changes inside the function affect the original object
- Immutable objects are generally thread-safe

## 3. How does memory management work in Python?

Python's memory management is handled automatically through a combination of reference counting, garbage collection, and memory allocation strategies:

**Reference Counting:**
- The primary mechanism for memory management
- Each object maintains a count of references pointing to it
- When the count reaches zero, the memory is immediately reclaimed
- Example: `x = [1, 2, 3]` creates a list with reference count 1
- If `y = x`, reference count becomes 2
- If `del x`, reference count decreases to 1
- If `del y`, reference count becomes 0 and the list is deallocated

**Garbage Collection:**
- Secondary mechanism to handle reference cycles
- Example cycle: `a = []; b = []; a.append(b); b.append(a)` - reference counts never reach zero
- Python periodically examines objects to find and clean up reference cycles
- Managed by the `gc` module
- Uses a generational algorithm (3 generations)
- Recently created objects (generation 0) are checked more frequently

**Memory Allocation:**
- Python uses a private heap space to manage objects
- Small objects (<512 bytes) use a special allocator called PyMalloc
- Maintains pools of fixed-size blocks for efficient allocation
- Uses "arenas" (chunks of 256KB) that are divided into pools and blocks
- Special optimizations for frequently used objects like small integers

**Memory Pooling:**
- Python maintains memory pools for efficient reuse
- Small integer objects (-5 to 256) are pre-allocated and reused
- Short strings may be interned (reused) depending on context
- Empty immutable containers (like `()` or `""`) are singleton objects

**Important Memory Management Features:**
- `sys.getrefcount(obj)` gives reference count (adds 1 temporarily for the function call)
- `gc.collect()` forces garbage collection
- `gc.disable()` / `gc.enable()` can control the automatic collection
- Memory profilers like `memory_profiler` or `tracemalloc` help identify leaks

## 4. Describe the difference between `list`, `tuple`, `set`, and `dict` in Python.

**List:**
- Ordered, mutable sequence
- Created with `[]` or `list()`
- Access elements by index: `my_list[0]`
- Supports operations like `append()`, `extend()`, `insert()`, `remove()`
- Time complexity:
  - Access by index: O(1)
  - Search (without index): O(n)
  - Insertion/deletion at end: O(1) amortized
  - Insertion/deletion elsewhere: O(n)
- Example: `fruits = ['apple', 'banana', 'cherry']`
- Use when: You need an ordered collection that might change, or need random access by position

**Tuple:**
- Ordered, immutable sequence
- Created with `()` or `tuple()`
- Access elements by index: `my_tuple[0]`
- Cannot be modified after creation
- Generally more memory efficient than lists
- Time complexity:
  - Access by index: O(1)
  - Search (without index): O(n)
- Example: `coordinates = (10.5, 20.8, 30.1)`
- Use when: You need an immutable ordered collection, especially for hashable composite keys

**Set:**
- Unordered collection of unique elements
- Created with `{}` or `set()`
- No indexing, access via membership test: `if x in my_set`
- Supports operations like `add()`, `remove()`, `union()`, `intersection()`
- Highly optimized for membership tests
- Time complexity:
  - Add/remove/membership test: O(1) average
- Example: `unique_visitors = {192, 195, 198, 203}`
- Use when: You need to ensure elements are unique or perform mathematical set operations

**Dict:**
- Unordered collection of key-value pairs (ordered since Python 3.7, but not guaranteed in language spec)
- Created with `{}` or `dict()`
- Access elements by key: `my_dict[key]` or `my_dict.get(key)`
- Keys must be immutable (hashable)
- Highly optimized for key lookups
- Time complexity:
  - Key access/insertion/deletion: O(1) average
- Example: `user = {'name': 'John', 'age': 30, 'email': 'john@example.com'}`
- Use when: You need to map keys to values for fast lookups

**Memory and Performance Comparisons:**
- Tuples use less memory than lists with the same elements
- Sets and dicts have overhead but provide O(1) lookups
- Lists have the fastest append operations
- Tuples are slightly faster to iterate through than lists

## 5. What is the Global Interpreter Lock (GIL) and how does it impact multithreaded Python programs?

The Global Interpreter Lock (GIL) is a mutex (lock) that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

**How the GIL Works:**
- Only one thread can execute Python bytecode at any given time
- The GIL is released during I/O operations, allowing other threads to run
- For CPU-bound tasks, the GIL switches between threads at regular intervals (by default every 5ms in modern Python)
- The GIL is specific to the CPython implementation (not PyPy, Jython, etc.)

**Impacts on Multithreaded Programs:**

*Negative Impacts:*
- CPU-bound programs cannot achieve true parallelism with threads
- May actually slow down CPU-intensive operations compared to single-threaded code due to thread switching overhead
- Makes multithreaded CPU-intensive code more complex with limited benefits

*Scenarios Where GIL is Not an Issue:*
- I/O-bound programs (network operations, file operations, etc.) - the GIL is released during I/O waits
- Programs using C extension modules that release the GIL during computation
- Single-threaded programs are obviously not affected

**Working Around the GIL:**
- Use `multiprocessing` module instead of `threading` for CPU-bound tasks
- Use external libraries that release the GIL (NumPy, Pandas operations often release the GIL)
- Use asynchronous programming with `asyncio` for I/O-bound concurrency
- Use alternative Python implementations like Jython or IronPython (which don't have a GIL)
- Leverage C extensions that release the GIL for computationally intensive parts

**Example of GIL Impact:**
```python
# CPU-bound task with threads - limited by GIL
import threading
import time

def cpu_bound(n):
    # CPU-intensive task
    count = 0
    for i in range(n):
        count += i
    return count

# Serial execution
start = time.time()
cpu_bound(10**8)
cpu_bound(10**8)
end = time.time()
print(f"Serial time: {end - start:.2f} seconds")

# Parallel execution with threads
start = time.time()
t1 = threading.Thread(target=cpu_bound, args=(10**8,))
t2 = threading.Thread(target=cpu_bound, args=(10**8,))
t1.start()
t2.start()
t1.join()
t2.join()
end = time.time()
print(f"Threaded time: {end - start:.2f} seconds")
# Threaded time is often similar or worse than serial time due to GIL
```

**GIL in Modern Python:**
- Python 3.2+ improved the GIL implementation to be more efficient
- Various proposals exist to remove the GIL but it's challenging due to backward compatibility
- Projects like Gilectomy have attempted to remove the GIL
- Python 3.9+ has some improvements in handling GIL for better multi-core performance

## 6. How would you implement a producer-consumer pattern in Python?

The producer-consumer pattern involves one or more producer threads that generate data and one or more consumer threads that process the data. In Python, there are several ways to implement this pattern:

### 1. Using Queue from the queue module

This is the most common and thread-safe approach:

```python
import threading
import queue
import time
import random

# Shared queue with a maximum size
buffer = queue.Queue(maxsize=10)

def producer(name):
    """Produces data and puts it into the shared queue"""
    count = 0
    while count < 20:  # produce 20 items
        # Simulate varying production time
        time.sleep(random.random())
        
        # Produce an item
        item = f"{name}-{count}"
        
        # Put it in the queue (will block if queue is full)
        buffer.put(item)
        print(f"{name} produced: {item}")
        
        count += 1
    
    # Signal that this producer is done
    buffer.put(None)
    print(f"{name} finished producing")

def consumer(name):
    """Consumes data from the shared queue"""
    while True:
        # Get an item from the queue (will block if queue is empty)
        item = buffer.get()
        
        # Check for termination signal
        if item is None:
            print(f"{name} received termination signal")
            # Put back the termination signal for other consumers
            buffer.put(None)
            break
            
        # Simulate varying consumption time
        time.sleep(random.random() * 2)
        
        # Process the item
        print(f"{name} consumed: {item}")
        
        # Mark task as done
        buffer.task_done()

# Create producer and consumer threads
producers = [
    threading.Thread(target=producer, args=(f'Producer-{i}',)) 
    for i in range(2)
]

consumers = [
    threading.Thread(target=consumer, args=(f'Consumer-{i}',))
    for i in range(3)
]

# Start the threads
for p in producers:
    p.start()

for c in consumers:
    c.start()

# Wait for all producers to finish
for p in producers:
    p.join()

# Wait for all consumers to finish
for c in consumers:
    c.join()

print("All done!")
```

### 2. Using Condition objects

For more complex synchronization:

```python
import threading
import time
import random
import collections

# Maximum buffer size
MAX_SIZE = 10

class ProducerConsumer:
    def __init__(self):
        self.buffer = collections.deque()
        self.condition = threading.Condition()
        self.done = False
    
    def produce(self, name):
        count = 0
        while count < 20:  # produce 20 items
            # Simulate production time
            time.sleep(random.random())
            
            # Produce an item
            item = f"{name}-{count}"
            
            with self.condition:
                # Wait if buffer is full
                while len(self.buffer) >= MAX_SIZE:
                    self.condition.wait()
                
                # Add to buffer
                self.buffer.append(item)
                print(f"{name} produced: {item}")
                
                # Notify consumers waiting for items
                self.condition.notify()
            
            count += 1
        
        with self.condition:
            # Signal that we're done producing
            self.done = True
            self.condition.notify_all()
    
    def consume(self, name):
        while True:
            with self.condition:
                # Wait if buffer is empty and producers aren't done
                while not self.buffer and not self.done:
                    self.condition.wait()
                
                # If buffer is empty and producers are done, exit
                if not self.buffer and self.done:
                    break
                
                # Get item from buffer
                item = self.buffer.popleft()
                print(f"{name} consumed: {item}")
                
                # Notify producers waiting for space
                self.condition.notify()
            
            # Simulate consumption time
            time.sleep(random.random() * 2)
```

### 3. Using asyncio for an asynchronous approach

```python
import asyncio
import random

async def producer(queue, name):
    for i in range(20):
        # Simulate production time
        await asyncio.sleep(random.random())
        
        # Produce an item
        item = f"{name}-{i}"
        
        # Put in queue
        await queue.put(item)
        print(f"{name} produced: {item}")
    
    # Signal completion
    await queue.put(None)

async def consumer(queue, name):
    while True:
        # Get item from queue
        item = await queue.get()
        
        # Check for termination signal
        if item is None:
            print(f"{name} received termination signal")
            queue.task_done()
            break
        
        # Simulate consumption time
        await asyncio.sleep(random.random() * 2)
        
        # Process the item
        print(f"{name} consumed: {item}")
        
        # Mark task as done
        queue.task_done()

async def main():
    # Create queue
    queue = asyncio.Queue(maxsize=10)
    
    # Create producer and consumer tasks
    producers = [asyncio.create_task(producer(queue, f'Producer-{i}'))
                 for i in range(2)]
    
    consumers = [asyncio.create_task(consumer(queue, f'Consumer-{i}'))
                 for i in range(3)]
    
    # Wait for producers to finish
    await asyncio.gather(*producers)
    
    # Wait for consumers to process all items
    await queue.join()
    
    # Cancel consumers
    for c in consumers:
        c.cancel()
```

### 4. Using multiprocessing for true parallelism

# Implementing a Producer-Consumer Pattern with Python Multiprocessing

The producer-consumer pattern with multiprocessing in Python is ideal for CPU-bound tasks that need true parallelism (bypassing the GIL). Here's how to implement it:

```python
import multiprocessing as mp
import time
import random

def producer(queue, producer_id, num_items):
    """Producer function that places items into a shared queue."""
    for i in range(num_items):
        # Create an item to produce
        item = f"Producer {producer_id} - Item {i}"
        
        # Simulate variable production time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Place item in the queue
        queue.put(item)
        print(f"[Producer {producer_id}] Produced: {item}")
    
    # Indicate this producer is done
    queue.put(None)  # Sentinel value
    print(f"[Producer {producer_id}] Done producing.")

def consumer(queue, consumer_id):
    """Consumer function that processes items from a shared queue."""
    while True:
        # Get an item from the queue, waiting if necessary
        item = queue.get()
        
        # Check for sentinel value (end of production)
        if item is None:
            # Put the sentinel back for other consumers and exit
            queue.put(None)
            break
            
        # Process the item (simulate work)
        print(f"[**Consumer** {consumer_id}] Processing: {item}")
        time.sleep(random.uniform(0.2, 0.8))  # Simulate processing time
        
        # print(f"[Consumer {consumer_id}] Finished: {item}")

def main():
    # Create a multiprocessing queue for thread-safe communication
    queue = mp.Queue(maxsize=10)  # Limit queue size to prevent memory issues
    
    # Number of producers and consumers
    num_producers = 3
    num_consumers = 2
    items_per_producer = 5
    
    # Create and start producer processes
    producers = []
    for i in range(num_producers):
        p = mp.Process(target=producer, args=(queue, i, items_per_producer))
        producers.append(p)
        p.start()
    
    # Create and start consumer processes
    consumers = []
    for i in range(num_consumers):
        c = mp.Process(target=consumer, args=(queue, i))
        consumers.append(c)
        c.start()
    
    # Wait for all producers to finish
    for p in producers:
        p.join()
    
    # Wait for all consumers to finish
    for c in consumers:
        c.join()
    
    print("All processes completed.")

if __name__ == "__main__":
    # This guard is essential for multiprocessing
    main()
```

## Key Features of This Implementation

1. **True Parallelism**: Uses `multiprocessing` instead of `threading` to bypass Python's GIL, allowing parallel execution on multiple CPU cores.

2. **Process-Safe Queue**: `multiprocessing.Queue` provides process-safe communication between producers and consumers.

3. **Bounded Queue Size**: The queue's `maxsize` parameter limits memory usage and creates back-pressure if producers outpace consumers.

4. **Graceful Termination**: Uses `None` as a sentinel value to signal when production is complete.

5. **Multiple Producers and Consumers**: Supports any number of producers and consumers working in parallel.

## Use Cases in Data Engineering

This pattern is particularly useful for:

- Processing large files in parallel chunks
- ETL workloads with CPU-intensive transformations
- Distributed data processing pipelines
- Image or video processing
- Parallel API data ingestion

Would you like me to expand on any part of this implementation or discuss variations for specific data processing scenarios?

Let me create a Mermaid diagram to visualize the producer-consumer pattern implementation with multiprocessing.

Here's a visualization of the producer-consumer pattern using multiprocessing:

```mermaid

flowchart LR
    subgraph main["Main Process"]
        A[Start Main] --> B[Create MP Queue]
        B --> C[Create Producer Processes]
        C --> D[Create Consumer Processes]
        D --> E[Wait for processes to complete]
        E --> F[End Main]
    end

    subgraph producers["Producer Processes"]
        P1[Producer 1] --> |"put(item1)"| Q[(MP Queue)]
        P2[Producer 2] --> |"put(item2)"| Q
        P3[Producer 3] --> |"put(item3)"| Q
        
        P1 --> |"put(None)"| Q
        P2 --> |"put(None)"| Q
        P3 --> |"put(None)"| Q
    end

    subgraph consumers["Consumer Processes"]
        Q --> |"get()"| C1[Consumer 1]
        Q --> |"get()"| C2[Consumer 2]
        
        C1 --> |"Process item"| C1_done[Done]
        C2 --> |"Process item"| C2_done[Done]
        
        C1 --> |"If None, put(None) back"| Q
        C2 --> |"If None, put(None) back"| Q
    end
    
    C --> producers
    D --> consumers
```

The diagram illustrates how:

1. The main process creates a multiprocessing Queue that serves as the communication channel
2. Multiple producer processes independently generate items and put them into the shared queue
3. Consumer processes take items from the queue and process them concurrently
4. Producers add sentinel values (None) when they finish to signal completion
5. When consumers encounter a sentinel value, they put it back in the queue for other consumers and terminate

This architecture achieves true parallelism because each producer and consumer runs in a separate Python process with its own memory space and interpreter, completely bypassing the GIL (Global Interpreter Lock).

The bounded queue creates natural back-pressure in the system - if producers generate data faster than consumers can process it, the queue will eventually fill up, causing producers to block until space becomes available. This helps prevent memory issues when processing large datasets.

Would you like me to explain any specific aspect of the diagram or the underlying implementation in more detail?



**Pros and Cons of Each Approach:**

1. **Queue Module**:
   - Pros: Simple, thread-safe, has blocking operations built-in
   - Cons: Limited to threading (GIL constraints)

2. **Condition Objects**:
   - Pros: More fine-grained control over synchronization
   - Cons: More complex, still limited by GIL

3. **Asyncio**:
   - Pros: Efficient for I/O-bound tasks, modern syntax with async/await
   - Cons: Not suitable for CPU-bound tasks due to single-threaded nature

4. **Multiprocessing**:
   - Pros: True parallelism, bypasses GIL limitations
   - Cons: Higher memory overhead, slower interprocess communication

## 7. Explain the time and space complexity of common operations on Python dictionaries, lists, and sets.

### Dictionary Operations:

| Operation               | Average Case | Worst Case | Notes                                 |
|-------------------------|--------------|------------|---------------------------------------|
| d[key]                  | O(1)         | O(n)       | Lookup by key                         |
| d[key] = value          | O(1)         | O(n)       | Insertion or update                   |
| key in d                | O(1)         | O(n)       | Membership test                       |
| del d[key]              | O(1)         | O(n)       | Deletion                              |
| len(d)                  | O(1)         | O(1)       | Size                                  |
| d.keys(), d.values()    | O(1)         | O(1)       | Views (iterators in Python 3)         |
| d.items()               | O(1)         | O(1)       | View of key-value pairs               |
| d.get(key, default)     | O(1)         | O(n)       | Lookup with default                   |

**Space Complexity**: O(n) where n is the number of items

**Notes**:
- Dictionaries are implemented as hash tables
- Worst case O(n) occurs with many hash collisions, but this is rare with Python's implementation
- Dictionaries maintain insertion order since Python 3.7
- Average case performance is what you'll experience in practice

### List Operations:

| Operation               | Average Case | Worst Case | Notes                                 |
|-------------------------|--------------|------------|---------------------------------------|
| l[i]                    | O(1)         | O(1)       | Access by index                       |
| l[i] = value            | O(1)         | O(1)       | Assignment by index                   |
| l.append(value)         | O(1)         | O(1)*      | Add to end                            |
| l.insert(i, value)      | O(n)         | O(n)       | Insert at position                    |
| l.pop()                 | O(1)         | O(1)       | Remove from end                       |
| l.pop(i)                | O(n)         | O(n)       | Remove at position                    |
| l.remove(value)         | O(n)         | O(n)       | Remove by value (first occurrence)    |
| value in l              | O(n)         | O(n)       | Membership test                       |
| len(l)                  | O(1)         | O(1)       | Size                                  |
| l + l2                  | O(n+m)       | O(n+m)     | Concatenation (n, m = lengths)        |
| l.sort()                | O(n log n)   | O(n log n) | In-place sort                         |
| sorted(l)               | O(n log n)   | O(n log n) | New sorted list                       |
| min(l), max(l)          | O(n)         | O(n)       | Find minimum/maximum                  |

**Space Complexity**: O(n) where n is the number of items

**Notes**:
- Lists are implemented as dynamic arrays
- *Amortized O(1) for append (occasional resizing)
- Operations at the beginning or middle of the list are more expensive than at the end
- Lists maintain order and allow duplicates

### Set Operations:

| Operation               | Average Case | Worst Case | Notes                                 |
|-------------------------|--------------|------------|---------------------------------------|
| s.add(value)            | O(1)         | O(n)       | Add element                           |
| s.remove(value)         | O(1)         | O(n)       | Remove element                        |
| value in s              | O(1)         | O(n)       | Membership test                       |
| len(s)                  | O(1)         | O(1)       | Size                                  |
| s.union(s2)             | O(n+m)       | O(n+m)     | Union of sets (n, m = sizes)          |
| s.intersection(s2)      | O(min(n,m))  | O(n*m)     | Intersection of sets                  |
| s.difference(s2)        | O(n)         | O(n)       | Difference of sets                    |
| s.issubset(s2)          | O(n)         | O(n)       | Test if subset                        |

**Space Complexity**: O(n) where n is the number of items

**Notes**:
- Sets are implemented using hash tables (like dictionaries but without values)
- Sets do not maintain order and do not allow duplicates
- Set operations like union and intersection are highly optimized
- Worst case rarely occurs in practice

### Comparison and Trade-offs:

1. **Lookup Operations**:
   - Dictionary/Set: O(1) average for key/element lookup
   - List: O(n) for value lookup without index

2. **Memory Usage**:
   - Dictionaries and sets have higher overhead per element
   - Lists have lower overhead, more memory-efficient for small items
   - Size increases: lists > dictionaries > sets (for same number of comparable elements)

3. **Ordering**:
   - Lists: Maintain insertion order and allow access by position
   - Dictionaries: Maintain insertion order (since Python 3.7)
   - Sets: Do not guarantee any ordering

4. **Use Cases**:
   - Lists: When order matters and you need frequent iteration
   - Dictionaries: When you need key-value mappings
   - Sets: When you need uniqueness and frequent membership testing

## 8. How would you optimize a Python function that processes large datasets?

When optimizing Python functions for large datasets, consider these techniques:

Optimizing Python functions for large datasets involves multiple approaches, from algorithm selection to hardware utilization:

### 1. Algorithmic Optimization

**Choose the Right Data Structures:**
```python
# Less efficient for membership testing
def check_membership_list(item, data):
    return item in data  # O(n) for lists

# More efficient for membership testing
def check_membership_set(item, data_set):
    return item in data_set  # O(1) average for sets
```

**Use Appropriate Algorithms:**
```python
# Less efficient sorting approach
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# More efficient built-in sort
def efficient_sort(arr):
    return sorted(arr)  # Uses Timsort, O(n log n)
```

### 2. Vectorization and Specialized Libraries

**Use NumPy for Numerical Operations:**
```python
# Slow Python loops
def sum_squares(data):
    result = 0
    for x in data:
        result += x * x
    return result

# Fast NumPy vectorization
import numpy as np
def sum_squares_numpy(data):
    data_array = np.array(data)
    return np.sum(data_array ** 2)
```

**Use Pandas for Data Manipulation:**
```python
import pandas as pd

# More efficient than manual iteration
def process_data_pandas(data):
    df = pd.DataFrame(data)
    # Group operations are faster than explicit loops
    result = df.groupby('category').agg({
        'value': ['mean', 'sum', 'count']
    })
    return result
```

### 3. Memory Management

**Process Data in Chunks:**
```python
def process_large_file(filename, chunk_size=100000):
    results = []
    # Read and process in manageable chunks
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process each chunk
        processed = process_chunk(chunk)
        results.append(processed)
    
    # Combine results
    return pd.concat(results)
```

**Use Generators for Lazy Evaluation:**
```python
# Memory-inefficient - loads entire dataset
def process_all_at_once(data):
    processed = [transform(x) for x in data]
    filtered = [x for x in processed if condition(x)]
    return sum(filtered)

# Memory-efficient - processes one item at a time
def process_with_generators(data):
    processed = (transform(x) for x in data)
    filtered = (x for x in processed if condition(x))
    return sum(filtered)
```

### 4. Parallel Processing

**Use Multiprocessing for CPU-bound Tasks:**
```python
from multiprocessing import Pool

def process_item(item):
    # CPU-intensive computation
    return complex_calculation(item)

def process_parallel(data, num_processes=None):
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_item, data)
    return results
```

**Use Threading for I/O-bound Tasks:**
```python
from concurrent.futures import ThreadPoolExecutor

def fetch_data(url):
    # I/O-bound operation (network request)
    return requests.get(url).json()

def fetch_all_data(urls, max_workers=20):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_data, url): url for url in urls}
        for future in as_completed(futures):
            results.append(future.result())
    return results
```

### 5. Code-Level Optimizations

**Use Built-in Functions and Methods:**
```python
# Slower manual implementation
def find_max_manual(data):
    max_val = data[0]
    for x in data[1:]:
        if x > max_val:
            max_val = x
    return max_val

# Faster built-in function
def find_max_builtin(data):
    return max(data)  # Implemented in C, much faster
```

**Local Variable Optimization:**
```python
# Slower with attribute lookups
def process_points_slow(points):
    result = 0
    for point in points:
        result += math.sqrt(point.x ** 2 + point.y ** 2)
    return result

# Faster with local function reference
def process_points_fast(points):
    sqrt = math.sqrt  # Local reference
    result = 0
    for point in points:
        result += sqrt(point.x ** 2 + point.y ** 2)
    return result
```

### 6. Specialized Techniques

**Use Numba for JIT Compilation:**
```python
from numba import jit

@jit(nopython=True)
def compute_intensive_task(data):
    result = 0
    for i in range(len(data)):
        for j in range(len(data)):
            result += math.sin(data[i]) * math.cos(data[j])
    return result
```

**Use Cython for Performance-Critical Sections:**
```python
# In .pyx file
def fast_computation(double[:] data):
    cdef double result = 0
    cdef int i, n = data.shape[0]
    for i in range(n):
        result += data[i] * data[i]
    return result
```

### 7. Profiling and Benchmarking

**Use Profiling to Identify Bottlenecks:**
```python
import cProfile

def optimize_function():
    # First profile to find bottlenecks
    cProfile.run('original_function(large_dataset)')
    
    # Then focus optimization efforts on the bottlenecks
```

**Benchmark Different Approaches:**
```python
import timeit

def benchmark_approaches():
    approaches = [
        ("Original", lambda: original_function(data)),
        ("Optimized v1", lambda: optimized_v1(data)),
        ("Optimized v2", lambda: optimized_v2(data))
    ]
    
    for name, func in approaches:
        time = timeit.timeit(func, number=100)
        print(f"{name}: {time:.4f} seconds")
```

### 8. Real-World Example: Combining Techniques

```python
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os

def optimize_data_processing(input_file, output_file):
    # 1. Process data in chunks to manage memory
    chunk_size = 500000
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    # 2. Use multiprocessing for parallel processing
    num_processes = os.cpu_count() - 1
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # 3. Combine results efficiently
    final_result = pd.concat(results)
    final_result.to_csv(output_file, index=False)
    
def process_chunk(chunk):
    # 4. Use vectorized operations with NumPy/Pandas
    # Convert categorical data to numerical
    chunk['category_code'] = chunk['category'].astype('category').cat.codes
    
    # 5. Efficient filtering
    filtered = chunk[chunk['value'] > chunk['value'].mean()]
    
    # 6. Use built-in methods for aggregations
    grouped = filtered.groupby('category').agg({
        'value': ['sum', 'mean'],
        'timestamp': 'count'
    })
    
    return grouped

## 9. Describe different methods for deep copying objects in Python.

Deep copying in Python creates a completely independent clone of an object, including all nested objects. Here are the different methods available:

### 1. Using the `copy` Module

The most straightforward and recommended approach:

```python
import copy

# Original nested object
original = [1, 2, [3, 4], {'a': 5}]

# Create a deep copy
deep_copied = copy.deepcopy(original)

# Modifying the deep copy won't affect the original
deep_copied[2][0] = 'X'
deep_copied[3]['a'] = 'Y'

print(original)      # [1, 2, [3, 4], {'a': 5}]
print(deep_copied)   # [1, 2, ['X', 4], {'a': 'Y'}]
```

The `copy.deepcopy()` function recursively clones each object, creating new objects at each level of nesting.

### 2. Using JSON Serialization

Useful for simple data structures with JSON-compatible types:

```python
import json

# Original data (must be JSON-serializable)
original = {'name': 'John', 'scores': [90, 85, 88], 'details': {'age': 25}}

# Deep copy using JSON serialization/deserialization
deep_copied = json.loads(json.dumps(original))

# Modify the copy
deep_copied['scores'][0] = 95
deep_copied['details']['age'] = 26

print(original)      # {'name': 'John', 'scores': [90, 85, 88], 'details': {'age': 25}}
print(deep_copied)   # {'name': 'John', 'scores': [95, 85, 88], 'details': {'age': 26}}
```

**Limitations:**
- Only works with JSON-serializable types (strings, numbers, lists, dictionaries, booleans, None)
- Doesn't preserve custom objects, functions, tuples (become lists), sets, etc.
- May lose precision on certain numeric types

### 3. Using Pickle Serialization

Provides broader support for Python objects:

```python
import pickle

# Original data (can include custom objects)
original = {'name': 'John', 'scores': [90, 85, 88], 'data': set([1, 2, 3])}

# Deep copy using pickle serialization/deserialization
deep_copied = pickle.loads(pickle.dumps(original))

# Modify the copy
deep_copied['scores'][0] = 95
deep_copied['data'].add(4)

print(original)      # {'name': 'John', 'scores': [90, 85, 88], 'data': {1, 2, 3}}
print(deep_copied)   # {'name': 'John', 'scores': [95, 85, 88], 'data': {1, 2, 3, 4}}
```

**Advantages over JSON:**
- Supports more Python types (tuples, sets, custom classes, etc.)
- Preserves object types more accurately

**Limitations:**
- Security concerns with untrusted data
- Less portable across Python versions

### 4. Manual Deep Copy Implementation

For specific needs or custom objects:

```python
def custom_deep_copy(obj):
    """A recursive deep copy function for common data structures."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        # Immutable types can be returned as-is
        return obj
    elif isinstance(obj, list):
        # Recursively copy lists
        return [custom_deep_copy(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively copy tuples
        return tuple(custom_deep_copy(item) for item in obj)
    elif isinstance(obj, dict):
        # Recursively copy dictionaries
        return {custom_deep_copy(key): custom_deep_copy(value) 
                for key, value in obj.items()}
    elif isinstance(obj, set):
        # Recursively copy sets
        return {custom_deep_copy(item) for item in obj}
    else:
        # For custom objects, you might need specific logic
        raise TypeError(f"Cannot deep copy object of type {type(obj)}")
```

**Use cases:**
- When you need very specific copying behavior
- When optimizing for performance in specific scenarios
- When dealing with custom objects with special requirements

### 5. Using the `__deepcopy__` Protocol

For custom classes, you can implement the `__deepcopy__` method:

```python
import copy

class CustomObject:
    def __init__(self, data, reference):
        self.data = data            # Will be deep copied
        self.reference = reference  # Will not be deep copied
        self.calculated = data * 2  # Will be recalculated
    
    def __deepcopy__(self, memo):
        """Control how this object is deep copied."""
        # Create a new instance without calling __init__
        new_obj = self.__class__.__new__(self.__class__)
        
        # Add the new object to the memo dictionary to handle recursive references
        memo[id(self)] = new_obj
        
        # Deep copy the data attribute
        new_obj.data = copy.deepcopy(self.data, memo)
        
        # Share the reference attribute (not deep copied)
        new_obj.reference = self.reference
        
        # Recalculate instead of copying
        new_obj.calculated = new_obj.data * 2
        
        return new_obj

# Usage
shared_ref = {"shared": "reference"}
original = CustomObject([1, 2, 3], shared_ref)
cloned = copy.deepcopy(original)

# Modifications
cloned.data[0] = 100

print(original.data)       # [1, 2, 3]
print(cloned.data)         # [100, 2, 3]

# Check that reference is shared
original.reference["shared"] = "modified"
print(cloned.reference)    # {"shared": "modified"}
```

### 6. Performance Considerations

Different deep copying methods have different performance characteristics:

| Method           | Speed           | Memory Usage    | Feature Support  | Use Case                            |
|------------------|-----------------|-----------------|------------------|-------------------------------------|
| `copy.deepcopy`  | Moderate        | Moderate        | Excellent        | General purpose, handles all types  |
| JSON             | Fast            | Low             | Limited          | Simple data structures, serialization|
| Pickle           | Moderate        | Moderate        | Good             | Complex data with standard types    |
| Custom           | Varies          | Varies          | Custom           | Specific optimization needs         |

### 7. Special Considerations

- **Cyclic References**: `copy.deepcopy()` handles circular references automatically by keeping track of already-copied objects.
- **Performance with Large Datasets**: For very large data structures, consider if a deep copy is actually necessary or if a shallow copy or partial deep copy would suffice.
- **Immutable Objects**: Technically don't need deep copying, but their containers might.
- **Custom Classes**: Always implement `__deepcopy__` for complex custom classes to control the copying behavior.

## 10. How would you implement a custom caching mechanism in Python?

Implementing a custom caching mechanism in Python involves creating a system that stores computed results to avoid redundant calculations. Here are several approaches:

### 1. Basic Function Caching with a Dictionary

```python
def create_cached_function(func):
    """
    Creates a cached version of a function using a dictionary.
    """
    cache = {}
    
    def cached_func(*args):
        # Convert args to a hashable type for dictionary key
        # (assuming args contains only hashable types)
        key = args
        
        if key in cache:
            print(f"Cache hit for {key}")
            return cache[key]
        else:
            print(f"Cache miss for {key}")
            result = func(*args)
            cache[key] = result
            return result
    
    return cached_func

# Example usage
def expensive_computation(x, y):
    print(f"Computing {x} + {y}...")
    # Simulate expensive operation
    import time
    time.sleep(1)  
    return x + y

# Create cached version
cached_computation = create_cached_function(expensive_computation)

# First call - cache miss
print(cached_computation(1, 2))  # Computing 1 + 2... Cache miss for (1, 2)
                                 # 3

# Second call with same args - cache hit
print(cached_computation(1, 2))  # Cache hit for (1, 2)
                                 # 3

# Different args - cache miss
print(cached_computation(2, 3))  # Computing 2 + 3... Cache miss for (2, 3)
                                 # 5
```

### 2. Using Python's `functools.lru_cache` Decorator

The standard library provides an efficient LRU (Least Recently Used) cache:

```python
import functools

@functools.lru_cache(maxsize=128)
def fibonacci(n):
    """Compute the nth Fibonacci number."""
    print(f"Computing fibonacci({n})...")
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First time calculations
print(fibonacci(10))  # Calculates all fibonacci values from 0 to 10

# Second call is cached
print(fibonacci(10))  # Direct cache hit

# Partially cached calculation
print(fibonacci(11))  # Only needs to calculate fibonacci(11), uses cached values for the rest
```

### 3. Implementing a Time-Based Cache (TTL Cache)

```python
import time
from datetime import datetime, timedelta

class TTLCache:
    """A time-to-live cache that expires entries after a specified duration."""
    
    def __init__(self, ttl_seconds=60):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        """Retrieve an item from cache if it exists and hasn't expired."""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        now = datetime.now()
        
        # Check if the entry has expired
        if now - timestamp > timedelta(seconds=self.ttl):
            # Remove expired entry
            del self.cache[key]
            return None
            
        return value
    
    def set(self, key, value):
        """Add an item to the cache with current timestamp."""
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    def clean_expired(self):
        """Remove all expired entries from the cache."""
        now = datetime.now()
        # Create a list of expired keys
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items() 
            if now - timestamp > timedelta(seconds=self.ttl)
        ]
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)  # Return number of removed entries

# Example usage
def get_weather_data(city):
    """Simulate an API call to get weather data."""
    print(f"Fetching weather data for {city}...")
    time.sleep(2)  # Simulate API request
    return f"{city}: 72°F, Sunny"

# Create TTL cache with 30-second expiration
weather_cache = TTLCache(ttl_seconds=30)

def get_cached_weather(city):
    """Get weather with caching."""
    # Try to get from cache first
    result = weather_cache.get(city)
    
    if result is None:
        # Cache miss or expired, fetch new data
        result = get_weather_data(city)
        weather_cache.set(city, result)
    else:
        print(f"Using cached data for {city}")
    
    return result

# First call - cache miss
print(get_cached_weather("New York"))  # Fetching weather data...

# Second call within TTL - cache hit
print(get_cached_weather("New York"))  # Using cached data...

# Wait for cache to expire
print("Waiting 31 seconds for cache to expire...")
time.sleep(31)

# After expiration - cache miss
print(get_cached_weather("New York"))  # Fetching weather data...
```

### 4. Implementing an LRU Cache from Scratch

```python
class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.
    Evicts least recently used items when capacity is reached.
    """
    
    class _Node:
        """Doubly linked list node to track usage order."""
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity=100):
        self.capacity = max(1, capacity)
        self.cache = {}  # Maps keys to nodes
        # Dummy head and tail nodes for the doubly linked list
        self.head = self._Node(None, None)  # Most recently used
        self.tail = self._Node(None, None)  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head (most recently used position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_front(self, node):
        """Move an existing node to the front (most recently used)."""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove and return the tail node (least recently used)."""
        node = self.tail.prev
        self._remove_node(node)
        return node
    
    def get(self, key):
        """Get value for key and mark as recently used."""
        if key not in self.cache:
            return None
        
        # Move to front of the list (mark as most recently used)
        node = self.cache[key]
        self._move_to_front(node)
        return node.value
    
    def put(self, key, value):
        """Add or update a key-value pair, maintaining LRU property."""
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            # Create new node
            node = self._Node(key, value)
            self.cache[key] = node
            self._add_node(node)
            
            # Evict least recently used item if over capacity
            if len(self.cache) > self.capacity:
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def __len__(self):
        """Return number of items in cache."""
        return len(self.cache)


# Example usage
cache = LRUCache(capacity=2)

cache.put("A", "Value A")
cache.put("B", "Value B")

print(cache.get("A"))  # Value A (now most recently used)

# This will evict B since A was recently accessed
cache.put("C", "Value C")

print(cache.get("B"))  # None (was evicted)
print(cache.get("C"))  # Value C
```

### 5. Disk-Based Caching for Larger Datasets

```python
import os
import pickle
import hashlib
import time

class DiskCache:
    """A cache that stores items on disk for persistence."""
    
    def __init__(self, cache_dir="./cache", ttl_seconds=3600):
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, key):
        """Convert a key to a file path."""
        # Hash the key to create a filename
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def get(self, key):
        """Retrieve an item from the disk cache."""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
            
        # Check file modification time for TTL
        mod_time = os.path.getmtime(cache_path)
        if time.time() - mod_time > self.ttl:
            # Cache expired, remove file
            os.remove(cache_path)
            return None
            
        # Load cached data
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            # Handle corrupted cache file
            os.remove(cache_path)
            return None
    
    def set(self, key, value):
        """Store an item in the disk cache."""
        cache_path = self._get_cache_path(key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)
    
    def delete(self, key):
        """Delete an item from the cache."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
    
    def clear(self):
        """Clear all items from the cache."""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.cache'):
                os.remove(file_path)

# Example usage with a function decorator
def disk_cached(cache_instance):
    """Decorator to cache function results to disk."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a key from function name and arguments
            key = str((func.__name__, args, sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache_instance.get(key)
            if result is not None:
                print("Cache hit!")
                return result
                
            # Cache miss, compute result
            print("Cache miss!")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_instance.set(key, result)
            return result
        return wrapper
    return decorator

# Usage example
cache = DiskCache(ttl_seconds=60)

@disk_cached(cache)
def expensive_computation(a, b, c):
    print("Computing...")
    time.sleep(2)  # Simulate expensive computation
    return a + b + c

# First call - cache miss
print(expensive_computation(1, 2, 3))  # Computing... Cache miss!
                                       # 6

# Second call with same args - cache hit
print(expensive_computation(1, 2, 3))  # Cache hit!
                                       # 6
```

### 6. Implementing a Cache with Memoization Pattern in a Class

```python
class MemoizedCalculator:
    """A calculator that caches results of expensive calculations."""
    
    def __init__(self):
        self._cache = {}
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache = {}
    
    def factorial(self, n):
        """Calculate factorial with memoization."""
        # Check if result is in cache
        if n in self._cache:
            return self._cache[n]
        
        # Base case
        if n <= 1:
            return 1
        
        # Calculate result and store in cache
        result = n * self.factorial(n - 1)
        self._cache[n] = result
        return result
    
    def fibonacci(self, n):
        """Calculate Fibonacci number with memoization."""
        # Create method-specific cache if needed
        if 'fibonacci' not in self._cache:
            self._cache['fibonacci'] = {}
        
        # Method-specific cache
        fib_cache = self._cache['fibonacci']
        
        # Check if result is in cache
        if n in fib_cache:
            return fib_cache[n]
        
        # Base cases
        if n <= 0:
            return 0
        if n == 1:
            return 1
        
        # Calculate result and store in cache
        result = self.fibonacci(n - 1) + self.fibonacci(n - 2)
        fib_cache[n] = result
        return result

# Usage
calc = MemoizedCalculator()
print(calc.factorial(20))  # Calculates and caches all factorials from 1 to 20
print(calc.factorial(15))  # Instant result from cache
print(calc.fibonacci(30))  # Calculates efficiently with memoization
```

### 7. Building a Redis-Based Distributed Cache

For distributed applications, you might need a shared cache across multiple instances:

```python
import redis
import pickle
import json
import hashlib

class RedisCache:
    """A distributed cache using Redis."""
    
    def __init__(self, host='localhost', port=6379, db=0, prefix='cache:', ttl=3600):
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
        self.ttl = ttl
    
    def _get_key(self, key):
        """Generate a prefixed Redis key."""
        # Hash complex keys
        if not isinstance(key, str):
            key = hashlib.md5(str(key).encode()).hexdigest()
        return f"{self.prefix}{key}"
    
    def get(self, key):
        """Get value from Redis cache."""
        redis_key = self._get_key(key)
        data = self.redis.get(redis_key)
        if data is None:
            return None
        
        try:
            return pickle.loads(data)
        except:
            # Fallback if pickle fails
            try:
                return json.loads(data)
            except:
                # Return raw data if all else fails
                return data
    
    def set(self, key, value, ttl=None):
        """Set value in Redis cache with optional custom TTL."""
        redis_key = self._get_key(key)
        ttl = ttl if ttl is not None else self.ttl
        
        try:
            data = pickle.dumps(value)
        except:
            # Fallback to JSON for serializable objects
            try:
                data = json.dumps(value)
            except:
                raise ValueError("Value not serializable")
        
        self.redis.set(redis_key, data, ex=ttl)
    
    def delete(self, key):
        """Delete key from cache."""
        redis_key = self._get_key(key)
        self.redis.delete(redis_key)
    
    def clear(self):
        """Clear all keys with this cache's prefix."""
        for key in self.redis.scan_iter(f"{self.prefix}*"):
            self.redis.delete(key)
    
    def exists(self, key):
        """Check if key exists in cache."""
        redis_key = self._get_key(key)
        return self.redis.exists(redis_key) > 0

# Usage in a distributed application
# redis_cache = RedisCache(host='redis-server.example.com')
# 
# def get_user_data(user_id):
#     # Try cache first
#     user_data = redis_cache.get(f"user:{user_id}")
#     if user_data is not None:
#         return user_data
#     
#     # Cache miss, query database
#     user_data = database.query(f"SELECT * FROM users WHERE id = {user_id}")
#     
#     # Cache the result
#     redis_cache.set(f"user:{user_id}", user_data)
#     return user_data
```

### 8. Caching Best Practices and Considerations

1. **Choose the right caching strategy:**
   - Memory-based for speed but limited capacity
   - Disk-based for larger datasets but slower access
   - Distributed for multi-instance applications

2. **Cache invalidation strategies:**
   - Time-based expiration (TTL)
   - Explicit invalidation on update
   - Version/ETag-based invalidation

3. **Memory management:**
   - Set appropriate size limits
   - Use eviction policies (LRU, LFU, FIFO)
   - Monitor memory usage

4. **Thread safety:**
   - Use thread-safe data structures or locks
   - Consider concurrent access patterns

5. **Cache warming:**
   - Pre-populate cache with frequently accessed items
   - Background refresh of soon-to-expire items

6. **Monitoring and metrics:**
   - Track hit/miss ratios
   - Measure cache response times
   - Log cache operations for debugging

7. **Error handling:**
   - Graceful fallback when cache fails
   - Circuit breaker pattern for distributed caches# Answers to Python Interview Questions 1-10

## 1. What are the key differences between Python 2 and Python 3?

Python 3 introduced several important changes that were not backward compatible with Python 2:

**Print Statement vs Function:**
- Python 2: `print "Hello"` (statement)
- Python 3: `print("Hello")` (function)

**Integer Division:**
- Python 2: `5 / 2` returns `2` (integer division)
- Python 3: `5 / 2` returns `2.5` (true division)
- Python 3: `5 // 2` returns `2` (floor division)

**Unicode Handling:**
- Python 2: Strings are ASCII by default, Unicode strings with `u""`
- Python 3: All strings are Unicode by default, bytes with `b""`

**Error Handling:**
- Python 2: `except Exception, e`
- Python 3: `except Exception as e`

**xrange vs range:**
- Python 2: `range()` returns a list, `xrange()` returns an iterator
- Python 3: `range()` returns an iterator for memory efficiency

**Other Important Differences:**
- Improved function annotations and type hinting in Python 3
- Dictionary views instead of lists for methods like `.keys()`
- Removal of `<>` in favor of `!=` for inequality
- Ordering comparisons between different types (like `1 < "2"`) raise TypeError in Python 3
- Changed handling of metaclasses
- Improved asynchronous capabilities with `async`/`await` (Python 3.5+)

## 2. Explain mutable vs immutable data types in Python with examples.

**Immutable Data Types:**
These cannot be changed after creation. When you "modify" them, you're actually creating a new object.

Examples:
- `int`: `x = 5` - integers cannot be modified in-place
- `float`: `y = 3.14` - floats cannot be modified in-place
- `str`: `s = "hello"` - strings create new objects when modified
- `tuple`: `t = (1, 2, 3)` - tuples cannot be modified after creation
- `frozenset`: `fs = frozenset([1, 2, 3])` - immutable version of set
- `bool`: `b = True` - booleans are immutable

Demonstration with strings:
```python
s = "hello"
id_before = id(s)
s += " world"
id_after = id(s)
print(id_before == id_after)  # False - a new string object was created
```

**Mutable Data Types:**
These can be modified in-place after creation.

Examples:
- `list`: `l = [1, 2, 3]` - you can modify elements or add/remove them
- `dict`: `d = {"a": 1, "b": 2}` - keys and values can be added/changed/removed
- `set`: `s = {1, 2, 3}` - elements can be added or removed
- Custom class instances (unless designed otherwise)

Demonstration with lists:
```python
l = [1, 2, 3]
id_before = id(l)
l.append(4)
id_after = id(l)
print(id_before == id_after)  # True - same object was modified
```

**Important Implications:**
- Immutable objects are hashable (can be dictionary keys or set members)
- Mutable objects cannot be used as dictionary keys
- When passing mutable objects to functions, changes inside the function affect the original object
- Immutable objects are generally thread-safe

## 3. How does memory management work in Python?

Python's memory management is handled automatically through a combination of reference counting, garbage collection, and memory allocation strategies:

**Reference Counting:**
- The primary mechanism for memory management
- Each object maintains a count of references pointing to it
- When the count reaches zero, the memory is immediately reclaimed
- Example: `x = [1, 2, 3]` creates a list with reference count 1
- If `y = x`, reference count becomes 2
- If `del x`, reference count decreases to 1
- If `del y`, reference count becomes 0 and the list is deallocated

**Garbage Collection:**
- Secondary mechanism to handle reference cycles
- Example cycle: `a = []; b = []; a.append(b); b.append(a)` - reference counts never reach zero
- Python periodically examines objects to find and clean up reference cycles
- Managed by the `gc` module
- Uses a generational algorithm (3 generations)
- Recently created objects (generation 0) are checked more frequently

**Memory Allocation:**
- Python uses a private heap space to manage objects
- Small objects (<512 bytes) use a special allocator called PyMalloc
- Maintains pools of fixed-size blocks for efficient allocation
- Uses "arenas" (chunks of 256KB) that are divided into pools and blocks
- Special optimizations for frequently used objects like small integers

**Memory Pooling:**
- Python maintains memory pools for efficient reuse
- Small integer objects (-5 to 256) are pre-allocated and reused
- Short strings may be interned (reused) depending on context
- Empty immutable containers (like `()` or `""`) are singleton objects

**Important Memory Management Features:**
- `sys.getrefcount(obj)` gives reference count (adds 1 temporarily for the function call)
- `gc.collect()` forces garbage collection
- `gc.disable()` / `gc.enable()` can control the automatic collection
- Memory profilers like `memory_profiler` or `tracemalloc` help identify leaks

## 4. Describe the difference between `list`, `tuple`, `set`, and `dict` in Python.

**List:**
- Ordered, mutable sequence
- Created with `[]` or `list()`
- Access elements by index: `my_list[0]`
- Supports operations like `append()`, `extend()`, `insert()`, `remove()`
- Time complexity:
  - Access by index: O(1)
  - Search (without index): O(n)
  - Insertion/deletion at end: O(1) amortized
  - Insertion/deletion elsewhere: O(n)
- Example: `fruits = ['apple', 'banana', 'cherry']`
- Use when: You need an ordered collection that might change, or need random access by position

**Tuple:**
- Ordered, immutable sequence
- Created with `()` or `tuple()`
- Access elements by index: `my_tuple[0]`
- Cannot be modified after creation
- Generally more memory efficient than lists
- Time complexity:
  - Access by index: O(1)
  - Search (without index): O(n)
- Example: `coordinates = (10.5, 20.8, 30.1)`
- Use when: You need an immutable ordered collection, especially for hashable composite keys

**Set:**
- Unordered collection of unique elements
- Created with `{}` or `set()`
- No indexing, access via membership test: `if x in my_set`
- Supports operations like `add()`, `remove()`, `union()`, `intersection()`
- Highly optimized for membership tests
- Time complexity:
  - Add/remove/membership test: O(1) average
- Example: `unique_visitors = {192, 195, 198, 203}`
- Use when: You need to ensure elements are unique or perform mathematical set operations

**Dict:**
- Unordered collection of key-value pairs (ordered since Python 3.7, but not guaranteed in language spec)
- Created with `{}` or `dict()`
- Access elements by key: `my_dict[key]` or `my_dict.get(key)`
- Keys must be immutable (hashable)
- Highly optimized for key lookups
- Time complexity:
  - Key access/insertion/deletion: O(1) average
- Example: `user = {'name': 'John', 'age': 30, 'email': 'john@example.com'}`
- Use when: You need to map keys to values for fast lookups

**Memory and Performance Comparisons:**
- Tuples use less memory than lists with the same elements
- Sets and dicts have overhead but provide O(1) lookups
- Lists have the fastest append operations
- Tuples are slightly faster to iterate through than lists

## 5. What is the Global Interpreter Lock (GIL) and how does it impact multithreaded Python programs?

The Global Interpreter Lock (GIL) is a mutex (lock) that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

**How the GIL Works:**
- Only one thread can execute Python bytecode at any given time
- The GIL is released during I/O operations, allowing other threads to run
- For CPU-bound tasks, the GIL switches between threads at regular intervals (by default every 5ms in modern Python)
- The GIL is specific to the CPython implementation (not PyPy, Jython, etc.)

**Impacts on Multithreaded Programs:**

*Negative Impacts:*
- CPU-bound programs cannot achieve true parallelism with threads
- May actually slow down CPU-intensive operations compared to single-threaded code due to thread switching overhead
- Makes multithreaded CPU-intensive code more complex with limited benefits

*Scenarios Where GIL is Not an Issue:*
- I/O-bound programs (network operations, file operations, etc.) - the GIL is released during I/O waits
- Programs using C extension modules that release the GIL during computation
- Single-threaded programs are obviously not affected

**Working Around the GIL:**
- Use `multiprocessing` module instead of `threading` for CPU-bound tasks
- Use external libraries that release the GIL (NumPy, Pandas operations often release the GIL)
- Use asynchronous programming with `asyncio` for I/O-bound concurrency
- Use alternative Python implementations like Jython or IronPython (which don't have a GIL)
- Leverage C extensions that release the GIL for computationally intensive parts

**Example of GIL Impact:**
```python
# CPU-bound task with threads - limited by GIL
import threading
import time

def cpu_bound(n):
    # CPU-intensive task
    count = 0
    for i in range(n):
        count += i
    return count

# Serial execution
start = time.time()
cpu_bound(10**8)
cpu_bound(10**8)
end = time.time()
print(f"Serial time: {end - start:.2f} seconds")

# Parallel execution with threads
start = time.time()
t1 = threading.Thread(target=cpu_bound, args=(10**8,))
t2 = threading.Thread(target=cpu_bound, args=(10**8,))
t1.start()
t2.start()
t1.join()
t2.join()
end = time.time()
print(f"Threaded time: {end - start:.2f} seconds")
# Threaded time is often similar or worse than serial time due to GIL
```

**GIL in Modern Python:**
- Python 3.2+ improved the GIL implementation to be more efficient
- Various proposals exist to remove the GIL but it's challenging due to backward compatibility
- Projects like Gilectomy have attempted to remove the GIL
- Python 3.9+ has some improvements in handling GIL for better multi-core performance

## 6. How would you implement a producer-consumer pattern in Python?

The producer-consumer pattern involves one or more producer threads that generate data and one or more consumer threads that process the data. In Python, there are several ways to implement this pattern:

### 1. Using Queue from the queue module

This is the most common and thread-safe approach:

```python
import threading
import queue
import time
import random

# Shared queue with a maximum size
buffer = queue.Queue(maxsize=10)

def producer(name):
    """Produces data and puts it into the shared queue"""
    count = 0
    while count < 20:  # produce 20 items
        # Simulate varying production time
        time.sleep(random.random())
        
        # Produce an item
        item = f"{name}-{count}"
        
        # Put it in the queue (will block if queue is full)
        buffer.put(item)
        print(f"{name} produced: {item}")
        
        count += 1
    
    # Signal that this producer is done
    buffer.put(None)
    print(f"{name} finished producing")

def consumer(name):
    """Consumes data from the shared queue"""
    while True:
        # Get an item from the queue (will block if queue is empty)
        item = buffer.get()
        
        # Check for termination signal
        if item is None:
            print(f"{name} received termination signal")
            # Put back the termination signal for other consumers
            buffer.put(None)
            break
            
        # Simulate varying consumption time
        time.sleep(random.random() * 2)
        
        # Process the item
        print(f"{name} consumed: {item}")
        
        # Mark task as done
        buffer.task_done()

# Create producer and consumer threads
producers = [
    threading.Thread(target=producer, args=(f'Producer-{i}',)) 
    for i in range(2)
]

consumers = [
    threading.Thread(target=consumer, args=(f'Consumer-{i}',))
    for i in range(3)
]

# Start the threads
for p in producers:
    p.start()

for c in consumers:
    c.start()

# Wait for all producers to finish
for p in producers:
    p.join()

# Wait for all consumers to finish
for c in consumers:
    c.join()

print("All done!")
```

### 2. Using Condition objects

For more complex synchronization:

```python
import threading
import time
import random
import collections

# Maximum buffer size
MAX_SIZE = 10

class ProducerConsumer:
    def __init__(self):
        self.buffer = collections.deque()
        self.condition = threading.Condition()
        self.done = False
    
    def produce(self, name):
        count = 0
        while count < 20:  # produce 20 items
            # Simulate production time
            time.sleep(random.random())
            
            # Produce an item
            item = f"{name}-{count}"
            
            with self.condition:
                # Wait if buffer is full
                while len(self.buffer) >= MAX_SIZE:
                    self.condition.wait()
                
                # Add to buffer
                self.buffer.append(item)
                print(f"{name} produced: {item}")
                
                # Notify consumers waiting for items
                self.condition.notify()
            
            count += 1
        
        with self.condition:
            # Signal that we're done producing
            self.done = True
            self.condition.notify_all()
    
    def consume(self, name):
        while True:
            with self.condition:
                # Wait if buffer is empty and producers aren't done
                while not self.buffer and not self.done:
                    self.condition.wait()
                
                # If buffer is empty and producers are done, exit
                if not self.buffer and self.done:
                    break
                
                # Get item from buffer
                item = self.buffer.popleft()
                print(f"{name} consumed: {item}")
                
                # Notify producers waiting for space
                self.condition.notify()
            
            # Simulate consumption time
            time.sleep(random.random() * 2)
```

### 3. Using asyncio for an asynchronous approach

```python
import asyncio
import random

async def producer(queue, name):
    for i in range(20):
        # Simulate production time
        await asyncio.sleep(random.random())
        
        # Produce an item
        item = f"{name}-{i}"
        
        # Put in queue
        await queue.put(item)
        print(f"{name} produced: {item}")
    
    # Signal completion
    await queue.put(None)

async def consumer(queue, name):
    while True:
        # Get item from queue
        item = await queue.get()
        
        # Check for termination signal
        if item is None:
            print(f"{name} received termination signal")
            queue.task_done()
            break
        
        # Simulate consumption time
        await asyncio.sleep(random.random() * 2)
        
        # Process the item
        print(f"{name} consumed: {item}")
        
        # Mark task as done
        queue.task_done()

async def main():
    # Create queue
    queue = asyncio.Queue(maxsize=10)
    
    # Create producer and consumer tasks
    producers = [asyncio.create_task(producer(queue, f'Producer-{i}'))
                 for i in range(2)]
    
    consumers = [asyncio.create_task(consumer(queue, f'Consumer-{i}'))
                 for i in range(3)]
    
    # Wait for producers to finish
    await asyncio.gather(*producers)
    
    # Wait for consumers to process all items
    await queue.join()
    
    # Cancel consumers
    for c in consumers:
        c.cancel()
```

### 4. Using multiprocessing for true parallelism

```python
import multiprocessing as mp
import time
import random

def producer(queue, name):
    for i in range(20):
        # Simulate production time
        time.sleep(random.random())
        
        # Produce item
        item = f"{name}-{i}"
        
        # Put in queue
        queue.put(item)
        print(f"{name} produced: {item}")
    
    # Signal completion
    queue.put(None)

def consumer(queue, name):
    while True:
        # Get item from queue
        item = queue.get()
        
        # Check for termination signal
        if item is None:
            print(f"{name} received termination signal")
            queue.put(None)  # Put back for other consumers
            break
        
        # Simulate consumption time
        time.sleep(random.random() * 2)
        
        # Process item
        print(f"{name} consumed: {item}")
```

**Pros and Cons of Each Approach:**

1. **Queue Module**:
   - Pros: Simple, thread-safe, has blocking operations built-in
   - Cons: Limited to threading (GIL constraints)

2. **Condition Objects**:
   - Pros: More fine-grained control over synchronization
   - Cons: More complex, still limited by GIL

3. **Asyncio**:
   - Pros: Efficient for I/O-bound tasks, modern syntax with async/await
   - Cons: Not suitable for CPU-bound tasks due to single-threaded nature

4. **Multiprocessing**:
   - Pros: True parallelism, bypasses GIL limitations
   - Cons: Higher memory overhead, slower interprocess communication

## 7. Explain the time and space complexity of common operations on Python dictionaries, lists, and sets.

### Dictionary Operations:

| Operation               | Average Case | Worst Case | Notes                                 |
|-------------------------|--------------|------------|---------------------------------------|
| d[key]                  | O(1)         | O(n)       | Lookup by key                         |
| d[key] = value          | O(1)         | O(n)       | Insertion or update                   |
| key in d                | O(1)         | O(n)       | Membership test                       |
| del d[key]              | O(1)         | O(n)       | Deletion                              |
| len(d)                  | O(1)         | O(1)       | Size                                  |
| d.keys(), d.values()    | O(1)         | O(1)       | Views (iterators in Python 3)         |
| d.items()               | O(1)         | O(1)       | View of key-value pairs               |
| d.get(key, default)     | O(1)         | O(n)       | Lookup with default                   |

**Space Complexity**: O(n) where n is the number of items

**Notes**:
- Dictionaries are implemented as hash tables
- Worst case O(n) occurs with many hash collisions, but this is rare with Python's implementation
- Dictionaries maintain insertion order since Python 3.7
- Average case performance is what you'll experience in practice

### List Operations:

| Operation               | Average Case | Worst Case | Notes                                 |
|-------------------------|--------------|------------|---------------------------------------|
| l[i]                    | O(1)         | O(1)       | Access by index                       |
| l[i] = value            | O(1)         | O(1)       | Assignment by index                   |
| l.append(value)         | O(1)         | O(1)*      | Add to end                            |
| l.insert(i, value)      | O(n)         | O(n)       | Insert at position                    |
| l.pop()                 | O(1)         | O(1)       | Remove from end                       |
| l.pop(i)                | O(n)         | O(n)       | Remove at position                    |
| l.remove(value)         | O(n)         | O(n)       | Remove by value (first occurrence)    |
| value in l              | O(n)         | O(n)       | Membership test                       |
| len(l)                  | O(1)         | O(1)       | Size                                  |
| l + l2                  | O(n+m)       | O(n+m)     | Concatenation (n, m = lengths)        |
| l.sort()                | O(n log n)   | O(n log n) | In-place sort                         |
| sorted(l)               | O(n log n)   | O(n log n) | New sorted list                       |
| min(l), max(l)          | O(n)         | O(n)       | Find minimum/maximum                  |

**Space Complexity**: O(n) where n is the number of items

**Notes**:
- Lists are implemented as dynamic arrays
- *Amortized O(1) for append (occasional resizing)
- Operations at the beginning or middle of the list are more expensive than at the end
- Lists maintain order and allow duplicates

### Set Operations:

| Operation               | Average Case | Worst Case | Notes                                 |
|-------------------------|--------------|------------|---------------------------------------|
| s.add(value)            | O(1)         | O(n)       | Add element                           |
| s.remove(value)         | O(1)         | O(n)       | Remove element                        |
| value in s              | O(1)         | O(n)       | Membership test                       |
| len(s)                  | O(1)         | O(1)       | Size                                  |
| s.union(s2)             | O(n+m)       | O(n+m)     | Union of sets (n, m = sizes)          |
| s.intersection(s2)      | O(min(n,m))  | O(n*m)     | Intersection of sets                  |
| s.difference(s2)        | O(n)         | O(n)       | Difference of sets                    |
| s.issubset(s2)          | O(n)         | O(n)       | Test if subset                        |

**Space Complexity**: O(n) where n is the number of items

**Notes**:
- Sets are implemented using hash tables (like dictionaries but without values)
- Sets do not maintain order and do not allow duplicates
- Set operations like union and intersection are highly optimized
- Worst case rarely occurs in practice

### Comparison and Trade-offs:

1. **Lookup Operations**:
   - Dictionary/Set: O(1) average for key/element lookup
   - List: O(n) for value lookup without index

2. **Memory Usage**:
   - Dictionaries and sets have higher overhead per element
   - Lists have lower overhead, more memory-efficient for small items
   - Size increases: lists > dictionaries > sets (for same number of comparable elements)

3. **Ordering**:
   - Lists: Maintain insertion order and allow access by position
   - Dictionaries: Maintain insertion order (since Python 3.7)
   - Sets: Do not guarantee any ordering

4. **Use Cases**:
   - Lists: When order matters and you need frequent iteration
   - Dictionaries: When you need key-value mappings
   - Sets: When you need uniqueness and frequent membership testing

## 8. How would you optimize a Python function that processes large datasets?

When optimizing Python functions for large datasets, consider these techniques:

Optimizing Python functions for large datasets involves multiple approaches, from algorithm selection to hardware utilization:

### 1. Algorithmic Optimization

**Choose the Right Data Structures:**
```python
# Less efficient for membership testing
def check_membership_list(item, data):
    return item in data  # O(n) for lists

# More efficient for membership testing
def check_membership_set(item, data_set):
    return item in data_set  # O(1) average for sets
```

**Use Appropriate Algorithms:**
```python
# Less efficient sorting approach
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# More efficient built-in sort
def efficient_sort(arr):
    return sorted(arr)  # Uses Timsort, O(n log n)
```

### 2. Vectorization and Specialized Libraries

**Use NumPy for Numerical Operations:**
```python
# Slow Python loops
def sum_squares(data):
    result = 0
    for x in data:
        result += x * x
    return result

# Fast NumPy vectorization
import numpy as np
def sum_squares_numpy(data):
    data_array = np.array(data)
    return np.sum(data_array ** 2)
```

**Use Pandas for Data Manipulation:**
```python
import pandas as pd

# More efficient than manual iteration
def process_data_pandas(data):
    df = pd.DataFrame(data)
    # Group operations are faster than explicit loops
    result = df.groupby('category').agg({
        'value': ['mean', 'sum', 'count']
    })
    return result
```

### 3. Memory Management

**Process Data in Chunks:**
```python
def process_large_file(filename, chunk_size=100000):
    results = []
    # Read and process in manageable chunks
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process each chunk
        processed = process_chunk(chunk)
        results.append(processed)
    
    # Combine results
    return pd.concat(results)
```

**Use Generators for Lazy Evaluation:**
```python
# Memory-inefficient - loads entire dataset
def process_all_at_once(data):
    processed = [transform(x) for x in data]
    filtered = [x for x in processed if condition(x)]
    return sum(filtered)

# Memory-efficient - processes one item at a time
def process_with_generators(data):
    processed = (transform(x) for x in data)
    filtered = (x for x in processed if condition(x))
    return sum(filtered)
```

### 4. Parallel Processing

**Use Multiprocessing for CPU-bound Tasks:**
```python
from multiprocessing import Pool

def process_item(item):
    # CPU-intensive computation
    return complex_calculation(item)

def process_parallel(data, num_processes=None):
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_item, data)
    return results
```

**Use Threading for I/O-bound Tasks:**
```python
from concurrent.futures import ThreadPoolExecutor

def fetch_data(url):
    # I/O-bound operation (network request)
    return requests.get(url).json()

def fetch_all_data(urls, max_workers=20):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_data, url): url for url in urls}
        for future in as_completed(futures):
            results.append(future.result())
    return results
```

### 5. Code-Level Optimizations

**Use Built-in Functions and Methods:**
```python
# Slower manual implementation
def find_max_manual(data):
    max_val = data[0]
    for x in data[1:]:
        if x > max_val:
            max_val = x
    return max_val

# Faster built-in function
def find_max_builtin(data):
    return max(data)  # Implemented in C, much faster
```

**Local Variable Optimization:**
```python
# Slower with attribute lookups
def process_points_slow(points):
    result = 0
    for point in points:
        result += math.sqrt(point.x ** 2 + point.y ** 2)
    return result

# Faster with local function reference
def process_points_fast(points):
    sqrt = math.sqrt  # Local reference
    result = 0
    for point in points:
        result += sqrt(point.x ** 2 + point.y ** 2)
    return result
```

### 6. Specialized Techniques

**Use Numba for JIT Compilation:**
```python
from numba import jit

@jit(nopython=True)
def compute_intensive_task(data):
    result = 0
    for i in range(len(data)):
        for j in range(len(data)):
            result += math.sin(data[i]) * math.cos(data[j])
    return result
```

**Use Cython for Performance-Critical Sections:**
```python
# In .pyx file
def fast_computation(double[:] data):
    cdef double result = 0
    cdef int i, n = data.shape[0]
    for i in range(n):
        result += data[i] * data[i]
    return result
```

### 7. Profiling and Benchmarking

**Use Profiling to Identify Bottlenecks:**
```python
import cProfile

def optimize_function():
    # First profile to find bottlenecks
    cProfile.run('original_function(large_dataset)')
    
    # Then focus optimization efforts on the bottlenecks
```

**Benchmark Different Approaches:**
```python
import timeit

def benchmark_approaches():
    approaches = [
        ("Original", lambda: original_function(data)),
        ("Optimized v1", lambda: optimized_v1(data)),
        ("Optimized v2", lambda: optimized_v2(data))
    ]
    
    for name, func in approaches:
        time = timeit.timeit(func, number=100)
        print(f"{name}: {time:.4f} seconds")
```

### 8. Real-World Example: Combining Techniques

```python
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os

def optimize_data_processing(input_file, output_file):
    # 1. Process data in chunks to manage memory
    chunk_size = 500000
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    # 2. Use multiprocessing for parallel processing
    num_processes = os.cpu_count() - 1
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # 3. Combine results efficiently
    final_result = pd.concat(results)
    final_result.to_csv(output_file, index=False)
    
def process_chunk(chunk):
    # 4. Use vectorized operations with NumPy/Pandas
    # Convert categorical data to numerical
    chunk['category_code'] = chunk['category'].astype('category').cat.codes
    
    # 5. Efficient filtering
    filtered = chunk[chunk['value'] > chunk['value'].mean()]
    
    # 6. Use built-in methods for aggregations
    grouped = filtered.groupby('category').agg({
        'value': ['sum', 'mean'],
        'timestamp': 'count'
    })
    
    return grouped



### What are Lambda Functions?

Lambda functions in Python are small, anonymous functions defined using the `lambda` keyword. They can have any number of arguments but only one expression. The expression is evaluated and returned when the lambda function is called.

**Syntax**:
```python
lambda arguments: expression
```

**Basic examples**:
```python
# Simple addition lambda
add = lambda x, y: x + y
print(add(5, 3))  # 8

# Squaring lambda
square = lambda x: x**2
print(square(4))  # 16

# Lambda with conditional expression
is_even = lambda x: x % 2 == 0
print(is_even(6))  # True
print(is_even(7))  # False
```

### Key Characteristics of Lambda Functions

1. **Anonymous**: They don't need a name (though they can be assigned to variables)
2. **Single expression**: They can only contain one expression
3. **Inline**: They are typically used where a function is needed for a short period
4. **Functional**: They are commonly used in functional programming constructs

### Common Uses of Lambda Functions

#### 1. Sorting with custom keys

```python
# Sort a list of tuples by the second element
data = [(1, 5), (3, 2), (2, 8), (4, 1)]
sorted_data = sorted(data, key=lambda x: x[1])
print(sorted_data)  # [(4, 1), (3, 2), (1, 5), (2, 8)]

# Sort strings by length
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
sorted_words = sorted(words, key=lambda s: len(s))
print(sorted_words)  # ['date', 'apple', 'cherry', 'banana', 'elderberry']
```

#### 2. Filtering with `filter()`

```python
# Filter out odd numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4, 6, 8, 10]

# Filter strings by length
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
long_words = list(filter(lambda s: len(s) >= 6, words))
print(long_words)  # ['banana', 'elderberry']
```

#### 3. Transforming with `map()`

```python
# Square all numbers
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Convert temperatures from Celsius to Fahrenheit
celsius = [0, 10, 20, 30, 40]
fahrenheit = list(map(lambda c: (9/5) * c + 32, celsius))
print(fahrenheit)  # [32.0, 50.0, 68.0, 86.0, 104.0]
```

#### 4. In higher-order functions

```python
def apply_operation(x, y, operation):
    return operation(x, y)

# Use with different lambda functions
print(apply_operation(5, 3, lambda x, y: x + y))  # 8 (addition)
print(apply_operation(5, 3, lambda x, y: x * y))  # 15 (multiplication)
print(apply_operation(5, 3, lambda x, y: x ** y))  # 125 (exponentiation)
```

### Practical Use Cases in Data Engineering

#### 1. ETL Data Transformations

```python
# Sample data records (e.g., from a database query)
raw_data = [
    {'id': 1, 'name': 'John Doe', 'salary': 50000, 'department': 'IT'},
    {'id': 2, 'name': 'Jane Smith', 'salary': 60000, 'department': 'HR'},
    {'id': 3, 'name': 'Bob Johnson', 'salary': 55000, 'department': 'IT'},
    {'id': 4, 'name': 'Alice Brown', 'salary': 65000, 'department': 'Finance'}
]

# Transform data for a data warehouse loading
transformed_data = list(map(
    lambda record: {
        'employee_id': f"EMP{record['id']:04d}",  # Format ID with padding
        'full_name': record['name'].upper(),      # Standardize names
        'annual_salary': record['salary'],
        'monthly_salary': round(record['salary'] / 12, 2),
        'department_code': record['department'].lower(),
        'data_date': '2023-05-15'                 # Add ETL run date
    },
    raw_data
))

# Filter for only IT department employees
it_employees = list(filter(
    lambda record: record['department'] == 'IT',
    raw_data
))

# Calculate average salary by department
from itertools import groupby
from operator import itemgetter

# Sort by department for groupby
sorted_by_dept = sorted(raw_data, key=lambda r: r['department'])

# Group and calculate averages
dept_avg_salary = {
    dept: sum(map(lambda r: r['salary'], records)) / len(list(records))
    for dept, records in groupby(sorted_by_dept, key=lambda r: r['department'])
}
```

#### 2. Data Quality Checks

```python
def validate_dataset(data, validation_rules):
    """
    Validate a dataset against a set of rules.
    
    Args:
        data: List of dictionaries (records)
        validation_rules: Dict mapping field names to validation functions
    
    Returns:
        List of validation failures
    """
    failures = []
    
    for i, record in enumerate(data):
        for field, validator in validation_rules.items():
            if field in record:
                if not validator(record[field]):
                    failures.append({
                        'record_index': i,
                        'field': field,
                        'value': record[field],
                        'message': f"Validation failed for {field}"
                    })
    
    return failures

# Define validation rules using lambda functions
validation_rules = {
    'id': lambda x: isinstance(x, int) and x > 0,
    'name': lambda x: isinstance(x, str) and len(x) > 0,
    'salary': lambda x: isinstance(x, (int, float)) and 20000 <= x <= 200000,
    'department': lambda x: x in ('IT', 'HR', 'Finance', 'Marketing')
}

# Run validation
validation_results = validate_dataset(raw_data, validation_rules)
```

#### 3. Data Pipeline Processing Logic

```python
class DataPipeline:
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, transformation):
        """Add a transformation function to the pipeline."""
        self.transformations.append(transformation)
    
    def process(self, data):
        """Process data through all transformations in the pipeline."""
        result = data
        for transform in self.transformations:
            result = transform(result)
        return result

# Create a data pipeline
pipeline = DataPipeline()

# Add transformations using lambda functions
pipeline.add_transformation(
    lambda data: [record for record in data if record['salary'] > 0]  # Filter invalid salaries
)

pipeline.add_transformation(
    lambda data: [  # Add calculated fields
        {**record, 'tax': record['salary'] * 0.3}
        for record in data
    ]
)

pipeline.add_transformation(
    lambda data: sorted(data, key=lambda r: r['salary'], reverse=True)  # Sort by salary
)

# Process data through the pipeline
processed_data = pipeline.process(raw_data)
```

#### 4. Custom Aggregation Functions in Pandas

```python
import pandas as pd
import numpy as np

# Sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'product': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'D'],
    'region': ['East', 'West', 'East', 'West', 'East', 'West', 'East', 'West', 'East', 'West'],
    'sales': [100, 200, 150, 300, 250, 180, 320, 220, 170, 400],
    'quantity': [5, 10, 7, 15, 12, 9, 16, 11, 8, 20]
})

# Use lambda functions in groupby aggregations
result = df.groupby(['product', 'region']).agg({
    'sales': ['sum', 'mean', lambda x: x.max() - x.min()],  # Custom range calculation
    'quantity': ['sum', lambda x: np.percentile(x, 75)]     # 75th percentile
})

# Rename columns for clarity
result.columns = ['sales_sum', 'sales_mean', 'sales_range', 'quantity_sum', 'quantity_p75']

# Use lambda in apply for custom calculations
df['sales_per_unit'] = df.apply(lambda row: row['sales'] / row['quantity'], axis=1)

# Complex transformation with multiple conditions
df['performance'] = df.apply(
    lambda row: 'Excellent' if row['sales'] > 300 else 
                'Good' if row['sales'] > 200 else 
                'Average' if row['sales'] > 100 else 'Poor',
    axis=1
)
```

#### 5. Feature Engineering in ML Pipelines

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

# Sample DataFrame (e.g., customer data)
df = pd.DataFrame({
    'age': [25, 40, 33, 55, 28],
    'income': [50000, 80000, 65000, 120000, 45000],
    'education_years': [16, 18, 16, 20, 12],
    'purchases_last_month': [2, 5, 3, 8, 1],
    'churn': [0, 0, 0, 1, 1]
})

# Define feature engineering functions using lambda
age_group_transformer = FunctionTransformer(
    lambda X: pd.DataFrame({
        'age_group': pd.cut(X['age'], bins=[0, 30, 45, 60, 100], 
                            labels=['Young', 'MiddleAge', 'Senior', 'Elderly'])
    })
)

income_log_transformer = FunctionTransformer(
    lambda X: pd.DataFrame({
        'income_log': np.log1p(X['income'])
    })
)

purchase_category_transformer = FunctionTransformer(
    lambda X: pd.DataFrame({
        'purchase_category': pd.cut(X['purchases_last_month'], bins=[-1, 1, 3, 10],
                                    labels=['Low', 'Medium', 'High'])
    })
)

# Create feature engineering pipeline
feature_engineering = Pipeline([
    ('age_group', age_group_transformer),
    ('income_log', income_log_transformer),
    ('purchase_cat', purchase_category_transformer)
])

# Apply transformations (in a real pipeline, would connect to model training)
engineered_features = feature_engineering.fit_transform(df)
```

### Advantages and Limitations of Lambda Functions

**Advantages:**
1. **Conciseness**: Compact syntax for simple functions
2. **Readability**: Can make code more readable for simple operations
3. **Convenience**: Eliminates need for separate function definitions
4. **Functional programming**: Enables functional programming patterns

**Limitations:**
1. **Single expression only**: Cannot contain multiple statements
2. **Limited complexity**: Not suitable for complex logic
3. **No docstrings**: Cannot include documentation within the function
4. **Debugging challenges**: Harder to debug due to lack of name

### Lambda vs. Regular Functions - When to Use Each

**Use lambda functions when:**
- The function is simple and fits in one line
- The function is used only once or in a very limited scope
- You're using higher-order functions like `map()`, `filter()`, or `sorted()`
- You need to define a function inline within another expression

**Use regular functions when:**
- The function contains multiple statements or complex logic
- You need to reuse the function in multiple places
- Documentation (docstrings) is important
- The function needs to be easily debugged
- The function name adds clarity to the code

```python
# Good lambda use case
sorted_employees = sorted(employees, key=lambda e: e.salary)

# Better as a regular function
def calculate_complex_metric(employee):
    """Calculate a complex performance metric for an employee."""
    base_score = employee.productivity * employee.quality
    adjustment = employee.tenure / 5 if employee.tenure < 5 else 1
    return base_score * adjustment * (1 + 0.1 * employee.trainings)

evaluated_employees = [(e, calculate_complex_metric(e)) for e in employees]
```

## 15. How would you implement a higher-order function in Python?

A higher-order function is a function that either takes one or more functions as arguments or returns a function as its result (or both). Python's flexibility makes it easy to implement higher-order functions, which are a core concept in functional programming.

### Key Concepts of Higher-Order Functions

1. **Functions as arguments**: Passing functions to other functions
2. **Functions as return values**: Returning functions from other functions
3. **Function composition**: Creating new functions by combining existing ones
4. **Closures**: Functions that capture their enclosing environment

### 1. Functions as Arguments

The simplest form of higher-order functions takes other functions as parameters:

```python
def apply_function(func, value):
    """Apply a function to a value and return the result."""
    return func(value)

# Example usage
def square(x):
    return x ** 2

def double(x):
    return x * 2

print(apply_function(square, 5))  # 25
print(apply_function(double, 5))  # 10
print(apply_function(lambda x: x + 10, 5))  # 15
```

A more complex example with multiple arguments:

```python
def apply_operation(func, values, *args, **kwargs):
    """Apply a function to each value in a list with additional arguments."""
    return [func(value, *args, **kwargs) for value in values]

# Usage
def power(base, exponent):
    return base ** exponent

numbers = [1, 2, 3, 4, 5]
squares = apply_operation(power, numbers, 2)  # [1, 4, 9, 16, 25]
cubes = apply_operation(power, numbers, 3)    # [1, 8, 27, 64, 125]
```

In data engineering, this pattern is commonly used for processing data:

```python
def process_dataset(data, transformations):
    """Apply a series of transformations to a dataset."""
    result = data
    for transformation in transformations:
        result = transformation(result)
    return result

# Example transformations
def clean_missing_values(df):
    return df.dropna()

def normalize_columns(df):
    return (df - df.mean()) / df.std()

def add_derived_features(df):
    df['total'] = df['column1'] + df['column2']
    return df

# Process data
processed_data = process_dataset(
    original_data,
    [clean_missing_values, normalize_columns, add_derived_features]
)
```

### 2. Functions as Return Values

Higher-order functions can also generate and return new functions:

```python
def create_multiplier(factor):
    """Create a function that multiplies its argument by a given factor."""
    def multiplier(x):
        return x * factor
    return multiplier

# Create specific multiplier functions
double = create_multiplier(2)
triple = create_multiplier(3)

print(double(10))  # 20
print(triple(10))  # 30
```

This pattern enables creating specialized functions dynamically:

```python
def create_validator(validation_rule):
    """Create a validation function based on a rule."""
    if validation_rule == 'positive':
        return lambda x: x > 0
    elif validation_rule == 'even':
        return lambda x: x % 2 == 0
    elif validation_rule == 'in_range':
        return lambda x, min_val, max_val: min_val <= x <= max_val
    else:
        raise ValueError(f"Unknown validation rule: {validation_rule}")

# Create and use validators
is_positive = create_validator('positive')
is_even = create_validator('even')
is_in_range = create_validator('in_range')

print(is_positive(5))         # True
print(is_even(5))             # False
print(is_in_range(5, 1, 10))  # True
```

A data engineering example for creating custom data filters:

```python
def create_data_filter(column, operator, value):
    """Create a filter function for a DataFrame based on criteria."""
    if operator == 'equals':
        return lambda df: df[df[column] == value]
    elif operator == 'greater_than':
        return lambda df: df[df[column] > value]
    elif operator == 'less_than':
        return lambda df: df[df[column] < value]
    elif operator == 'contains':
        return lambda df: df[df[column].str.contains(value)]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

# Create filters
high_salary_filter = create_data_filter('salary', 'greater_than', 75000)
marketing_dept_filter = create_data_filter('department', 'equals', 'Marketing')
name_filter = create_data_filter('name', 'contains', 'Smith')

# Apply filters
high_salary_employees = high_salary_filter(employees_df)
marketing_employees = marketing_dept_filter(employees_df)
```

### 3. Function Composition

Function composition involves combining multiple functions to create a new function:

```python
def compose(f, g):
    """Compose two functions: compose(f, g)(x) = f(g(x))."""
    return lambda x: f(g(x))

# Example functions
def double(x):
    return x * 2

def square(x):
    return x ** 2

# Create composed functions
square_then_double = compose(double, square)
double_then_square = compose(square, double)

print(square_then_double(3))  # double(square(3)) = double(9) = 18
print(double_then_square(3))  # square(double(3)) = square(6) = 36
```

We can extend this to compose multiple functions:

```python
def compose_all(*functions):
    """
    Compose multiple functions: compose_all(f, g, h)(x) = f(g(h(x)))
    
    Functions are applied from right to left.
    """
    def compose_two(f, g):
        return lambda x: f(g(x))
    
    if not functions:
        return lambda x: x  # Identity function
    
    return functools.reduce(compose_two, functions)

# Example
def add_one(x):
    return x + 1

def double(x):
    return x * 2

def square(x):
    return x ** 2

# Compose multiple functions
transform = compose_all(add_one, double, square)
# This is equivalent to: add_one(double(square(x)))

print(transform(3))  # add_one(double(square(3))) = add_one(double(9)) = add_one(18) = 19
```

In data engineering, function composition can be used for building data transformation pipelines:

```python
def build_transformation_pipeline(*transforms):
    """Build a data transformation pipeline from component transforms."""
    def pipeline(data):
        result = data
        for transform in transforms:
            result = transform(result)
        return result
    
    return pipeline

# Example transformations
def clean_data(df):
    return df.dropna()

def normalize(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df

def add_features(df):
    df['feature_a'] = df['col1'] * df['col2']
    df['feature_b'] = df['col3'] / df['col4']
    return df

# Create pipeline
data_pipeline = build_transformation_pipeline(
    clean_data,
    normalize,
    add_features
)

# Process data
processed_data = data_pipeline(raw_data)
```

### 4. Implementing Decorators (a Special Case of Higher-Order Functions)

Decorators are a common use case for higher-order functions in Python:

```python
def timing_decorator(func):
    """Decorator that measures execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    
    return wrapper

# Apply decorator
@timing_decorator
def process_data(data):
    # Simulate processing
    time.sleep(1)
    return data * 2

result = process_data(10)  # Outputs timing information
```

### 5. Implementing Partial Application

Partial application is a technique where we create a new function by fixing some arguments of an existing function:

```python
import functools

def multiply(x, y):
    return x * y

# Create a new function with x fixed to 2
double = functools.partial(multiply, 2)

print(double(5))  # 10
```

Custom implementation:

```python
def partial(func, *fixed_args, **fixed_kwargs):
    """
    Partially apply a function by fixing some arguments.
    """
    def wrapper(*args, **kwargs):
        # Combine fixed and new arguments
        all_args = fixed_args + args
        
        # Combine fixed and new keyword arguments
        all_kwargs = fixed_kwargs.copy()
        all_kwargs.update(kwargs)
        
        # Call the original function with all arguments
        return func(*all_args, **all_kwargs)
    
    return wrapper

# Usage example
def format_string(template, name, value):
    return template.format(name=name, value=value)

# Create a partially applied function with template fixed
alert_format = partial(format_string, "ALERT: {name} exceeded threshold: {value}")

# Use the new function
print(alert_format("CPU Usage", "95%"))  # ALERT: CPU Usage exceeded threshold: 95%
```

### 6. Implementing Currying

Currying transforms a function that takes multiple arguments into a sequence of functions that each take a single argument:

```python
def curry(func, arity):
    """
    Curry a function of multiple arguments.
    
    Args:
        func: The function to curry
        arity: Number of arguments the function takes
    """
    def curried(*args):
        if len(args) >= arity:
            return func(*args)
        
        def inner(*more_args):
            return curried(*(args + more_args))
        
        return inner
    
    return curried

# Example usage
def add3(a, b, c):
    return a + b + c

# Curry the function
curried_add3 = curry(add3, 3)

# Different ways to call
print(curried_add3(1, 2, 3))    # 6
print(curried_add3(1)(2)(3))    # 6
print(curried_add3(1, 2)(3))    # 6
print(curried_add3(1)(2, 3))    # 6
```

### 7. Creating a Memoization Function

Memoization is a technique to cache function results:

```python
def memoize(func):
    """
    Create a memoized version of a function.
    """
    cache = {}
    
    def memoized(*args):
        # Convert args to a hashable key
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    # Add a method to clear the cache
    memoized.clear_cache = lambda: cache.clear()
    
    return memoized

# Example usage
@memoize
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First call calculates
print(fibonacci(30))  # 832040 (calculated efficiently due to memoization)

# Second call is retrieved from cache
print(fibonacci(30))  # 832040 (immediately from cache)
```

### 8. Creating a Function Pipeline Class

A more structured approach to function composition using classes:

```python
class FunctionPipeline:
    """
    A class representing a pipeline of functions.
    """
    def __init__(self, functions=None):
        self.functions = functions or []
    
    def add(self, function):
        """Add a function to the pipeline."""
        self.functions.append(function)
        return self  # For method chaining
    
    def __call__(self, x):
        """Execute the pipeline on an input."""
        result = x
        for func in self.functions:
            result = func(result)
        return result

# Example usage
pipeline = (
    FunctionPipeline()
    .add(lambda x: x + 1)   # Add 1
    .add(lambda x: x * 2)   # Then multiply by 2
    .add(lambda x: x ** 2)  # Then square the result
)

print(pipeline(3))  # ((3 + 1) * 2) ^ 2 = 8 ^ 2 = 64
```

### Real-World Data Engineering Examples

#### 1. Custom Data Processing Framework

```python
class DataProcessor:
    """Framework for processing data through transformation steps."""
    
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, name, transform_func):
        """Add a named transformation to the pipeline."""
        self.transformations.append((name, transform_func))
        return self  # For method chaining
    
    def create_pipeline(self, subset=None):
        """
        Create a processing pipeline using all or a subset of transformations.
        
        Args:
            subset: Optional list of transformation names to include
        """
        # Filter transformations if subset is specified
        if subset:
            selected = [(name, func) for name, func in self.transformations 
                        if name in subset]
        else:
            selected = self.transformations
        
        # Create the pipeline function
        def pipeline(data):
            result = data
            for name, func in selected:
                print(f"Applying transformation: {name}")
                result = func(result)
            return result
        
        return pipeline

# Example usage with pandas
processor = DataProcessor()

# Add transformations
processor.add_transformation(
    "clean_missing", 
    lambda df: df.dropna()
)
processor.add_transformation(
    "normalize", 
    lambda df: (df - df.mean()) / df.std()
)
processor.add_transformation(
    "add_features", 
    lambda df: df.assign(total=df['a'] + df['b'])
)

# Create different pipelines for different purposes
full_pipeline = processor.create_pipeline()
cleaning_only = processor.create_pipeline(subset=["clean_missing"])
feature_pipeline = processor.create_pipeline(subset=["clean_missing", "add_features"])

# Process data with different pipelines
clean_data = cleaning_only(raw_data)
feature_data = feature_pipeline(raw_data)
```

#### 2. Custom Map-Reduce Framework

```python
def map_reduce(data, map_func, reduce_func, initial_value=None):
    """
    Implement a simple map-reduce framework.
    
    Args:
        data: The input data iterable
        map_func: Function to apply to each data item
        reduce_func: Function to combine mapped results
        initial_value: Initial value for reduction
    """
    # Map phase
    mapped_data = map(map_func, data)
    
    # Reduce phase
    if initial_value is not None:
        result = functools.reduce(reduce_func, mapped_data, initial_value)
    else:
        result = functools.reduce(reduce_func, mapped_data)
    
    return result

# Example: Count words in documents
documents = [
    "This is a sample document",
    "Another document with more words",
    "A third document with even more sample words"
]

# Map function: Count words in a document
def count_words(doc):
    words = doc.lower().split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

# Reduce function: Combine word count dictionaries
def combine_counts(dict1, dict2):
    result = dict1.copy()
    for word, count in dict2.items():
        result[word] = result.get(word, 0) + count
    return result

# Perform map-reduce
word_counts = map_reduce(documents, count_words, combine_counts, {})
print(word_counts)
```

#### 3. Strategy Pattern Implementation

```python
class DataStrategy:
    """Class that applies different strategies to process data."""
    
    def __init__(self, default_strategy=None):
        self.strategies = {}
        self.default_strategy = default_strategy
    
    def register_strategy(self, name, strategy_func):
        """Register a new strategy function."""
        self.strategies[name] = strategy_func
        return self
    
    def process(self, data, strategy_name=None):
        """
        Process data using the named strategy or default strategy.
        """
        # Determine which strategy to use
        if strategy_name is not None and strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
        elif self.default_strategy is not None:
            strategy = self.default_strategy
        else:
            raise ValueError("No strategy specified and no default strategy")
        
        # Apply the strategy
        return strategy(data)
    
    def create_processor(self, strategy_name=None):
        """Create a processor function with a fixed strategy."""
        def processor(data):
            return self.process(data, strategy_name)
        return processor

# Example usage for data aggregation
aggregator = DataStrategy()

# Register different aggregation strategies
aggregator.register_strategy(
    "sum", 
    lambda data: sum(data)
)
aggregator.register_strategy(
    "mean", 
    lambda data: sum(data) / len(data) if data else 0
)
aggregator.register_strategy(
    "median", 
    lambda data: sorted(data)[len(data)//2] if data else 0
)

# Create specialized processor functions
sum_processor = aggregator.create_processor("sum")
mean_processor = aggregator.create_processor("mean")

# Process data
data = [1, 5, 3, 9, 2, 6]
print(sum_processor(data))   # 26
print(mean_processor(data))  # 4.333...
```

### Best Practices for Higher-Order Functions

1. **Keep it simple**:
   Higher-order functions should follow the single responsibility principle.

2. **Clear naming**:
   Use descriptive names for functions and arguments to improve readability.

3. **Type hints**:
   Use type hints to document expected function types for higher-order functions.# Answers to Python Interview Questions 11-20

## 11. Explain list comprehensions and generator expressions. When would you use one over the other?

### List Comprehensions

List comprehensions provide a concise way to create lists based on existing iterables. They follow the format:

```python
[expression for item in iterable if condition]
```

**Example:**
```python
# Traditional way
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)

# With list comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]
```

**Characteristics:**
- Creates a new list in memory immediately
- Evaluates all elements at once
- Takes more memory but provides immediate access to all elements
- Supports nested loops: `[x+y for x in range(3) for y in range(3)]`

### Generator Expressions

Generator expressions have a similar syntax but use parentheses instead of brackets:

```python
(expression for item in iterable if condition)
```

**Example:**
```python
# Generator expression
squares_gen = (x**2 for x in range(10) if x % 2 == 0)

# Iterating over the generator
for square in squares_gen:
    print(square)  # Prints 0, 4, 16, 36, 64
```

**Characteristics:**
- Creates a generator object that produces values on demand (lazy evaluation)
- Evaluates elements one at a time when needed
- Uses minimal memory regardless of the size of the output
- Can only be iterated over once
- Can be infinite (e.g., `(i for i in itertools.count())`)

### When to Use List Comprehensions:

1. **When you need the entire result set at once:**
   ```python
   # Find all files in a directory
   all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
   print(f"Found {len(all_files)} files")  # Need whole list for len()
   ```

2. **When you need to use the result multiple times:**
   ```python
   data = [process(x) for x in input_data]
   
   # Use data multiple times
   avg = sum(data) / len(data)
   max_val = max(data)
   min_val = min(data)
   ```

3. **When working with small to moderate data sizes:**
   ```python
   # Small dataset, acceptable memory usage
   words = [word.lower() for word in text.split() if len(word) > 3]
   ```

4. **When you need list-specific methods:**
   ```python
   # Using list methods like sort
   sorted_results = [x**2 for x in numbers]
   sorted_results.sort()  # In-place sorting
   ```

### When to Use Generator Expressions:

1. **When working with large or infinite sequences:**
   ```python
   # Processing a large file line by line
   line_count = sum(1 for line in open('huge_file.txt'))
   ```

2. **When you only need to iterate once:**
   ```python
   # Find first match
   first_match = next((x for x in data if is_match(x)), None)
   ```

3. **When memory efficiency is critical:**
   ```python
   # Memory efficient processing of large data
   avg = sum(float(line) for line in open('measurements.txt')) / \
         sum(1 for _ in open('measurements.txt'))
   ```

4. **When pipelining operations:**
   ```python
   # Chain multiple generators together
   processed_data = (transform(x) for x in 
                     (parse(line) for line in open('data.csv')))
   ```

5. **When the results are consumed by another iterator:**
   ```python
   # Feeding a generator to a function that consumes iterators
   max_value = max(x**2 for x in range(1000000))
   ```

### Performance Comparison:

```python
import sys
import time

# Memory usage
numbers = range(1000000)
list_comp = [x for x in numbers]  # Creates a list with 1M elements
gen_exp = (x for x in numbers)    # Creates a generator object

print(f"List comprehension size: {sys.getsizeof(list_comp)} bytes")
print(f"Generator expression size: {sys.getsizeof(gen_exp)} bytes")

# Execution time comparison for summation
numbers = range(10000000)

start = time.time()
sum_list = sum([x for x in numbers])  # Creates list in memory first
list_time = time.time() - start

start = time.time()
sum_gen = sum(x for x in numbers)     # Streams values directly to sum()
gen_time = time.time() - start

print(f"List comprehension time: {list_time:.4f} seconds")
print(f"Generator expression time: {gen_time:.4f} seconds")
```

Typical output would show that generator expressions use significantly less memory and can be faster for operations that consume iterators directly, like `sum()` or `max()`.

### Real-World Data Engineering Example:

```python
def process_large_dataset(file_path):
    """Process a large CSV file without loading it all into memory."""
    
    # Bad approach - loads everything into memory
    # with open(file_path) as f:
    #     data = [process_line(line) for line in f]
    #     return analyze(data)
    
    # Good approach - streams data
    with open(file_path) as f:
        # Process file line by line using generator expressions
        line_count = sum(1 for _ in f)
    
    with open(file_path) as f:
        # Calculate average of some value
        total = sum(float(line.split(',')[2]) for line in f 
                   if line.strip() and not line.startswith('#'))
    
    with open(file_path) as f:
        # Find records matching a criterion
        matches = (line for line in f if "important" in line)
        # Process first 10 matches
        for i, match in enumerate(matches):
            if i >= 10:
                break
            process_match(match)
            
    return total / line_count
```

## 12. How do you use `map()`, `filter()`, and `reduce()` functions in Python?

These three functions are cornerstones of functional programming in Python, allowing for elegant data transformations and processing.

### 1. `map()` Function

**Purpose**: Applies a function to each item in an iterable and returns a map object (iterator).

**Syntax**: 
```python
map(function, iterable, [iterable2, iterable3, ...])
```

**Examples**:

```python
# Basic usage - convert strings to integers
numbers = ['1', '2', '3', '4']
int_numbers = list(map(int, numbers))
print(int_numbers)  # [1, 2, 3, 4]

# With a lambda function
squared = list(map(lambda x: x**2, range(5)))
print(squared)  # [0, 1, 4, 9, 16]

# With multiple iterables
first = [1, 2, 3]
second = [4, 5, 6]
sums = list(map(lambda x, y: x + y, first, second))
print(sums)  # [5, 7, 9]

# With a named function
def celsius_to_fahrenheit(c):
    return (9/5) * c + 32

temperatures_c = [0, 10, 20, 30, 40]
temperatures_f = list(map(celsius_to_fahrenheit, temperatures_c))
print(temperatures_f)  # [32.0, 50.0, 68.0, 86.0, 104.0]
```

**Data Engineering Example**:

```python
# Parse and transform log entries
log_entries = [
    "2021-05-12,ERROR,Connection failed",
    "2021-05-12,INFO,Server started",
    "2021-05-13,WARNING,Disk space low"
]

def parse_log(entry):
    """Parse a log entry into a structured dictionary."""
    date, level, message = entry.split(',', 2)
    return {
        'date': date,
        'level': level,
        'message': message,
        'timestamp': pd.to_datetime(date)
    }

# Parse all logs with map
parsed_logs = list(map(parse_log, log_entries))
```

### 2. `filter()` Function

**Purpose**: Filters elements from an iterable based on a function that returns True/False.

**Syntax**: 
```python
filter(function, iterable)
```

**Examples**:

```python
# Basic usage - keep only even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4, 6, 8, 10]

# Filter out None and empty values
values = [0, None, False, '', 'hello', [], [1, 2]]
non_empty = list(filter(None, values))  # Passing None as function filters out "falsy" values
print(non_empty)  # [0, 'hello', [1, 2]]

# With a named function
def is_positive(num):
    return num > 0

numbers = [-3, -2, -1, 0, 1, 2, 3]
positive_numbers = list(filter(is_positive, numbers))
print(positive_numbers)  # [1, 2, 3]
```

**Data Engineering Example**:

```python
# Filter log entries by error level
def is_error(log):
    return log['level'] == 'ERROR'

# Get only error logs
error_logs = list(filter(is_error, parsed_logs))

# Practical example: filter out bad data
def is_valid_reading(reading):
    """Check if a sensor reading is valid."""
    # Check for missing values
    if reading['value'] is None:
        return False
    
    # Check for values outside valid range
    if not (0 <= reading['value'] <= 100):
        return False
    
    # Check for valid timestamp
    if reading['timestamp'] < pd.Timestamp('2020-01-01'):
        return False
    
    return True

# Filter out invalid readings
valid_readings = filter(is_valid_reading, sensor_data)
```

### 3. `reduce()` Function

**Purpose**: Applies a function of two arguments cumulatively to the items of an iterable, reducing it to a single value.

**Syntax**: 
```python
functools.reduce(function, iterable[, initializer])
```

**Note**: In Python 3, `reduce()` was moved to the `functools` module.

**Examples**:

```python
from functools import reduce

# Sum all numbers
numbers = [1, 2, 3, 4, 5]
sum_result = reduce(lambda x, y: x + y, numbers)
print(sum_result)  # 15

# With an initializer
sum_with_start = reduce(lambda x, y: x + y, numbers, 10)
print(sum_with_start)  # 25 (10 + 1 + 2 + 3 + 4 + 5)

# Find maximum value
max_value = reduce(lambda x, y: x if x > y else y, numbers)
print(max_value)  # 5

# Flatten a list of lists
nested = [[1, 2], [3, 4], [5, 6]]
flattened = reduce(lambda x, y: x + y, nested)
print(flattened)  # [1, 2, 3, 4, 5, 6]

# Composing functions
def double(x):
    return x * 2

def increment(x):
    return x + 1

functions = [double, increment, double]
# Apply functions from right to left
composed = reduce(lambda f, g: lambda x: f(g(x)), functions)
print(composed(5))  # double(increment(double(5))) = double(increment(10)) = double(11) = 22
```

**Data Engineering Example**:

```python
# Calculate total size of all files in directories
def get_dir_size(path):
    return sum(os.path.getsize(os.path.join(path, f)) 
               for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))

directories = ['logs', 'data', 'temp']
total_size = reduce(lambda total, dir: total + get_dir_size(dir), directories, 0)

# More complex reduction for data aggregation
def combine_metrics(acc, new_data):
    """Combine running metrics with new data point."""
    return {
        'count': acc['count'] + 1,
        'sum': acc['sum'] + new_data['value'],
        'min': min(acc['min'], new_data['value']),
        'max': max(acc['max'], new_data['value']),
        # Track values for later std_dev calculation
        'values': acc['values'] + [new_data['value']]
    }

# Initial accumulator
initial = {'count': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf'), 'values': []}

# Reduce data to summary metrics
metrics = reduce(combine_metrics, data_points, initial)
metrics['avg'] = metrics['sum'] / metrics['count']
```

### Combining map, filter, and reduce

These functions can be combined for powerful data processing:

```python
from functools import reduce

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Find the sum of squares of even numbers
result = reduce(
    lambda x, y: x + y,           # reduce: sum the values
    map(lambda x: x**2,           # map: square the values
        filter(lambda x: x % 2 == 0, data)  # filter: keep even numbers
    )
)
print(result)  # 220 (2² + 4² + 6² + 8² + 10²)
```

### Modern Alternatives in Python

While `map()`, `filter()`, and `reduce()` are powerful, Python provides alternative syntax that is often more readable:

1. **List Comprehensions** instead of `map()` and `filter()`:
   ```python
   # Using map and filter
   result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
   
   # Using list comprehension
   result = [x**2 for x in numbers if x % 2 == 0]
   ```

2. **Built-in functions** instead of `reduce()`:
   ```python
   # Using reduce for sum
   from functools import reduce
   sum_result = reduce(lambda x, y: x + y, numbers)
   
   # Using built-in sum
   sum_result = sum(numbers)
   ```

3. **Generator expressions** for memory efficiency:
   ```python
   # Process data efficiently with generators
   result = sum(x**2 for x in numbers if x % 2 == 0)
   ```

### Parallel Processing with `map`

For CPU-intensive operations, the `multiprocessing` module provides a parallel version of `map`:

```python
from multiprocessing import Pool

def expensive_operation(x):
    # Simulate CPU-intensive task
    return x**3

numbers = range(10000)

with Pool() as pool:
    # Process in parallel across multiple cores
    results = pool.map(expensive_operation, numbers)
```

## 13. What are decorators in Python and how would you implement a custom one?

### Understanding Decorators

Decorators are a powerful feature in Python that allows you to modify the behavior of a function or method without changing its source code. At their core, decorators are a design pattern that allow wrapping one piece of code with another.

In Python, decorators use the `@decorator_name` syntax placed above function or method definitions.

### Basic Decorator Structure

A decorator is a function that takes another function as an argument, adds some functionality, and returns a new function:

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Code to execute before the function
        print("Before function call")
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Code to execute after the function
        print("After function call")
        
        # Return the result
        return result
    
    # Return the wrapper function
    return wrapper

# Using the decorator
@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

# This is equivalent to:
# say_hello = my_decorator(say_hello)

# When we call the function
say_hello("Alice")
# Output:
# Before function call
# Hello, Alice!
# After function call
```

### Implementing Custom Decorators

Here are several types of custom decorators for different purposes:

#### 1. Timing Decorator (Measuring Performance)

```python
import time
import functools

def timer(func):
    """Measure and print the execution time of the decorated function."""
    @functools.wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.time()
        
        # Print execution time
        print(f"{func.__name__} ran in {end_time - start_time:.6f} seconds")
        
        return result
    
    return wrapper

# Usage
@timer
def expensive_operation(n):
    """Simulate an expensive operation."""
    total = 0
    for i in range(n):
        total += i**2
    return total

# Test the decorated function
result = expensive_operation(1000000)
# Output: expensive_operation ran in 0.123456 seconds
```

#### 2. Logging Decorator

```python
import functools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_function_call(func):
    """Log information about function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call with arguments
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        logger.info(f"Calling {func.__name__}({signature})")
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log successful completion
            logger.info(f"{func.__name__} returned {result!r}")
            
            return result
        except Exception as e:
            # Log any exceptions
            logger.exception(f"Exception raised in {func.__name__}: {str(e)}")
            raise
    
    return wrapper

# Usage
@log_function_call
def divide(a, b):
    """Divide a by b."""
    return a / b

# Test with normal case
divide(10, 2)  # Logs: Calling divide(10, 2) and divide returned 5.0

# Test with exception
try:
    divide(10, 0)  # Logs the exception
except ZeroDivisionError:
    pass
```

#### 3. Retry Decorator

```python
import functools
import time
import random

def retry(max_attempts=3, delay_seconds=1):
    """
    Retry a function if it raises an exception.
    
    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Delay between attempts in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        # Re-raise the exception if we've reached max attempts
                        raise
                    
                    # Log the failed attempt
                    print(f"Attempt {attempts} failed: {str(e)}")
                    print(f"Retrying in {delay_seconds} seconds...")
                    
                    # Wait before retrying
                    time.sleep(delay_seconds)
            
        return wrapper
    return decorator

# Usage
@retry(max_attempts=5, delay_seconds=2)
def unreliable_api_call():
    """Simulate an unreliable API call that sometimes fails."""
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("API connection failed")
    return "Success!"

# Test the decorated function
try:
    result = unreliable_api_call()
    print(f"Final result: {result}")
except ConnectionError:
    print("All attempts failed")
```

#### 4. Caching/Memoization Decorator

```python
import functools

def memoize(func):
    """Cache the results of a function for the same inputs."""
    # Cache to store results
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        # For kwargs, we need to sort items to ensure consistent key
        key = str(args) + str(sorted(kwargs.items()))
        
        # Check if result is already in cache
        if key not in cache:
            # Call the function and store result in cache
            cache[key] = func(*args, **kwargs)
            print(f"Cache miss for {key}")
        else:
            print(f"Cache hit for {key}")
            
        return cache[key]
    
    # Add a method to clear the cache
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper

# Usage
@memoize
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    print(f"Computing fibonacci({n})...")
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the decorated function
print(fibonacci(10))  # Computes and caches all values
print(fibonacci(5))   # Uses cached value
print(fibonacci(12))  # Reuses cached values for fib(10) and fib(11)

# Clear the cache
fibonacci.clear_cache()
print(fibonacci(5))   # Recomputes after cache clear
```

#### 5. Permission/Authentication Decorator

```python
import functools

def require_permission(permission):
    """Check if user has required permission."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(user, *args, **kwargs):
            # Check if user has the required permission
            if permission not in user.get('permissions', []):
                raise PermissionError(f"User does not have '{permission}' permission")
            
            # If user has permission, call the original function
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_permission('admin')
def delete_user(current_user, user_id):
    print(f"User {user_id} deleted by {current_user['name']}")

# Test with admin user
admin_user = {'name': 'Admin', 'permissions': ['read', 'write', 'admin']}
delete_user(admin_user, 123)  # Works fine

# Test with regular user
regular_user = {'name': 'User', 'permissions': ['read']}
try:
    delete_user(regular_user, 123)  # Raises PermissionError
except PermissionError as e:
    print(str(e))
```

#### 6. Parameter Validation Decorator

```python
import functools

def validate_args(*types, **kwarg_types):
    """
    Validate function arguments against specified types.
    
    Usage:
    @validate_args(int, str, bool, optional_param=dict)
    def my_func(a, b, c, optional_param=None):
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check positional arguments
            for i, (arg, expected_type) in enumerate(zip(args, types)):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} should be {expected_type.__name__}, "
                                    f"got {type(arg).__name__}")
            
            # Check keyword arguments
            for param_name, arg in kwargs.items():
                if param_name in kwarg_types and not isinstance(arg, kwarg_types[param_name]):
                    raise TypeError(f"Argument '{param_name}' should be {kwarg_types[param_name].__name__}, "
                                    f"got {type(arg).__name__}")
            
            # Call the original function with validated arguments
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@validate_args(int, int, operation=str)
def calculate(a, b, operation='add'):
    """Perform a calculation on a and b."""
    if operation == 'add':
        return a + b
    elif operation == 'multiply':
        return a * b
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Test with valid arguments
result = calculate(5, 3)
print(result)  # 8

# Test with invalid arguments
try:
    result = calculate('5', 3)  # Raises TypeError
except TypeError as e:
    print(str(e))
```

### Decorators with Arguments

As shown in some examples above, you can create decorators that accept arguments:

```python
def repeat(num_times):
    """Repeat the function call num_times times."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(num_times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

# Usage
@repeat(3)
def greet(name):
    return f"Hello, {name}!"

# Test
print(greet("World"))  # ['Hello, World!', 'Hello, World!', 'Hello, World!']
```

### Class-Based Decorators

Decorators can also be implemented as classes:

```python
class CountCalls:
    """Count the number of calls to the decorated function."""
    
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.num_calls = 0
    
    def __call__(self, *args, **kwargs):
        """Called instead of the decorated function."""
        self.num_calls += 1
        print(f"Call {self.num_calls} of {self.func.__name__}")
        return self.func(*args, **kwargs)

# Usage
@CountCalls
def say_hello():
    print("Hello!")

# Test
say_hello()  # Call 1 of say_hello
say_hello()  # Call 2 of say_hello
```

### Real-World Data Engineering Examples

#### 1. Profiling Data Processing Functions

```python
import functools
import time
import resource

def profile_resource_usage(func):
    """Profile memory and CPU usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial resource usage
        start_time = time.time()
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get final resource usage
        end_time = time.time()
        end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Calculate differences
        elapsed_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Print performance metrics
        print(f"{func.__name__} performance:")
        print(f"  Time: {elapsed_time:.4f} seconds")
        print(f"  Memory: {memory_used / 1024:.2f} MB")
        
        return result
    return wrapper

# Usage in data engineering context
@profile_resource_usage
def process_large_dataframe(df):
    """Process a large DataFrame."""
    # Simulate data processing operations
    result = df.groupby('category').agg({'value': ['sum', 'mean']})
    result = result.reset_index()
    return result
```

#### 2. Pipeline Component Decorator

```python
def pipeline_component(name=None, requires=None):
    """
    Decorator for data pipeline components.
    
    Args:
        name: Name of the component
        requires: List of components that must run before this one
    """
    def decorator(func):
        # Set component metadata
        func.is_pipeline_component = True
        func.pipeline_name = name or func.__name__
        func.pipeline_requires = requires or []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Running pipeline component: {func.pipeline_name}")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Usage in a data pipeline
@pipeline_component(name="extract_data", requires=[])
def extract_data():
    """Extract data from source."""
    # Implementation
    return "raw_data"

@pipeline_component(name="transform_data", requires=["extract_data"])
def transform_data(data):
    """Transform the data."""
    # Implementation
    return f"transformed_{data}"

@pipeline_component(name="load_data", requires=["transform_data"])
def load_data(data):
    """Load the data to destination."""
    # Implementation
    print(f"Loaded: {data}")
    return "success"

# Execute the pipeline
def run_pipeline():
    data = extract_data()
    transformed = transform_data(data)
    result = load_data(transformed)
    return result
```

#### 3. Data Validation Decorator

```python
def validate_dataframe(required_columns=None, column_types=None, non_empty=True):
    """
    Validate pandas DataFrame arguments.
    
    Args:
        required_columns: List of columns that must be present
        column_types: Dict mapping column names to expected types
        non_empty: Whether the DataFrame should have at least one row
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame arguments
            import pandas as pd
            df_args = [arg for arg in args if isinstance(arg, pd.DataFrame)]
            df_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, pd.DataFrame)}
            
            # Validate each DataFrame
            for i, df in enumerate(df_args):
                validate_df(df, f"positional argument {i}", required_columns, column_types, non_empty)
            
            for name, df in df_kwargs.items():
                validate_df(df, f"keyword argument '{name}'", required_columns, column_types, non_empty)
            
            # Call the function if validation passes
            return func(*args, **kwargs)
        
        def validate_df(df, arg_name, required_columns, column_types, non_empty):
            # Check if empty
            if non_empty and len(df) == 0:
                raise ValueError(f"DataFrame {arg_name} cannot be empty")
            
            # Check required columns
            if required_columns:
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise ValueError(f"DataFrame {arg_name} missing required columns: {missing}")
            
            # Check column types
            if column_types:
                for col, expected_type in column_types.items():
                    if col in df.columns:
                        # Check if the column contains values of the expected type
                        if not all(isinstance(x, expected_type) for x in df[col].dropna()):
                            raise TypeError(f"Column '{col}' in {arg_name} should contain only {expected_type.__name__}")
        
        return wrapper
    return decorator

# Usage
import pandas as pd

@validate_dataframe(
    required_columns=['date', 'value', 'category'],
    column_types={'value': (int, float)},
    non_empty=True
)
def analyze_timeseries(df):
    """Analyze time series data."""
    # Implementation
    return df.groupby('category')['value'].mean()
```

### Combining Multiple Decorators

You can apply multiple decorators to a single function:

```python
@timer
@log_function_call
@retry(max_attempts=3)
def process_data(data):
    """Process data with multiple decorators applied."""
    # Implementation
    return processed_data
```

Decorators are applied from bottom to top (retry, then log, then timer).

### Best Practices for Decorators

1. **Use `functools.wraps`**:
   Always use `@functools.wraps(func)` in your decorator to preserve the metadata of the original function (name, docstring, etc.).

2. **Keep decorators focused**:
   Each decorator should have a single responsibility for better maintainability.

3. **Handle arguments correctly**:
   Use `*args` and `**kwargs` to ensure your decorator works with any function signature.

4. **Document decorators**:
   Add clear docstrings to your decorators explaining their purpose and usage.

5. **Error handling**:
   Include proper error handling in decorators to make debugging easier.

6. **Testing**:
   Write tests specifically for your decorators to ensure they work as expected.

## 14. Explain lambda functions and provide a practical use case in data engineering.

### What are Lambda Functions?

Lambda functions in Python are small, anonymous functions defined using the `lambda




**Advantages**:
- Ensures initialization happens only once
- More intuitive behavior for users
- Simple implementation

**Disadvantages**:
- Not thread-safe by default
- Uses a flag variable that might be accidentally modified
- Requires careful design for inheritance

### Real-World Data Engineering Examples

#### Example 1: Database Connection Pool as a Singleton

```python
import threading
import time
import random

class DatabaseConnectionPool:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:  # Thread-safe singleton creation
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, host='localhost', port=5432, max_connections=5):
        # Only initialize once
        if not hasattr(self, '_initialized') or not self._initialized:
            self.host = host
            self.port = port
            self.max_connections = max_connections
            self.connections = []
            self.available_connections = []
            self.connection_lock = threading.Lock()
            self._initialized = True
            print(f"Initializing connection pool to {host}:{port}")
    
    def _create_connection(self):
        """Create a new database connection."""
        connection_id = len(self.connections) + 1
        connection = f"Connection-{connection_id} to {self.host}:{self.port}"
        print(f"Creating {connection}")
        time.sleep(0.1)  # Simulate connection time
        return connection
    
    def get_connection(self):
        """Get a connection from the pool or create a new one."""
        with self.connection_lock:
            if self.available_connections:
                connection = self.available_connections.pop()
                print(f"Reusing {connection}")
                return connection
            
            if len(self.connections) < self.max_connections:
                connection = self._create_connection()
                self.connections.append(connection)
                return connection
            
            # No available connections and pool is full
            print("Connection pool exhausted, waiting...")
            while not self.available_connections:
                # Release lock and wait
                self.connection_lock.release()
                time.sleep(0.1)
                self.connection_lock.acquire()
            
            connection = self.available_connections.pop()
            print(f"Reusing {connection} after waiting")
            return connection
    
    def release_connection(self, connection):
        """Return a connection to the pool."""
        with self.connection_lock:
            if connection in self.connections and connection not in self.available_connections:
                print(f"Releasing {connection} back to pool")
                self.available_connections.append(connection)
            else:
                print(f"Warning: Attempting to release an invalid connection")

# Usage in a multithreaded application
def worker(pool, worker_id):
    """Simulate a worker that needs a database connection."""
    print(f"Worker {worker_id} starting")
    connection = pool.get_connection()
    
    # Simulate using the connection
    print(f"Worker {worker_id} using {connection}")
    time.sleep(random.uniform(0.5, 1.5))
    
    # Release the connection
    pool.release_connection(connection)
    print(f"Worker {worker_id} finished")

# Create the connection pool singleton
pool1 = DatabaseConnectionPool(host="db.example.com", max_connections=3)
pool2 = DatabaseConnectionPool()  # Same instance, configuration not changed

print(f"pool1 is pool2: {pool1 is pool2}")  # True

# Simulate multiple workers
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(pool1, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

#### Example 2: Configuration Manager as a Singleton

```python
import json
import os

class ConfigManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path=None):
        # Only initialize once
        if not self._initialized:
            self.config_path = config_path or os.environ.get('CONFIG_PATH', 'config.json')
            self.config = {}
            self.load_config()
            self._initialized = True
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Loaded configuration from {self.config_path}")
            else:
                print(f"Config file {self.config_path} not found, using defaults")
                self.config = {
                    'database': {
                        'host': 'localhost',
                        'port': 5432,
                        'name': 'mydb'
                    },
                    'api': {
                        'url': 'https://api.example.com',
                        'timeout': 30
                    },
                    'logging': {
                        'level': 'INFO',
                        'path': 'logs/'
                    }
                }
        except Exception as e:
            print(f"Error loading config: {e}")
            # Use defaults on error
            self.config = {}
    
    def get(self, section, key, default=None):
        """Get a configuration value."""
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default
    
    def set(self, section, key, value):
        """Set a configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

# Usage across different parts of an application
def setup_database():
    config = ConfigManager()
    host = config.get('database', 'host', 'localhost')
    port = config.get('database', 'port', 5432)
    db_name = config.get('database', 'name', 'mydb')
    
    print(f"Setting up database connection to {host}:{port}/{db_name}")
    # In a real application, this would set up the actual database connection

def setup_api_client():
    config = ConfigManager()
    api_url = config.get('api', 'url', 'https://api.default.com')
    timeout = config.get('api', 'timeout', 60)
    
    print(f"Setting up API client for {api_url} with timeout {timeout}")
    # In a real application, this would set up the API client

# Create the configuration manager
config = ConfigManager()

# Use it in different components
setup_database()
setup_api_client()

# Update a configuration value
config.set('api', 'timeout', 45)
print(f"Updated API timeout to {config.get('api', 'timeout')}")

# Another part of the application gets the same instance
another_config = ConfigManager()
print(f"API timeout from another component: {another_config.get('api', 'timeout')}")  # 45
```

#### Example 3: ETL Job Registry as a Singleton

```python
class ETLJobRegistry:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.jobs = {}
            self.job_dependencies = {}
            self.running_jobs = set()
            self._initialized = True
    
    def register_job(self, job_id, job_func, dependencies=None):
        """Register an ETL job with optional dependencies."""
        self.jobs[job_id] = job_func
        self.job_dependencies[job_id] = dependencies or []
        print(f"Registered job {job_id} with dependencies {dependencies}")
    
    def get_job(self, job_id):
        """Get a registered job."""
        return self.jobs.get(job_id)
    
    def get_dependencies(self, job_id):
        """Get dependencies for a job."""
        return self.job_dependencies.get(job_id, [])
    
    def start_job(self, job_id, *args, **kwargs):
        """Start a registered job if dependencies are met."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not registered")
        
        # Check dependencies
        for dep_id in self.job_dependencies[job_id]:
            if dep_id in self.running_jobs:
                raise RuntimeError(f"Cannot start job {job_id}, dependency {dep_id} is still running")
        
        # Mark job as running
        self.running_jobs.add(job_id)
        print(f"Starting job {job_id}")
        
        try:
            # Run the job
            result = self.jobs[job_id](*args, **kwargs)
            print(f"Job {job_id} completed")
            return result
        finally:
            # Mark job as completed
            self.running_jobs.remove(job_id)
    
    def get_runnable_jobs(self):
        """Get jobs that can be run (dependencies not running)."""
        runnable_jobs = []
        for job_id in self.jobs:
            if job_id not in self.running_jobs:
                dependencies_met = True
                for dep_id in self.job_dependencies[job_id]:
                    if dep_id in self.running_jobs:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    runnable_jobs.append(job_id)
        
        return runnable_jobs

# Example ETL jobs
def extract_data(source):
    print(f"Extracting data from {source}")
    return f"data from {source}"

def transform_data(data):
    print(f"Transforming {data}")
    return f"transformed {data}"

def load_data(data, destination):
    print(f"Loading {data} to {destination}")
    return f"loaded to {destination}"

# Register jobs in the registry
registry = ETLJobRegistry()
registry.register_job('extract', extract_data, [])
registry.register_job('transform', transform_data, ['extract'])
registry.register_job('load', load_data, ['transform'])

# Another part of the application gets the same registry
another_registry = ETLJobRegistry()
print(f"Registry has {len(another_registry.jobs)} jobs")  # 3

# Run the ETL jobs in dependency order
data = registry.start_job('extract', 'database')
transformed_data = registry.start_job('transform', data)
result = registry.start_job('load', transformed_data, 'data_warehouse')

print(f"ETL pipeline result: {result}")
```

### Best Practices for Singleton Implementation

1. **Choose the right approach**: Select the implementation that best fits your needs (module, class-based, metaclass, etc.).

2. **Thread safety**: Ensure thread-safe initialization for multithreaded applications.

3. **Initialization**: Handle one-time initialization correctly, especially with `__init__`.

4. **Testing**: Consider how to mock or replace the singleton during testing.

5. **Documentation**: Clearly document that a class is a singleton to avoid confusion.

6. **Subclassing**: Be careful with inheritance and singleton patterns.

7. **Dependency injection**: Consider using dependency injection instead of singletons for better testability.

8. **Use sparingly**: Singletons introduce global state, which can make testing and maintenance harder.

### When to Use the Singleton Pattern

**Good use cases**:
- Configuration management
- Database connection pools
- Logging facilities
- Caches
- Thread pools
- Dialog boxes, in UIs

**When to avoid**:
- When multiple instances might be needed in the future
- When global state makes testing difficult
- When different configurations are needed for different components

## 20. How would you implement a custom iterator in Python?

### Understanding Iterators in Python

An iterator in Python is an object that implements two methods: `__iter__()` and `__next__()`. These methods make it possible to iterate over the object using a `for` loop or other iteration tools.

- The `__iter__()` method returns the iterator object itself.
- The `__next__()` method returns the next value from the iterator. When there are no more items to return, it should raise a `StopIteration` exception.

### Basic Implementation of an Iterator

Here's a simple example of a custom iterator that generates a sequence of powers of a given number:

```python
class PowerIterator:
    def __init__(self, base, max_exp):
        self.base = base
        self.max_exp = max_exp
        self.exp = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.exp > self.max_exp:
            raise StopIteration
        
        result = self.base ** self.exp
        self.exp += 1
        return result

# Using the iterator
for power in PowerIterator(2, 5):
    print(power)
# Output: 1, 2, 4, 8, 16, 32

# Converting to a list
powers = list(PowerIterator(3, 4))
print(powers)  # [1, 3, 9, 27, 81]
```

### Creating an Iterable Class

An iterable is an object that has an `__iter__()` method which returns an iterator. It's common to separate the iterable from its iterator:

```python
class PowerSeries:
    def __init__(self, base, max_exp):
        self.base = base
        self.max_exp = max_exp
    
    def __iter__(self):
        return PowerIterator(self.base, self.max_exp)

class PowerIterator:
    def __init__(self, base, max_exp):
        self.base = base
        self.max_exp = max_exp
        self.exp = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.exp > self.max_exp:
            raise StopIteration
        
        result = self.base ** self.exp
        self.exp += 1
        return result

# Using the iterable
powers_of_2 = PowerSeries(2, 5)

# Multiple iterations are possible
for power in powers_of_2:
    print(power)  # 1, 2, 4, 8, 16, 32

# Each iteration gets a fresh iterator
print(list(powers_of_2))  # [1, 2, 4, 8, 16, 32]
```

### Using Generator Functions

Generators provide a simpler way to create iterators. A generator function uses `yield` instead of `return` to provide values:

```python
def power_generator(base, max_exp):
    """Generate powers of base from 0 to max_exp."""
    for exp in range(max_exp + 1):
        yield base ** exp

# Using the generator function
for power in power_generator(2, 5):
    print(power)  # 1, 2, 4, 8, 16, 32

# Converting to a list
powers = list(power_generator(3, 4))
print(powers)  # [1, 3, 9, 27, 81]
```

### Using Generator Expressions

For simple cases, generator expressions provide an even more concise syntax:

```python
# Generator expression for powers
powers_of_2 = (2 ** exp for exp in range(6))

for power in powers_of_2:
    print(power)  # 1, 2, 4, 8, 16, 32
```

### Implementing a More Realistic Example: DatasetIterator

Here's a more practical example for data engineering - an iterator that processes data in chunks:

```python
class DatasetIterator:
    def __init__(self, data, batch_size=2):
        self.data = data
        self.batch_size = batch_size
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        # Get the next batch
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        
        # Process the batch (in a real scenario, this might be more complex)
        return [item.upper() if isinstance(item, str) else item for item in batch]

# Sample data
dataset = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']

# Iterate over the dataset in batches
for batch in DatasetIterator(dataset, batch_size=3):
    print(f"Processing batch: {batch}")
# Output:
# Processing batch: ['APPLE', 'BANANA', 'CHERRY']
# Processing batch: ['DATE', 'ELDERBERRY', 'FIG']
```

### Lazy Evaluation and Infinite Iterators

One powerful feature of iterators is lazy evaluation - values are generated only when needed, allowing for efficient processing of large datasets or even infinite sequences:

```python
def fibonacci_generator():
    """Generate an infinite sequence of Fibonacci numbers."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using the infinite generator (with a limit to avoid infinite loop)
fib = fibonacci_generator()
for i in range(10):
    print(next(fib))  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

### Iterator Functions from the itertools Module

Python's `itertools` module provides many useful functions for working with iterators:

```python
import itertools

# Take first 5 Fibonacci numbers
fib = fibonacci_generator()
first_5 = list(itertools.islice(fib, 5))
print(first_5)  # [0, 1, 1, 2, 3]

# Generate combinations
colors = ['red', 'green', 'blue']
for combo in itertools.combinations(colors, 2):
    print(combo)  # ('red', 'green'), ('red', 'blue'), ('green', 'blue')

# Cycle through values indefinitely
for i, color in enumerate(itertools.cycle(colors)):
    if i >= 7:
        break
    print(color)  # red, green, blue, red, green, blue, red
```

### Advanced Example: CSV File Iterator

Here's a more advanced example that processes a CSV file in chunks, demonstrating how iterators can be used for efficient data processing:

```python
import csv
from typing import List, Dict, Iterator

class CSVIterator:
    """Iterator for processing CSV files in chunks."""
    
    def __init__(self, filename: str, chunk_size: int = 100, has_header: bool = True):
        self.filename = filename
        self.chunk_size = chunk_size
        self.has_header = has_header
        self.file = None
        self.reader = None
        self.header = None
    
    def __enter__(self):
        """Context manager entry."""
        self.file = open(self.filename, 'r', newline='')
        self.reader = csv.reader(self.file)
        
        if self.has_header:
            self.header = next(self.reader)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file:
            self.file.close()
    
    def __iter__(self) -> Iterator[List[Dict]]:
        """Return self as iterator."""
        if not self.file:
            raise ValueError("CSV file not opened. Use 'with' statement.")
        return self
    
    def __next__(self) -> List[Dict]:
        """Return the next chunk of data."""
        chunk = []
        
        for _ in range(self.chunk_size):
            try:
                row = next(self.reader)
                
                # Convert to dict if header exists
                if self.header:
                    row_dict = {self.header[i]: value for i, value in enumerate(row) if i < len(self.header)}
                    chunk.append(row_dict)
                else:
                    chunk.append(row)
                    
            except StopIteration:
                if not chunk:
                    # Re-raise StopIteration if no rows were read
                    raise
                break
        
        if not chunk:
            raise StopIteration
        
        return chunk

# Example usage (assuming 'data.csv' exists)
def process_large_csv():
    try:
        with CSVIterator('data.csv', chunk_size=500) as csv_iter:
            for i, chunk in enumerate(csv_iter):
                print(f"Processing chunk {i+1} with {len(chunk)} records")
                
                # Process each record in the chunk
                for record in chunk:
                    # In a real application, perform data transformations here
                    pass
    except FileNotFoundError:
        print("CSV file not found")
        # In a real application, handle this error appropriately

# Alternatively, we could use a generator function for similar functionality
def csv_chunk_generator(filename, chunk_size=100):
    """Generate chunks of data from a CSV file."""
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Assuming there's a header
        
        chunk = []
        for row in reader:
            # Convert row to dict
            row_dict = {header[i]: value for i, value in enumerate(row) if i < len(header)}
            chunk.append(row_dict)
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        # Yield the last chunk if it's not empty
        if chunk:
            yield chunk
```

### Real-World Data Engineering Examples

#### Example 1: Time Series Data Iterator

```python
import datetime
from typing import Dict, List, Union, Optional

class TimeSeriesIterator:
    """Iterator for time series data with various aggregation options."""
    
    def __init__(self, data: List[Dict], 
                 start_date: Optional[datetime.date] = None,
                 end_date: Optional[datetime.date] = None,
                 aggregation: str = 'daily'):
        """
        Initialize the time series iterator.
        
        Args:
            data: List of dictionaries with 'date' and 'value' keys
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            aggregation: Aggregation level ('daily', 'weekly', 'monthly')
        """
        self.raw_data = data
        self.start_date = start_date
        self.end_date = end_date
        self.aggregation = aggregation
        
        # Process and filter the data
        self.processed_data = self._preprocess_data()
        self.keys = list(self.processed_data.keys())
        self.index = 0
    
    def _preprocess_data(self) -> Dict[str, Union[float, List[Dict]]]:
        """Preprocess and aggregate the data."""
        # Filter by date if specified
        filtered_data = []
        for item in self.raw_data:
            date = datetime.datetime.strptime(item['date'], '%Y-%m-%d').date()
            
            if self.start_date and date < self.start_date:
                continue
            if self.end_date and date > self.end_date:
                continue
            
            filtered_data.append({
                'date': date,
                'value': float(item['value'])
            })
        
        # Aggregate the data
        aggregated = {}
        for item in filtered_data:
            date = item['date']
            value = item['value']
            
            # Get the aggregation key
            if self.aggregation == 'daily':
                key = date.isoformat()
            elif self.aggregation == 'weekly':
                # Get the Monday of the week
                start_of_week = date - datetime.timedelta(days=date.weekday())
                key = start_of_week.isoformat()
            elif self.aggregation == 'monthly':
                key = date.strftime('%Y-%m-01')
            else:
                key = date.isoformat()  # Default to daily
            
            # Add to the aggregation
            if key not in aggregated:
                aggregated[key] = {
                    'sum': value,
                    'count': 1,
                    'min': value,
                    'max': value,
                    'values': [item]
                }
            else:
                aggregated[key]['sum'] += value
                aggregated[key]['count'] += 1
                aggregated[key]['min'] = min(aggregated[key]['min'], value)
                aggregated[key]['max'] = max(aggregated[key]['max'], value)
                aggregated[key]['values'].append(item)
        
        return aggregated
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.keys):
            raise StopIteration
        
        key = self.keys[self.index]
        data = self.processed_data[key]
        
        result = {
            'period': key,
            'average': data['sum'] / data['count'],
            'sum': data['sum'],
            'count': data['count'],
            'min': data['min'],
            'max': data['max'],
            'raw_data': data['values']
        }
        
        self.index += 1
        return result

# Example usage
sample_data = [
    {'date': '2023-05-01', 'value': '10.5'},
    {'date': '2023-05-01', 'value': '12.3'},
    {'date': '2023-05-02', 'value': '9.8'},
    {'date': '2023-05-03', 'value': '11.2'},
    {'date': '2023-05-08', 'value': '10.0'},
    {'date': '2023-05-09', 'value': '9.5'},
    {'date': '2023-05-15', 'value': '11.8'},
    {'date': '2023-05-16', 'value': '12.1'},
]

# Daily aggregation
daily_iter = TimeSeriesIterator(sample_data, aggregation='daily')
print("Daily Data:")
for item in daily_iter:
    print(f"{item['period']}: Avg={item['average']:.2f}, Count={item['count']}")

# Weekly aggregation
weekly_iter = TimeSeriesIterator(sample_data, aggregation='weekly')
print("\nWeekly Data:")
for item in weekly_iter:
    print(f"Week of {item['period']}: Avg={item['average']:.2f}, Sum={item['sum']:.2f}, Count={item['count']}")
```

#### Example 2: Batch Processing Iterator for Large Datasets

```python
class BatchProcessor:
    """Process large datasets in batches with customizable transformations."""
    
    def __init__(self, data_source, batch_size=100, transformations=None):
        """
        Initialize the batch processor.
        
        Args:
            data_source: Data source (file, list, etc.)
            batch_size: Number of records per batch
            transformations: List of functions to apply to each record
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.transformations = transformations or []
        self.position = 0
    
    def __iter__(self):
        self.position = 0
        return self
    
    def __next__(self):
        # Get the next batch
        if isinstance(self.data_source, list):
            # Handle list data source
            if self.position >= len(self.data_source):
                raise StopIteration
            
            batch = self.data_source[self.position:self.position + self.batch_size]
            self.position += self.batch_size
        else:
            # Assume it's a file-like object
            batch = []
            for _ in range(self.batch_size):
                try:
                    line = next(self.data_source)
                    batch.append(line)
                except StopIteration:
                    if not batch:
                        raise
                    break
        
        # Apply transformations
        for transform in self.transformations:
            batch = [transform(item) for item in batch]
        
        return batch
    
    def process_all(self, callback=None):
        """
        Process all batches and optionally apply a callback to each batch.
        
        Args:
            callback: Function to call with each processed batch
        
        Returns:
            List of results if callback returns values, otherwise None
        """
        results = []
        batch_num = 0
        
        for batch in self:
            batch_num += 1
            print(f"Processing batch {batch_num} with {len(batch)} records")
            
            if callback:
                result = callback(batch)
                if result is not None:
                    results.append(result)
        
        return results if results else None

# Example usage with list data
def parse_int(x):
    return int(x)

def square(x):
    return x * x

data = list(range(1, 1001))  # Numbers from 1 to 1000
processor = BatchProcessor(data, batch_size=100, transformations=[parse_int, square])

# Process and sum each batch
def sum_batch(batch):
    return sum(batch)

batch_sums = processor.process_all(sum_batch)
print(f"Batch sums: {batch_sums}")
print(f"Total sum: {sum(batch_sums)}")

# Example usage with file data
def process_log_file(filename):
    def parse_log_line(line):
        # In a real case, this would parse the log format
        return {
            'timestamp': line[:19],
            ### Common Use Cases for `__new__`

1. **Singleton Pattern**: Ensuring only one instance of a class exists.

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, name):
        # This will be called every time, even though the same instance is returned
        self.name = name

# Creating instances
s1 = Singleton("First")
s2 = Singleton("Second")

print(s1 is s2)      # True (same instance)
print(s1.name)       # "Second" (the attribute was overwritten)
```

2. **Controlling Instance Creation**: Modifying or validating instances during creation.

```python
class PositiveInteger:
    def __new__(cls, value):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        if value <= 0:
            raise ValueError("Value must be positive")
        return super().__new__(cls)
    
    def __init__(self, value):
        self.value = value

# Creating instances
try:
    n1 = PositiveInteger(10)     # Works fine
    n2 = PositiveInteger(-5)     # Raises ValueError
except ValueError as e:
    print(e)  # Value must be positive
```

3. **Immutable Types**: Creating immutable objects.

```python
class ImmutablePoint:
    def __new__(cls, x, y):
        instance = super().__new__(cls)
        # Set attributes directly in __new__
        instance._x = x
        instance._y = y
        return instance
    
    # No __init__ method, as it would allow modification after creation
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    def __repr__(self):
        return f"ImmutablePoint({self._x}, {self._y})"

point = ImmutablePoint(3, 4)
print(point.x, point.y)  # 3 4
# Cannot modify attributes without explicit setters
```

4. **Class Method Alternative**: Using `__new__` as an alternative to class methods for creating instances.

```python
class Person:
    def __new__(cls, name=None, age=None, **kwargs):
        if kwargs.get('from_dict'):
            data = kwargs['from_dict']
            return super().__new__(cls)
        return super().__new__(cls)
    
    def __init__(self, name=None, age=None, **kwargs):
        if kwargs.get('from_dict'):
            data = kwargs['from_dict']
            self.name = data['name']
            self.age = data['age']
        else:
            self.name = name
            self.age = age
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

# Normal creation
p1 = Person("Alice", 30)

# Creation from dictionary
p2 = Person(from_dict={'name': 'Bob', 'age': 25})

print(p1)  # Person(name='Alice', age=30)
print(p2)  # Person(name='Bob', age=25)
```

5. **Return Different Types**: Returning instances of different types based on parameters.

```python
class Shape:
    def __new__(cls, shape_type, *args, **kwargs):
        if shape_type == 'circle':
            return Circle(*args, **kwargs)
        elif shape_type == 'rectangle':
            return Rectangle(*args, **kwargs)
        else:
            return super().__new__(cls)
    
    def __init__(self, shape_type, *args, **kwargs):
        self.shape_type = shape_type
    
    def area(self):
        return 0

class Circle:
    def __init__(self, radius, **kwargs):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2

class Rectangle:
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

# Creating shapes
c = Shape('circle', 5)
r = Shape('rectangle', 4, 6)

print(type(c))  # <class '__main__.Circle'>
print(type(r))  # <class '__main__.Rectangle'>
print(c.area())  # ~78.54
print(r.area())  # 24
```

### Common Use Cases for `__init__`

1. **Setting Instance Attributes**: The most common use case.

```python
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer = 0  # Default value
    
    def get_description(self):
        return f"{self.year} {self.make} {self.model}, {self.odometer} miles"

car = Car("Toyota", "Corolla", 2022)
print(car.get_description())  # "2022 Toyota Corolla, 0 miles"
```

2. **Input Validation**: Checking and processing inputs during initialization.

```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        if not isinstance(account_number, str):
            raise TypeError("Account number must be a string")
        
        if not account_number.isdigit() or len(account_number) != 10:
            raise ValueError("Account number must be a 10-digit number")
        
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        
        self.account_number = account_number
        self.balance = initial_balance
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

try:
    account = BankAccount("1234567890", 100)
    account.deposit(50)
    print(account.balance)  # 150
    
    account = BankAccount("123", 100)  # ValueError (invalid account number)
except ValueError as e:
    print(e)  # Account number must be a 10-digit number
```

3. **Setting Up Resources**: Initializing connections, files, or other resources.

```python
class DatabaseConnection:
    def __init__(self, host, port, username, password, database):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.connection = None
        
        # Connect to the database
        self.connect()
    
    def connect(self):
        print(f"Connecting to {self.database} on {self.host}:{self.port}...")
        # In a real implementation, this would establish the actual connection
        self.connection = f"Connection to {self.database}"
    
    def close(self):
        if self.connection:
            print(f"Closing connection to {self.database}")
            self.connection = None

# Create a connection
db = DatabaseConnection("localhost", 5432, "user", "password", "mydb")
# Output: Connecting to mydb on localhost:5432...
```

4. **Composition**: Setting up relationships with other objects.

```python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.running = False
    
    def start(self):
        self.running = True
        return "Engine started"
    
    def stop(self):
        self.running = False
        return "Engine stopped"

class Car:
    def __init__(self, make, model, year, engine_hp):
        self.make = make
        self.model = model
        self.year = year
        # Composition: Car has an Engine
        self.engine = Engine(engine_hp)
    
    def start(self):
        return f"{self.make} {self.model}: {self.engine.start()}"
    
    def stop(self):
        return f"{self.make} {self.model}: {self.engine.stop()}"

car = Car("Toyota", "Corolla", 2022, 170)
print(car.start())  # "Toyota Corolla: Engine started"
```

5. **Inheritance**: Initializing parent class attributes.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def describe(self):
        return f"{self.name}, {self.age} years old"

class Employee(Person):
    def __init__(self, name, age, employee_id, role):
        # Initialize the parent class
        super().__init__(name, age)
        
        # Add employee-specific attributes
        self.employee_id = employee_id
        self.role = role
    
    def describe(self):
        return f"{super().describe()}, {self.role} (ID: {self.employee_id})"

employee = Employee("Alice", 30, "E12345", "Software Engineer")
print(employee.describe())  # "Alice, 30 years old, Software Engineer (ID: E12345)"
```

### Combining `__new__` and `__init__` in Practice

When both methods are used together, it's crucial to understand their interactions:

```python
class LoggedClass:
    instances = 0
    
    def __new__(cls, *args, **kwargs):
        # Increment the instance counter
        cls.instances += 1
        instance = super().__new__(cls)
        
        # Log creation
        print(f"Creating instance #{cls.instances} of {cls.__name__}")
        instance._instance_number = cls.instances
        
        return instance
    
    def __init__(self, name):
        # Log initialization
        print(f"Initializing instance #{self._instance_number} with name: {name}")
        self.name = name

# Create instances
obj1 = LoggedClass("First")
# Output:
# Creating instance #1 of LoggedClass
# Initializing instance #1 with name: First

obj2 = LoggedClass("Second")
# Output:
# Creating instance #2 of LoggedClass
# Initializing instance #2 with name: Second
```

### Important Considerations

1. **Return Value of `__new__`**:
   - If `__new__` returns an instance of the class being constructed, `__init__` will be called on that instance.
   - If `__new__` returns an instance of a different class, `__init__` will not be called on it.

```python
class A:
    def __init__(self):
        print("A.__init__ called")
        self.value = "A"

class B:
    def __new__(cls, *args, **kwargs):
        print("B.__new__ called")
        # Return an instance of class A instead of B
        return A()
    
    def __init__(self):
        print("B.__init__ called")
        self.value = "B"

# Create an instance of B
obj = B()
# Output:
# B.__new__ called
# A.__init__ called

print(type(obj))  # <class '__main__.A'>
print(obj.value)  # "A"
```

2. **Inheritance**:
   - When overriding `__new__`, you typically need to call the parent class's `__new__` method.
   - When overriding `__init__`, you typically need to call the parent class's `__init__` method.

```python
class Parent:
    def __new__(cls, *args, **kwargs):
        print("Parent.__new__ called")
        return super().__new__(cls)
    
    def __init__(self, value):
        print("Parent.__init__ called")
        self.value = value

class Child(Parent):
    def __new__(cls, *args, **kwargs):
        print("Child.__new__ called")
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, value, extra):
        print("Child.__init__ called")
        super().__init__(value)
        self.extra = extra

# Create an instance
obj = Child("value", "extra")
# Output:
# Child.__new__ called
# Parent.__new__ called
# Child.__init__ called
# Parent.__init__ called
```

3. **Performance Implications**:
   - `__new__` is called before the object exists, so it can be more efficient for validating arguments before creating an object.
   - However, overriding `__new__` for simple use cases can add unnecessary complexity.

### Data Engineering Examples

#### Example 1: Cached Data Processor with Singleton Pattern

```python
class DataProcessor:
    _instances = {}
    
    def __new__(cls, dataset_id, *args, **kwargs):
        # Singleton per dataset_id
        if dataset_id not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[dataset_id] = instance
            # Flag for __init__ to know this is a new instance
            instance._is_new = True
        else:
            # Flag for __init__ to know this is an existing instance
            cls._instances[dataset_id]._is_new = False
        
        return cls._instances[dataset_id]
    
    def __init__(self, dataset_id, raw_data=None):
        # Only initialize once for each dataset_id
        if hasattr(self, '_is_new') and self._is_new:
            self.dataset_id = dataset_id
            self.raw_data = raw_data
            self.processed_data = None
            self.last_processed = None
            self._is_new = False  # Reset flag
    
    def process(self):
        """Process the raw data and cache the results."""
        if self.processed_data is None:
            print(f"Processing dataset {self.dataset_id}...")
            # Simulate data processing
            self.processed_data = f"Processed {self.raw_data}"
            self.last_processed = "2023-05-15"
        else:
            print(f"Using cached results for dataset {self.dataset_id}")
        
        return self.processed_data

# Example usage
dp1 = DataProcessor("dataset1", "raw data 1")
result1 = dp1.process()  # Processes the data
print(result1)  # "Processed raw data 1"

# Same dataset_id returns the same instance
dp2 = DataProcessor("dataset1", "new raw data")  # raw_data parameter is ignored
result2 = dp2.process()  # Uses cached results
print(result2)  # "Processed raw data 1"
print(dp1 is dp2)  # True

# Different dataset_id creates a new instance
dp3 = DataProcessor("dataset2", "raw data 2")
result3 = dp3.process()  # Processes the data
print(result3)  # "Processed raw data 2"
print(dp1 is dp3)  # False
```

#### Example 2: Immutable Dataset Class

```python
class ImmutableDataset:
    """
    An immutable dataset class that uses __new__ to enforce immutability.
    """
    def __new__(cls, name, data, metadata=None):
        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Dataset name must be a non-empty string")
        
        if not isinstance(data, (list, tuple)):
            raise ValueError("Data must be a list or tuple")
        
        # Create and configure the instance
        instance = super().__new__(cls)
        object.__setattr__(instance, '_name', name)
        object.__setattr__(instance, '_data', tuple(data))  # Ensure data is immutable
        object.__setattr__(instance, '_metadata', metadata)
        
        return instance
    
    # No __init__ method needed
    
    def __setattr__(self, name, value):
        # Prevent attribute modification
        raise AttributeError("Cannot modify attributes of an immutable dataset")
    
    @property
    def name(self):
        return self._name
    
    @property
    def data(self):
        return self._data
    
    @property
    def metadata(self):
        return self._metadata
    
    def __repr__(self):
        return f"ImmutableDataset(name='{self._name}', data={self._data}, metadata={self._metadata})"
    
    def filter(self, condition):
        """
        Returns a new dataset with filtered data.
        """
        filtered_data = [item for item in self._data if condition(item)]
        return ImmutableDataset(f"{self._name}_filtered", filtered_data, self._metadata)
    
    def map(self, func):
        """
        Returns a new dataset with transformed data.
        """
        transformed_data = [func(item) for item in self._data]
        return ImmutableDataset(f"{self._name}_mapped", transformed_data, self._metadata)

# Create an immutable dataset
dataset = ImmutableDataset("temperatures", [20, 22, 25, 18, 23], {"unit": "Celsius"})

# Get properties
print(dataset.name)  # "temperatures"
print(dataset.data)  # (20, 22, 25, 18, 23)

# Create derived datasets
filtered = dataset.filter(lambda x: x > 20)
print(filtered.data)  # (22, 25, 23)

transformed = dataset.map(lambda x: (x * 9/5) + 32)  # Convert to Fahrenheit
print(transformed.data)  # (68.0, 71.6, 77.0, 64.4, 73.4)

# Attempt to modify (will raise an error)
try:
    dataset.name = "new_name"
except AttributeError as e:
    print(e)  # Cannot modify attributes of an immutable dataset
```

#### Example 3: Factory Method for Data Connectors

```python
class DataConnector:
    """Base class for data connectors."""
    
    def __new__(cls, connector_type, *args, **kwargs):
        # Factory pattern: return different connector types based on connector_type
        if connector_type == 'csv':
            return CSVConnector(*args, **kwargs)
        elif connector_type == 'database':
            return DatabaseConnector(*args, **kwargs)
        elif connector_type == 'api':
            return APIConnector(*args, **kwargs)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")
    
    def connect(self):
        raise NotImplementedError("Subclasses must implement connect()")
    
    def fetch_data(self):
        raise NotImplementedError("Subclasses must implement fetch_data()")
    
    def close(self):
        raise NotImplementedError("Subclasses must implement close()")

class CSVConnector:
    def __init__(self, file_path, delimiter=','):
        self.file_path = file_path
        self.delimiter = delimiter
        self.file = None
    
    def connect(self):
        print(f"Opening CSV file: {self.file_path}")
        # In a real implementation, this would open the file
        self.file = f"Connection to {self.file_path}"
        return True
    
    def fetch_data(self):
        if not self.file:
            self.connect()
        return f"Data from CSV file {self.file_path}"
    
    def close(self):
        if self.file:
            print(f"Closing CSV file: {self.file_path}")
            self.file = None

class DatabaseConnector:
    def __init__(self, connection_string, query):
        self.connection_string = connection_string
        self.query = query
        self.connection = None
    
    def connect(self):
        print(f"Connecting to database: {self.connection_string}")
        # In a real implementation, this would establish a database connection
        self.connection = f"Connection to {self.connection_string}"
        return True
    
    def fetch_data(self):
        if not self.connection:
            self.connect()
        return f"Data from database query: {self.query}"
    
    def close(self):
        if self.connection:
            print(f"Closing database connection: {self.connection_string}")
            self.connection = None

class APIConnector:
    def __init__(self, api_url, api_key=None):
        self.api_url = api_url
        self.api_key = api_key
        self.session = None
    
    def connect(self):
        print(f"Connecting to API: {self.api_url}")
        # In a real implementation, this would create an API session
        self.session = f"Session to {self.api_url}"
        return True
    
    def fetch_data(self):
        if not self.session:
            self.connect()
        return f"Data from API: {self.api_url}"
    
    def close(self):
        if self.session:
            print(f"Closing API session: {self.api_url}")
            self.session = None

# Create connectors using the factory
csv_connector = DataConnector('csv', 'data.csv')
db_connector = DataConnector('database', 'postgresql://localhost/mydb', 'SELECT * FROM users')
api_connector = DataConnector('api', 'https://api.example.com/data', api_key='abc123')

# They are actually instances of the specific connector classes
print(type(csv_connector))  # <class '__main__.CSVConnector'>
print(type(db_connector))   # <class '__main__.DatabaseConnector'>
print(type(api_connector))  # <class '__main__.APIConnector'>

# Use them in a consistent way
for connector in [csv_connector, db_connector, api_connector]:
    data = connector.fetch_data()
    print(data)
    connector.close()
```

### Best Practices

1. **Use `__init__` for Most Cases**:
   - For most classes, overriding only `__init__` is sufficient.
   - Use `__new__` only when you need to control the instance creation process.

2. **Keep `__new__` Simple**:
   - When overriding `__new__`, focus on instance creation and avoid complex logic.
   - Perform most instance configuration in `__init__`.

3. **Remember to Call the Parent Methods**:
   - Always call `super().__new__(cls)` in your `__new__` method.
   - Always call `super().__init__(...)` in your `__init__` method when inheriting.

4. **Be Careful with Return Values**:
   - `__new__` must return an instance, typically of the class being constructed.
   - `__init__` must return `None`.

5. **Document Special Behavior**:
   - If your class has custom `__new__` behavior, document it clearly to avoid confusion.

## 19. How do you implement a singleton pattern in Python?

### Understanding the Singleton Pattern

The Singleton pattern is a design pattern that restricts the instantiation of a class to a single instance. This is useful when exactly one object is needed to coordinate actions across the system, such as a configuration manager, connection pool, or logger.

In Python, there are several ways to implement the Singleton pattern, each with its own advantages and trade-offs.

### Method 1: Using a Module

The simplest way to implement a Singleton in Python is to use a module. Python modules are imported only once, so any variables or functions defined in the module act as singletons.

**File: `singleton_module.py`**:
```python
# This module itself is a singleton

# Singleton state
_config = {}

# Singleton methods
def get_config():
    return _config

def set_config(key, value):
    _config[key] = value

def clear_config():
    _config.clear()
```

**Usage**:
```python
# In file1.py
import singleton_module

singleton_module.set_config('server', 'localhost')
singleton_module.set_config('port', 8080)

# In file2.py (even in a different part of the application)
import singleton_module

print(singleton_module.get_config())  # {'server': 'localhost', 'port': 8080}
```

**Advantages**:
- Simple and Pythonic
- No boilerplate code
- Thread-safe (module imports are thread-safe in Python)

**Disadvantages**:
- Limited to simple use cases
- No inheritance or instance methods
- Cannot be easily customized for different contexts

### Method 2: Using a Class with a Class Variable

This approach maintains a single instance as a class variable and returns it whenever the class is instantiated.

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Put initialization here
        return cls._instance
    
    def __init__(self, value=None):
        # Note: __init__ is called every time, even when returning the existing instance
        # Only set the value if it's provided
        if value is not None:
            self.value = value

# Usage
s1 = Singleton("first")
s2 = Singleton("second")

print(s1 is s2)      # True (same instance)
print(s1.value)      # "second" (the attribute was overwritten)
```

**Advantages**:
- Simple and easy to understand
- Works with inheritance
- Instance can have methods and properties

**Disadvantages**:
- `__init__` is called on every instantiation, which might be confusing
- Not thread-safe by default
- All subclasses share the same instance

### Method 3: Using a Metaclass

A more elegant approach is to use a metaclass, which controls the class creation process:

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self, value):
        self.value = value

# Usage
s1 = Singleton("first")
s2 = Singleton("second")

print(s1 is s2)      # True (same instance)
print(s1.value)      # "second" (the attribute was overwritten)
```

**Advantages**:
- Cleaner implementation (separation of concerns)
- Each subclass has its own instance
- More flexible and customizable

**Disadvantages**:
- More complex (uses metaclasses)
- `__init__` is still called on every instantiation
- Not thread-safe by default

### Method 4: Using a Decorator

A decorator can be used to transform any class into a singleton:

```python
def singleton(cls):
    """Decorator that makes a class a Singleton."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Config:
    def __init__(self, filename=None):
        self.filename = filename
        self.data = {}
        
        if filename:
            self.load(filename)
    
    def load(self, filename):
        # In a real implementation, this would load from a file
        self.data = {'key': 'value'}
        print(f"Loaded config from {filename}")
    
    def get(self, key, default=None):
        return self.data.get(key, default)

# Usage
config1 = Config("config.json")  # Loads the config
config2 = Config()               # Returns the existing instance, doesn't reload

print(config1 is config2)        # True
print(config1.get("key"))        # "value"
```

**Advantages**:
- Can be applied to any class without modifying its code
- Clear separation between singleton behavior and class functionality
- Easy to understand and use

**Disadvantages**:
- Changes the class behavior in a way that might not be obvious
- Not ideal for inheritance
- Not thread-safe by default

### Method 5: Thread-Safe Singleton

For multithreaded applications, a thread-safe implementation is necessary:

```python
import threading

class Singleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, value=None):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.value = value
            self.initialized = True

# Usage in multithreaded environment
def create_singleton(value):
    s = Singleton(value)
    print(f"Thread created singleton with value: {s.value}")

threads = []
for i in range(5):
    t = threading.Thread(target=create_singleton, args=(f"value-{i}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# All threads use the same instance, only the first initialization counts
s = Singleton()
print(f"Final singleton value: {s.value}")  # "value-0" (first thread to acquire the lock)
```

**Advantages**:
- Thread-safe singleton creation
- Ensures only one instance exists even with concurrent access
- Initialization happens only once

**Disadvantages**:
- More complex implementation
- Lock acquisition may impact performance
- Requires careful handling of initialization

### Method 6: Borg Pattern (Monostate)

The Borg pattern (also known as Monostate) is an alternative to the Singleton. Instead of having a single instance, it allows multiple instances that share the same state:

```python
class Borg:
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state

class Config(Borg):
    def __init__(self, settings=None):
        Borg.__init__(self)
        
        if settings and not hasattr(self, 'settings'):
            self.settings = settings
    
    def get_setting(self, key, default=None):
        return self.settings.get(key, default)

# Usage
config1 = Config({'debug': True, 'log_level': 'INFO'})
config2 = Config()  # No settings provided, but shares state with config1

print(config1 is config2)  # False (different instances)
print(config1.settings is config2.settings)  # True (shared state)
print(config2.get_setting('debug'))  # True
```

**Advantages**:
- More flexible than traditional singleton
- Allows multiple instances with shared state
- Easier to test and extend

**Disadvantages**:
- Conceptually more complex
- Not a true singleton (multiple instances)
- State sharing might lead to unexpected behavior

### Method 7: Using `__new__` with Instance Flag

A variant that avoids re-running `__init__` on subsequent calls:

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, value=None):
        # Only initialize once
        if not self._initialized:
            self.value = value
            self._initialized = True

# Usage
s1 = Singleton("first")
s2 = Singleton("second")  # Won't change the value

print(s1 is s2)      # True (same instance)
print(s1.value)      # "first" (initialization only happened once)# Answers to Python Interview Questions 16-20

## 16. Explain class inheritance and method overriding in Python.

### Class Inheritance in Python

Inheritance is a fundamental concept in object-oriented programming that allows a class (subclass) to inherit attributes and methods from another class (superclass). This promotes code reuse and establishes a hierarchical relationship between classes.

#### Basic Inheritance

```python
# Base class (superclass)
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def eat(self):
        return f"{self.name} is eating"

# Derived class (subclass)
class Dog(Animal):
    def speak(self):
        return "Woof!"
    
    def fetch(self):
        return f"{self.name} is fetching"

# Create instances
animal = Animal("Generic Animal")
dog = Dog("Buddy")

print(animal.speak())  # Some sound
print(dog.speak())     # Woof!
print(dog.eat())       # Buddy is eating (inherited from Animal)
print(dog.fetch())     # Buddy is fetching (specific to Dog)
```

In this example:
- `Animal` is the superclass (base class)
- `Dog` is the subclass (derived class)
- `Dog` inherits the `__init__` and `eat` methods from `Animal`
- `Dog` has its own specific method `fetch`
- `Dog` overrides the `speak` method from `Animal`

#### Types of Inheritance

Python supports several types of inheritance:

1. **Single Inheritance**: A subclass inherits from a single superclass (as shown above).

2. **Multiple Inheritance**: A subclass inherits from multiple superclasses.

```python
class Swimmer:
    def swim(self):
        return "Swimming"

class Flyer:
    def fly(self):
        return "Flying"

class Duck(Swimmer, Flyer):
    def speak(self):
        return "Quack!"

duck = Duck()
print(duck.swim())  # Swimming
print(duck.fly())   # Flying
print(duck.speak()) # Quack!
```

3. **Multilevel Inheritance**: A subclass inherits from a class which itself inherits from another class.

```python
class Animal:
    def breathe(self):
        return "Breathing"

class Mammal(Animal):
    def feed_young(self):
        return "Feeding young with milk"

class Dog(Mammal):
    def speak(self):
        return "Woof!"

dog = Dog()
print(dog.breathe())     # Breathing (from Animal)
print(dog.feed_young())  # Feeding young with milk (from Mammal)
print(dog.speak())       # Woof! (from Dog)
```

4. **Hierarchical Inheritance**: Multiple subclasses inherit from a single superclass.

```python
class Animal:
    def breathe(self):
        return "Breathing"

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog()
cat = Cat()
print(dog.breathe())  # Breathing
print(cat.breathe())  # Breathing
print(dog.speak())    # Woof!
print(cat.speak())    # Meow!
```

### Method Resolution Order (MRO)

When dealing with multiple inheritance, Python uses the C3 linearization algorithm to determine the order in which methods are resolved (Method Resolution Order or MRO).

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # B (follows MRO)
print(D.__mro__)   # (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

The MRO determines that `B.method()` will be called before `C.method()` because `B` is listed first in the inheritance list.

### Method Overriding

Method overriding occurs when a subclass provides a specific implementation for a method already defined in its superclass. This allows the subclass to change the behavior inherited from the superclass.

#### Basic Method Overriding

```python
class Shape:
    def area(self):
        return "Area calculation not implemented"
    
    def perimeter(self):
        return "Perimeter calculation not implemented"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):  # Override area method
        return self.width * self.height
    
    def perimeter(self):  # Override perimeter method
        return 2 * (self.width + self.height)

# Using the overridden methods
shape = Shape()
rectangle = Rectangle(5, 3)

print(shape.area())       # Area calculation not implemented
print(rectangle.area())   # 15
print(rectangle.perimeter())  # 16
```

#### Accessing Overridden Methods Using `super()`

The `super()` function allows you to call methods from the parent class. This is particularly useful when you want to extend rather than completely replace the parent's functionality.

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def make_sound(self):
        return "Some generic sound"

class Dog(Animal):
    def __init__(self, name, breed):
        # Call the parent's __init__ method
        super().__init__(name)
        self.breed = breed
    
    def make_sound(self):
        # Extend the parent's method behavior
        parent_sound = super().make_sound()
        return f"{parent_sound} and a Woof as I'm a {self.breed}"

# Create an instance
dog = Dog("Rex", "German Shepherd")
print(dog.name)         # Rex
print(dog.breed)        # German Shepherd
print(dog.make_sound()) # Some generic sound and a Woof as I'm a German Shepherd
```

#### The Diamond Problem and `super()`

When dealing with multiple inheritance, the diamond problem can occur (a class inherits from two classes that both inherit from a common base class). `super()` helps resolve this by following the MRO:

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return f"B, then {super().method()}"

class C(A):
    def method(self):
        return f"C, then {super().method()}"

class D(B, C):
    def method(self):
        return f"D, then {super().method()}"

d = D()
print(d.method())  # D, then B, then C, then A
print(D.__mro__)   # (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

This demonstrates how `super()` follows the MRO, calling each method in the inheritance chain.

### Abstract Base Classes

Python provides abstract base classes through the `abc` module, which allows you to define methods that must be implemented by subclasses.

```python
from abc import ABC, abstractmethod

class DatabaseConnector(ABC):
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def execute_query(self, query):
        pass
    
    def close(self):
        print("Closing connection")

class MySQLConnector(DatabaseConnector):
    def connect(self):
        return "Connected to MySQL"
    
    def execute_query(self, query):
        return f"Executing MySQL query: {query}"

# Cannot instantiate abstract class
try:
    db = DatabaseConnector()  # TypeError: Can't instantiate abstract class
except TypeError as e:
    print(e)

# Can instantiate concrete subclass
mysql = MySQLConnector()
print(mysql.connect())           # Connected to MySQL
print(mysql.execute_query("SELECT * FROM users"))  # Executing MySQL query: SELECT * FROM users
mysql.close()                    # Closing connection
```

### Practical Examples in Data Engineering

#### Example 1: Data Processor Hierarchy

```python
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        # Base implementation
        return self.data
    
    def validate(self):
        # Basic validation
        return len(self.data) > 0

class CSVProcessor(DataProcessor):
    def process(self):
        # CSV-specific processing
        processed_data = super().process()
        
        # Convert list of lists to list of dicts
        headers = processed_data[0]
        return [dict(zip(headers, row)) for row in processed_data[1:]]
    
    def validate(self):
        # First check basic validation
        if not super().validate():
            return False
        
        # CSV-specific validation
        return all(len(row) == len(self.data[0]) for row in self.data)

class JSONProcessor(DataProcessor):
    def process(self):
        # JSON-specific processing
        processed_data = super().process()
        
        # Perform additional transformations
        for item in processed_data:
            if 'date' in item:
                item['date'] = self._parse_date(item['date'])
        
        return processed_data
    
    def _parse_date(self, date_str):
        # Helper method specific to JSONProcessor
        return datetime.strptime(date_str, "%Y-%m-%d")
```

#### Example 2: ETL Pipeline Components

```python
class ETLComponent:
    def __init__(self, name):
        self.name = name
        self.status = "Ready"
    
    def execute(self, data):
        self.status = "Running"
        result = self._process(data)
        self.status = "Completed"
        return result
    
    def _process(self, data):
        # Default implementation
        return data
    
    def get_status(self):
        return f"{self.name}: {self.status}"

class Extractor(ETLComponent):
    def __init__(self, source, name="Extractor"):
        super().__init__(name)
        self.source = source
    
    def _process(self, data=None):
        # Override _process to extract data from source
        if self.source == "database":
            return "Data from database"
        elif self.source == "api":
            return "Data from API"
        else:
            return "Data from unknown source"

class Transformer(ETLComponent):
    def __init__(self, transformation_type, name="Transformer"):
        super().__init__(name)
        self.transformation_type = transformation_type
    
    def _process(self, data):
        # Override _process to transform data
        if self.transformation_type == "clean":
            return f"Cleaned: {data}"
        elif self.transformation_type == "enrich":
            return f"Enriched: {data}"
        else:
            return f"Unknown transformation: {data}"

class Loader(ETLComponent):
    def __init__(self, destination, name="Loader"):
        super().__init__(name)
        self.destination = destination
    
    def _process(self, data):
        # Override _process to load data
        return f"Loaded to {self.destination}: {data}"

# Using the ETL components
extractor = Extractor("database")
transformer = Transformer("clean")
loader = Loader("data_warehouse")

# Execute ETL pipeline
data = extractor.execute()
transformed_data = transformer.execute(data)
result = loader.execute(transformed_data)

print(result)  # Loaded to data_warehouse: Cleaned: Data from database
```

### Best Practices for Inheritance and Method Overriding

1. **Favor composition over inheritance**: Sometimes it's better to use composition (having an instance of another class as an attribute) rather than inheritance.

2. **Keep inheritance hierarchies shallow**: Deep inheritance hierarchies can become complex and difficult to understand.

3. **Follow the Liskov Substitution Principle**: Subclasses should be substitutable for their base classes without altering the correctness of the program.

4. **Use `super()` when overriding methods**: This ensures that parent class functionality is properly maintained.

5. **Document class hierarchies**: Clear documentation helps others understand the relationships between classes.

6. **Use abstract base classes** to define required interfaces for subclasses.

7. **Be cautious with multiple inheritance**: It can lead to complex behavior if not used carefully.

## 17. What are metaclasses in Python and when would you use them?

### Understanding Metaclasses

In Python, everything is an object, including classes themselves. A class is an instance of a metaclass, and by default, Python uses the built-in `type` metaclass to create classes. Metaclasses are "classes of classes" that define how classes behave.

To understand metaclasses, we need to first understand that:
1. Classes are objects
2. Classes are instances of the `type` metaclass
3. You can create classes dynamically using `type`

### How Classes are Created

When you define a class in Python, the interpreter uses a metaclass to create it:

```python
class MyClass:
    x = 1
    
    def hello(self):
        return "Hello, World!"
```

Behind the scenes, Python uses `type` (the default metaclass) to create `MyClass`:

```python
# Equivalent to the class definition above
MyClass = type('MyClass', (), {
    'x': 1,
    'hello': lambda self: "Hello, World!"
})
```

Here, `type` takes three arguments:
- The name of the class
- A tuple of parent classes (empty for base classes)
- A dictionary of attributes and methods

### Creating a Custom Metaclass

A metaclass is typically created by inheriting from `type` and overriding its methods, particularly `__new__` or `__init__`:

```python
class Meta(type):
    def __new__(cls, name, bases, attrs):
        # Print attributes before class creation
        print(f"Creating class {name} with attributes: {attrs.keys()}")
        
        # Add a new attribute to the class
        attrs['added_by_metaclass'] = True
        
        # Create and return the new class
        return super().__new__(cls, name, bases, attrs)

# Use the metaclass to create a new class
class MyClass(metaclass=Meta):
    x = 1
    
    def hello(self):
        return "Hello, World!"

# The metaclass has added an attribute
print(MyClass.added_by_metaclass)  # True
```

### Key Methods in Metaclasses

1. **`__new__(cls, name, bases, attrs)`**: Called before class creation to customize the class object before it's created.

2. **`__init__(cls, name, bases, attrs)`**: Called after class creation to initialize the class object.

3. **`__call__(cls, *args, **kwargs)`**: Called when the class is called to create an instance (e.g., `obj = MyClass()`).

### When to Use Metaclasses

Metaclasses are a powerful but advanced feature. They're useful in specific scenarios:

#### 1. Registering Classes in a Registry

```python
class PluginRegistry(type):
    plugins = {}
    
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        
        # Don't register the base Plugin class
        if name != 'Plugin':
            # Register the plugin class by its name
            cls.plugins[name] = new_class
        
        return new_class

class Plugin(metaclass=PluginRegistry):
    """Base class for plugins."""
    
    def run(self):
        raise NotImplementedError("Plugins must implement the run method")

# Define some plugins
class TextPlugin(Plugin):
    def run(self):
        return "Processing text"

class ImagePlugin(Plugin):
    def run(self):
        return "Processing image"

# Access the plugin registry
print(PluginRegistry.plugins)
# {'TextPlugin': <class '__main__.TextPlugin'>, 'ImagePlugin': <class '__main__.ImagePlugin'>}

# Create and run a plugin by name
plugin_name = "TextPlugin"
plugin_class = PluginRegistry.plugins[plugin_name]
plugin = plugin_class()
print(plugin.run())  # Processing text
```

#### 2. Enforcing Coding Standards or Design Patterns

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self, connection_string):
        self.connection_string = connection_string
        print(f"Connecting to database with {connection_string}")

# Create instances
db1 = DatabaseConnection("postgresql://localhost/db")
db2 = DatabaseConnection("postgresql://localhost/db")

# Both variables refer to the same instance
print(db1 is db2)  # True
```

#### 3. Automatically Adding Methods or Properties

```python
class AutoProperty(type):
    def __new__(cls, name, bases, attrs):
        # For each attribute that starts with "_x_"
        for key, value in list(attrs.items()):
            if key.startswith('_x_'):
                # Create a property name by removing the "_x_" prefix
                property_name = key[3:]
                
                # Define getter and setter methods
                def getter(self, key=key):
                    return getattr(self, key)
                
                def setter(self, value, key=key):
                    setattr(self, key, value)
                
                # Add the property to the class attributes
                attrs[property_name] = property(getter, setter)
        
        return super().__new__(cls, name, bases, attrs)

class Person(metaclass=AutoProperty):
    def __init__(self, name, age):
        self._x_name = name
        self._x_age = age

# Test the auto-property feature
person = Person("Alice", 30)
print(person.name)  # Alice (accessing via property)
print(person.age)   # 30

# Update via property
person.name = "Bob"
print(person._x_name)  # Bob (underlying attribute was updated)
```

#### 4. Code Generation or Aspect-Oriented Programming

```python
class LoggingMeta(type):
    def __new__(cls, name, bases, attrs):
        # For each method (callable attribute)
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and not attr_name.startswith('__'):
                # Replace method with a logging wrapper
                attrs[attr_name] = cls.add_logging(attr_name, attr_value)
        
        return super().__new__(cls, name, bases, attrs)
    
    @staticmethod
    def add_logging(name, method):
        def wrapper(self, *args, **kwargs):
            print(f"Calling method {name} with args: {args}, kwargs: {kwargs}")
            result = method(self, *args, **kwargs)
            print(f"Method {name} returned: {result}")
            return result
        return wrapper

class Service(metaclass=LoggingMeta):
    def process_data(self, data):
        return f"Processed: {data}"
    
    def calculate(self, x, y):
        return x + y

# All methods automatically log their calls and returns
service = Service()
service.process_data("some data")
# Outputs:
# Calling method process_data with args: ('some data',), kwargs: {}
# Method process_data returned: Processed: some data

service.calculate(5, 3)
# Outputs:
# Calling method calculate with args: (5, 3), kwargs: {}
# Method calculate returned: 8
```

#### 5. ORM (Object-Relational Mapping) Systems

Many ORM systems like Django's models or SQLAlchemy use metaclasses to transform class definitions into database operations.

```python
class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        # Skip the base Model class
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        
        # Extract field definitions
        fields = {}
        for key, value in list(attrs.items()):
            if isinstance(value, Field):
                fields[key] = value
        
        # Add fields to the class attrs
        attrs['_fields'] = fields
        
        # Create a table name from the class name
        attrs['_table'] = name.lower()
        
        # Add convenience methods
        def get_create_table_sql(cls):
            sql = [f"CREATE TABLE {cls._table} ("]
            for name, field in cls._fields.items():
                sql.append(f"  {name} {field.sql_type},")
            sql[-1] = sql[-1][:-1]  # Remove the last comma
            sql.append(");")
            return "\n".join(sql)
        
        attrs['get_create_table_sql'] = classmethod(get_create_table_sql)
        
        return super().__new__(cls, name, bases, attrs)

class Field:
    def __init__(self, sql_type):
        self.sql_type = sql_type

class Model(metaclass=ModelMeta):
    """Base class for all database models."""
    pass

# Define a model class
class User(Model):
    id = Field("INTEGER PRIMARY KEY")
    name = Field("TEXT")
    email = Field("TEXT")
    age = Field("INTEGER")

# The metaclass has added methods and attributes
print(User._table)  # user
print(User._fields)  # {'id': <__main__.Field object at ...>, 'name': <__main__.Field object at ...>, ...}

# Generate SQL
print(User.get_create_table_sql())
# CREATE TABLE user (
#   id INTEGER PRIMARY KEY,
#   name TEXT,
#   email TEXT,
#   age INTEGER
# );
```

### Metaclasses vs. Class Decorators

In many cases, class decorators can achieve similar results with simpler code:

```python
def add_repr(cls):
    """Class decorator to add a __repr__ method."""
    def __repr__(self):
        attributes = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{cls.__name__}({attributes})"
    
    cls.__repr__ = __repr__
    return cls

@add_repr
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)
print(person)  # Person(name='Alice', age=30)
```

### When to Use Metaclasses vs. Alternatives

1. **Use metaclasses when**:
   - You need to modify the class creation process itself
   - You want to intercept class definitions before they're created
   - You're building a framework or library where users will define many similar classes

2. **Consider alternatives when**:
   - You only need to modify existing classes (class decorators)
   - You need to modify instances (decorators or mixins)
   - Composition might solve the problem more clearly

### Data Engineering Examples

#### Example 1: Schema Validation for Data Processing

```python
class SchemaValidatorMeta(type):
    def __new__(cls, name, bases, attrs):
        # Skip base class
        if name == 'DataProcessor':
            return super().__new__(cls, name, bases, attrs)
        
        # Ensure schema is defined
        if 'schema' not in attrs:
            raise TypeError(f"{name} must define a 'schema' attribute")
        
        # Add validation method
        def validate_data(self, data):
            """Validate data against the schema."""
            errors = []
            for record in data:
                for field, field_type in self.schema.items():
                    if field not in record:
                        errors.append(f"Missing field '{field}'")
                    elif not isinstance(record[field], field_type):
                        errors.append(f"Field '{field}' has type {type(record[field]).__name__}, expected {field_type.__name__}")
            return errors
        
        attrs['validate_data'] = validate_data
        
        return super().__new__(cls, name, bases, attrs)

class DataProcessor(metaclass=SchemaValidatorMeta):
    """Base class for data processors."""
    
    def process(self, data):
        """Process data after validation."""
        errors = self.validate_data(data)
        if errors:
            raise ValueError(f"Data validation failed: {errors}")
        return self._process_impl(data)
    
    def _process_impl(self, data):
        """Implementation to be provided by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_impl")

# User-defined data processor
class UserProcessor(DataProcessor):
    schema = {
        'id': int,
        'name': str,
        'email': str
    }
    
    def _process_impl(self, data):
        return [{'id': record['id'], 'name': record['name'].upper()} for record in data]

# Valid data
valid_data = [
    {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
    {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
]

# Invalid data
invalid_data = [
    {'id': '1', 'name': 'Alice', 'email': 'alice@example.com'},  # id should be int
    {'id': 2, 'name': 'Bob'}  # missing email
]

processor = UserProcessor()
try:
    result = processor.process(valid_data)
    print(result)  # [{'id': 1, 'name': 'ALICE'}, {'id': 2, 'name': 'BOB'}]
    
    processor.process(invalid_data)  # Raises ValueError
except ValueError as e:
    print(e)  # Data validation failed: [...]
```

#### Example 2: Automatic Data Pipeline Component Registration

```python
class PipelineComponentMeta(type):
    components = {}
    
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        
        # Register component by name
        if name != 'PipelineComponent':
            # Extract component name and version from class
            component_name = attrs.get('component_name', name)
            component_version = attrs.get('component_version', '1.0')
            key = f"{component_name}-{component_version}"
            
            cls.components[key] = new_class
            print(f"Registered pipeline component: {key}")
        
        return new_class

class PipelineComponent(metaclass=PipelineComponentMeta):
    """Base class for pipeline components."""
    
    def process(self, data):
        """Process the data."""
        raise NotImplementedError("Subclasses must implement process")

# Define specialized components
class DataCleaner(PipelineComponent):
    component_name = "cleaner"
    component_version = "1.0"
    
    def process(self, data):
        return f"Cleaned: {data}"

class DataTransformer(PipelineComponent):
    component_name = "transformer"
    component_version = "2.0"
    
    def process(self, data):
        return f"Transformed: {data}"

# Create a pipeline using the registry
def create_pipeline(component_specs):
    """Create a pipeline from component specifications."""
    pipeline = []
    for spec in component_specs:
        key = f"{spec['name']}-{spec['version']}"
        if key in PipelineComponentMeta.components:
            component_class = PipelineComponentMeta.components[key]
            pipeline.append(component_class())
        else:
            raise ValueError(f"Unknown component: {key}")
    return pipeline

# Define a pipeline
pipeline_spec = [
    {"name": "cleaner", "version": "1.0"},
    {"name": "transformer", "version": "2.0"}
]

# Create and run the pipeline
pipeline = create_pipeline(pipeline_spec)
data = "raw data"

for component in pipeline:
    data = component.process(data)

print(data)  # Transformed: Cleaned: raw data
```

### Best Practices for Metaclasses

1. **Use sparingly**: Metaclasses are complex and can make code harder to understand.

2. **Document extensively**: If you use metaclasses, document how they work and why they're necessary.

3. **Consider alternatives**: Before using metaclasses, consider if decorators, mixins, or other approaches would work.

4. **Keep it simple**: A metaclass should have a single, clear responsibility.

5. **Test thoroughly**: Metaclasses can have subtle effects that require comprehensive testing.

## 18. Describe the difference between `__new__` and `__init__` methods.

### `__new__` vs. `__init__` in Python

In Python, object creation involves two key methods: `__new__` and `__init__`. While they both participate in the object creation process, they serve different purposes and operate at different stages.

### Overview of the Two Methods

#### `__new__` Method

`__new__` is responsible for creating and returning a new instance of the class. It is a static method (implicitly, even without the `@staticmethod` decorator) that gets called before `__init__`.

**Signature**:
```python
def __new__(cls, *args, **kwargs)
```

- `cls`: The class that is being instantiated
- `*args` and `**kwargs`: Arguments passed to the class constructor
- Returns: A new instance of the class (usually)

#### `__init__` Method

`__init__` is responsible for initializing the instance after it has been created. It doesn't create the instance; it configures it.

**Signature**:
```python
def __init__(self, *args, **kwargs)
```

- `self`: The instance that has already been created
- `*args` and `**kwargs`: Arguments passed to the class constructor
- Returns: `None` (cannot return any other value)

### Key Differences

1. **Purpose**:
   - `__new__`: Creates the instance
   - `__init__`: Initializes the created instance

2. **Return Value**:
   - `__new__`: Must return a new instance (typically of the class being constructed)
   - `__init__`: Must return `None` (any other return value raises `TypeError`)

3. **First Parameter**:
   - `__new__`: Receives the class (`cls`) as its first parameter
   - `__init__`: Receives the instance (`self`) as its first parameter

4. **When Called**:
   - `__new__`: Called first, when an object is being created
   - `__init__`: Called after `__new__`, only if `__new__` returns an instance of the proper class

5. **Inheritance Behavior**:
   - `__new__`: Not automatically called on child classes unless explicitly implemented
   - `__init__`: Automatically called on child classes through the normal method resolution order

### Basic Example

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        print(f"__new__ called with {cls}, {args}, {kwargs}")
        instance = super().__new__(cls)
        return instance
    
    def __init__(self, x, y):
        print(f"__init__ called with {self}, {x}, {y}")
        self.x = x
        self.y = y

# Creating an instance
obj = MyClass(1, 2)
# Output:
# __new__ called with <class '__main__.MyClass'>, (1, 2), {}
# __init__ called with <__main__.MyClass object at 0x...>, 1, 2



# 21-25

# Brief Answers to Python Interview Questions 21-25

## 21. How would you efficiently read and process a 10GB CSV file in Python?

**Key Concept**: Process large files in chunks to manage memory constraints.

**Solution**:
1. **Use chunking with pandas**:
   ```python
   import pandas as pd
   
   # Process file in chunks of 100,000 rows
   chunk_size = 100000
   results = []
   
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       # Process each chunk
       processed_chunk = process_chunk(chunk)
       results.append(processed_chunk)
   
   # Combine results if needed
   final_result = pd.concat(results)
   ```

2. **Use CSV reader with generators**:
   ```python
   import csv
   
   def process_csv(file_path):
       with open(file_path, 'r') as f:
           reader = csv.reader(f)
           header = next(reader)  # Get header row
           
           for row in reader:
               # Process one row at a time
               yield process_row(row, header)
   
   # Process the file
   for processed_row in process_csv('large_file.csv'):
       # Do something with each processed row
       store_or_aggregate(processed_row)
   ```

3. **Use Dask for parallel processing**:
   ```python
   import dask.dataframe as dd
   
   # Let Dask handle the chunking and parallel processing
   df = dd.read_csv('large_file.csv')
   result = df.apply(process_function, axis=1).compute()
   ```

4. **Memory optimization techniques**:
   - Select only needed columns
   - Use appropriate data types 
   - Filter early to reduce data size
   - Consider memory-mapped files for faster access

## 22. Compare Pandas and PySpark for large-scale data processing. When would you use each?

**Key Concept**: Pandas works in-memory on a single machine, while PySpark distributes processing across clusters.

**Pandas**:
- **Strengths**:
  - Easy to use with intuitive API
  - Excellent for data exploration and visualization
  - Efficient for datasets that fit in memory (typically <10GB)
  - Fast for complex operations on medium-sized data
  - Rich ecosystem with scikit-learn, matplotlib, etc.

- **Use Pandas when**:
  - Working with datasets that fit in RAM
  - Performing exploratory data analysis
  - Need for quick iteration and visualization
  - Working on a single machine
  - Processing doesn't require parallelization

**PySpark**:
- **Strengths**:
  - Distributed processing across multiple nodes
  - Handles datasets of any size (terabytes or petabytes)
  - Built-in fault tolerance
  - Seamless integration with big data ecosystem
  - Supports streaming data processing

- **Use PySpark when**:
  - Processing very large datasets (>10GB)
  - Need distributed computing resources
  - Working with data warehouse/lake
  - Need fault tolerance for long-running jobs
  - Processing streaming data in real-time

**Code Comparison**:
```python
# Pandas example
import pandas as pd

df = pd.read_csv('data.csv')
result = df.groupby('category').agg({'value': ['mean', 'sum']})

# PySpark equivalent
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("Analysis").getOrCreate()
df = spark.read.csv('data.csv', header=True, inferSchema=True)
result = df.groupby('category').agg(F.mean('value'), F.sum('value'))
```

## 23. How do you handle missing data in Pandas DataFrames?

**Key Concept**: Pandas provides various methods to detect, remove, or impute missing values.

**Detecting Missing Values**:
```python
# Check for missing values
df.isna().sum()  # Count missing values per column
df.isnull().any()  # Check if any value is missing in each column
```

**Removing Missing Values**:
```python
# Remove rows with any missing values
df_clean = df.dropna()

# Remove rows if all values are missing
df_clean = df.dropna(how='all')

# Remove columns with missing values above a threshold
df_clean = df.dropna(axis=1, thresh=len(df) * 0.7)  # At least 70% non-null
```

**Filling Missing Values**:
```python
# Fill with a specific value
df_filled = df.fillna(0)

# Fill with column mean/median
df_filled = df.fillna({
    'numeric_col': df['numeric_col'].mean(),
    'categorical_col': 'unknown'
})

# Forward fill (use previous value)
df_filled = df.fillna(method='ffill')

# Backward fill (use next value)
df_filled = df.fillna(method='bfill')
```

**Advanced Imputation**:
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation for numeric features
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

**Best Practices**:
1. Understand why data is missing before deciding how to handle it
2. Consider the impact of your imputation method on analysis
3. Document all missing data treatments
4. Consider creating "missing" indicator features
5. Use different strategies for different columns based on data type

## 24. Explain the concept of vectorization in NumPy and Pandas. Why is it important?

**Key Concept**: Vectorization replaces explicit loops with array operations, greatly improving performance.

**Vectorization Explained**:
- Vectorization is performing operations on entire arrays rather than individual elements
- It leverages optimized, compiled C code instead of Python's slower loops
- Significantly improves performance (often 10-100x faster)
- Takes advantage of CPU's SIMD instructions (Single Instruction, Multiple Data)

**NumPy Vectorization Example**:
```python
import numpy as np
import time

# Non-vectorized approach (slow)
def square_loop(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2
    return result

# Vectorized approach (fast)
def square_vectorized(arr):
    return arr ** 2

# Performance comparison
arr = np.random.rand(1000000)

start = time.time()
result1 = square_loop(arr)
print(f"Loop time: {time.time() - start:.6f} seconds")

start = time.time()
result2 = square_vectorized(arr)
print(f"Vectorized time: {time.time() - start:.6f} seconds")
```

**Pandas Vectorization**:
```python
import pandas as pd

df = pd.DataFrame({
    'A': range(1000000),
    'B': range(1000000)
})

# Slow: apply with a lambda
start = time.time()
df['C'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
print(f"Apply time: {time.time() - start:.6f} seconds")

# Fast: vectorized operation
start = time.time()
df['C'] = df['A'] + df['B']
print(f"Vectorized time: {time.time() - start:.6f} seconds")
```

**Benefits of Vectorization**:
1. **Performance**: Much faster execution
2. **Readability**: Code is often more concise
3. **Optimization**: Leverages optimized libraries (BLAS, LAPACK)
4. **Parallelization**: Operations can be automatically parallelized

**When Vectorization is Critical**:
- Large datasets processing
- Numerical computations and simulations
- Machine learning pipelines
- Real-time data processing
- Any performance-critical applications

## 25. How would you implement a custom Pandas UDF (User Defined Function)?

**Key Concept**: Custom UDFs allow you to apply your own functions to Pandas DataFrames, but vectorized operations are preferred when possible.

**Methods to Create Custom Functions in Pandas**:

1. **Using `apply()` (simplest but slower)**:
```python
import pandas as pd

def dollars_to_cents(dollar_value):
    return int(dollar_value * 100)

# Apply to a single column
df['cents'] = df['dollars'].apply(dollars_to_cents)

# Apply row-wise with multiple columns
df['full_name'] = df.apply(lambda row: f"{row['first_name']} {row['last_name']}", axis=1)
```

2. **Using NumPy vectorization (faster)**:
```python
import numpy as np

# Vectorized function
def normalize(series):
    return (series - series.mean()) / series.std()

# Apply to a column
df['normalized'] = normalize(df['value'])
```

3. **Using `pandas.Series.map()` for element-wise operations**:
```python
# Map values using a dictionary
mapping = {'A': 'Category 1', 'B': 'Category 2', 'C': 'Category 3'}
df['category_name'] = df['category_code'].map(mapping)

# Or with a function
df['uppercase'] = df['text'].map(str.upper)
```

4. **Using `transform()` for group operations**:
```python
# Calculate percentage of group sum
df['percentage'] = df.groupby('category')['value'].transform(
    lambda x: x / x.sum() * 100
)
```

5. **Using Numba for high-performance UDFs**:
```python
from numba import jit
import numpy as np

@jit(nopython=True)
def complex_calculation(array):
    result = np.zeros_like(array)
    for i in range(len(array)):
        # Some complex calculation
        result[i] = array[i] ** 2 + np.sin(array[i])
    return result

# Apply to DataFrame
df['result'] = complex_calculation(df['value'].values)
```

**Best Practices for Pandas UDFs**:
1. **Avoid UDFs when possible** - use built-in methods when available
2. **Vectorize when possible** rather than applying row-by-row
3. **Use `.values` or `.to_numpy()`** to convert to NumPy arrays for better performance
4. **Consider Numba** for computationally intensive functions
5. **Profile and benchmark** different implementations
6. **Use appropriate dtypes** to minimize memory usage and maximize speed


# 26-30


# Brief Answers to Python Interview Questions 26-30

## 26. Explain Python's exception handling mechanism with try, except, else, and finally blocks.

**Key Concept**: Python's exception handling allows graceful recovery from errors with a structured approach.

**Basic Structure**:
```python
try:
    # Code that might raise an exception
    result = risky_operation()
except ExceptionType as e:
    # Handle the specific exception
    print(f"Error: {e}")
except (AnotherError, YetAnotherError) as e:
    # Handle multiple exception types
    print(f"Another type of error: {e}")
except:
    # Handle any other exception (use sparingly)
    print("Unknown error occurred")
else:
    # Executes if no exception occurred in the try block
    print("Operation successful")
finally:
    # Always executes, regardless of whether an exception occurred
    cleanup_resources()
```

**Key Components**:
- **try**: Contains code that might raise an exception
- **except**: Catches and handles specific exception types
- **else**: Executes only if no exceptions occurred in the try block
- **finally**: Always executes, used for cleanup operations (closing files, connections)

**Practical Example**:
```python
def read_data_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
        result = process_data(data)  # Process data after file is closed
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except PermissionError:
        print(f"No permission to read {filename}.")
        return None
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
    else:
        print(f"Successfully processed {filename}")
        return result
    finally:
        print(f"Attempt to process {filename} completed")
```

**Best Practices**:
1. Catch specific exceptions, not general ones
2. Keep try blocks as small as possible
3. Use the context manager (`with` statement) for resources
4. Avoid empty except blocks
5. Consider creating custom exceptions for your applications

## 27. How would you debug memory leaks in a Python application?

**Key Concept**: Memory leaks in Python typically occur when objects remain referenced but unused. Specialized tools help identify these issues.

**Memory Profiling Tools**:

1. **Using memory_profiler**:
```python
# Install with: pip install memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_list = [i for i in range(10000000)]
    return sum(large_list)

# Run the function
memory_intensive_function()
```

2. **Using tracemalloc** (built-in since Python 3.4):
```python
import tracemalloc

# Start tracking memory
tracemalloc.start()

# Run your code
result = process_large_dataset()

# Get memory snapshot and statistics
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6}MB")
print(f"Peak memory usage: {peak / 10**6}MB")

# Get the top 10 memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

3. **Using objgraph** to visualize object references:
```python
# Install with: pip install objgraph
import objgraph

# Find objects of a particular type
objgraph.show_most_common_types()

# Show what's referencing a particular object
objgraph.show_backrefs(problematic_object, filename='backrefs.png')
```

**Common Memory Leak Causes and Solutions**:

1. **Cyclic References**:
   - Problem: Objects reference each other but aren't used
   - Solution: Use weak references (`weakref` module)
   ```python
   import weakref
   
   class Node:
       def __init__(self, value):
           self.value = value
           self.parent = None
           self.children = []
       
       def add_child(self, child):
           self.children.append(child)
           child.parent = weakref.ref(self)  # Weak reference
   ```

2. **Global Variables/Caches**:
   - Problem: Unbounded growth of global structures
   - Solution: Use bounded collections (e.g., `functools.lru_cache`)
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)  # Limits cache size
   def expensive_computation(n):
       return n ** 3
   ```

3. **Large Objects in Closures**:
   - Problem: Functions capturing large objects in their closure
   - Solution: Use function parameters instead of closure variables

4. **Forgotten File Descriptors or Connections**:
   - Problem: Not closing resources
   - Solution: Always use context managers
   ```python
   with open('file.txt', 'r') as f:
       data = f.read()  # File auto-closes after block
   ```

## 28. What tools do you use for profiling Python code?

**Key Concept**: Profilers help identify performance bottlenecks by measuring execution time and resource usage.

**Standard Library Profilers**:

1. **cProfile**: The standard Python profiler
```python
import cProfile
import pstats

# Basic profiling
cProfile.run('expensive_function()')

# More detailed profiling with output sorting
def main():
    # Your code here
    result = process_large_dataset()

# Run profiler
cProfile.run('main()', 'profile_output')

# Analyze results
stats = pstats.Stats('profile_output')
stats.strip_dirs().sort_stats('cumulative').print_stats(10)  # Show top 10 functions
```

2. **timeit**: For timing small code snippets
```python
import timeit

# Measure execution time
time_taken = timeit.timeit(
    'process_list([i for i in range(1000)])', 
    setup='from my_module import process_list',
    number=1000
)
print(f"Average execution time: {time_taken / 1000} seconds")
```

**Third-Party Profilers**:

1. **line_profiler**: For line-by-line profiling
```python
# Install with: pip install line_profiler
# Add decorator to function of interest
@profile  # Don't import this, it's injected by line_profiler
def my_function():
    # Code to profile
    pass

# Run: kernprof -l script.py
# Analyze: python -m line_profiler script.py.lprof
```

2. **py-spy**: Low-overhead sampling profiler
```bash
# Install with: pip install py-spy
# Run profiler
py-spy record -o profile.svg -- python my_script.py

# Live view
py-spy top -- python my_script.py
```

3. **Scalene**: CPU, memory, and GPU profiler
```bash
# Install with: pip install scalene
# Run profiler
python -m scalene my_script.py
```

**IPython/Jupyter Magic Commands**:
```python
# In Jupyter notebook
%timeit function_call()  # Time a single function

%%timeit                 # Time an entire cell
code_line_1
code_line_2

%prun function_call()    # Profile a function call

%load_ext line_profiler
%lprun -f function_name function_call()  # Line profile
```

**Advanced Performance Analysis**:
1. **Flame Graphs**: Visualize call stack and time spent
   - py-spy supports flame graphs out of the box
   
2. **System Monitoring**: Use psutil for system-wide monitoring
```python
import psutil

# Monitor process memory
process = psutil.Process()
print(f"Memory used: {process.memory_info().rss / 10**6} MB")
```

3. **Web Applications**: Django Debug Toolbar, Flask Debug Toolbar

## 29. How do you handle errors in a distributed Python application?

**Key Concept**: Distributed systems require robust error handling strategies beyond simple try/except, including retries, circuit breakers, and centralized logging.

**Error Handling Strategies**:

1. **Retry Logic with Backoff**:
```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=0.5, max_backoff=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = backoff_factor
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    
                    # Calculate backoff with jitter
                    sleep_time = min(backoff * (2 ** (retries - 1)) + random.uniform(0, 1), max_backoff)
                    print(f"Retry {retries} after {sleep_time:.2f}s due to {e}")
                    time.sleep(sleep_time)
        
        return wrapper
    return decorator

@retry_with_backoff()
def call_remote_service():
    # Call to remote service that might fail
    response = requests.get("https://api.example.com/data", timeout=5)
    response.raise_for_status()
    return response.json()
```

2. **Circuit Breaker Pattern**:
```python
import time
from functools import wraps

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = None
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "HALF-OPEN"
                else:
                    raise CircuitBreakerError(f"Circuit breaker is OPEN until {self.last_failure_time + self.recovery_timeout}")
            
            try:
                result = func(*args, **kwargs)
                
                # Success, reset if in HALF-OPEN
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold or self.state == "HALF-OPEN":
                    self.state = "OPEN"
                
                raise
        
        return wrapper

class CircuitBreakerError(Exception):
    pass

@CircuitBreaker(failure_threshold=3, recovery_timeout=10)
def call_unreliable_service():
    # Service call that might fail
    pass
```

3. **Centralized Logging and Monitoring**:
```python
import logging
from logging.handlers import RotatingFileHandler
import traceback
import socket

def setup_logging(service_name):
    """Set up logging with unique host identifier."""
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    
    # Add host info to format
    hostname = socket.gethostname()
    formatter = logging.Formatter(
        f'%(asctime)s - {hostname} - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f"{service_name}.log", maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Optional: add Sentry, ELK, or other handlers
    
    logger.addHandler(file_handler)
    return logger

logger = setup_logging("user-service")

def handle_request(request_data):
    try:
        # Process request
        result = process_data(request_data)
        logger.info(f"Successfully processed request {request_data['id']}")
        return result
    except Exception as e:
        # Log full exception details
        logger.error(
            f"Error processing request {request_data.get('id', 'unknown')}: "
            f"{str(e)}\n{traceback.format_exc()}"
        )
        # Re-raise or return error response
        raise
```

4. **Dead Letter Queues**:
```python
# Using RabbitMQ with pika
import pika
import json

def publish_to_dead_letter(message, error):
    """Publish failed message to dead letter queue."""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')
    )
    channel = connection.channel()
    
    # Ensure dead letter queue exists
    channel.queue_declare(queue='dead_letter_queue', durable=True)
    
    # Add error details
    message_with_error = {
        'original_message': message,
        'error': str(error),
        'timestamp': time.time()
    }
    
    # Publish to dead letter queue
    channel.basic_publish(
        exchange='',
        routing_key='dead_letter_queue',
        body=json.dumps(message_with_error),
        properties=pika.BasicProperties(
            delivery_mode=2,  # Make persistent
        )
    )
    connection.close()
```

5. **Health Checks and Self-Healing**:
```python
def health_check():
    """Check if all dependencies are available."""
    checks = {
        'database': check_database_connection(),
        'cache': check_cache_connection(),
        'message_queue': check_queue_connection()
    }
    
    # If any check fails, attempt recovery
    for service, status in checks.items():
        if not status['healthy']:
            try:
                recovery_functions[service]()
                logger.info(f"Recovered {service} connection")
            except Exception as e:
                logger.error(f"Failed to recover {service}: {e}")
    
    return all(status['healthy'] for status in checks.values())
```

## 30. Explain context managers (`with` statement) and how to implement a custom one.

**Key Concept**: Context managers provide a clean way to manage resources by automatically handling setup and teardown operations.

**How Context Managers Work**:
- The `with` statement establishes a context
- Resources are initialized in `__enter__`
- Resources are cleaned up in `__exit__`, even if an exception occurs

**Built-in Context Managers**:
```python
# File handling
with open('file.txt', 'r') as f:
    content = f.read()
    # File automatically closes when block exits

# Lock management
import threading
lock = threading.Lock()
with lock:
    # Critical section
    modify_shared_resource()
    # Lock automatically released when block exits
```

**Implementing a Custom Context Manager**:

1. **Using a Class**:
```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to database at {self.connection_string}")
        self.connection = create_connection(self.connection_string)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            print("Closing database connection")
            self.connection.close()
        
        # Return True to suppress exceptions, False to propagate them
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
            return False  # Propagate exception
        return True

# Usage
with DatabaseConnection("postgresql://localhost/mydb") as conn:
    data = conn.execute("SELECT * FROM users")
    # Connection automatically closed
```

2. **Using `contextlib.contextmanager` Decorator**:
```python
from contextlib import contextmanager

@contextmanager
def database_connection(connection_string):
    connection = None
    try:
        # __enter__ logic
        print(f"Connecting to database at {connection_string}")
        connection = create_connection(connection_string)
        yield connection  # Provide the connection to the with block
    finally:
        # __exit__ logic
        if connection:
            print("Closing database connection")
            connection.close()

# Usage
with database_connection("postgresql://localhost/mydb") as conn:
    data = conn.execute("SELECT * FROM users")
```

**Practical Examples**:

1. **Timing Context Manager**:
```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    print(f"{name} took {elapsed:.6f} seconds")

# Usage
with timer("Data processing"):
    process_large_dataset()
```

2. **Temporary Directory Context Manager**:
```python
import tempfile
import shutil
from contextlib import contextmanager
import os

@contextmanager
def temporary_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

# Usage
with temporary_directory() as temp_dir:
    # Work with temporary files
    with open(os.path.join(temp_dir, "temp_file.txt"), "w") as f:
        f.write("Temporary data")
    # Directory automatically deleted
```

3. **Transaction Context Manager**:
```python
@contextmanager
def transaction(session):
    """Database transaction as a context manager."""
    try:
        yield session
        session.commit()
        print("Transaction committed")
    except Exception as e:
        session.rollback()
        print(f"Transaction rolled back due to {e}")
        raise

# Usage
with transaction(database.session) as session:
    user = User(name="John", email="john@example.com")
    session.add(user)
    # Automatically committed or rolled back
```

4. **Resource Pool**:
```python
class ResourcePool:
    def __init__(self, factory, max_resources=5):
        self.resources = []
        self.factory = factory
        self.max_resources = max_resources
        self.in_use = set()
    
    @contextmanager
    def acquire(self):
        resource = self._get_resource()
        self.in_use.add(resource)
        try:
            yield resource
        finally:
            self.in_use.remove(resource)
            self._return_resource(resource)
    
    def _get_resource(self):
        if self.resources:
            return self.resources.pop()
        if len(self.in_use) < self.max_resources:
            return self.factory()
        raise RuntimeError("Resource pool exhausted")
    
    def _return_resource(self, resource):
        self.resources.append(resource)

# Usage
def create_connection():
    return {"connection": "Database connection"}

pool = ResourcePool(create_connection, max_resources=3)

with pool.acquire() as conn:
    # Use the connection
    print(f"Using {conn}")
```

**Best Practices**:
1. Use context managers for any resource that needs cleanup
2. Ensure `__exit__` handles exceptions properly
3. Keep context managers focused on a single responsibility
4. Use the `contextlib` module for simple cases
5. Document your context managers clearly, especially their behavior with exceptions


# 31-35

**6. Using a Multiprocessing Counter for True Parallelism**:
```python
import multiprocessing as mp

class ProcessSafeCounter:
    def __init__(self):
        self.value = mp.Value('i', 0)  # Shared integer
    
    def increment(self, amount=1):
        with self.value.get_lock():
            self.value.value += amount
    
    def get_value(self):
        with self.value.get_lock():
            return self.value.value

def worker(counter, num_increments):
    for _ in range(num_increments):
        counter.increment()

if __name__ == "__main__":
    # Create shared counter
    counter = ProcessSafeCounter()
    
    # Create and start processes
    processes = []
    for _ in range(4):
        p = mp.Process(target=worker, args=(counter, 100000))
        processes.append(p)
        p.start()
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    print(f"Final count: {counter.get_value()}")  # Should be 400,000
```

**7. Using Redis for Distributed Counting**:
```python
import redis
import threading

class RedisCounter:
    def __init__(self, counter_name, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.counter_name = counter_name
        # Initialize counter if it doesn't exist
        self.redis_client.setnx(self.counter_name, 0)
    
    def increment(self, amount=1):
        # Atomic increment in Redis
        return self.redis_client.incrby(self.counter_name, amount)
    
    def get_value(self):
        return int(self.redis_client.get(self.counter_name))
    
    def reset(self):
        self.redis_client.set(self.counter_name, 0)

# Example usage with distributed workers
counter = RedisCounter("my_counter")
counter.reset()  # Start fresh

def worker():
    for _ in range(1000):
        counter.increment()

# Simulate multiple workers
threads = []
for _ in range(10):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final count: {counter.get_value()}")  # Should be 10,000
```

**8. Performance Comparison of Different Approaches**:
```python
import threading
import time
import multiprocessing as mp
from threading import atomic  # Python 3.12+

def benchmark_counter(counter_class, num_threads, increments_per_thread):
    counter = counter_class()
    
    def worker():
        for _ in range(increments_per_thread):
            counter.increment()
    
    start_time = time.time()
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    print(f"{counter_class.__name__}: {counter.get_value()} increments in {elapsed:.6f} seconds")
    print(f"  Expected: {num_threads * increments_per_thread}")
    print(f"  Throughput: {counter.get_value() / elapsed:.2f} ops/sec")
    
    return elapsed, counter.get_value()

# Counter implementations (simplified)
class LockCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
    
    def get_value(self):
        return self.value

class AtomicCounter:
    def __init__(self):
        self.value = atomic.int64(0)
    
    def increment(self):
        self.value.inc()
    
    def get_value(self):
        return self.value.get()

# Run benchmarks
NUM_THREADS = 10
INCREMENTS = 100000

print(f"Benchmarking with {NUM_THREADS} threads and {INCREMENTS} increments per thread\n")

lock_time, lock_count = benchmark_counter(LockCounter, NUM_THREADS, INCREMENTS)
atomic_time, atomic_count = benchmark_counter(AtomicCounter, NUM_THREADS, INCREMENTS)

print(f"\nAtomic operations were {lock_time/atomic_time:.2f}x faster than locks")
```

**Data Engineering Example: Pipeline Processing Counter**:
```python
import threading
import time
import queue
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class PipelineMetrics:
    """Thread-safe metrics for data pipeline monitoring."""
    
    def __init__(self):
        self.lock = threading.RLock()
        self._records_processed = 0
        self._errors = 0
        self._processing_times: List[float] = []
        self._stage_counters: Dict[str, int] = {}
        self._start_time = time.time()
    
    def record_processed(self, count=1):
        """Record that N records were processed."""
        with self.lock:
            self._records_processed += count
    
    def record_error(self):
        """Record an error."""
        with self.lock:
            self._errors += 1
    
    def record_processing_time(self, seconds):
        """Record processing time for a batch."""
        with self.lock:
            self._processing_times.append(seconds)
    
    def increment_stage_counter(self, stage_name, count=1):
        """Increment counter for a specific pipeline stage."""
        with self.lock:
            if stage_name not in self._stage_counters:
                self._stage_counters[stage_name] = 0
            self._stage_counters[stage_name] += count
    
    def get_metrics(self):
        """Get all metrics as a dictionary."""
        with self.lock:
            elapsed = time.time() - self._start_time
            
            avg_time = 0
            if self._processing_times:
                avg_time = sum(self._processing_times) / len(self._processing_times)
            
            return {
                'records_processed': self._records_processed,
                'errors': self._errors,
                'error_rate': (self._errors / max(1, self._records_processed + self._errors)) * 100,
                'avg_processing_time': avg_time,
                'records_per_second': self._records_processed / max(0.001, elapsed),
                'stage_counters': dict(self._stage_counters),
                'elapsed_seconds': elapsed
            }

# Example data pipeline with metrics
def data_pipeline(input_queue, output_queue, metrics):
    while True:
        try:
            # Get next batch
            batch = input_queue.get(timeout=1)
            if batch is None:  # Poison pill
                break
                
            start_time = time.time()
            
            # Process each record in the batch
            metrics.increment_stage_counter('batches')
            processed = []
            
            for record in batch:
                try:
                    # Extract
                    metrics.increment_stage_counter('extract')
                    extracted = extract_data(record)
                    
                    # Transform
                    metrics.increment_stage_counter('transform')
                    transformed = transform_data(extracted)
                    
                    # Add to processed records
                    processed.append(transformed)
                    metrics.record_processed()
                
                except Exception as e:
                    metrics.record_error()
            
            # Put processed records in output queue
            if processed:
                output_queue.put(processed)
            
            # Record processing time for this batch
            processing_time = time.time() - start_time
            metrics.record_processing_time(processing_time)
            
            # Mark task as done
            input_queue.task_done()
            
        except queue.Empty:
            pass
```

**Best Practices for Thread-Safe Counters**:

1. **Choose the Right Tool**:
   - For simple in-process counting, use locks or atomic operations
   - For thread-local counting with aggregation, use `threading.local`
   - For cross-process counting, use multiprocessing shared values
   - For distributed counting, consider Redis or a database

2. **Minimize Lock Contention**:
   - Use atomic operations when available
   - Keep critical sections as small as possible
   - Consider batch incrementing (accumulate locally, update shared counter less frequently)

3. **Use Higher-Level Abstractions**:
   - Use queue-based solutions for event counting
   - Consider actor-based models for complex counting scenarios

4. **Consider Performance Requirements**:
   - Profile different approaches under expected load
   - Be aware of throughput vs. accuracy trade-offs

5. **Monitoring and Debugging**:
   - Add logging for complex counting scenarios
   - Implement validation checks to verify count integrity
   - Build monitoring for high-volume counters# Brief Answers to Python Interview Questions 31-35

## 31. Explain the difference between multithreading and multiprocessing in Python.

**Key Concept**: Multithreading uses multiple threads within a single process (limited by GIL), while multiprocessing uses multiple processes with separate memory spaces (bypassing GIL).

**Multithreading**:
- Uses `threading` module
- Shares memory space between threads
- Limited by Global Interpreter Lock (GIL) for CPU-bound tasks
- Good for I/O-bound tasks (network, file operations)
- Lightweight and less resource-intensive

```python
import threading
import time

def worker(name):
    print(f"Thread {name} starting")
    time.sleep(2)  # Simulate I/O operation
    print(f"Thread {name} finished")

# Create and start threads
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()
```

**Multiprocessing**:
- Uses `multiprocessing` module
- Each process has its own Python interpreter and memory space
- Bypasses GIL limitations
- Good for CPU-bound tasks (computation, data processing)
- More resource-intensive

```python
import multiprocessing as mp
import time

def cpu_bound_task(n):
    """A CPU-intensive calculation."""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

if __name__ == "__main__":
    # Create a pool of worker processes
    with mp.Pool(processes=4) as pool:
        # Distribute tasks to the pool
        results = pool.map(cpu_bound_task, [10000000, 20000000, 30000000, 40000000])
        print(results)
```

**Comparison**:

| Feature | Multithreading | Multiprocessing |
|---------|---------------|----------------|
| Module | `threading` | `multiprocessing` |
| Memory | Shared memory | Separate memory |
| GIL Impact | Limited by GIL | Not limited by GIL |
| Best for | I/O-bound tasks | CPU-bound tasks |
| Communication | Direct via shared objects | Inter-process (pipes, queues) |
| Resource usage | Lower | Higher |
| Complexity | Simpler to implement | More complex |

**When to Use Each**:
- **Use Multithreading for**:
  - Network requests
  - File operations
  - Database queries
  - GUI applications

- **Use Multiprocessing for**:
  - Data processing
  - Numerical computations
  - CPU-intensive algorithms
  - Parallel processing of large datasets

## 32. How would you implement parallel data processing using Python's `concurrent.futures`?

**Key Concept**: `concurrent.futures` provides a high-level interface for asynchronous execution with thread and process pools.

**Basic Usage Pattern**:
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def process_chunk(chunk):
    """Process a chunk of data."""
    # Simulate processing time
    time.sleep(1)
    return sum(chunk)

# Sample data
data = [list(range(i, i+1000)) for i in range(0, 10000, 1000)]

# Choose executor based on task type
# ThreadPoolExecutor for I/O-bound tasks
# ProcessPoolExecutor for CPU-bound tasks
with ProcessPoolExecutor(max_workers=4) as executor:
    # Map processes data in parallel
    results = list(executor.map(process_chunk, data))
    
    # Or submit tasks individually
    futures = [executor.submit(process_chunk, chunk) for chunk in data]
    results = [future.result() for future in futures]

print(f"Processed {len(data)} chunks with result: {sum(results)}")
```

**Advanced Features**:

1. **Handling Task Completion as they Finish**:
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random

def process_item(item):
    # Simulate varying processing times
    processing_time = random.uniform(0.5, 3)
    time.sleep(processing_time)
    return item, item * item, processing_time

items = list(range(20))

with ProcessPoolExecutor(max_workers=4) as executor:
    # Submit all tasks
    future_to_item = {executor.submit(process_item, item): item for item in items}
    
    # Process results as they complete (not in submission order)
    for future in as_completed(future_to_item):
        item = future_to_item[future]
        try:
            data = future.result()
            print(f"Item {item} with result {data[1]} took {data[2]:.2f} seconds")
        except Exception as e:
            print(f"Item {item} generated an exception: {e}")
```

2. **Cancelling Tasks**:
```python
def process_with_timeout(data, timeout=10):
    with ProcessPoolExecutor() as executor:
        future = executor.submit(long_running_task, data)
        
        try:
            # Wait for the result with a timeout
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            # Cancel the task if it takes too long
            future.cancel()
            print("Task timed out and was cancelled")
            return None
```

3. **Combining ThreadPoolExecutor and ProcessPoolExecutor**:
```python
def download_and_process_files(urls):
    downloaded_files = []
    processed_results = []
    
    # Download files (I/O-bound)
    with ThreadPoolExecutor(max_workers=8) as downloader:
        future_to_url = {downloader.submit(download_file, url): url for url in urls}
        for future in as_completed(future_to_url):
            downloaded_files.append(future.result())
    
    # Process files (CPU-bound)
    with ProcessPoolExecutor(max_workers=4) as processor:
        future_to_file = {processor.submit(process_file, file): file for file in downloaded_files}
        for future in as_completed(future_to_file):
            processed_results.append(future.result())
    
    return processed_results
```

**Practical Data Processing Example**:
```python
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os

def process_csv_chunk(filename):
    """Process a single CSV file."""
    df = pd.read_csv(filename)
    
    # Perform data transformations
    df['value_squared'] = df['value'] ** 2
    df['category_code'] = df['category'].astype('category').cat.codes
    
    # Calculate aggregations
    result = {
        'filename': filename,
        'row_count': len(df),
        'sum': df['value'].sum(),
        'mean': df['value'].mean(),
        'categories': df['category'].nunique()
    }
    
    return result

def process_csv_directory(directory, pattern='*.csv', max_workers=None):
    """Process all CSV files in a directory in parallel."""
    import glob
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory, pattern))
    print(f"Found {len(csv_files)} CSV files to process")
    
    results = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_csv_chunk, file): file 
                           for file in csv_files}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Processed {file} with {result['row_count']} rows")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Combine results
    summary = pd.DataFrame(results)
    return summary
```

**Best Practices**:
1. Choose the right executor (ThreadPoolExecutor for I/O, ProcessPoolExecutor for CPU)
2. Set an appropriate number of workers (typically CPU count for ProcessPoolExecutor)
3. Handle exceptions properly within worker functions
4. Use `as_completed()` for processing results as they arrive
5. Consider using timeouts for long-running tasks
6. Avoid sharing large data structures between processes
7. Chunk data appropriately for your hardware

## 33. Explain Python's asyncio and when you would use it in data engineering.

**Key Concept**: `asyncio` enables asynchronous, non-blocking I/O operations using coroutines with `async`/`await` syntax, ideal for I/O-bound tasks.

**Basic Asyncio Structure**:
```python
import asyncio

async def fetch_data(url):
    """Asynchronous function to fetch data."""
    print(f"Start fetching from {url}")
    # Simulate network delay
    await asyncio.sleep(2)  # Non-blocking sleep
    print(f"Finished fetching from {url}")
    return f"Data from {url}"

async def main():
    # Create tasks for concurrent execution
    tasks = [
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com")
    ]
    
    # Execute all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)
    return results

# Run the event loop
if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
```

**Key Asyncio Components**:
- **Coroutines**: Functions defined with `async def` that can be paused and resumed
- **Tasks**: Wrappers around coroutines to manage their execution
- **Event Loop**: Manages and distributes the execution of tasks
- **`await`**: Suspends execution of the coroutine until the awaitable completes
- **`asyncio.gather()`**: Run awaitables concurrently and collect their results

**Data Engineering Use Cases**:

1. **Parallel API Requests**:
```python
import aiohttp
import asyncio
import time

async def fetch_api_data(session, api_url):
    """Fetch data from an API endpoint."""
    async with session.get(api_url) as response:
        return await response.json()

async def fetch_all_apis(api_urls):
    """Fetch data from multiple APIs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_api_data(session, url) for url in api_urls]
        return await asyncio.gather(*tasks)

# Example usage
async def main():
    start = time.time()
    
    api_urls = [
        'https://api.example.com/data/1',
        'https://api.example.com/data/2',
        'https://api.example.com/data/3',
        # More URLs...
    ]
    
    results = await fetch_all_apis(api_urls)
    
    end = time.time()
    print(f"Fetched {len(results)} APIs in {end - start:.2f} seconds")
    
    # Process the results
    for i, result in enumerate(results):
        process_data(result, i)

asyncio.run(main())
```

2. **Asynchronous Database Operations**:
```python
import asyncio
import asyncpg

async def fetch_user_data(user_ids):
    """Fetch user data from PostgreSQL database asynchronously."""
    connection = await asyncpg.connect('postgresql://user:password@localhost/database')
    
    try:
        # Prepare statement once
        stmt = await connection.prepare(
            'SELECT id, name, email FROM users WHERE id = $1'
        )
        
        # Execute for all user_ids concurrently
        results = await asyncio.gather(
            *[stmt.fetchrow(user_id) for user_id in user_ids]
        )
        
        return [dict(r) if r else None for r in results]
    finally:
        await connection.close()
```

3. **Asynchronous File Processing**:
```python
import aiofiles
import asyncio
import os

async def process_file(filename):
    """Process a file asynchronously."""
    async with aiofiles.open(filename, 'r') as f:
        content = await f.read()
        # Process content
        result = len(content.split())
        return filename, result

async def process_directory(directory):
    """Process all files in a directory concurrently."""
    files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if os.path.isfile(os.path.join(directory, f))
    ]
    
    tasks = [process_file(f) for f in files]
    return await asyncio.gather(*tasks)
```

4. **Data ETL Pipeline**:
```python
async def extract(sources):
    """Extract data from multiple sources concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for source in sources:
            if source['type'] == 'api':
                tasks.append(fetch_api_data(session, source['url']))
            elif source['type'] == 'database':
                tasks.append(fetch_db_data(source['connection_string']))
            elif source['type'] == 'file':
                tasks.append(read_file_data(source['path']))
        
        return await asyncio.gather(*tasks)

async def transform(data_list):
    """Transform extracted data concurrently."""
    # Apply transformations to each dataset
    tasks = [apply_transformations(data) for data in data_list]
    return await asyncio.gather(*tasks)

async def load(transformed_data, destinations):
    """Load data to multiple destinations concurrently."""
    load_tasks = []
    for i, dest in enumerate(destinations):
        if dest['type'] == 'database':
            load_tasks.append(load_to_database(transformed_data[i], dest))
        elif dest['type'] == 'file':
            load_tasks.append(write_to_file(transformed_data[i], dest))
        elif dest['type'] == 'api':
            load_tasks.append(send_to_api(transformed_data[i], dest))
    
    await asyncio.gather(*load_tasks)

async def etl_pipeline(sources, destinations):
    """Execute full ETL pipeline asynchronously."""
    # Extract
    raw_data = await extract(sources)
    
    # Transform
    transformed_data = await transform(raw_data)
    
    # Load
    await load(transformed_data, destinations)
    
    return "Pipeline completed successfully"
```

**Asyncio vs. Threads vs. Processes**:

| Feature | Asyncio | Threads | Processes |
|---------|---------|---------|-----------|
| Best for | I/O-bound with many connections | I/O-bound with few connections | CPU-bound tasks |
| Concurrency model | Cooperative multitasking | Preemptive multitasking | Separate processes |
| GIL impact | Not affected | Limited by GIL | Not limited by GIL |
| Memory usage | Low | Medium | High |
| Context switching | Explicit (await) | Implicit (OS managed) | OS managed |
| Complexity | Requires async code | Simpler API | Simple API |
| Debugging | More complex | Moderate | Easier |

**When to Use Asyncio in Data Engineering**:
- Fetching data from multiple APIs/services concurrently
- Handling many concurrent database connections
- Processing numerous files with I/O bottlenecks
- Building high-throughput data streaming pipelines
- Creating real-time data processing services
- Implementing high-performance web scraping

**When Not to Use Asyncio**:
- For CPU-bound tasks (use multiprocessing instead)
- When working with synchronous-only libraries
- When code simplicity is more important than performance
- When you need thread-local storage

## 34. How do you handle race conditions in concurrent Python applications?

**Key Concept**: Race conditions occur when multiple threads/processes access shared resources with non-atomic operations. Python provides synchronization primitives to prevent these issues.

**Common Synchronization Mechanisms**:

1. **Locks (Mutexes)**:
```python
import threading
import time

class Counter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:  # Acquire the lock
            current = self.count
            time.sleep(0.001)  # Simulate processing
            self.count = current + 1
            # Lock automatically released after block

# Test with and without lock
counter = Counter()

def worker():
    for _ in range(100):
        counter.increment()

threads = []
for _ in range(10):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final count: {counter.count}")  # Will be 1000 with lock
```

2. **RLock (Reentrant Lock)**:
```python
import threading

class ReentrantResource:
    def __init__(self):
        self.lock = threading.RLock()  # Can be acquired multiple times by same thread
        self.data = {}
    
    def update_value(self, key, value):
        with self.lock:
            self.data[key] = value
            # Can call other methods that also acquire the lock
            self.log_update(key, value)
    
    def log_update(self, key, value):
        with self.lock:  # Same thread can acquire again
            print(f"Updated {key} to {value}")
```

3. **Semaphores**:
```python
import threading
import time
import random

# Limit concurrent access to a resource
pool_semaphore = threading.Semaphore(3)  # Allow 3 concurrent accesses

def worker(worker_id):
    with pool_semaphore:
        print(f"Worker {worker_id} acquired the semaphore")
        # Simulate using a limited resource
        time.sleep(random.uniform(1, 3))
        print(f"Worker {worker_id} released the semaphore")

# Start 10 workers but only 3 can run at a time
threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

4. **Events**:
```python
import threading
import time

# Used for signaling between threads
start_event = threading.Event()

def worker(worker_id):
    # Wait for the start signal
    print(f"Worker {worker_id} waiting for start signal...")
    start_event.wait()
    print(f"Worker {worker_id} started processing")
    # Perform work...

# Create and start threads
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Simulate preparation work
time.sleep(2)
print("Preparation complete. Starting all workers...")

# Signal all threads to start
start_event.set()

for t in threads:
    t.join()
```

5. **Condition Variables**:
```python
import threading
import queue
import time
import random

# Producer-consumer pattern with condition variable
class Buffer:
    def __init__(self, size):
        self.queue = queue.Queue(size)
        self.condition = threading.Condition()
        self.max_size = size
    
    def produce(self, item):
        with self.condition:
            # Wait until there's space in the buffer
            while self.queue.full():
                print("Buffer full, producer waiting...")
                self.condition.wait()
            
            # Add item and notify consumers
            self.queue.put(item)
            print(f"Produced: {item}")
            self.condition.notify()
    
    def consume(self):
        with self.condition:
            # Wait until there's an item to consume
            while self.queue.empty():
                print("Buffer empty, consumer waiting...")
                self.condition.wait()
            
            # Get item and notify producers
            item = self.queue.get()
            print(f"Consumed: {item}")
            self.condition.notify()
            return item

# Example usage
buffer = Buffer(5)

def producer():
    for i in range(10):
        time.sleep(random.uniform(0.1, 0.5))
        buffer.produce(i)

def consumer():
    for _ in range(10):
        time.sleep(random.uniform(0.2, 0.6))
        buffer.consume()

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t1.start()
t2.start()
t1.join()
t2.join()
```

6. **Queue**:
```python
import queue
import threading
import time

# Thread-safe queue for producer-consumer pattern
task_queue = queue.Queue()

def producer():
    for i in range(10):
        task_queue.put(f"Task {i}")
        print(f"Produced: Task {i}")
        time.sleep(0.5)
    # Signal end of tasks
    task_queue.put(None)

def consumer():
    while True:
        task = task_queue.get()
        if task is None:
            break
        print(f"Processing: {task}")
        time.sleep(1)
        task_queue.task_done()

# Start threads
p = threading.Thread(target=producer)
c = threading.Thread(target=consumer)
p.start()
c.start()

# Wait for all tasks to be processed
task_queue.join()
p.join()
c.join()
```

7. **Atomic Operations with `threading.atomic`** (Python 3.12+):
```python
from threading import atomic
import threading

# Thread-safe counter using atomic operations
counter = atomic.int64(0)

def increment():
    for _ in range(10000):
        counter.inc()  # Atomic increment

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Final count: {counter.get()}")  # Will be 100000
```

8. **For Processes: Using Shared Memory and Locks**:
```python
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

def update_shared_array(shm_name, index, value, lock):
    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    # Create a NumPy array using the shared memory buffer
    shared_array = np.ndarray((10,), dtype=np.int64, buffer=existing_shm.buf)
    
    # Update value with lock protection
    with lock:
        shared_array[index] += value
    
    # Clean up
    existing_shm.close()

if __name__ == "__main__":
    # Create shared memory
    shape = (10,)
    shm = shared_memory.SharedMemory(create=True, size=np.zeros(shape, dtype=np.int64).nbytes)
    shared_array = np.ndarray(shape, dtype=np.int64, buffer=shm.buf)
    shared_array[:] = 0  # Initialize with zeros
    
    # Create a lock for synchronization
    lock = mp.Lock()
    
    # Create and start processes
    processes = []
    for i in range(5):
        p = mp.Process(target=update_shared_array, args=(shm.name, i % 10, 1, lock))
        processes.append(p)
        p.start()
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    # Print results
    print(f"Final array: {shared_array}")
    
    # Clean up
    shm.close()
    shm.unlink()
```

**Data Engineering Example: Thread-Safe DataFrame Updates**:
```python
import pandas as pd
import threading
import time

class ThreadSafeDataFrame:
    """Thread-safe wrapper for pandas DataFrame operations."""
    
    def __init__(self, initial_df=None):
        self.df = initial_df if initial_df is not None else pd.DataFrame()
        self.lock = threading.RLock()
    
    def apply_transformation(self, transformation_func):
        """Apply a transformation function to the DataFrame."""
        with self.lock:
            self.df = transformation_func(self.df)
    
    def append_row(self, row_dict):
        """Append a row to the DataFrame."""
        with self.lock:
            self.df = pd.concat([self.df, pd.DataFrame([row_dict])], ignore_index=True)
    
    def get_stats(self):
        """Get statistics about the DataFrame."""
        with self.lock:
            return {
                'rows': len(self.df),
                'columns': list(self.df.columns),
                'memory': self.df.memory_usage(deep=True).sum()
            }
    
    def get_snapshot(self):
        """Get a copy of the current DataFrame."""
        with self.lock:
            return self.df.copy()

# Example usage
def worker(shared_df, worker_id):
    """Update the shared DataFrame."""
    for i in range(5):
        row = {'worker_id': worker_id, 'value': i, 'timestamp': time.time()}
        shared_df.append_row(row)
        time.sleep(0.01)

# Create shared DataFrame
df = ThreadSafeDataFrame(pd.DataFrame(columns=['worker_id', 'value', 'timestamp']))

# Start multiple threads updating the same DataFrame
threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(df, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Get the result
result = df.get_snapshot()
print(f"Final DataFrame has {len(result)} rows")
print(result.head())
```

**Best Practices**:
1. Use the highest-level synchronization primitive that meets your needs
2. Keep critical sections as small as possible
3. Avoid nested locks to prevent deadlocks
4. Consider using thread-safe data structures (`queue.Queue`, `collections.deque`)
5. For simple counters, consider `itertools.count` or atomic operations
6. Use resource pools for managing limited resources
7. Document synchronization requirements clearly
8. Consider using higher-level concurrent programming models (e.g., actors)

## 35. Describe strategies for implementing a thread-safe counter in Python.

**Key Concept**: Thread-safe counters ensure accurate counting in concurrent environments by using synchronization mechanisms to prevent race conditions.

**1. Using Locks**:
```python
import threading

class ThreadSafeCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = threading.Lock()
    
    def increment(self, amount=1):
        with self.lock:
            self.value += amount
            return self.value
    
    def decrement(self, amount=1):
        with self.lock:
            self.value -= amount
            return self.value
    
    def get_value(self):
        with self.lock:
            return self.value

# Example usage
counter = ThreadSafeCounter()

def worker():
    for _ in range(100000):
        counter.increment()

threads = []
for _ in range(10):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final count: {counter.get_value()}")  # Should be 1,000,000
```

**2. Using RLock for Reentrant Access**:
```python
import threading

class ReentrantCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.RLock()  # Reentrant lock
    
    def increment_and_get(self):
        with self.lock:
            self.value += 1
            return self.get_value()  # Can call another locked method
    
    def get_value(self):
        with self.lock:  # Same thread can acquire again
            return self.value
```

**3. Using `threading.local()` for Thread-Local Counters**:
```python
import threading
import queue

class ThreadLocalCounter:
    def __init__(self):
        self.local = threading.local()
        self.local.counter = 0
        self.result_queue = queue.Queue()
    
    def increment(self, amount=1):
        self.local.counter += amount
    
    def get_local_value(self):
        return getattr(self.local, 'counter', 0)
    
    def submit_local_count(self):
        self.result_queue.put(self.local.counter)
    
    def aggregate_counts(self):
        total = 0
        while not self.result_queue.empty():
            total += self.result_queue.get()
        return total

# Usage for per-thread counting
counter = ThreadLocalCounter()

def worker(worker_id):
    # Each thread has its own counter
    for _ in range(1000):
        counter.increment()
    
    local_count = counter.get_local_value()
    print(f"Worker {worker_id} counted to {local_count}")
    counter.submit_local_count()

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Get total across all threads
total = counter.aggregate_counts()
print(f"Total count: {total}")  # Should be 5,000
```

**4. Using Atomic Operations with `threading.atomic`** (Python 3.12+):
```python
from threading import atomic
import threading

# Create atomic counter
counter = atomic.int64(0)

def increment_atomic():
    for _ in range(100000):
        counter.inc()  # Atomic increment

threads = []
for _ in range(10):
    t = threading.Thread(target=increment_atomic)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final count: {counter.get()}")  # Should be 1,000,000
```

**5. Using Queue for Worker Counts**:
```python
import threading
import queue

def count_events(event_queue, stop_event):
    """Count events from a thread-safe queue."""
    count = 0
    while not stop_event.is_set():
        try:
            # Try to get an event with timeout to check stop_event periodically
            event = event_queue.get(timeout=0.1)
            count += 1
            event_queue.task_done()
        except queue.Empty:
            continue
    
    # Return the final count
    return count

# Setup shared queue and control event
event_queue = queue.Queue()
stop_event = threading.Event()

# Start counter thread
counter_thread = threading.Thread(
    target=lambda: count_events(event_queue, stop_event)
)
counter_thread.daemon = True
counter_thread.start()

# Generate events from multiple threads
def generate_events(n):
    for _ in range(n):
        event_queue.put(1)  # 1 represents an event

# Start multiple producer threads
producers = []
for _ in range(5):
    t = threading.Thread(target=generate_events, args=(1000,))
    producers.append(t)
    t.start()

# Wait for producers to finish
for p in producers:
    p.join()

# Wait for queue to be processed
event_queue.join()

# Signal counter to stop and get the count
stop_event.set()
counter_thread.join()

# Final count would be 5000 (5 threads × 1000 events)



# 36-40 

# Brief Answers to Python Interview Questions 36-40

## 36. How do you manage dependencies in a Python project?

**Key Concept**: Effective dependency management ensures reproducible environments and simplifies collaboration.

### 1. Using Virtual Environments

Virtual environments isolate project dependencies from the system Python:

```bash
# Create virtual environment
python -m venv myproject_env

# Activate (on Windows)
myproject_env\Scripts\activate

# Activate (on Unix/MacOS)
source myproject_env/bin/activate

# Deactivate when done
deactivate
```

### 2. Package Management with pip

`pip` is Python's standard package installer:

```bash
# Install packages
pip install package-name
pip install package-name==1.2.3  # Specific version
pip install 'package-name>=1.0,<2.0'  # Version range

# Install from requirements file
pip install -r requirements.txt

# Generate requirements file
pip freeze > requirements.txt
```

Example `requirements.txt`:
```
numpy==1.21.5
pandas>=1.3.0,<2.0.0
scikit-learn==1.0.2
```

### 3. Using Conda for Environment Management

Conda is popular for data science projects and manages non-Python dependencies:

```bash
# Create conda environment
conda create --name myproject python=3.9

# Activate environment
conda activate myproject

# Install packages
conda install numpy pandas scikit-learn

# Export environment
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml
```

### 4. Dependency Management with Poetry

Poetry offers more advanced dependency management:

```bash
# Initialize project
poetry new myproject
cd myproject

# Add dependencies
poetry add pandas
poetry add numpy@^1.20.0
poetry add --dev pytest  # Development dependency

# Install dependencies
poetry install

# Update dependencies
poetry update

# Run commands in the environment
poetry run python script.py
```

Example `pyproject.toml`:
```toml
[tool.poetry]
name = "myproject"
version = "0.1.0"
description = "Data processing project"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
pandas = "^1.4.0"
numpy = "^1.22.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### 5. Using Pipenv

Pipenv combines pip and virtualenv:

```bash
# Install Pipenv
pip install pipenv

# Install packages
pipenv install pandas numpy
pipenv install pytest --dev

# Activate environment
pipenv shell

# Install dependencies from Pipfile
pipenv install

# Generate Pipfile.lock
pipenv lock
```

### 6. Project-Specific Dependency Management

For data engineering projects:

```python
# setup.py for package development
from setuptools import setup, find_packages

setup(
    name="data-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
        "aws": ["boto3", "s3fs"],
    },
    entry_points={
        "console_scripts": [
            "my-command=my_package.cli:main",
        ],
    },
    include_package_data=True,
)
        "pandas>=1.4.0",
        "sqlalchemy>=1.4.0",
        "apache-airflow>=2.2.0",
        "prefect>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "pylint"],
        "aws": ["boto3", "s3fs"],
        "gcp": ["google-cloud-storage", "google-cloud-bigquery"],
    }
)
```

### 7. Best Practices for Dependency Management

1. **Define version constraints**:
   - Use semantic versioning (e.g., `>=1.0.0,<2.0.0`)
   - Pin exact versions for reproducibility in production

2. **Separate production and development dependencies**

3. **Use dependency isolation**:
   - Always use virtual environments
   - Consider containerization (Docker)

4. **Regularly update dependencies**:
   - Check for security vulnerabilities
   - Use tools like `pip-audit` or `safety`

5. **Document dependency decisions**

6. **Use lock files** for deterministic installs:
   - `Pipfile.lock`, `poetry.lock`, etc.

Example Docker approach:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

## 37. Explain virtual environments in Python and why they are important.

**Key Concept**: Virtual environments create isolated Python environments, preventing dependency conflicts between projects.

### What Virtual Environments Solve

1. **Dependency Conflicts**: Different projects may need different versions of the same package
2. **System Protection**: Avoid modifying system-wide Python installation
3. **Reproducibility**: Ensure consistent development environments
4. **Isolation**: Test code with specific package versions

### Standard Library Solution: venv

```python
# Create a virtual environment
python -m venv my_project_env

# Activate on Windows
my_project_env\Scripts\activate

# Activate on macOS/Linux
source my_project_env/bin/activate

# Verify activation
which python  # Should point to virtual environment
pip list      # Should show only base packages

# Install project-specific dependencies
pip install pandas matplotlib seaborn

# When finished
deactivate
```

### How Virtual Environments Work

1. **Structure**: Creates a directory with:
   - Python interpreter copy/symlink
   - pip and setuptools
   - `site-packages` directory for packages

2. **Activation**: Changes environment variables:
   - Modifies `PATH` to prioritize venv binaries
   - Sets `VIRTUAL_ENV` environment variable

3. **Isolation**: Packages installed in venv's `site-packages`, not globally

### Alternative Tools

1. **virtualenv**: More features than venv (supports older Python)
```bash
pip install virtualenv
virtualenv my_env
```

2. **Conda**: Environment manager that handles non-Python dependencies
```bash
conda create --name data_science python=3.9 numpy pandas
conda activate data_science
```

3. **pyenv**: Manage multiple Python versions
```bash
pyenv install 3.9.7
pyenv virtualenv 3.9.7 my_project
pyenv activate my_project
```

4. **pipenv**: Combines pip and virtualenv
```bash
pipenv install pandas
pipenv shell
```

5. **poetry**: Dependency management with virtual environments
```bash
poetry env use python3.9
poetry shell
```

### Virtual Environments in IDEs

1. **VS Code**: Select interpreter from environment
2. **PyCharm**: Create/select virtual environment when creating project
3. **Jupyter Notebook**: Create kernel from environment
```bash
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

### Best Practices

1. **One environment per project**

2. **Include activation in project documentation**:
```markdown
## Setup
1. Create virtual environment: `python -m venv env`
2. Activate environment: `source env/bin/activate` (Linux/Mac) or `env\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
```

3. **Don't commit virtual environment directories**:
```gitignore
# .gitignore
env/
venv/
.env/
.venv/
```

4. **Standardize environment locations**:
   - Project-local: `.venv/` in project directory
   - Centralized: `~/.virtualenvs/project_name/`

5. **Record dependencies**:
```bash
pip freeze > requirements.txt
```

### Data Engineering Context

Virtual environments are critical for:
1. ETL pipelines with specific dependencies
2. ML model training with reproducible package versions
3. Development vs. production environment parity
4. Working with conflicting dependencies (e.g., TensorFlow vs. PyTorch)
5. Simplified deployment through containerization with virtual environments

## 38. How would you structure a complex data engineering Python project?

**Key Concept**: Well-structured data engineering projects use modular organization, clear separation of concerns, and follow industry best practices.

### Core Project Structure

```
data_pipeline/
├── README.md             # Project documentation
├── setup.py              # Package installation
├── requirements.txt      # Dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore file
├── data_pipeline/        # Main package
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration handling
│   ├── logging_conf.py   # Logging configuration
│   ├── main.py           # Entry point
│   ├── exceptions.py     # Custom exceptions
│   ├── extractors/       # Data extraction modules
│   │   ├── __init__.py
│   │   ├── base.py       # Base extractor class
│   │   ├── api.py        # API data extractor
│   │   ├── database.py   # Database extractor
│   │   └── file.py       # File data extractor
│   ├── transformers/     # Data transformation modules
│   │   ├── __init__.py
│   │   ├── base.py       # Base transformer
│   │   └── cleaners.py   # Data cleaning
│   ├── loaders/          # Data loading modules
│   │   ├── __init__.py
│   │   ├── base.py       # Base loader class
│   │   ├── database.py   # Database loader
│   │   └── file.py       # File loader
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   ├── date_utils.py # Date/time utilities
│   │   └── validators.py # Data validation
│   ├── models/           # Data models 
│   │   ├── __init__.py
│   │   └── schema.py     # Schema definitions
│   └── cli.py            # Command-line interface
├── config/               # Configuration files
│   ├── dev.yaml          # Development config
│   ├── prod.yaml         # Production config
│   └── logging.yaml      # Logging config
├── scripts/              # Utility scripts
│   ├── setup_env.sh      # Environment setup
│   └── deploy.py         # Deployment script
├── notebooks/            # Jupyter notebooks
│   └── exploration.ipynb # Data exploration
├── tests/                # Test suite
│   ├── __init__.py
│   ├── conftest.py       # Test fixtures
│   ├── test_extractors/
│   ├── test_transformers/
│   └── test_loaders/
└── docs/                 # Documentation
    ├── architecture.md   # System architecture
    ├── usage.md          # Usage guide
    └── maintenance.md    # Maintenance docs
```

### Core Application Components

1. **Configuration Management**:
```python
# config.py
import os
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path=None):
        self.config = {}
        
        # Load default config
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_path.exists():
            with open(default_path) as f:
                self.config = yaml.safe_load(f)
        
        # Load environment-specific config
        env = os.getenv("ENV", "dev")
        env_path = Path(__file__).parent.parent / "config" / f"{env}.yaml"
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f)
                self.config.update(env_config)
        
        # Override with custom config if provided
        if config_path:
            with open(config_path) as f:
                custom_config = yaml.safe_load(f)
                self.config.update(custom_config)
        
        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith("APP_"):
                # Convert APP_DATABASE_HOST to config["database"]["host"]
                parts = key[4:].lower().split("_")
                current = self.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
    
    def get(self, key, default=None):
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if k not in value:
                return default
            value = value[k]
        return value
```

2. **Logging Setup**:
```python
# logging_conf.py
import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(config_path=None, default_level=logging.INFO):
    """Setup logging configuration"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Default configuration
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
```

3. **Base Classes for ETL Components**:
```python
# extractors/base.py
from abc import ABC, abstractmethod
import logging

class BaseExtractor(ABC):
    """Base class for all data extractors."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def extract(self, *args, **kwargs):
        """Extract data from source."""
        pass
    
    def validate_source(self):
        """Validate that source is available."""
        pass
```

4. **ETL Pipeline Class**:
```python
# pipeline.py
import logging
from datetime import datetime

class Pipeline:
    """Data pipeline orchestration."""
    
    def __init__(self, name, extractor, transformer, loader, config):
        self.name = name
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def run(self, context=None):
        """Execute the pipeline."""
        start_time = datetime.now()
        self.logger.info(f"Starting pipeline: {self.name}")
        
        try:
            # Extract
            self.logger.info("Extraction phase")
            raw_data = self.extractor.extract(context)
            
            # Transform
            self.logger.info("Transformation phase")
            transformed_data = self.transformer.transform(raw_data, context)
            
            # Load
            self.logger.info("Loading phase")
            result = self.loader.load(transformed_data, context)
            
            # Complete
            elapsed = datetime.now() - start_time
            self.logger.info(f"Pipeline {self.name} completed in {elapsed}")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline {self.name} failed: {str(e)}")
            raise
```

5. **Command-Line Interface**:
```python
# cli.py
import argparse
import logging
import sys
from data_pipeline.config import Config
from data_pipeline.logging_conf import setup_logging
from data_pipeline.main import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Data Pipeline CLI")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--log-config", help="Path to logging config")
    parser.add_argument("--pipeline", required=True, help="Pipeline to run")
    parser.add_argument("--date", help="Processing date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_config)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        result = run_pipeline(args.pipeline, config, date=args.date)
        logger.info(f"Pipeline completed with result: {result}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Dependency Management

Use `setup.py` for proper packaging:

```python
from setuptools import setup, find_packages

setup(
    name="data-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "requests>=2.27.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "pylint", "mypy"],
        "aws": ["boto3", "s3fs"],
    },
    entry_points={
        "console_scripts": [
            "data-pipeline=data_pipeline.cli:main",
        ],
    },
)
```

### Testing Strategy

1. **Unit tests** for components:
```python
# tests/test_extractors/test_api.py
import pytest
from unittest.mock import patch, MagicMock
from data_pipeline.extractors.api import APIExtractor

def test_api_extractor_extract():
    # Mock the requests library
    with patch('requests.get') as mock_get:
        # Configure the mock
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": 1, "value": "test"}]}
        
        # Create extractor with mock config
        config = {"api": {"url": "https://example.com/api", "key": "test_key"}}
        extractor = APIExtractor(config)
        
        # Execute the method
        result = extractor.extract()
        
        # Assertions
        assert result == {"data": [{"id": 1, "value": "test"}]}
        mock_get.assert_called_once_with(
            "https://example.com/api",
            headers={"Authorization": "Bearer test_key"}
        )
```

2. **Integration tests** with actual components:
```python
# tests/test_integration/test_pipeline.py
import pytest
import tempfile
import os
from data_pipeline.extractors.file import CSVExtractor
from data_pipeline.transformers.cleaners import DataCleaner
from data_pipeline.loaders.file import CSVLoader
from data_pipeline.pipeline import Pipeline

@pytest.fixture
def sample_data():
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("id,name,value\n")
        f.write("1,Item 1,100\n")
        f.write("2,Item 2,200\n")
        f.write("3,,300\n")  # Missing name to be cleaned
    
    filename = f.name
    yield filename
    os.unlink(filename)

def test_csv_etl_pipeline(sample_data):
    # Create config
    config = {
        "extractor": {"file_path": sample_data},
        "transformer": {"fill_nulls": True},
        "loader": {"output_path": "output.csv"}
    }
    
    # Create pipeline components
    extractor = CSVExtractor(config)
    transformer = DataCleaner(config)
    loader = CSVLoader(config)
    
    # Create and run pipeline
    pipeline = Pipeline("test", extractor, transformer, loader, config)
    result = pipeline.run()
    
    # Verify output file exists and has correct data
    assert os.path.exists("output.csv")
    
    # Clean up
    os.unlink("output.csv")
```

### Documentation Standards

1. **Code Documentation**:
   - Module docstrings
   - Class and function docstrings
   - Type hints

2. **Project Documentation**:
   - README.md with project overview, setup, and usage
   - Architecture documentation
   - API documentation 
   - Deployment guides

3. **Example Documentation**:
```python
def transform_data(
    data: pd.DataFrame, 
    columns_to_clean: List[str],
    null_strategy: str = "mean"
) -> pd.DataFrame:
    """
    Clean and transform the input DataFrame.
    
    Args:
        data: Input DataFrame to transform
        columns_to_clean: List of column names to apply cleaning to
        null_strategy: Strategy for handling null values ('mean', 'median', 'drop', 'fill')
    
    Returns:
        Transformed DataFrame
        
    Raises:
        ValueError: If null_strategy is invalid
    
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
        >>> transform_data(df, ['A', 'B'], 'mean')
          A    B
        0 1.0  4.0
        1 2.0  5.0
        2 1.5  6.0
    """
    # Implementation
```

### Best Practices

1. **Modular Design**: 
   - Separation of concerns 
   - Single responsibility principle
   - Small, testable functions

2. **Abstraction Layers**:
   - Abstract base classes
   - Interface definitions
   - Dependency injection

3. **Error Handling**:
   - Custom exceptions
   - Proper logging
   - Graceful failure modes

4. **Code Quality**:
   - Consistent code style (Black, isort)
   - Linting (pylint, flake8)
   - Type checking (mypy)
   - Code reviews

5. **Scalability Considerations**:
   - Chunked processing
   - Memory-efficient operations
   - Parallelization options

## 39. What is the purpose of `__init__.py` files in Python packages?

**Key Concept**: `__init__.py` files define directories as Python packages and control the package's namespace.

### Core Functions of `__init__.py`

1. **Package Declaration**: 
   - Marks a directory as a Python package
   - Makes the directory importable

2. **Initialization Code**: 
   - Runs when package is imported
   - Performs package setup

3. **Namespace Management**:
   - Controls what names are exposed with `import *`
   - Creates package-level convenience imports

4. **Package Structure Definition**:
   - Can include subpackage imports
   - Defines package hierarchy

### Basic Usage

1. **Empty `__init__.py`**:
```python
# simplest form - empty file
# just marks directory as a package
```

2. **Package Variables**:
```python
# mypackage/__init__.py
__version__ = '1.0.0'
__author__ = 'Your Name'
```

3. **Exposing Submodule Functionality**:
```python
# mypackage/__init__.py
from .submodule1 import useful_function, UsefulClass
from .submodule2 import another_function

__all__ = ['useful_function', 'UsefulClass', 'another_function']
```

### Advanced Uses

1. **Lazy Loading with `__getattr__`** (Python 3.7+):
```python
# Defer importing heavy modules until needed
def __getattr__(name):
    if name == 'heavy_module':
        import .heavy_module as heavy
        return heavy
    raise AttributeError(f"module {__name__} has no attribute {name}")
```

2. **Package Initialization**:
```python
# database/__init__.py
import logging
from .connection import create_connection

# Initialize logging
logger = logging.getLogger(__name__)

# Global connection pool
connection_pool = None

def init_app(config):
    """Initialize the database connection pool."""
    global connection_pool
    logger.info("Initializing database connection pool")
    connection_pool = create_connection(
        host=config.get('DB_HOST'),
        port=config.get('DB_PORT'),
        user=config.get('DB_USER'),
        password=config.get('DB_PASSWORD'),
        database=config.get('DB_NAME'),
        pool_size=config.get('DB_POOL_SIZE', 5)
    )
    return connection_pool
```

3. **Creating Package-Level Classes/Functions**:
```python
# analytics/__init__.py
from .metrics import calculate_mean, calculate_median
from .visualizations import plot_histogram

class Analytics:
    """Main analytics interface combining functionality from submodules."""
    
    def __init__(self, data):
        self.data = data
    
    def basic_stats(self):
        """Calculate basic statistics for the data."""
        return {
            'mean': calculate_mean(self.data),
            'median': calculate_median(self.data)
        }
    
    def visualize(self, output_path=None):
        """Create visualizations of the data."""
        return plot_histogram(self.data, output_path)

# Allow from analytics import Analytics
__all__ = ['Analytics', 'calculate_mean', 'calculate_median', 'plot_histogram']
```

### Package Structure Examples

1. **Simple Package**:
```
mypackage/
├── __init__.py
├── module1.py
└── module2.py
```

```python
# mypackage/__init__.py
from .module1 import function1
from .module2 import function2

__all__ = ['function1', 'function2']
```

2. **Nested Package**:
```
data_processing/
├── __init__.py
├── readers/
│   ├── __init__.py
│   ├── csv_reader.py
│   └── json_reader.py
└── writers/
    ├── __init__.py
    ├── csv_writer.py
    └── json_writer.py
```

```python
# data_processing/__init__.py
from .readers.csv_reader import read_csv
from .readers.json_reader import read_json
from .writers.csv_writer import write_csv
from .writers.json_writer import write_json

__all__ = ['read_csv', 'read_json', 'write_csv', 'write_json']
```

```python
# data_processing/readers/__init__.py
from .csv_reader import read_csv
from .json_reader import read_json

__all__ = ['read_csv', 'read_json']
```

3. **Data Engineering Package**:
```
etl_framework/
├── __init__.py
├── extractors/
│   ├── __init__.py
│   ├── base.py
│   ├── database.py
│   └── api.py
├── transformers/
│   ├── __init__.py
│   └── ...
└── loaders/
    ├── __init__.py
    └── ...
```

```python
# etl_framework/__init__.py
from .extractors import DatabaseExtractor, APIExtractor
from .transformers import DataTransformer
from .loaders import DatabaseLoader

# Create simple ETL pipeline function
def create_pipeline(extractor, transformer, loader):
    """Create and return an ETL pipeline."""
    def pipeline():
        data = extractor.extract()
        transformed = transformer.transform(data)
        loader.load(transformed)
    return pipeline

__all__ = [
    'DatabaseExtractor', 'APIExtractor',
    'DataTransformer',
    'DatabaseLoader',
    'create_pipeline'
]
```

### Best Practices

1. **Keep `__init__.py` files simple and focused**
   - Avoid complex logic
   - Move implementation to dedicated modules

2. **Use `__all__` to control exports**
   - List explicitly what should be available with `from package import *`
   - Only expose the public API

3. **Use relative imports within packages**
   - `from .module import function`
   - `from ..sibling_package import function`

4. **Avoid circular imports**
   - Use forward references or import inside functions

5. **Version numbering in the base package**
   - `__version__ = '1.2.3'`

6. **Consider namespace packages** (Python 3.3+)
   - Directories without `__init__.py` can still be packages
   - Useful for distributing parts of a namespace across multiple installations

## 40. How do you create and distribute a custom Python package?

**Key Concept**: Creating and distributing a Python package involves setting up proper directory structure, defining metadata, and publishing to a package repository like PyPI.

### Package Creation Process

#### 1. Directory Structure

```
my_package/
├── LICENSE
├── README.md
├── pyproject.toml  # Modern build configuration
├── setup.py        # Traditional setup script
├── setup.cfg       # Configuration for setup.py
├── MANIFEST.in     # Additional files to include
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── subpackage/
│       ├── __init__.py
│       └── module2.py
└── tests/
    ├── test_module1.py
    └── test_subpackage.py
```

#### 2. Package Metadata and Setup

**Modern Approach (recommended)** - Using `pyproject.toml`:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
version = "0.1.0"
description = "A data processing toolkit"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.7"
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "black",
    "mypy",
]
aws = [
    "boto3",
    "s3fs",
]

[project.urls]
Homepage = "https://github.com/yourusername/my_package"
Documentation = "https://my_package.readthedocs.io/"
Issues = "https://github.com/yourusername/my_package/issues"

[project.scripts]
my-command = "my_package.cli:main"
```

**Traditional Approach** - Using `setup.py`:

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="my_package",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A data processing toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[


# 41-45 

def __init__(self, x, y, category):
        self.x = x
        self.y = y
        self.category = category

# Memory comparison
import sys

points_normal = [DataPoint(i, i*2, "A") for i in range(1000000)]
points_slots = [DataPointSlots(i, i*2, "A") for i in range(1000000)]

size_normal = sys.getsizeof(points_normal[0]) * len(points_normal)
size_slots = sys.getsizeof(points_slots[0]) * len(points_slots)

print(f"Normal: {size_normal / 1e6:.2f} MB")
print(f"With slots: {size_slots / 1e6:.2f} MB")
```

### 12. Data Pipeline Memory Optimization

Putting it all together in a data pipeline:

```python
import pandas as pd
import numpy as np
from datetime import datetime
import os

def process_large_dataset(input_dir, output_file, chunk_size=100000):
    """Process a collection of large CSV files with memory optimization."""
    
    # Get all CSV files
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.endswith('.csv')]
    
    # Create an empty results DataFrame
    all_results = pd.DataFrame()
    
    # Process each file
    for file_idx, file_path in enumerate(csv_files):
        print(f"Processing file {file_idx+1}/{len(csv_files)}: {file_path}")
        
        # Process one file at a time in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            print(f"  Processing chunk {chunk_idx+1}")
            
            # Optimize memory usage: convert dtypes
            chunk = optimize_dtypes(chunk)
            
            # Process chunk
            result = process_chunk(chunk)
            
            # Append to results
            if len(all_results) == 0:
                all_results = result
            else:
                all_results = pd.concat([all_results, result], ignore_index=True)
            
            # Periodic save to avoid excessive memory usage
            if len(all_results) >= 1000000:
                intermediate_file = f"{output_file}.part{file_idx}.{chunk_idx}"
                all_results.to_csv(intermediate_file, index=False)
                print(f"  Saved intermediate results to {intermediate_file}")
                all_results = pd.DataFrame()  # Clear memory
    
    # Final save or combine intermediate files
    if len(all_results) > 0:
        all_results.to_csv(output_file, index=False)
    else:
        # Combine intermediate files
        combine_intermediate_files(output_file)

def optimize_dtypes(df):
    """Optimize DataFrame dtypes to reduce memory usage."""
    # Implementation from earlier example
    # ...
    return df

def process_chunk(chunk):
    """Process a single chunk of data."""
    # Implement your data processing logic here
    # Return a processed DataFrame
    return processed_chunk

def combine_intermediate_files(output_file):
    """Combine intermediate files into a single output file."""
    # Implementation details
    # ...
```

## 44. How do you profile and identify bottlenecks in a Python data pipeline?

**Key Concept**: Profiling helps identify performance bottlenecks in data pipelines by measuring execution time, memory usage, and resource consumption at various stages.

### 1. Basic Time Profiling

Start with simple timing measurements:

```python
import time

def time_function(func, *args, **kwargs):
    """Time the execution of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
    return result, execution_time

# Usage example
result, execution_time = time_function(process_data, large_dataset)
```

### 2. Function-Level Profiling with `cProfile`

Identify which functions consume the most time:

```python
import cProfile
import pstats
import io

def profile_function(func, *args, **kwargs):
    """Profile a function and print sorted stats."""
    # Create a profile object
    pr = cProfile.Profile()
    
    # Start profiling
    pr.enable()
    
    # Call the function
    result = func(*args, **kwargs)
    
    # End profiling
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 time-consuming functions
    print(s.getvalue())
    
    return result

# Usage
profile_function(process_large_dataset, 'data.csv')
```

### 3. Line-by-Line Profiling with `line_profiler`

Identify specific slow lines of code:

```python
# Install with: pip install line_profiler
# Add @profile decorator to the function you want to profile
@profile
def process_chunk(chunk):
    """Process a chunk of data."""
    # Preprocessing
    chunk = chunk.copy()
    
    # Clean data
    chunk['col1'] = chunk['col1'].str.strip()
    chunk['col2'] = chunk['col2'].fillna(0)
    
    # Complex calculation
    for idx, row in chunk.iterrows():
        chunk.loc[idx, 'result'] = complex_calculation(row)
    
    # Aggregate
    result = chunk.groupby('category').agg({'result': 'mean'})
    
    return result

# Run with: kernprof -l script.py
# Then view results: python -m line_profiler script.py.lprof
```

### 4. Memory Profiling with `memory_profiler`

Track memory usage:

```python
# Install with: pip install memory_profiler
from memory_profiler import profile as memory_profile

@memory_profile
def load_and_process_data(file_path):
    """Load and process a large dataset."""
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean data
    df = clean_data(df)
    
    # Transform
    df = transform_data(df)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    return stats

# Run normally: python script.py
# Output will show memory usage per line
```

### 5. Profiling a Multi-Stage Data Pipeline

Break down profiling by pipeline stages:

```python
import time
import pandas as pd
import psutil
import os

class ProfiledPipeline:
    def __init__(self, name):
        self.name = name
        self.stage_times = {}
        self.stage_memory = {}
    
    def run_stage(self, stage_name, stage_func, *args, **kwargs):
        """Run and profile a pipeline stage."""
        # Record start metrics
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run the stage
        result = stage_func(*args, **kwargs)
        
        # Record end metrics
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate and store metrics
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        self.stage_times[stage_name] = execution_time
        self.stage_memory[stage_name] = memory_used
        
        print(f"Stage '{stage_name}' completed in {execution_time:.2f} seconds")
        print(f"Memory change: {memory_used:.2f} MB")
        
        return result
    
    def print_summary(self):
        """Print a summary of all stage metrics."""
        print(f"\nPipeline: {self.name} - Performance Summary")
        print("-" * 50)
        print(f"{'Stage':<20} {'Time (s)':<10} {'Memory (MB)':<15}")
        print("-" * 50)
        
        total_time = 0
        for stage, time_taken in self.stage_times.items():
            memory = self.stage_memory[stage]
            print(f"{stage:<20} {time_taken:<10.2f} {memory:<15.2f}")
            total_time += time_taken
        
        print("-" * 50)
        print(f"{'Total':<20} {total_time:<10.2f}")

# Usage example
def run_data_pipeline(input_file):
    pipeline = ProfiledPipeline("DataProcessing")
    
    # Stage 1: Load data
    df = pipeline.run_stage("data_loading", pd.read_csv, input_file)
    
    # Stage 2: Clean data
    df = pipeline.run_stage("data_cleaning", clean_data, df)
    
    # Stage 3: Transform data
    df = pipeline.run_stage("transformation", transform_data, df)
    
    # Stage 4: Aggregate data
    results = pipeline.run_stage("aggregation", aggregate_data, df)
    
    # Print performance summary
    pipeline.print_summary()
    
    return results
```

### 6. Database Query Profiling

Profile database interactions in your pipeline:

```python
import time
import pandas as pd
from sqlalchemy import create_engine, text

def profile_sql_query(query, engine, params=None):
    """Profile a SQL query execution time."""
    start_time = time.time()
    
    # Execute query
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        data = result.fetchall()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Query took {execution_time:.4f} seconds")
    print(f"Returned {len(data)} rows")
    
    # Create DataFrame from results
    df = pd.DataFrame(data, columns=result.keys())
    return df, execution_time

# Usage example
engine = create_engine("postgresql://username:password@localhost/database")

queries_to_profile = [
    "SELECT * FROM large_table WHERE category = 'A'",
    "SELECT category, COUNT(*) FROM large_table GROUP BY category",
    """
    SELECT t1.id, t1.name, t2.value 
    FROM table1 t1 
    JOIN table2 t2 ON t1.id = t2.id 
    WHERE t2.value > 100
    """
]

for i, query in enumerate(queries_to_profile):
    print(f"\nQuery {i+1}:")
    print(query)
    df, time_taken = profile_sql_query(query, engine)
    print(f"Result shape: {df.shape}")
```

### 7. Visualizing Profiling Results

Convert profiling data into visualizations:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_profiling_results(profile_data):
    """Create visualizations of profiling results."""
    # Prepare data
    stages = list(profile_data['time'].keys())
    times = list(profile_data['time'].values())
    memory = list(profile_data['memory'].values())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times
    bars1 = ax1.bar(stages, times, color='skyblue')
    ax1.set_title('Execution Time by Pipeline Stage')
    ax1.set_xlabel('Pipeline Stage')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xticklabels(stages, rotation=45, ha='right')
    
    # Add time labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # Plot memory usage
    bars2 = ax2.bar(stages, memory, color='lightgreen')
    ax2.set_title('Memory Usage by Pipeline Stage')
    ax2.set_xlabel('Pipeline Stage')
    ax2.set_ylabel('Memory Change (MB)')
    ax2.set_xticklabels(stages, rotation=45, ha='right')
    
    # Add memory labels on bars
    for bar in bars2:
        height = bar.get_height()
        label = f'{height:.1f} MB'
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, height + np.sign(height)*0.1,
                label, ha='center', va=va)
    
    plt.tight_layout()
    plt.savefig('pipeline_profile.png')
    plt.show()

# Example usage
profile_data = {
    'time': {
        'Extract': 2.45,
        'Clean': 5.12,
        'Transform': 8.67,
        'Aggregate': 1.23,
        'Load': 3.89
    },
    'memory': {
        'Extract': 245.3,
        'Clean': -120.7,
        'Transform': 567.2,
        'Aggregate': -320.5,
        'Load': 89.4
    }
}

visualize_profiling_results(profile_data)
```

### 8. Profiling Distributed Data Processing

Profile tasks running on distributed workers:

```python
import time
from concurrent.futures import ProcessPoolExecutor
import psutil
import os

class DistributedProfiler:
    def __init__(self):
        self.task_profiles = {}
    
    def profile_task(self, task_id, func, *args, **kwargs):
        """Profile a task to be executed in a worker process."""
        def profiled_func(*args, **kwargs):
            # Initialize metrics
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss
            
            # Execute task
            result = func(*args, **kwargs)
            
            # Collect metrics
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss
            
            # Return result and metrics
            return {
                'result': result,
                'profile': {
                    'task_id': task_id,
                    'execution_time': end_time - start_time,
                    'memory_usage': (end_memory - start_memory) / (1024 * 1024),  # MB
                    'worker_pid': os.getpid()
                }
            }
        
        return profiled_func
    
    def run_distributed_tasks(self, tasks, max_workers=None):
        """Run and profile a set of tasks using a process pool."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit profiled tasks
            future_to_task = {
                executor.submit(self.profile_task(i, task['func']), *task['args'], **task.get('kwargs', {})): i
                for i, task in enumerate(tasks)
            }
            
            # Collect results and profiles
            for future in future_to_task:
                result_with_profile = future.result()
                task_id = future_to_task[future]
                
                # Store profile data
                self.task_profiles[task_id] = result_with_profile['profile']
                
                # Store actual result
                results.append(result_with_profile['result'])
        
        return results
    
    def print_task_profiles(self):
        """Print profile data for all tasks."""
        print("\nTask Profiles:")
        print("-" * 80)
        print(f"{'Task ID':<10} {'Worker PID':<12} {'Time (s)':<10} {'Memory (MB)':<15}")
        print("-" * 80)
        
        for task_id, profile in sorted(self.task_profiles.items()):
            print(f"{profile['task_id']:<10} {profile['worker_pid']:<12} "
                  f"{profile['execution_time']:<10.2f} {profile['memory_usage']:<15.2f}")

# Usage example
def process_chunk(chunk_data, chunk_id):
    """Process a chunk of data."""
    # Simulate processing
    time.sleep(chunk_id * 0.1)  # Different execution times for illustration
    return f"Processed chunk {chunk_id}: {len(chunk_data)} items"

# Create sample chunks
chunks = [list(range(i*1000, (i+1)*1000)) for i in range(10)]

# Define tasks
tasks = [
    {'func': process_chunk, 'args': (chunk, i)}
    for i, chunk in enumerate(chunks)
]

# Profile and run distributed tasks
profiler = DistributedProfiler()
results = profiler.run_distributed_tasks(tasks, max_workers=4)

# Print task profiles
profiler.print_task_profiles()
```

### 9. Pandas Performance Analysis

Analyze pandas operations performance:

```python
import pandas as pd
import time

def profile_pandas_operation(name, operation, df, n_repeats=3):
    """Profile a pandas operation with multiple repeats."""
    times = []
    
    for i in range(n_repeats):
        start = time.time()
        result = operation(df.copy())
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Operation '{name}' average time: {avg_time:.4f}s")
    
    return result, avg_time

# Example usage
df = pd.read_csv('large_file.csv')

# Define operations to profile
operations = {
    'Filter rows': lambda df: df[df['value'] > df['value'].mean()],
    'Group by with agg': lambda df: df.groupby('category').agg({'value': ['mean', 'sum', 'count']}),
    'Apply function': lambda df: df.apply(lambda row: complex_calculation(row), axis=1),
    'Join datasets': lambda df: pd.merge(df, df.groupby('category').mean().reset_index(), on='category'),
    'Sort large dataset': lambda df: df.sort_values(['category', 'value']),
}

results = {}
for name, operation in operations.items():
    _, time_taken = profile_pandas_operation(name, operation, df)
    results[name] = time_taken

# Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(list(results.keys()), list(results.values()), color='skyblue')
plt.xlabel('Time (seconds)')
plt.title('Pandas Operations Performance')
plt.tight_layout()
plt.show()
```

### 10. Profiling Disk I/O Operations

Measure file operations performance:

```python
import os
import time
import pandas as pd

def profile_io_operation(operation_name, func, *args, **kwargs):
    """Profile I/O operations."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Try to get file size if applicable
    file_size_mb = None
    if 'path' in kwargs and os.path.exists(kwargs['path']):
        file_size_mb = os.path.getsize(kwargs['path']) / (1024 * 1024)
    elif len(args) > 0 and isinstance(args[0], str) and os.path.exists(args[0]):
        file_size_mb = os.path.getsize(args[0]) / (1024 * 1024)
    
    if file_size_mb is not None:
        throughput = file_size_mb / execution_time
        print(f"{operation_name}: {execution_time:.2f}s for {file_size_mb:.2f}MB ({throughput:.2f}MB/s)")
    else:
        print(f"{operation_name}: {execution_time:.2f}s")
    
    return result, execution_time

# Example usage
def test_file_operations():
    # Create test data
    df = pd.DataFrame({
        'A': range(1000000),
        'B': range(1000000, 2000000),
        'C': ['value'] * 1000000
    })
    
    # Profile CSV write
    _, write_time = profile_io_operation(
        "CSV Write", 
        df.to_csv, 
        'test_large.csv', 
        index=False
    )
    
    # Profile CSV read
    _, read_time = profile_io_operation(
        "CSV Read", 
        pd.read_csv, 
        'test_large.csv'
    )
    
    # Profile Parquet write
    _, parquet_write_time = profile_io_operation(
        "Parquet Write", 
        df.to_parquet, 
        'test_large.parquet', 
        index=False
    )
    
    # Profile Parquet read
    _, parquet_read_time = profile_io_operation(
        "Parquet Read", 
        pd.read_parquet, 
        'test_large.parquet'
    )
    
    # Compare formats
    print("\nFormat Comparison:")
    print(f"CSV Read/Write: {read_time:.2f}s / {write_time:.2f}s")
    print(f"Parquet Read/Write: {parquet_read_time:.2f}s / {parquet_write_time:.2f}s")
    print(f"Parquet is {read_time/parquet_read_time:.1f}x faster for reading")
    print(f"Parquet is {write_time/parquet_write_time:.1f}x faster for writing")
    
    # Clean up
    for file in ['test_large.csv', 'test_large.parquet']:
        if os.path.exists(file):
            os.remove(file)

test_file_operations()
```

### 11. Combining Profiling Methods for ETL Pipeline

Comprehensive profiling of an ETL pipeline:

```python
import time
import pandas as pd
import psutil
import os
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import cProfile
import pstats
import io

class ETLProfiler:
    """Profile the performance of ETL pipeline stages."""
    
    def __init__(self, name):
        self.name = name
        self.profiles = {}
    
    def profile_stage(self, stage_name, func, *args, **kwargs):
        """Profile a pipeline stage comprehensively."""
        profile = {}
        
        # Time execution
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run cProfile
        pr = cProfile.Profile()
        pr.enable()
        
        # Execute function with memory profiling
        mem_usage, result = self._profile_memory_usage(func, *args, **kwargs)
        
        # Stop cProfile
        pr.disable()
        
        # Calculate metrics
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Collect metrics
        profile['execution_time'] = end_time - start_time
        profile['memory_start'] = start_memory
        profile['memory_end'] = end_memory
        profile['memory_diff'] = end_memory - start_memory
        profile['memory_profile'] = mem_usage
        
        # Get cProfile stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        profile['cprofile_stats'] = s.getvalue()
        
        # Store profile
        self.profiles[stage_name] = profile
        
        # Print summary
        print(f"Stage: {stage_name}")
        print(f"  Time: {profile['execution_time']:.2f}s")
        print(f"  Memory change: {profile['memory_diff']:.2f}MB")
        print(f"  Peak memory: {max(mem_usage) - min(mem_usage):.2f}MB")
        
        return result
    
    def _profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function."""
        # Define wrapper function
        def wrapper():
            return func(*args, **kwargs)
        
        # Profile memory usage
        mem_usage = memory_usage(wrapper, interval=0.1, timeout=None, 
                                max_iterations=1, include_children=True)
        
        return mem_usage, wrapper()
    
    def print_stage_cprofile(self, stage_name):
        """Print detailed cProfile stats for a stage."""
        if stage_name in self.profiles:
            print(f"\nDetailed Profile for {stage_name}:")
            print("-" * 80)
            print(self.profiles[stage_name]['cprofile_stats'])
        else:
            print(f"No profile data found for stage: {stage_name}")
    
    def visualize_results(self):
        """Create visualizations of profiling results."""
        if not self.profiles:
            print("No profiling data to visualize")
            return
        
        stages = list(self.profiles.keys())
        times = [self.profiles[s]['execution_time'] for s in stages]
        memory_diffs = [self.profiles[s]['memory_diff'] for s in stages]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot execution times
        bars1 = ax1.bar(stages, times, color='skyblue')
        ax1.set_title('Execution Time by Pipeline Stage')
        ax1.set_xlabel('Pipeline Stage')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticklabels(stages, rotation=45, ha='right')
        
        # Add time labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Plot memory usage
        bars2 = ax2.bar(stages, memory_diffs, color='lightgreen')
        ax2.set_title('Memory Usage Change by Pipeline Stage')
        ax2.set_xlabel('Pipeline Stage')
        ax2.set_ylabel('Memory Change (MB)')
        ax2.set_xticklabels(stages, rotation=45, ha='right')
        
        # Add memory labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{height:.2f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.name}_profile.png")
        plt.show()
        
        # Create memory usage over time visualization
        plt.figure(figsize=(12, 6))
        for stage in stages:
            mem_profile = self.profiles[stage]['memory_profile']
            timeline = [i * 0.1 for i in range(len(mem_profile))]  # 0.1s intervals
            plt.plot(timeline, mem_profile, label=stage)
        
        plt.title('Memory Usage Over Time by Stage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.name}_memory_timeline.png")
        plt.show()

# Example usage in an ETL pipeline
def run_etl_pipeline():
    profiler = ETLProfiler("MyETLPipeline")
    
    # Extract
    data = profiler.profile_stage("Extract", 
                                extract_data, 
                                source="database", 
                                query="SELECT * FROM large_table")
    
    # Transform
    transformed = profiler.profile_stage("Transform",
                                        transform_data,
                                        data)
    
    # Load
    profiler.profile_stage("Load",
                          load_data,
                          transformed,
                          destination="warehouse")
    
    # Print detailed profile for the slowest stage
    stages = list(profiler.profiles.keys())
    times = [profiler.profiles[s]['execution_time'] for s in stages]
    slowest_stage = stages[times.index(max(times))]
    profiler.print_stage_cprofile(slowest_stage)
    
    # Visualize results
    profiler.visualize_results()
```

## 45. Explain lazy evaluation in Python and how it can improve performance.

**Key Concept**: Lazy evaluation defers computation until results are actually needed, improving memory efficiency and potentially reducing unnecessary work.

### 1. Understanding Lazy Evaluation

Lazy evaluation contrasts with eager evaluation:

```python
# Eager evaluation: 
# - Computes all values immediately
# - Stores all results in memory
squares_eager = [x**2 for x in range(1000000)]  # Creates a list with 1M elements

# Lazy evaluation:
# - Computes values only when needed
# - Doesn't store all results in memory at once
squares_lazy = (x**2 for x in range(1000000))  # Creates a generator object

# Using the lazy version
for i, square in enumerate(squares_lazy):
    if i < 5:
        print(square)
    else:
        break  # Only computes the first 5 squares
```

### 2. Generators for Lazy Evaluation

Generator functions and expressions are the primary way to implement lazy evaluation:

```python
# Generator function
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Consume values lazily
fib_gen = fibonacci()
for _ in range(10):
    print(next(fib_gen))  # Computes only what's needed

# Generator expression
even_squares = (x**2 for x in range(1000) if x % 2 == 0)

# Process values one at a time
for square in even_squares:
    process(square)
```

### 3. Lazy Evaluation in File Processing

Process large files without loading entirely into memory:

```python
def process_large_file(filename):
    """Process a large file line by line."""
    with open(filename, 'r') as file:
        for line in file:  # File objects are iterators - lazy evaluation
            yield line.strip()

def filter_lines(lines, keyword):
    """Filter lines containing a keyword."""
    for line in lines:
        if keyword in line:
            yield line

def transform_lines(lines, transform_func):
    """Apply transformation to lines."""
    for line in lines:
        yield transform_func(line)

# Pipeline of lazy operations
def process_log_file(filename, keyword):
    lines = process_large_file(filename)
    filtered_lines = filter_lines(lines, keyword)
    transformed_lines = transform_lines(filtered_lines, str.upper)
    
    # Only now are the operations actually performed, and only on lines that pass through the pipeline
    for line in transformed_lines:
        print(line)
```

### 4. Lazy Evaluation with `itertools`

The `itertools` module provides functions for lazy iterator operations:

```python
import itertools

# Generate infinite sequence lazily
numbers = itertools.count(start=0, step=2)  # Even numbers: 0, 2, 4...

# Take only what's needed
first_five = list(itertools.islice(numbers, 5))
print(first_five)  # [0, 2, 4, 6, 8]

# Infinite cycle through a finite sequence
colors = itertools.cycle(['red', 'green', 'blue'])
for _ in range(7):
    print(next(colors))  # Prints: red, green, blue, red, green, blue, red

# Chain multiple iterables lazily
combined = itertools.chain(range(3), 'ABC', [10, 20, 30])
print(list(combined))  # [0, 1, 2, 'A', 'B', 'C', 10, 20, 30]

# Group consecutive items
data = "AAAABBBCCDAABBB"
for key, group in itertools.groupby(data):
    print(key, list(group))

# Combine multiple iterables (like zip, but continues until all are exhausted)
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35, 40]  # One more item than names
for name, age in itertools.zip_longest(names, ages, fillvalue='Unknown'):
    print(f"{name}: {age}")
```

### 5. Lazy Evaluation with Pandas

Utilize lazy operations in Pandas:

```python
import pandas as pd

# Using chunksize parameter for lazy reading
def process_csv_lazily(filepath, chunksize=10000):
    # Returns an iterator over DataFrame chunks
    chunk_iterator = pd.read_csv(filepath, chunksize=chunksize)
    
    results = []
    for chunk in chunk_iterator:
        # Process each chunk
        processed = process_chunk(chunk)
        results.append(processed)
    
    # Combine results only at the end
    return pd.concat(results)

# Dask enables lazy evaluation with pandas-like API
import dask.dataframe as dd

def process_with_dask(filepath):
    # Reading is lazy - nothing is loaded yet
    ddf = dd.read_csv(filepath)
    
    # Transformations are lazy - just building the task graph
    result = (ddf
              .assign(C=ddf.A + ddf.B)
              .groupby('key')
              .agg({'C': 'mean'}))
    
    # Computation is performed only when explicitly requested
    return result.compute()
```

### 6. Building Custom Lazy Data Processing Pipelines

Create reusable components for lazy operations:

```python
class LazyPipeline:
    """A pipeline that lazily processes data through a series of transformations."""
    
    def __init__(self, source_data):
        """Initialize with a data source (can be any iterable)."""
        self.source = iter(source_data)
        self.operations = []
    
    def map(self, func):
        """Add a mapping operation to the pipeline."""
        self.operations.append(lambda data: map(func, data))
        return self
    
    def filter(self, predicate):
        """Add a filtering operation to the pipeline."""
        self.operations.append(lambda data: filter(predicate, data))
        return self
    
    def take(self, n):
        """Limit the output to n items."""
        self.operations.append(lambda data: itertools.islice(data, n))
        return self
    
    def reduce(self, func, initial=None):
        """Reduce the pipeline to a single value."""
        # Execute the pipeline
        result = self.execute()
        
        # Apply the reduce function
        if initial is not None:
            return functools.reduce(func, result, initial)
        return functools.reduce(func, result)
    
    def execute(self):
        """Execute the pipeline and return an iterator of results."""
        data = self.source
        for operation in self.operations:
            data = operation(data)
        return data
    
    def collect(self):
        """Execute the pipeline and collect results into a list."""
        return list(self.execute())

# Example usage
numbers = range(1, 1000000)  # Large range

# Create a pipeline
result = (LazyPipeline(numbers)
          .filter(lambda x: x % 3 == 0)  # Divisible by 3
          .map(lambda x: x**2)           # Square the values
          .filter(lambda x: x < 10000)   # Filter large values
          .take(5)                       # Take only first 5 results
          .collect())                    # Materialize the results

print(result)  # [9, 36, 81, 144, 225]
```

### 7. Lazy Property Evaluation in Classes

Implement lazily evaluated properties:

```python
class DataProcessor:
    """Class with lazily evaluated properties."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self._data = None  # Will be loaded lazily
        
    @property
    def data(self):
        """Lazily load data when first accessed."""
        if self._data is None:
            print("Loading data from file...")
            self._data = pd.read_csv(self.filepath)
        return self._data
    
    @property
    def summary_stats(self):
        """Lazily compute summary statistics."""
        return self.data.describe()
    
    @property
    def missing_values(self):
        """Lazily compute missing value counts."""
        return self.data.isna().sum()
    
    def process(self):
        """Only now do we actually load data and compute statistics."""
        print("Processing data:")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values:\n{self.missing_values}")
        print(f"Summary:\n{self.summary_stats}")

# Create processor
processor = DataProcessor('large_file.csv')

# No data is loaded yet
print("Created processor, but data not loaded yet")

# Data is loaded only when needed
processor.process()
```

### 8. Lazy Database Queries with SQLAlchemy

Delay database queries until data is actually needed:

```python
from sqlalchemy import create_engine, Column, Integer, String, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

# Set up database connection
engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def lazy_database_operations():
    session = Session()
    
    # Build query - lazily evaluated, nothing executed yet
    query = select(User).where(User.name.like('A%'))
    
    # Query executed only when results are needed
    users = session.execute(query)
    
    # Processing users one at a time
    for user in users:
        process_user(user)
        
    # Close session
    session.close()
```

### 9. Real-World Example: Lazy ETL Pipeline

A complete example of lazy evaluation in an ETL pipeline:

```python
import csv
import itertools
from datetime import datetime

class LazyETLPipeline:
    """An ETL pipeline using lazy evaluation throughout."""
    
    def extract(self, source):
        """Extract data from source (file, API, database, etc.)."""
        if source.endswith('.csv'):
            return self._extract_from_csv(source)
        elif source.endswith('.json'):
            return self._extract_from_json(source)
        else:
            raise ValueError(f"Unsupported source: {source}")
    
    def _extract_from_csv(self, filepath):
        """Lazily extract data from CSV file."""
        def csv_reader():
            with open(filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
        
        return csv_reader()
    
    def transform(self, data, transformations):
        """Apply transformations lazily."""
        result = data
        for transform in transformations:
            result = transform(result)
        return result
    
    def load(self, data, destination):
        """Load transformed data to destination."""
        if destination.endswith('.csv'):
            return self._load_to_csv(data, destination)
        else:
            raise ValueError(f"Unsupported destination: {destination}")
    
    def _load_to_csv(self, data, filepath):
        """Lazily write data to CSV file."""
        # Get first row to determine headers
        try:
            first_row = next(data)
        except StopIteration:
            print("No data to write")
            return 0
        
        # Write data
        count = 1  # Count the first row
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=first_row.keys())
            writer.writeheader()
            writer.writerow(first_row)
            
            for row in data:
                writer.writerow(row)
                count += 1
        
        return count

# Example transformations
def clean_dates(data):
    """Convert date strings to datetime objects."""
    for row in data:
        if 'date' in row:
            try:
                row['date'] = datetime.strptime(row['date'], '%Y-%m-%d')
            except ValueError:
                row['date'] = None
        yield row

def filter_invalid_records(data):
    """Filter out records with missing required fields."""
    for row in data:
        if row.get('id') and row.get('value') is not None:
            yield row

def add_calculated_fields(data):
    """Add calculated fields to each record."""
    for row in data:
        if 'value' in row:
            row['value_squared'] = float(row['value']) ** 2
        yield row

def limit_records(data, max_records):
    """Limit the number of records processed."""
    return itertools.islice(data, max_records)

# Using the pipeline
def main():
    pipeline = LazyETLPipeline()
    
    # Extract data
    source_data = pipeline.extract('source.csv')
    
    # Transform data lazily
    transformations = [
        clean_dates,
        filter_invalid_records,
        add_calculated_fields,
        lambda data: limit_records(data, 1000)  # Process only 1000 records
    ]
    transformed_data = pipeline.transform(source_data, transformations)
    
    # Load transformed data
    records_written = pipeline.load(transformed_data, 'destination.csv')
    print(f"Processed {records_written} records")

# This pipeline processes data lazily, loading and transforming only what's needed
# and conserving memory throughout the entire process
```

### 10. Performance Benefits of Lazy Evaluation

Let's examine specific performance advantages of lazy evaluation:

#### Memory Efficiency

```python
import sys
import itertools

# Eager evaluation (list)
eager = [x**2 for x in range(1000000)]
eager_size = sys.getsizeof(eager) + sum(sys.getsizeof(x) for x in eager[:10])

# Lazy evaluation (generator)
lazy = (x**2 for x in range(1000000))
lazy_size = sys.getsizeof(lazy)

print(f"Eager evaluation size: {eager_size / 1024 / 1024:.2f} MB")
print(f"Lazy evaluation size: {lazy_size / 1024:.2f} KB")

# Result would show that lazy evaluation uses orders of magnitude less memory
```

#### Processing Speed for Filtered Data

```python
import time

def benchmark_eager_vs_lazy(data_size, filter_ratio):
    """Compare eager vs lazy for filtered data processing."""
    # Create data
    data = range(data_size)
    
    # Eager approach
    start = time.time()
    result_eager = [x**2 for x in data if x % filter_ratio == 0]
    eager_time = time.time() - start
    
    # Lazy approach
    start = time.time()
    result_lazy = list(x**2 for x in data if x % filter_ratio == 0)
    lazy_time = time.time() - start
    
    print(f"Data size: {data_size}, Filter ratio: 1/{filter_ratio}")
    print(f"Eager time: {eager_time:.4f}s")
    print(f"Lazy time: {lazy_time:.4f}s")
    print(f"Ratio: {eager_time/lazy_time:.2f}x")

# Run benchmarks
benchmark_eager_vs_lazy(10000000, 10)  # Only process 1/10 of the data
benchmark_eager_vs_lazy(10000000, 100)  # Only process 1/100 of the data
benchmark_eager_vs_lazy(10000000, 1000)  # Only process 1/1000 of the data

# The lazy approach should show increasingly better performance as the filter ratio increases
```

#### Short-Circuit Evaluation

```python
import time

def slow_function(x):
    """Simulate a slow computation."""
    time.sleep(0.1)
    return x > 100

# Eager evaluation must process all items
def eager_approach(data):
    start = time.time()
    result = all([slow_function(x) for x in data])  # Computes all values first
    end = time.time()
    return result, end - start

# Lazy evaluation can short-circuit
def lazy_approach(data):
    start = time.time()
    result = all(slow_function(x) for x in data)  # Stops at first False
    end = time.time()
    return result, end - start

# Test with data containing an early failure
test_data = [150, 200, 80, 300, 400, 500]  # 80 will cause `all` to return False

eager_result, eager_time = eager_approach(test_data)
lazy_result, lazy_time = lazy_approach(test_data)

print(f"Eager: {eager_result}, Time: {eager_time:.2f}s")
print(f"Lazy: {lazy_result}, Time: {lazy_time:.2f}s")

# Lazy evaluation should be about 3x faster by stopping after the third item
```

### 11. Best Practices for Lazy Evaluation

1. **Use generators for large data transformations**
   - Generator functions with `yield`
   - Generator expressions with parentheses `()`

2. **Chain operations in a pipeline**
   - Compose multiple lazy operations before consuming results
   - Process data in a single pass when possible

3. **Process data incrementally**
   - Avoid collecting all results if only some are needed
   - Use `itertools` functions like `islice`, `takewhile`

4. **Be careful with caching effects**
   - Remember generators are one-time use
   - Use `itertools.tee` to create multiple iterators from one generator

5. **Monitor memory usage**
   - Lazy evaluation helps avoid memory spikes
   - Watch for accidental materialization of entire datasets

6. **Consider tools built for lazy evaluation**
   - Dask for pandas-like lazy operations
   - SQLAlchemy for lazy database queries
   - Vaex for out-of-core dataframes# Brief Answers to Python Interview Questions 41-45

## 41. What strategies would you use to optimize Python code performance?

**Key Concept**: Python performance optimization involves choosing efficient algorithms, leveraging built-in functions, using appropriate data structures, and integrating with optimized libraries.

### 1. Algorithmic Optimization

The most significant performance improvements come from using better algorithms:

```python
# Inefficient: O(n²) approach
def find_duplicates_inefficient(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

# Efficient: O(n) approach using set
def find_duplicates_efficient(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)
```

### 2. Use Built-in Functions and Libraries

Python's built-in functions and standard library are implemented in C and highly optimized:

```python
# Slow: Manual implementation
def calculate_average_manual(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# Fast: Using built-in functions
def calculate_average_builtin(numbers):
    return sum(numbers) / len(numbers)

# Even faster: Using NumPy
import numpy as np
def calculate_average_numpy(numbers):
    return np.mean(numbers)
```

### 3. Vectorization with NumPy/Pandas

Replace loops with vectorized operations:

```python
import numpy as np
import pandas as pd

# Slow: Using loops
def normalize_loop(data):
    result = []
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    for val in data:
        normalized = (val - min_val) / range_val
        result.append(normalized)
    return result

# Fast: Using NumPy vectorization
def normalize_numpy(data):
    data_array = np.array(data)
    return (data_array - data_array.min()) / (data_array.max() - data_array.min())

# For DataFrames: Use Pandas vectorized operations
def normalize_pandas(df, columns):
    for col in columns:
        df[f"{col}_normalized"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df
```

### 4. Use Appropriate Data Structures

Choose data structures based on common operations:

```python
# Membership testing
# Slow for large datasets
items_list = [1, 2, 3, 4, 5, ..., 10000]
if 9876 in items_list:  # O(n) operation
    print("Found")

# Fast for large datasets
items_set = set([1, 2, 3, 4, 5, ..., 10000])
if 9876 in items_set:  # O(1) operation
    print("Found")

# Dictionary for lookups
# Slow: Using list to store and lookup key-value pairs
def get_value_list(key, data_list):
    for k, v in data_list:
        if k == key:
            return v
    return None

# Fast: Using dictionary for O(1) lookups
def get_value_dict(key, data_dict):
    return data_dict.get(key)
```

### 5. Use Generators for Memory Efficiency

Generators calculate values on-demand, reducing memory usage:

```python
# Memory-intensive: Creates full list in memory
def read_large_file_list(file_path):
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            result.append(line.strip().split(','))
    return result

# Memory-efficient: Yields one item at a time
def read_large_file_generator(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip().split(',')

# Usage:
for row in read_large_file_generator('large_data.csv'):
    process_row(row)
```

### 6. Caching and Memoization

Cache expensive function calls to avoid redundant computation:

```python
from functools import lru_cache

# Without caching - slow for repeated calls
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# With caching - much faster for repeated calls
@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)
```

### 7. Use List Comprehensions

List comprehensions are faster than manual loops for creating lists:

```python
# Slower: Using manual loop
squares = []
for i in range(1000):
    squares.append(i**2)

# Faster: Using list comprehension
squares = [i**2 for i in range(1000)]

# For filtering:
even_squares = [i**2 for i in range(1000) if i % 2 == 0]
```

### 8. Minimize Function Calls in Loops

Move operations outside of loops when possible:

```python
# Less efficient: Function calls inside loop
import math
def calculate_distances(points):
    distances = []
    for x, y in points:
        distances.append(math.sqrt(x**2 + y**2))
    return distances

# More efficient: Avoid function lookup in loop
from math import sqrt
def calculate_distances_optimized(points):
    sqrt_func = sqrt  # Store function reference
    distances = []
    for x, y in points:
        distances.append(sqrt_func(x**2 + y**2))
    return distances
```

### 9. Use `map()` and `filter()` for Simple Operations

These functions can be faster than loops for simple operations:

```python
# Applying a function to each item
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))

# Filtering elements
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
```

### 10. Profiling and Measurement

Always measure before and after optimization:

```python
import time
import cProfile

# Simple timing
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return result

# Detailed profiling
def profile_function(func, *args, **kwargs):
    cProfile.runctx('func(*args, **kwargs)', globals(), locals(), sort='cumtime')
```

### 11. Concurrency and Parallelism

Use multiple cores or handle I/O operations concurrently:

```python
# Multiprocessing for CPU-bound tasks
from multiprocessing import Pool

def process_chunk(chunk):
    return [item**2 for item in chunk]

def parallel_processing(data, num_processes=4):
    # Split data into chunks
    chunk_size = len(data) // num_processes
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    return [item for chunk_result in results for item in chunk_result]

# Async I/O for I/O-bound tasks
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### 12. JIT Compilation with Numba

Accelerate numerical functions with just-in-time compilation:

```python
from numba import jit
import numpy as np

# Regular Python function - slow
def monte_carlo_pi(samples):
    count = 0
    for i in range(samples):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1.0:
            count += 1
    return 4.0 * count / samples

# JIT-compiled function - much faster
@jit(nopython=True)
def monte_carlo_pi_numba(samples):
    count = 0
    for i in range(samples):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1.0:
            count += 1
    return 4.0 * count / samples
```

### 13. Optimize String Operations

String operations in loops can be expensive:

```python
# Slow: String concatenation in loop
def build_string_slow(items):
    result = ""
    for item in items:
        result += str(item) + ", "
    return result[:-2]  # Remove trailing comma and space

# Fast: Using join
def build_string_fast(items):
    return ", ".join(str(item) for item in items)
```

### 14. Use pandas optimally for Data Processing

Apply pandas best practices:

```python
import pandas as pd

# Less efficient
def process_dataframe_slow(df):
    result = pd.DataFrame()
    for idx, row in df.iterrows():  # Iterating through rows is slow
        result.loc[idx, 'result'] = row['value'] * 2
    return result

# More efficient
def process_dataframe_fast(df):
    # Vectorized operation
    return df.assign(result=df['value'] * 2)

# For groupby operations
def aggregate_data_efficient(df):
    # Use optimized methods
    return df.groupby('category').agg({
        'value': ['mean', 'sum', 'count']
    })
```

### 15. Data Engineering Optimizations

For data engineering specifically:

```python
# Process large datasets in chunks
def process_large_csv(filepath, chunk_size=100000):
    results = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Process each chunk
        processed = process_chunk(chunk)
        results.append(processed)
    
    # Combine results if needed
    return pd.concat(results)

# Use SQL for operations that databases do well
from sqlalchemy import create_engine

def efficient_aggregation(df, group_by_cols):
    # For very large datasets, pushing computation to the database
    engine = create_engine('sqlite:///:memory:')
    df.to_sql('data', engine, index=False)
    
    query = f"""
    SELECT {','.join(group_by_cols)},
           AVG(value) as avg_value,
           SUM(value) as sum_value,
           COUNT(*) as count
    FROM data
    GROUP BY {','.join(group_by_cols)}
    """
    
    return pd.read_sql(query, engine)
```

## 42. Explain how to use Cython to improve performance of Python code.

**Key Concept**: Cython is a language that compiles Python-like code to C, dramatically improving performance for CPU-intensive operations by removing Python's interpreter overhead.

### 1. Basic Cython Usage

**Step 1**: Create a `.pyx` file with Cython code:

```python
# factorial.pyx
def factorial(int n):
    cdef int i
    cdef int result = 1
    for i in range(2, n+1):
        result *= i
    return result
```

**Step 2**: Create `setup.py` for compilation:

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("factorial.pyx")
)
```

**Step 3**: Compile the Cython code:

```bash
python setup.py build_ext --inplace
```

**Step 4**: Import and use in Python:

```python
import factorial

# Use the compiled function
result = factorial.factorial(10)
print(result)  # 3628800
```

### 2. Key Cython Features

#### Static Type Declarations

```python
# Without type declarations
def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

# With type declarations - much faster
def euclidean_distance_cy(double[:] a, double[:] b):
    cdef int i
    cdef int n = a.shape[0]
    cdef double sum_squared = 0.0
    
    for i in range(n):
        sum_squared += (a[i] - b[i]) ** 2
    
    return sum_squared ** 0.5
```

#### Using C Data Types

```python
# cython: language_level=3
import cython

def process_array(double[:] data):
    cdef int i
    cdef int n = data.shape[0]
    cdef double result = 0.0
    
    for i in range(n):
        result += data[i]
    
    return result / n
```

#### Memory Views (Typed NumPy arrays)

```python
# Efficient array manipulation
def matrix_multiply(double[:,:] A, double[:,:] B):
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int p = B.shape[1]
    
    cdef double[:,:] C = numpy.zeros((m, p), dtype=numpy.double)
    
    cdef int i, j, k
    cdef double temp
    
    for i in range(m):
        for j in range(p):
            temp = 0
            for k in range(n):
                temp += A[i, k] * B[k, j]
            C[i, j] = temp
    
    return numpy.asarray(C)
```

#### Pure C Functions with `cdef`

```python
# Pure C function (not callable from Python directly)
cdef double _square(double x):
    return x * x

# Python-accessible wrapper
def square(double x):
    return _square(x)
```

### 3. Optimizing Existing Python Code

**Gradual approach** - Start with regular Python, then add Cython optimizations:

```python
# Original Python function
def calculate_distances(points):
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distance = ((points[i][0] - points[j][0])**2 + 
                         (points[i][1] - points[j][1])**2)**0.5
            distances.append(distance)
    return distances

# Cythonized version
# distances.pyx
import numpy as np
cimport numpy as np

def calculate_distances(np.ndarray[np.float64_t, ndim=2] points):
    cdef int i, j
    cdef int n = points.shape[0]
    cdef double dx, dy, distance
    cdef list distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distance = (dx*dx + dy*dy)**0.5
            distances.append(distance)
    
    return distances
```

### 4. Using Cython with NumPy

```python
# fast_operations.pyx
import numpy as np
cimport numpy as np

def vector_multiply(np.ndarray[np.float64_t, ndim=1] a, 
                    np.ndarray[np.float64_t, ndim=1] b):
    cdef int i
    cdef int n = a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        result[i] = a[i] * b[i]
    
    return result
```

### 5. Profiling to Target Bottlenecks

Identify slow parts with profiling, then use Cython tactically:

```python
import cProfile
import pstats
import io

def profile_func(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())
    
    return result

# Profile the function
profile_func(my_slow_function, large_dataset)
```

### 6. Cython within a Data Engineering Project

Example: Converting a data processing function to Cython

```python
# Original Python (process_data.py)
def process_data(data, threshold):
    result = []
    for item in data:
        if item['value'] > threshold:
            # Complex calculation
            processed = calculate_metric(item)
            result.append(processed)
    return result

def calculate_metric(item):
    # Compute-intensive calculation
    return complex_formula(item['value'], item['weight'])

# Cythonized version (process_data.pyx)
import numpy as np
cimport numpy as np

def process_data(list data, double threshold):
    cdef list result = []
    cdef dict item
    cdef double processed
    
    for item in data:
        if item['value'] > threshold:
            processed = calculate_metric(item)
            result.append(processed)
    return result

cdef double calculate_metric(dict item):
    # Now in C-speed
    return complex_formula(item['value'], item['weight'])

cpdef double complex_formula(double value, double weight):
    cdef double result = 0.0
    cdef int i
    
    for i in range(1000):
        result += (value * weight) / (i + 1)
    
    return result
```

### 7. Integration with Python Package

```python
# setup.py for a project using Cython
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name="fast_data_processing",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize([
        "fast_data_processing/core_functions.pyx",
        "fast_data_processing/math_utils.pyx"
    ]),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0"
    ],
)
```

### 8. Best Practices for Cython

1. **Profile first**: Only cythonize bottlenecks
2. **Iterative approach**: Start simple, add more type declarations as needed
3. **Type declarations**: Add types to most frequently accessed variables
4. **Minimize Python interactions**: Avoid calling Python functions in tight loops
5. **Use memory views**: For efficient array access
6. **Avoid dictionary/list access in tight loops**: Pre-compute or use C arrays
7. **Check your gains**: Benchmark before and after cythonizing

## 43. What are Python's memory optimization techniques for large datasets?

**Key Concept**: When working with large datasets in Python, various techniques can reduce memory usage while maintaining processing capabilities.

### 1. Generators and Iterators

Process data without loading everything into memory:

```python
# Memory-intensive: Loads entire file into memory
def read_large_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()  # Returns all lines as a list

# Memory-efficient: Yields one line at a time
def read_large_file_iter(file_path):
    with open(file_path, 'r') as f:
        for line in f:  # File object is an iterator
            yield line.strip()

# Usage example
for line in read_large_file_iter('large_file.txt'):
    process_line(line)
```

### 2. Chunking with Pandas

Process large files in manageable chunks:

```python
import pandas as pd

def process_large_csv(filepath, chunk_size=100000):
    # Create an iterator over file chunks
    chunk_iter = pd.read_csv(filepath, chunksize=chunk_size)
    
    # Process each chunk
    results = []
    for chunk in chunk_iter:
        # Process chunk
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    # Combine results (if needed)
    if results:
        return pd.concat(results)
    return pd.DataFrame()

def process_chunk(chunk):
    # Example processing
    # Filter rows, perform calculations, etc.
    return chunk[chunk['value'] > 100]
```

### 3. Memory-Efficient Data Types

Use appropriate data types to reduce memory usage:

```python
import pandas as pd
import numpy as np

# Original DataFrame with default dtypes
df = pd.read_csv('data.csv')
print(f"Memory usage: {df.memory_usage().sum() / 1e6:.2f} MB")

# Optimize numeric columns
def optimize_df(df):
    df_optimized = df.copy()
    
    # Optimize integers
    int_columns = df_optimized.select_dtypes(include=['int']).columns
    for col in int_columns:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        # Choose the smallest integer type that can represent the data
        if col_min >= 0:
            if col_max < 2**8:
                df_optimized[col] = df_optimized[col].astype(np.uint8)
            elif col_max < 2**16:
                df_optimized[col] = df_optimized[col].astype(np.uint16)
            elif col_max < 2**32:
                df_optimized[col] = df_optimized[col].astype(np.uint32)
        else:
            if col_min > -2**7 and col_max < 2**7:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif col_min > -2**15 and col_max < 2**15:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif col_min > -2**31 and col_max < 2**31:
                df_optimized[col] = df_optimized[col].astype(np.int32)
    
    # Optimize float columns
    float_columns = df_optimized.select_dtypes(include=['float']).columns
    for col in float_columns:
        df_optimized[col] = df_optimized[col].astype(np.float32)
    
    # Convert object columns to categories if appropriate
    obj_columns = df_optimized.select_dtypes(include=['object']).columns
    for col in obj_columns:
        num_unique_values = len(df_optimized[col].unique())
        num_total_values = len(df_optimized[col])
        if num_unique_values / num_total_values < 0.5:  # Rule of thumb
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

# Apply optimization
df_optimized = optimize_df(df)
print(f"Memory usage after optimization: {df_optimized.memory_usage().sum() / 1e6:.2f} MB")
```

### 4. NumPy/Pandas Memory Views

Avoid copies with memory views:

```python
import numpy as np

# Create large array
large_array = np.random.rand(1000000)

# Memory-intensive: Creates a copy
subset1 = large_array[1000:50000].copy()

# Memory-efficient: Creates a view (no copy)
subset2 = large_array[1000:50000]  # This is a view

# Verify it's a view
large_array[1000] = 999
print(subset2[0])  # 999 - shows it's a view

# Be careful with views - modifications affect original data
```

### 5. Using HDF5 for Out-of-Memory Processing

Store and process data in a format designed for large datasets:

```python
import h5py
import numpy as np

# Create HDF5 file for large dataset
def create_large_hdf5(filename, shape=(10000, 10000)):
    with h5py.File(filename, 'w') as f:
        # Create dataset with compression
        dset = f.create_dataset("data", shape, dtype='float32',
                                chunks=(1000, 1000), compression="gzip")
        
        # Fill with random data in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, shape[0], chunk_size):
            end_i = min(i + chunk_size, shape[0])
            for j in range(0, shape[1], chunk_size):
                end_j = min(j + chunk_size, shape[1])
                data_chunk = np.random.rand(end_i - i, end_j - j).astype('float32')
                dset[i:end_i, j:end_j] = data_chunk

# Process large dataset without loading it entirely
def process_hdf5_by_chunks(filename, chunk_size=1000):
    results = []
    with h5py.File(filename, 'r') as f:
        dset = f["data"]
        shape = dset.shape
        
        # Process data in chunks
        for i in range(0, shape[0], chunk_size):
            end_i = min(i + chunk_size, shape[0])
            chunk = dset[i:end_i, :]  # Read a chunk
            
            # Process chunk
            processed = process_chunk(chunk)
            results.append(processed)
    
    # Combine results
    return np.concatenate(results)
```

### 6. Memory-Mapped Files

Use memory mapping for efficient file access:

```python
import numpy as np

# Create a memory-mapped array
def create_memmap(filename, shape=(1000000, 10), dtype=np.float32):
    # Create a memory-mapped array
    mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    
    # Write data in chunks to avoid memory issues
    chunk_size = 10000
    for i in range(0, shape[0], chunk_size):
        end = min(i + chunk_size, shape[0])
        mmap_array[i:end] = np.random.rand(end - i, shape[1]).astype(dtype)
    
    # Flush to disk
    mmap_array.flush()
    return mmap_array

# Process a memory-mapped array
def process_memmap(filename, shape=(1000000, 10), dtype=np.float32):
    # Open existing memory-mapped array
    mmap_array = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    
    # Process in chunks without loading entire array into memory
    results = []
    chunk_size = 10000
    for i in range(0, shape[0], chunk_size):
        end = min(i + chunk_size, shape[0])
        # Read and process chunk
        chunk = mmap_array[i:end]
        result = process_chunk(chunk)
        results.append(result)
    
    return results
```

### 7. Using Sparse Data Structures

For data with many zeros or missing values:

```python
import numpy as np
import scipy.sparse as sp

# Dense matrix - uses a lot of memory for mostly zero data
dense_matrix = np.zeros((10000, 10000))
for i in range(10000):
    # Only 0.1% of elements are non-zero
    dense_matrix[i, np.random.choice(10000, 10)] = np.random.rand(10)

print(f"Dense matrix: {dense_matrix.nbytes / 1e6:.2f} MB")

# Sparse matrix - much more memory efficient
# COO format for fast construction
sparse_matrix = sp.lil_matrix((10000, 10000))
for i in range(10000):
    sparse_matrix[i, np.random.choice(10000, 10)] = np.random.rand(10)

# Convert to CSR for efficient operations
sparse_matrix = sparse_matrix.tocsr()

# Compare memory usage - shows only non-zero elements + some overhead
print(f"Sparse matrix: {sparse_matrix.data.nbytes / 1e6:.2f} MB")
```

### 8. Dask for Parallel Out-of-Core Processing

Process datasets larger than RAM:

```python
import dask.dataframe as dd
import pandas as pd

# Create a Dask DataFrame from a large CSV file
def process_with_dask(csv_path):
    # Read the CSV into a Dask DataFrame (doesn't load into memory yet)
    ddf = dd.read_csv(csv_path)
    
    # Perform operations (executed lazily)
    result = ddf[ddf['value'] > 0].groupby('category').agg({
        'value': ['mean', 'sum', 'count']
    })
    
    # Compute the result (executes the operations)
    return result.compute()
```

### 9. Pandas Query Method for Filtering

Use optimized methods for filtering:

```python
import pandas as pd

# Less memory-efficient approach
def filter_standard(df):
    # Creates intermediate boolean mask in memory
    return df[(df['value'] > 100) & (df['category'] == 'A')]

# More memory-efficient approach
def filter_query(df):
    # More efficient for complex conditions
    return df.query("value > 100 and category == 'A'")
```

### 10. Managing Pandas Operations to Avoid Copies

Be careful with pandas operations that create copies:

```python
import pandas as pd

def memory_efficient_transform(df):
    # Create a list to store column names
    new_names = []
    
    # Iterate through columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Use .loc to update in-place
            df.loc[:, col] = df[col].str.upper()
        elif df[col].dtype == 'float64':
            # Use .loc to update in-place
            df.loc[:, col] = df[col].round(2)
        
        # Store new name
        new_names.append(f"{col}_processed")
    
    # Rename columns in-place
    df.columns = new_names
    
    return df
```

### 11. Using `__slots__` for Classes with Many Instances

For classes with millions of instances:

```python
# Memory-inefficient class
class DataPoint:
    def __init__(self, x, y, category):
        self.x = x
        self.y = y
        self.category = category

# Memory-efficient class using __slots__
class DataPointSlots:
    __slots__ = ['x', 'y', 'category']
    
    def __init


# 46-50

```

### 7. Mocking File Operations

```python
import pytest
import pandas as pd
import io
from myapp.file_processor import process_csv_file

def test_process_csv_file(mocker):
    # Create a mock for open function
    mock_file_content = """id,value,category
1,10,A
2,20,B
3,30,C
"""
    mock_open = mocker.mock_open(read_data=mock_file_content)
    mocker.patch('builtins.open', mock_open)
    
    # Mock pd.read_csv to read from the string buffer
    def mock_read_csv(*args, **kwargs):
        return pd.read_csv(io.StringIO(mock_file_content))
    
    mocker.patch('pandas.read_csv', side_effect=mock_read_csv)
    
    # Call the function being tested
    result = process_csv_file('dummy_file.csv')
    
    # Verify the result
    assert len(result) == 3
    assert result['value'].sum() == 60
    
    # Verify file was opened correctly
    mock_open.assert_called_once_with('dummy_file.csv', 'r')
```

### 8. Mocking External Services

```python
import pytest
from unittest.mock import Mock
import requests
from myapp.weather_service import get_weather, get_forecast

@pytest.fixture
def mock_weather_api(monkeypatch):
    """Fixture to mock the weather API responses."""
    def mock_get(*args, **kwargs):
        mock_response = Mock()
        
        # Mock different endpoints
        if 'current' in args[0]:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'temperature': 25.5,
                'humidity': 65,
                'conditions': 'Sunny'
            }
        elif 'forecast' in args[0]:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'daily': [
                    {'day': 'Monday', 'temp': 26, 'conditions': 'Sunny'},
                    {'day': 'Tuesday', 'temp': 24, 'conditions': 'Cloudy'},
                    {'day': 'Wednesday', 'temp': 21, 'conditions': 'Rainy'}
                ]
            }
        else:
            # Default response for other endpoints
            mock_response.status_code = 404
        
        return mock_response
    
    # Replace requests.get with our mock function
    monkeypatch.setattr(requests, 'get', mock_get)

def test_get_weather(mock_weather_api):
    """Test getting current weather."""
    weather = get_weather('London')
    
    assert weather['temperature'] == 25.5
    assert weather['conditions'] == 'Sunny'

def test_get_forecast(mock_weather_api):
    """Test getting weather forecast."""
    forecast = get_forecast('London', days=3)
    
    assert len(forecast) == 3
    assert forecast[0]['day'] == 'Monday'
    assert forecast[2]['conditions'] == 'Rainy'
```

### 9. Spying on Function Calls

```python
import pytest
from myapp.data_analyzer import analyze_data

def test_analyze_data_calls_correct_functions(mocker):
    """Test that analyze_data calls the expected helper functions."""
    # Create spies for helper functions
    clean_data_spy = mocker.patch('myapp.data_analyzer.clean_data')
    normalize_spy = mocker.patch('myapp.data_analyzer.normalize_values')
    calculate_metrics_spy = mocker.patch('myapp.data_analyzer.calculate_metrics')
    
    # Configure return values
    clean_data_spy.return_value = {'cleaned': True}
    normalize_spy.return_value = {'normalized': True}
    calculate_metrics_spy.return_value = {'metric1': 0.5, 'metric2': 0.7}
    
    # Call the function being tested
    result = analyze_data({'raw': 'data'})
    
    # Verify each function was called with the right arguments
    clean_data_spy.assert_called_once_with({'raw': 'data'})
    normalize_spy.assert_called_once_with({'cleaned': True})
    calculate_metrics_spy.assert_called_once_with({'normalized': True})
    
    # Verify the call order
    mocker.call_order(
        clean_data_spy,
        normalize_spy,
        calculate_metrics_spy
    )
    
    # Verify the result
    assert result == {'metric1': 0.5, 'metric2': 0.7}
```

### 10. Advanced Mocking Techniques

```python
import pytest
from unittest.mock import call

def test_complex_mocking_scenario(mocker):
    """Demonstrate advanced mocking techniques."""
    
    # 1. Mock with dynamic return values
    mock_db = mocker.patch('myapp.services.database')
    
    # Return different values on each call
    mock_db.get_user.side_effect = [
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': 'Bob'},
        None  # Third call returns None
    ]
    
    # 2. Mock to track all calls
    mock_logger = mocker.patch('myapp.services.logger')
    
    # 3. Mock to raise an exception on the third call
    mock_api = mocker.patch('myapp.services.api')
    mock_api.get_data.side_effect = [
        ['item1', 'item2'],
        ['item3', 'item4'],
        ConnectionError("API unavailable")
    ]
    
    # Call code that uses these mocks
    # ...
    
    # Verify specific calls
    assert mock_db.get_user.call_count == 3
    
    # Verify call arguments
    mock_db.get_user.assert_has_calls([
        call(1),
        call(2),
        call(3)
    ])
    
    # Verify logger calls contain specific messages
    for call_args in mock_logger.info.call_args_list:
        message = call_args[0][0]
        assert 'Processing' in message or 'Completed' in message
```

### 11. Best Practices for Mocking

1. **Mock at the right level**:
   - Mock at the boundary of your system
   - Don't mock what you don't own
   - Mock dependencies, not the system under test

2. **Make mocks just smart enough**:
   - Return the minimum necessary to test your code
   - Avoid complex mock implementations

3. **Verify interactions, not implementations**:
   - Test that the right calls are made
   - Don't test how the mocked function works internally

4. **Be cautious with extensive mocking**:
   - Too many mocks can reduce test value
   - Consider integration tests for complex interactions

5. **Document your mocks**:
   - Make it clear what's being mocked
   - Explain why mocking is necessary

## 48. What is your approach to code reviews for Python projects?

**Key Concept**: Code reviews ensure code quality, knowledge sharing, and team cohesion by systematically examining code changes before they're merged into the main codebase.

### 1. Code Review Process

A structured approach to reviewing Python code:

1. **Understand the requirements**:
   - What problem is the code solving?
   - Are there any specs or acceptance criteria?

2. **First pass - overall structure**:
   - Architecture and design patterns
   - Component interactions
   - Directory structure and module organization

3. **Second pass - code quality**:
   - Correctness: Does the code do what it's supposed to do?
   - Performance: Are there any obvious inefficiencies?
   - Maintainability: Is the code easy to understand and modify?
   - Error handling: Are edge cases handled appropriately?

4. **Third pass - details**:
   - Style consistency
   - Documentation
   - Tests
   - Security considerations

5. **Provide feedback**:
   - Separate issues by severity (critical, major, minor)
   - Be constructive and specific
   - Offer solutions or alternatives when possible

### 2. Code Review Checklist for Python Projects

```markdown
# Python Code Review Checklist

## Functionality
- [ ] Code correctly implements the requirements
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive and meaningful
- [ ] Backward compatibility is maintained (if required)

## Code Structure
- [ ] Functions/methods have single responsibility
- [ ] Classes are cohesive and follow OOP principles
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Appropriate design patterns are used

## Python-Specific
- [ ] Follows PEP 8 style guidelines
- [ ] Uses Python idioms appropriately (list comprehensions, generators, etc.)
- [ ] Uses appropriate data structures (lists, dicts, sets, etc.)
- [ ] Handles Python-specific issues (mutable default arguments, etc.)

## Performance
- [ ] Efficient algorithms and data structures
- [ ] Avoids unnecessary computations
- [ ] Uses generators for large datasets
- [ ] Considers memory usage
- [ ] Database queries are optimized (if applicable)

## Testing
- [ ] Unit tests cover all public functions/methods
- [ ] Edge cases are tested
- [ ] Integration tests for component interactions
- [ ] Tests are readable and maintainable
- [ ] Tests use appropriate fixtures and mocks

## Documentation
- [ ] Docstrings for modules, classes, and functions
- [ ] Inline comments for complex logic
- [ ] Updated README and other documentation
- [ ] Type hints where appropriate

## Security
- [ ] Input validation
- [ ] SQL injection prevention (if applicable)
- [ ] Secrets are not hardcoded
- [ ] Authentication/authorization checks (if applicable)

## Data Engineering Specific
- [ ] Data quality checks
- [ ] Error handling for data pipeline
- [ ] Logging of processing steps and errors
- [ ] Appropriate data transformations
- [ ] Resource usage considerations
```

### 3. Tools for Python Code Reviews

```python
# Sample pre-review automation script
import subprocess
import sys

def run_checks():
    """Run pre-review checks on code."""
    errors = []
    
    # Check code style
    print("Running flake8...")
    result = subprocess.run(["flake8", "."], capture_output=True, text=True)
    if result.returncode != 0:
        errors.append("flake8 found issues:\n" + result.stdout)
    
    # Check type hints
    print("Running mypy...")
    result = subprocess.run(["mypy", "."], capture_output=True, text=True)
    if result.returncode != 0:
        errors.append("mypy found issues:\n" + result.stdout)
    
    # Run tests
    print("Running tests...")
    result = subprocess.run(["pytest"], capture_output=True, text=True)
    if result.returncode != 0:
        errors.append("Tests failed:\n" + result.stdout)
    
    # Check test coverage
    print("Checking test coverage...")
    result = subprocess.run(["pytest", "--cov=.", "--cov-fail-under=80"], 
                        capture_output=True, text=True)
    if result.returncode != 0:
        errors.append("Coverage below threshold:\n" + result.stdout)
    
    # Report results
    if errors:
        print("\n" + "=" * 50)
        print("Issues found during pre-review checks:")
        for error in errors:
            print("-" * 50)
            print(error)
        sys.exit(1)
    else:
        print("All checks passed!")

if __name__ == "__main__":
    run_checks()
```

Recommended tools:

1. **Style and Formatting**:
   - `flake8`: Style guide enforcement
   - `black`: Code formatter
   - `isort`: Import sorter

2. **Static Analysis**:
   - `pylint`: Static code analysis
   - `mypy`: Type checking
   - `bandit`: Security issues detection

3. **Test Coverage**:
   - `pytest-cov`: Test coverage reports

4. **Code Quality Metrics**:
   - `radon`: Cyclomatic complexity
   - `xenon`: Code quality threshold checker

### 4. Common Python Issues to Look For

```python
# 1. Mutable default arguments
def bad_function(data=[]): # BAD: empty list as default argument
    data.append(1)
    return data

def good_function(data=None): # GOOD: None as default
    if data is None:
        data = []
    data.append(1)
    return data

# 2. Inefficient string concatenation
def bad_string_building(items):
    result = ""
    for item in items:
        result += str(item) + ", "  # BAD: creates new string each time
    return result

def good_string_building(items):
    return ", ".join(str(item) for item in items)  # GOOD: efficient joining

# 3. Not using context managers
def bad_file_handling(filename):
    f = open(filename, 'r')  # BAD: file might not be closed
    data = f.read()
    f.close()
    return data

def good_file_handling(filename):
    with open(filename, 'r') as f:  # GOOD: ensures file will be closed
        data = f.read()
    return data

# 4. Catching too general exceptions
def bad_exception_handling():
    try:
        process_data()
    except Exception as e:  # BAD: catches everything
        print(f"An error occurred: {e}")

def good_exception_handling():
    try:
        process_data()
    except (ValueError, KeyError) as e:  # GOOD: specific exceptions
        print(f"Data error: {e}")
    except IOError as e:
        print(f"I/O error: {e}")

# 5. Not using pandas vectorization
def bad_pandas_processing(df):
    for i, row in df.iterrows():  # BAD: slow row-by-row processing
        df.loc[i, 'new_col'] = row['col1'] * row['col2']
    return df

def good_pandas_processing(df):
    df['new_col'] = df['col1'] * df['col2']  # GOOD: vectorized operation
    return df
```

### 5. Data Engineering Specific Review Items

For data engineering code, additional review focus areas:

```python
# 1. Check for proper memory management
def bad_memory_usage(large_csv):
    # BAD: Loads entire file into memory
    df = pd.read_csv(large_csv)
    processed = some_processing(df)
    return processed

def good_memory_usage(large_csv):
    # GOOD: Processes in chunks
    chunk_list = []
    for chunk in pd.read_csv(large_csv, chunksize=100000):
        processed_chunk = some_processing(chunk)
        chunk_list.append(processed_chunk)
    return pd.concat(chunk_list)

# 2. Check for appropriate error handling in pipelines
def bad_pipeline_error_handling(data):
    # BAD: No error handling, whole pipeline fails if one stage fails
    extracted = extract_data(data)
    transformed = transform_data(extracted)
    load_data(transformed)

def good_pipeline_error_handling(data):
    # GOOD: Graceful error handling and logging
    try:
        extracted = extract_data(data)
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return False
    
    try:
        transformed = transform_data(extracted)
    except Exception as e:
        logging.error(f"Transformation failed: {e}")
        return False
    
    try:
        load_data(transformed)
    except Exception as e:
        logging.error(f"Loading failed: {e}")
        return False
    
    return True

# 3. Check for proper data validation
def bad_data_validation(df):
    # BAD: No validation, assumes data is correct
    result = df['value'] / df['total']
    return result

def good_data_validation(df):
    # GOOD: Validates data before processing
    # Check for required columns
    required_cols = ['value', 'total']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Check for nulls in critical columns
    if df['total'].isnull().any():
        raise ValueError("Null values found in 'total' column")
    
    # Check for zeros in denominator
    if (df['total'] == 0).any():
        raise ValueError("Zero values found in 'total' column")
    
    # Proceed with calculation
    result = df['value'] / df['total']
    return result
```

### 6. Giving Constructive Feedback

Examples of how to provide effective feedback:

```
# Not helpful:
This code is inefficient.

# Helpful:
The nested loops in process_data() have O(n²) complexity. Consider using a dictionary lookup to reduce this to O(n).

# Not helpful:
Your error handling is wrong.

# Helpful:
The try/except block on line 42 catches all exceptions, which could hide unexpected errors. Consider catching specific exceptions (ValueError, KeyError) that we expect might occur.

# Not helpful:
Use better variable names.

# Helpful:
The variable 'x' on line 28 could be renamed to 'customer_record' to better reflect its purpose and make the code more readable.
```

### 7. Automated and Manual Reviews

Balancing automated checks with manual reviews:

```python
# .github/workflows/code-review.yml
name: Code Review

on:
  pull_request:
    branches: [ main, develop ]

jobs:
  automate-checks:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 black mypy pytest pytest-cov bandit
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Check formatting with black
      run: black --check .
    - name: Lint with flake8
      run: flake8 .
    - name: Type check with mypy
      run: mypy .
    - name: Check for security issues with bandit
      run: bandit -r .
    - name: Run tests
      run: pytest --cov=. --cov-report=xml
    - name: Upload coverage report
      uses: codecov/codecov-action@v1
```

### 8. Code Review for Junior vs. Senior Engineers

Different focus areas based on experience level:

**For Junior Engineers**:
- **Focus on fundamentals**: Python idioms, language features
- **Provide context**: Explain why certain approaches are better
- **Educational opportunities**: Suggest resources for learning
- **Encourage questions**: Create a safe environment for discussion

**For Senior Engineers**:
- **Focus on architecture**: System design, scalability, maintainability
- **Discuss trade-offs**: Different approaches and their implications
- **Challenge assumptions**: Question design decisions
- **Collaboration**: Approach as a peer discussion

### 9. Data Engineering Code Review Example

A real-world data engineering code review example:

```python
# Original code
def process_logs(log_file):
    """Process server logs to extract error patterns."""
    # Read the entire log file
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    # Process each line
    errors = []
    for line in logs:
        if 'ERROR' in line:
            errors.append(line)
    
    # Count occurrences of each error type
    error_counts = {}
    for error in errors:
        error_type = error.split(': ')[1].split()[0]
        if error_type in error_counts:
            error_counts[error_type] += 1
        else:
            error_counts[error_type] = 1
    
    return error_counts

# Code review comments:
"""
1. Memory usage concern: Reading the entire log file into memory could cause issues 
   with very large log files. Consider processing line by line.

2. Error handling: No validation or error handling for file operations or parsing errors.

3. Performance: Multiple passes through the data (once to filter, once to count).
   This could be done in a single pass.

4. Robustness: The parsing logic assumes a specific format. What if the format changes?

5. Documentation: The function docstring lacks parameter and return value descriptions.
"""

# Improved code after review
def process_logs(log_file):
    """
    Process server logs to extract error patterns.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        dict: Dictionary mapping error types to occurrence counts
        
    Raises:
        FileNotFoundError: If log_file doesn't exist
        ValueError: If log format is invalid
    """
    # Validate input
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    # Process file line by line to avoid memory issues
    error_counts = {}
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'ERROR' in line:
                    try:
                        # More robust parsing with error handling
                        parts = line.split(': ', 1)
                        if len(parts) < 2:
                            continue  # Skip malformed lines
                        
                        error_message = parts[1]
                        error_type = error_message.split()[0]
                        
                        # Use get() to handle new error types
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    except IndexError:
                        # Log parsing errors but continue processing
                        logging.warning(f"Could not parse error line: {line.strip()}")
    except Exception as e:
        logging.error(f"Error processing log file: {e}")
        raise
    
    return error_counts
```

### 10. Code Review Best Practices

1. **Be timely**:
   - Respond to review requests promptly
   - Set aside dedicated time for reviews

2. **Be respectful**:
   - Focus on the code, not the coder
   - Assume good intentions
   - Phrase feedback as suggestions, not commands

3. **Be thorough but reasonable**:
   - Don't nitpick on trivial issues
   - Focus on important issues: correctness, performance, maintainability

4. **Prioritize feedback**:
   - Critical: Bugs, security issues, performance problems
   - Major: Design issues, maintainability concerns
   - Minor: Style, documentation

5. **Follow up**:
   - Ensure critical issues are addressed
   - Be available for questions and clarifications

6. **Share knowledge**:
   - Explain the reasoning behind suggestions
   - Link to resources for further learning

7. **Use code review as a learning opportunity**:
   - Both reviewer and reviewee should learn from the process

## 49. How do you ensure code quality in a large Python codebase?

**Key Concept**: Maintaining quality in large Python codebases requires a combination of coding standards, automated tools, architectural patterns, and team practices.

### 1. Coding Standards and Style Guides

Establish and enforce consistent coding standards:

```python
# bad_code.py - Inconsistent, hard to maintain
def  someFunction( x,y ):
    if x==None: return
    z=x+y
    if z>10:
        print('Big number')
    else:print('Small number')
    return z

# good_code.py - Following PEP 8
def add_numbers(x, y):
    """Add two numbers and classify the result."""
    if x is None:
        return None
    
    result = x + y
    
    if result > 10:
        print('Big number')
    else:
        print('Small number')
    
    return result
```

**Tools for enforcing standards**:
- **Black**: Automatic code formatting
- **isort**: Sort imports
- **flake8**: Style guide enforcement
- **YAPF**: Code formatter with configuration options

**Configuration examples**:

```ini
# setup.cfg
[flake8]
max-line-length = 88
exclude = .git,__pycache__,build,dist
ignore = E203, W503

[isort]
profile = black
line_length = 88
```

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?# Brief Answers to Python Interview Questions 46-50

## 46. How do you write unit tests for a data processing pipeline in Python?

**Key Concept**: Unit testing data pipelines requires testing individual components in isolation, creating mock data, validating transformations, and handling complex dependencies.

### 1. Basic Structure for Testing Data Pipelines

Use pytest for data pipeline testing:

```python
import pytest
import pandas as pd
import numpy as np
from mypackage.pipeline import extract_data, transform_data, load_data

# Fixtures provide test data and dependencies
@pytest.fixture
def sample_data():
    """Create a small, representative test dataset."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'date': pd.date_range(start='2023-01-01', periods=5)
    })

@pytest.fixture
def expected_result():
    """Expected output after transformation."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'date': pd.date_range(start='2023-01-01', periods=5),
        'value_normalized': [0.0, 0.25, 0.5, 0.75, 1.0]
    })

# Test individual pipeline components
def test_extract_data(mocker):
    """Test data extraction from source."""
    # Mock the data source
    mock_source = mocker.patch('mypackage.pipeline.DataSource')
    mock_source.return_value.get_data.return_value = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })
    
    # Call the extraction function
    result = extract_data(source='database', query='SELECT * FROM test')
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ['id', 'value']
    
    # Verify the source was called correctly
    mock_source.assert_called_once_with('database')
    mock_source.return_value.get_data.assert_called_once_with(query='SELECT * FROM test')

def test_transform_data(sample_data, expected_result):
    """Test data transformation logic."""
    # Apply transformation
    result = transform_data(sample_data)
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    assert 'value_normalized' in result.columns
    
    # Compare with expected result (considering floating point precision)
    pd.testing.assert_frame_equal(result, expected_result, atol=1e-10)

def test_load_data(sample_data, mocker):
    """Test data loading to destination."""
    # Mock the destination
    mock_destination = mocker.patch('mypackage.pipeline.DataDestination')
    
    # Call the load function
    result = load_data(sample_data, destination='output.csv')
    
    # Verify the destination was called correctly
    mock_destination.assert_called_once()
    mock_destination.return_value.write_data.assert_called_once_with(sample_data)
    
    # Verify the result
    assert result == True  # Assuming load_data returns True on success
```

### 2. Testing with Synthetic Data

Generate test data that covers edge cases:

```python
import pandas as pd
import numpy as np

class TestDataGenerator:
    """Generate synthetic data for testing."""
    
    @staticmethod
    def normal_dataset(size=100, seed=42):
        """Generate a normally distributed dataset."""
        rng = np.random.RandomState(seed)
        return pd.DataFrame({
            'id': range(1, size + 1),
            'value': rng.normal(loc=100, scale=15, size=size),
            'category': rng.choice(['A', 'B', 'C'], size=size),
            'timestamp': pd.date_range(start='2023-01-01', periods=size)
        })
    
    @staticmethod
    def dataset_with_nulls(size=100, null_ratio=0.1, seed=42):
        """Generate a dataset with null values."""
        df = TestDataGenerator.normal_dataset(size, seed)
        
        # Add nulls randomly
        rng = np.random.RandomState(seed)
        for col in ['value', 'category']:
            null_mask = rng.random(size=size) < null_ratio
            df.loc[null_mask, col] = None
        
        return df
    
    @staticmethod
    def dataset_with_outliers(size=100, outlier_ratio=0.05, seed=42):
        """Generate a dataset with outliers."""
        df = TestDataGenerator.normal_dataset(size, seed)
        
        # Add outliers
        rng = np.random.RandomState(seed + 1)  # Different seed for outliers
        num_outliers = int(size * outlier_ratio)
        outlier_indices = rng.choice(size, num_outliers, replace=False)
        
        # Make outliers 3-5 standard deviations away
        df.loc[outlier_indices, 'value'] = rng.normal(
            loc=100, 
            scale=15 * rng.uniform(3, 5, size=num_outliers)
        )
        
        return df

# Test a data cleaning function with synthetic data
def test_outlier_detection():
    # Generate data with outliers
    df = TestDataGenerator.dataset_with_outliers(size=1000, outlier_ratio=0.05)
    
    # Apply outlier detection function
    outliers = detect_outliers(df, column='value')
    
    # Verify results
    assert len(outliers) > 0
    assert len(outliers) < len(df)  # Not all rows are outliers
    assert 0.03 < len(outliers) / len(df) < 0.07  # Approximately 5% outliers
```

### 3. Testing Each Pipeline Stage

Organize tests by pipeline stage:

```python
# tests/test_extract.py
import pytest
import pandas as pd
from mypackage.pipeline.extract import extract_from_csv, extract_from_database

class TestExtract:
    def test_extract_from_csv(self, tmp_path):
        """Test CSV extraction."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df.to_csv(csv_path, index=False)
        
        # Test extraction
        result = extract_from_csv(csv_path)
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert len(result) == 2
    
    def test_extract_from_database(self, mocker):
        """Test database extraction."""
        # Mock the database connection
        mock_conn = mocker.patch('mypackage.pipeline.extract.create_db_connection')
        mock_conn.return_value.execute.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        # Test extraction
        result = extract_from_database(
            connection_string="postgresql://test:test@localhost/test",
            query="SELECT * FROM test_table"
        )
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Verify connection was used correctly
        mock_conn.assert_called_once()
        mock_conn.return_value.execute.assert_called_once_with("SELECT * FROM test_table")

# tests/test_transform.py
import pytest
import pandas as pd
import numpy as np
from mypackage.pipeline.transform import clean_data, normalize_values, add_features

class TestTransform:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, None, 30, 40, 50],
            'category': ['A', 'B', None, 'C', 'B'],
        })
    
    def test_clean_data(self, sample_data):
        """Test data cleaning function."""
        result = clean_data(sample_data)
        
        # Verify nulls were handled
        assert result['value'].isna().sum() == 0
        assert result['category'].isna().sum() == 0
        
        # Verify data integrity
        assert len(result) == len(sample_data)
        assert list(result.columns) == list(sample_data.columns)
    
    def test_normalize_values(self):
        """Test value normalization."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        result = normalize_values(df, column='value')
        
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_almost_equal(result['value_normalized'].values, expected)
    
    def test_add_features(self):
        """Test feature engineering."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = add_features(df)
        
        assert 'A_plus_B' in result.columns
        assert 'A_times_B' in result.columns
        np.testing.assert_array_equal(result['A_plus_B'].values, np.array([5, 7, 9]))
        np.testing.assert_array_equal(result['A_times_B'].values, np.array([4, 10, 18]))

# tests/test_load.py
# Similar structure for testing the load phase
```

### 4. Integration Tests for Pipeline

Test how components work together:

```python
import pytest
import pandas as pd
import tempfile
import os
from mypackage.pipeline import run_pipeline

class TestPipelineIntegration:
    def test_end_to_end_pipeline(self, mocker):
        """Test the entire pipeline from end to end."""
        # Create test input file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            input_path = tmp.name
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'value': [10, 20, 30, 40, 50],
                'category': ['A', 'B', 'A', 'C', 'B']
            }).to_csv(input_path, index=False)
        
        # Create output path
        output_path = input_path + '.output.csv'
        
        # Run the pipeline
        result = run_pipeline(
            input_path=input_path,
            output_path=output_path,
            config={'normalize': True, 'add_features': True}
        )
        
        # Verify pipeline completed successfully
        assert result == True
        
        # Verify output file exists and has correct data
        assert os.path.exists(output_path)
        
        # Check output data
        output_data = pd.read_csv(output_path)
        assert len(output_data) == 5
        assert 'value_normalized' in output_data.columns
        
        # Clean up
        os.unlink(input_path)
        os.unlink(output_path)
```

### 5. Testing with Mocks and Fixtures

Create reusable test fixtures:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Fixtures for common test scenarios
@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    mock_db = Mock()
    
    # Define behavior for mock query execution
    def execute_query(query):
        if "user" in query.lower():
            return pd.DataFrame({
                'user_id': range(1, 6),
                'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
            })
        elif "transaction" in query.lower():
            return pd.DataFrame({
                'transaction_id': range(101, 111),
                'user_id': [1, 2, 3, 1, 2, 3, 4, 5, 4, 5],
                'amount': [10.5, 20.3, 15.7, 8.2, 5.1, 3.8, 12.4, 18.9, 7.6, 9.3]
            })
        else:
            return pd.DataFrame()
    
    mock_db.execute = execute_query
    return mock_db

@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    return {
        'extract': {
            'source': 'database',
            'query': 'SELECT * FROM transactions'
        },
        'transform': {
            'clean_nulls': True,
            'normalize': ['amount'],
            'features': ['day_of_week', 'is_weekend']
        },
        'load': {
            'destination': 'output.csv',
            'if_exists': 'replace'
        }
    }

# Test configuration loading
def test_config_loading(sample_config, tmp_path):
    # Write config to temporary file
    config_path = tmp_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(sample_config, f)
    
    # Test loading configuration
    loaded_config = load_pipeline_config(config_path)
    
    # Verify
    assert loaded_config == sample_config

# Test database extraction with mock
def test_database_extraction(mock_database):
    with patch('mypackage.pipeline.create_db_connection', return_value=mock_database):
        # Extract data
        data = extract_from_database(query="SELECT * FROM transactions")
        
        # Verify
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert 'transaction_id' in data.columns
        assert 'amount' in data.columns
```

### 6. Testing Error Handling

Verify how the pipeline handles errors:

```python
def test_extraction_error_handling(mocker):
    """Test how extraction handles errors."""
    # Mock database that raises an exception
    mock_db = mocker.patch('mypackage.pipeline.create_db_connection')
    mock_db.side_effect = ConnectionError("Database connection failed")
    
    # Test extraction with error handling
    with pytest.raises(ConnectionError) as excinfo:
        extract_from_database(query="SELECT * FROM test")
    
    # Verify error message
    assert "Database connection failed" in str(excinfo.value)

def test_transform_with_invalid_data():
    """Test transformation with invalid data."""
    # Create invalid data
    invalid_data = pd.DataFrame({
        'id': [1, 2, 3],
        'invalid_column': ['a', 'b', 'c']  # Missing expected column
    })
    
    # Test transformation with validation
    with pytest.raises(ValueError) as excinfo:
        transform_data(invalid_data)
    
    # Verify error message
    assert "Missing required column" in str(excinfo.value)

def test_pipeline_retry_mechanism(mocker):
    """Test that pipeline retries on temporary failures."""
    # Mock a function that fails on first call, succeeds on second
    mock_extract = mocker.patch('mypackage.pipeline.extract_data')
    
    # Set up side effect to fail first, then succeed
    mock_extract.side_effect = [
        ConnectionError("Temporary failure"),
        pd.DataFrame({'data': [1, 2, 3]})
    ]
    
    # Run pipeline with retry
    result = run_pipeline_with_retry(
        input_path="test_input",
        max_retries=3,
        retry_delay=0.1
    )
    
    # Verify pipeline completed after retry
    assert result is not None
    assert mock_extract.call_count == 2  # Called twice
```

### 7. Performance Testing

Test the performance of your pipeline:

```python
import time
import pytest
import pandas as pd
import numpy as np

@pytest.mark.slow
def test_pipeline_performance():
    """Test performance with larger dataset."""
    # Generate large test dataset
    large_df = pd.DataFrame({
        'id': range(100000),
        'value': np.random.normal(size=100000),
        'category': np.random.choice(['A', 'B', 'C'], size=100000)
    })
    
    # Measure performance
    start_time = time.time()
    result = process_large_dataset(large_df)
    execution_time = time.time() - start_time
    
    # Verify performance meets expectations
    assert execution_time < 5.0  # Should process within 5 seconds
    assert len(result) == len(large_df)

@pytest.mark.parametrize("data_size", [10, 100, 1000, 10000])
def test_scaling_behavior(data_size):
    """Test how processing time scales with data size."""
    # Generate test data
    df = pd.DataFrame({
        'id': range(data_size),
        'value': np.random.normal(size=data_size)
    })
    
    # Measure execution time
    start_time = time.time()
    process_data(df)
    execution_time = time.time() - start_time
    
    # Log results for analysis
    print(f"Size: {data_size}, Time: {execution_time:.4f}s")
    
    # We expect roughly linear scaling, but with some overhead
    # This is a simple check that execution doesn't explode unexpectedly
    if data_size > 100:
        assert execution_time < (data_size / 100) * 2  # Allow some margin
```

### 8. Testing Data Quality

Verify data quality at each pipeline stage:

```python
import pandas as pd
import numpy as np
import pytest
from mypackage.pipeline import process_data
from mypackage.validation import validate_schema, check_data_quality

def test_output_schema():
    """Test that output conforms to expected schema."""
    # Input data
    input_data = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30],
        'category': ['A', 'B', 'C']
    })
    
    # Expected schema
    expected_schema = {
        'id': 'int',
        'value': 'float',
        'category': 'object',
        'value_normalized': 'float',
        'category_encoded': 'int'
    }
    
    # Process data
    result = process_data(input_data)
    
    # Validate schema
    schema_validation = validate_schema(result, expected_schema)
    assert schema_validation['valid'] == True, schema_validation['errors']
    
    # Check column presence
    for column in expected_schema:
        assert column in result.columns

def test_data_quality_metrics():
    """Test data quality metrics on pipeline output."""
    # Generated test data with known characteristics
    input_data = pd.DataFrame({
        'id': range(1000),
        'value': np.random.normal(100, 15, size=1000),
        'category': np.random.choice(['A', 'B', 'C'], size=1000, p=[0.6, 0.3, 0.1])
    })
    
    # Process data
    output_data = process_data(input_data)
    
    # Check data quality
    quality_report = check_data_quality(output_data)
    
    # Verify quality metrics
    assert quality_report['row_count'] == 1000
    assert quality_report['null_count']['value'] == 0
    assert abs(quality_report['mean']['value'] - 100) < 2  # Expected mean with some margin
    assert abs(quality_report['std']['value'] - 15) < 2  # Expected std with some margin
    
    # Check category distribution (with tolerance)
    cat_dist = quality_report['category_distribution']
    assert 0.55 < cat_dist['A'] < 0.65  # Expected around 0.6
    assert 0.25 < cat_dist['B'] < 0.35  # Expected around 0.3
    assert 0.05 < cat_dist['C'] < 0.15  # Expected around 0.1
```

### 9. CI/CD Integration

Configure tests for continuous integration:

```python
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Tests that take longer to run
    performance: Performance tests

# .github/workflows/python-tests.yml
name: Python Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
        pip install -e .
    - name: Run unit tests
      run: |
        pytest tests/ -m "unit" -v
    - name: Run integration tests
      run: |
        pytest tests/ -m "integration" -v
    - name: Run performance tests
      run: |
        pytest tests/ -m "performance" -v
```

### 10. Best Practices for Data Pipeline Testing

1. **Test at multiple levels**:
   - Unit tests for individual functions
   - Integration tests for component interactions
   - End-to-end tests for complete pipeline

2. **Use appropriate test data**:
   - Small datasets for unit tests
   - Realistic datasets for integration tests
   - Edge cases to test error handling

3. **Parameterize tests**:
   - Test multiple scenarios efficiently
   - Test with different configurations

4. **Mock external dependencies**:
   - Databases, APIs, file systems
   - Focus tests on your code, not external systems

5. **Validate data quality**:
   - Test for schema compliance
   - Check statistical properties
   - Verify business rules

6. **Test performance**:
   - Set performance expectations
   - Monitor scaling behavior
   - Create performance regression tests

7. **Test failure modes**:
   - Network failures
   - Invalid input data
   - Missing permissions

8. **Document test coverage**:
   - Use tools like pytest-cov to track coverage
   - Ensure critical paths are well-tested

## 47. Explain mocking in Python testing. When and how would you use it?

**Key Concept**: Mocking replaces real dependencies with simulated objects to create isolated, controlled testing environments, enabling testing without relying on external systems.

### 1. What is Mocking and Why Use It

Mocking is creating simulated objects that mimic the behavior of real components:

- **Why use mocking**:
  - Isolate the code being tested
  - Remove external dependencies (databases, APIs, file systems)
  - Test specific scenarios or error conditions
  - Increase test speed and reliability
  - Test components that aren't fully developed yet

```python
# Without mocking: Tests depend on actual database
def test_get_user_without_mock():
    user = get_user_from_database(user_id=123)
    assert user.name == "John"  # Breaks if database changes

# With mocking: Tests are isolated from the database
def test_get_user_with_mock(mocker):
    # Mock the database function
    mock_db = mocker.patch('myapp.database.query')
    mock_db.return_value = {'id': 123, 'name': 'John', 'email': 'john@example.com'}
    
    # Now test function that uses database
    user = get_user_from_database(user_id=123)
    assert user.name == "John"  # Works regardless of actual database
```

### 2. Different Types of Test Doubles

```python
# 1. Dummy: Objects passed but never used
dummy_logger = None

# 2. Fake: Working implementations, but not suitable for production
class FakeDatabase:
    def __init__(self):
        self.data = {}
    
    def save(self, key, value):
        self.data[key] = value
    
    def get(self, key):
        return self.data.get(key)

# 3. Stubs: Provide canned answers
class UserServiceStub:
    def get_user(self, user_id):
        return {'id': user_id, 'name': 'Test User'}

# 4. Spies: Record calls made
class DatabaseSpy:
    def __init__(self):
        self.queries = []
    
    def execute(self, query):
        self.queries.append(query)
        return []
    
    def get_executed_queries(self):
        return self.queries

# 5. Mocks: Pre-programmed with expectations
# (Usually created with mocking frameworks)
```

### 3. Using the `unittest.mock` Library

```python
import unittest
from unittest.mock import Mock, MagicMock, patch
import requests
from myapp.user_service import get_user_profile

class TestUserService(unittest.TestCase):
    def test_get_user_profile_with_mock(self):
        # Create a mock for the requests.get function
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 123,
            'name': 'John Doe',
            'email': 'john@example.com'
        }
        
        # Replace requests.get with our mock
        with patch('requests.get', return_value=mock_response):
            # Call function that uses requests.get
            user = get_user_profile(user_id=123)
            
            # Assert on the result
            self.assertEqual(user['name'], 'John Doe')
    
    def test_get_user_profile_connection_error(self):
        # Make requests.get raise an exception when called
        with patch('requests.get', side_effect=requests.ConnectionError("API unavailable")):
            # Test error handling behavior
            with self.assertRaises(ValueError) as context:
                get_user_profile(user_id=123)
            
            self.assertIn("API connection failed", str(context.exception))

if __name__ == '__main__':
    unittest.main()
```

### 4. Using pytest-mock

```python
import pytest
import pandas as pd
from myapp.data_processor import process_csv_file

def test_process_csv_file(mocker):
    # Create a mock for pandas.read_csv
    mock_df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })
    mocker.patch('pandas.read_csv', return_value=mock_df)
    
    # Mock os.path.exists to return True
    mocker.patch('os.path.exists', return_value=True)
    
    # Call the function we're testing
    result = process_csv_file('dummy_file.csv')
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'id' in result.columns
    assert 'value_squared' in result.columns  # Assuming the function adds this column
    assert result['value_squared'].tolist() == [100, 400, 900]
```

### 5. Mocking Methods in Classes

```python
from unittest.mock import patch
import pytest
from myapp.database import Database
from myapp.user_repository import UserRepository

class TestUserRepository:
    def test_get_user(self, mocker):
        # Mock the Database class's get_connection method
        mock_connection = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        
        # Configure the cursor's fetchone method to return user data
        mock_cursor.fetchone.return_value = (1, "John Doe", "john@example.com")
        
        # Configure connection to return our cursor
        mock_connection.cursor.return_value = mock_cursor
        
        # Patch the Database.get_connection method
        with patch.object(Database, 'get_connection', return_value=mock_connection):
            # Create repository and call method
            repository = UserRepository()
            user = repository.get_user(1)
            
            # Assertions
            assert user.id == 1
            assert user.name == "John Doe"
            assert user.email == "john@example.com"
            
            # Verify SQL query was executed correctly
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "SELECT * FROM users WHERE id = ?" in call_args[0]
            assert call_args[1] == (1,)
```

### 6. Mocking in Data Engineering Pipelines

```python
import pytest
import pandas as pd
from myapp.etl_pipeline import extract, transform, load

class TestETLPipeline:
    def test_extract_from_api(self, mocker):
        # Mock API client
        mock_api = mocker.patch('myapp.etl_pipeline.APIClient')
        mock_api.return_value.get_data.return_value = [
            {'id': 1, 'name': 'Product A', 'price': 10.99},
            {'id': 2, 'name': 'Product B', 'price': 24.99}
        ]
        
        # Call extract function
        data = extract(source='api', api_key='dummy_key')
        
        # Verify API was called correctly
        mock_api.assert_called_once_with(api_key='dummy_key')
        mock_api.return_value.get_data.assert_called_once()
        
        # Verify extracted data
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert data['price'].sum() == 35.98
    
    def test_transform(self):
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2],
            'price': [10.99, 24.99]
        })
        
        # Mock database connection
        mock_db = mocker.patch('myapp.etl_pipeline.Database')
        mock_connection = mock_db.return_value.get_connection.return_value
        
        # Call load function
        result = load(data, destination='database')
        
        # Verify database was called correctly
        mock_db.assert_called_once()
        mock_connection.execute.assert_called()
        
        # Typically, we'd verify the SQL, but it depends on implementation
        assert result == True  # Assuming load returns True on success,
            'category': ['A', 'B']
        })
        
        # Call transform function
        result = transform(data)
        
        # Verify transformation
        assert 'price_normalized' in result.columns
        assert result['price_normalized'].max() <= 1.0
        assert result['price_normalized'].min() >= 0.0
    
    def test_load_to_database(self, mocker):
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2],
            'price': [10.99, 24.99]
exclude = '''
/(
    \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

### 2. Static Analysis and Type Checking

Catch issues before runtime:

```python
# Without type hints
def process_data(data, factor, enable_feature):
    # What types are these parameters supposed to be?
    if enable_feature:
        return data * factor
    return data

# With type hints
from typing import Union, List, Dict, Any

def process_data(
    data: Union[List[float], Dict[str, float]], 
    factor: float, 
    enable_feature: bool = False
) -> Union[List[float], Dict[str, float]]:
    """
    Process numeric data by applying a multiplication factor.
    
    Args:
        data: List of numbers or dictionary with numeric values
        factor: Multiplication factor to apply
        enable_feature: Whether to apply the factor
        
    Returns:
        Processed data in the same format as input
    """
    if not enable_feature:
        return data
    
    if isinstance(data, list):
        return [item * factor for item in data]
    elif isinstance(data, dict):
        return {key: value * factor for key, value in data.items()}
    else:
        raise TypeError("Data must be a list or dictionary")
```

**Static analysis tools**:
- **mypy**: Type checking
- **pyright/Pylance**: Microsoft's Python type checker
- **pylint**: Static code analysis
- **bandit**: Security-focused static analysis

**Example mypy configuration**:
```ini
# mypy.ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False

[mypy.plugins.numpy.*]
follow_imports = skip

[mypy-pandas.*]
ignore_missing_imports = True
```

### 3. Automated Testing

Implement comprehensive testing strategy:

```python
# src/data_processor.py
class DataProcessor:
    def process_values(self, values):
        """Process a list of numeric values."""
        if not values:
            return []
        
        # Remove None values
        filtered = [v for v in values if v is not None]
        
        # Calculate average
        avg = sum(filtered) / len(filtered) if filtered else 0
        
        # Apply transformation
        result = [self._transform(v, avg) for v in filtered]
        return result
    
    def _transform(self, value, reference):
        """Apply a transformation based on the reference value."""
        if value > reference * 2:
            return value * 0.8  # Reduce outliers
        elif value < reference * 0.5:
            return value * 1.2  # Boost low values
        else:
            return value  # Keep normal values

# tests/test_data_processor.py
import pytest
from src.data_processor import DataProcessor

class TestDataProcessor:
    def setup_method(self):
        self.processor = DataProcessor()
    
    def test_empty_list(self):
        """Test with empty input."""
        assert self.processor.process_values([]) == []
    
    def test_none_values(self):
        """Test with None values in input."""
        assert self.processor.process_values([None, None]) == []
        assert self.processor.process_values([1, None, 2]) == [1, 2]
    
    def test_normal_values(self):
        """Test with normal values."""
        values = [10, 20, 30, 40, 50]
        result = self.processor.process_values(values)
        assert len(result) == 5
        assert sum(result) != sum(values)  # Transformation should change values
    
    @pytest.mark.parametrize("values,expected_len", [
        ([1, 2, 3], 3),
        ([10, 20, None, 30], 3),
        ([], 0)
    ])
    def test_output_length(self, values, expected_len):
        """Test output length with various inputs."""
        result = self.processor.process_values(values)
        assert len(result) == expected_len
    
    def test_transform_outliers(self):
        """Test the transformation of outlier values."""
        # Create values where some are clear outliers
        values = [10, 10, 10, 10, 100]  # Average is 28
        result = self.processor.process_values(values)
        
        # Verify the outlier was reduced
        assert result[4] < 100
        
        # Verify the normal values stayed the same
        assert result[0] == 10
```

**Testing best practices**:
- **Test pyramid**: Unit tests > Integration tests > End-to-end tests
- **Test coverage**: Aim for high coverage of critical code paths
- **Property-based testing**: Use tools like Hypothesis to find edge cases
- **CI integration**: Run tests automatically on every commit

### 4. Documentation

Document code thoroughly at different levels:

```python
"""
Customer Data Processing Module
===============================

This module provides functionality for processing customer transaction data.
It handles data cleaning, transformation, and aggregation for reporting purposes.

Components:
    - DataCleaner: Cleans raw customer data
    - TransactionProcessor: Processes transaction records
    - ReportGenerator: Generates summary reports
"""

from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

class TransactionProcessor:
    """
    Process customer transaction data.
    
    This class handles the core business logic for processing transaction
    records, including categorization, filtering, and aggregation.
    
    Attributes:
        config (Dict): Configuration parameters
        categories (List[str]): Valid transaction categories
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the transaction processor.
        
        Args:
            config: Configuration dictionary containing:
                - default_currency: Default currency code
                - exchange_rates: Dictionary of exchange rates
                - categories: List of valid transaction categories
        
        Raises:
            ValueError: If required configuration options are missing
        """
        if 'categories' not in config:
            raise ValueError("Configuration# Brief Answers to Python Interview Questions 46-50

## 46. How do you write unit tests for a data processing pipeline in Python?

**Key Concept**: Unit testing data pipelines requires testing individual components in isolation, creating mock data, validating transformations, and handling complex dependencies.

### 1. Basic Structure for Testing Data Pipelines

Use pytest for data pipeline testing:

```python
import pytest
import pandas as pd
import numpy as np
from mypackage.pipeline import extract_data, transform_data, load_data

# Fixtures provide test data and dependencies
@pytest.fixture
def sample_data():
    """Create a small, representative test dataset."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'date': pd.date_range(start='2023-01-01', periods=5)
    })

@pytest.fixture
def expected_result():
    """Expected output after transformation."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'date': pd.date_range(start='2023-01-01', periods=5),
        'value_normalized': [0.0, 0.25, 0.5, 0.75, 1.0]
    })

# Test individual pipeline components
def test_extract_data(mocker):
    """Test data extraction from source."""
    # Mock the data source
    mock_source = mocker.patch('mypackage.pipeline.DataSource')
    mock_source.return_value.get_data.return_value = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })
    
    # Call the extraction function
    result = extract_data(source='database', query='SELECT * FROM test')
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ['id', 'value']
    
    # Verify the source was called correctly
    mock_source.assert_called_once_with('database')
    mock_source.return_value.get_data.assert_called_once_with(query='SELECT * FROM test')

def test_transform_data(sample_data, expected_result):
    """Test data transformation logic."""
    # Apply transformation
    result = transform_data(sample_data)
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    assert 'value_normalized' in result.columns
    
    # Compare with expected result (considering floating point precision)
    pd.testing.assert_frame_equal(result, expected_result, atol=1e-10)

def test_load_data(sample_data, mocker):
    """Test data loading to destination."""
    # Mock the destination
    mock_destination = mocker.patch('mypackage.pipeline.DataDestination')
    
    # Call the load function
    result = load_data(sample_data, destination='output.csv')
    
    # Verify the destination was called correctly
    mock_destination.assert_called_once()
    mock_destination.return_value.write_data.assert_called_once_with(sample_data)
    
    # Verify the result
    assert result == True  # Assuming load_data returns True on success
```

### 2. Testing with Synthetic Data

Generate test data that covers edge cases:

```python
import pandas as pd
import numpy as np

class TestDataGenerator:
    """Generate synthetic data for testing."""
    
    @staticmethod
    def normal_dataset(size=100, seed=42):
        """Generate a normally distributed dataset."""
        rng = np.random.RandomState(seed)
        return pd.DataFrame({
            'id': range(1, size + 1),
            'value': rng.normal(loc=100, scale=15, size=size),
            'category': rng.choice(['A', 'B', 'C'], size=size),
            'timestamp': pd.date_range(start='2023-01-01', periods=size)
        })
    
    @staticmethod
    def dataset_with_nulls(size=100, null_ratio=0.1, seed=42):
        """Generate a dataset with null values."""
        df = TestDataGenerator.normal_dataset(size, seed)
        
        # Add nulls randomly
        rng = np.random.RandomState(seed)
        for col in ['value', 'category']:
            null_mask = rng.random(size=size) < null_ratio
            df.loc[null_mask, col] = None
        
        return df
    
    @staticmethod
    def dataset_with_outliers(size=100, outlier_ratio=0.05, seed=42):
        """Generate a dataset with outliers."""
        df = TestDataGenerator.normal_dataset(size, seed)
        
        # Add outliers
        rng = np.random.RandomState(seed + 1)  # Different seed for outliers
        num_outliers = int(size * outlier_ratio)
        outlier_indices = rng.choice(size, num_outliers, replace=False)
        
        # Make outliers 3-5 standard deviations away
        df.loc[outlier_indices, 'value'] = rng.normal(
            loc=100, 
            scale=15 * rng.uniform(3, 5, size=num_outliers)
        )
        
        return df

# Test a data cleaning function with synthetic data
def test_outlier_detection():
    # Generate data with outliers
    df = TestDataGenerator.dataset_with_outliers(size=1000, outlier_ratio=0.05)
    
    # Apply outlier detection function
    outliers = detect_outliers(df, column='value')
    
    # Verify results
    assert len(outliers) > 0
    assert len(outliers) < len(df)  # Not all rows are outliers
    assert 0.03 < len(outliers) / len(df) < 0.07  # Approximately 5% outliers
```

### 3. Testing Each Pipeline Stage

Organize tests by pipeline stage:

```python
# tests/test_extract.py
import pytest
import pandas as pd
from mypackage.pipeline.extract import extract_from_csv, extract_from_database

class TestExtract:
    def test_extract_from_csv(self, tmp_path):
        """Test CSV extraction."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df.to_csv(csv_path, index=False)
        
        # Test extraction
        result = extract_from_csv(csv_path)
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert len(result) == 2
    
    def test_extract_from_database(self, mocker):
        """Test database extraction."""
        # Mock the database connection
        mock_conn = mocker.patch('mypackage.pipeline.extract.create_db_connection')
        mock_conn.return_value.execute.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        # Test extraction
        result = extract_from_database(
            connection_string="postgresql://test:test@localhost/test",
            query="SELECT * FROM test_table"
        )
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Verify connection was used correctly
        mock_conn.assert_called_once()
        mock_conn.return_value.execute.assert_called_once_with("SELECT * FROM test_table")

# tests/test_transform.py
import pytest
import pandas as pd
import numpy as np
from mypackage.pipeline.transform import clean_data, normalize_values, add_features

class TestTransform:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, None, 30, 40, 50],
            'category': ['A', 'B', None, 'C', 'B'],
        })
    
    def test_clean_data(self, sample_data):
        """Test data cleaning function."""
        result = clean_data(sample_data)
        
        # Verify nulls were handled
        assert result['value'].isna().sum() == 0
        assert result['category'].isna().sum() == 0
        
        # Verify data integrity
        assert len(result) == len(sample_data)
        assert list(result.columns) == list(sample_data.columns)
    
    def test_normalize_values(self):
        """Test value normalization."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        result = normalize_values(df, column='value')
        
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_almost_equal(result['value_normalized'].values, expected)
    
    def test_add_features(self):
        """Test feature engineering."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = add_features(df)
        
        assert 'A_plus_B' in result.columns
        assert 'A_times_B' in result.columns
        np.testing.assert_array_equal(result['A_plus_B'].values, np.array([5, 7, 9]))
        np.testing.assert_array_equal(result['A_times_B'].values, np.array([4, 10, 18]))

# tests/test_load.py
# Similar structure for testing the load phase
```

### 4. Integration Tests for Pipeline

Test how components work together:

```python
import pytest
import pandas as pd
import tempfile
import os
from mypackage.pipeline import run_pipeline

class TestPipelineIntegration:
    def test_end_to_end_pipeline(self, mocker):
        """Test the entire pipeline from end to end."""
        # Create test input file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            input_path = tmp.name
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'value': [10, 20, 30, 40, 50],
                'category': ['A', 'B', 'A', 'C', 'B']
            }).to_csv(input_path, index=False)
        
        # Create output path
        output_path = input_path + '.output.csv'
        
        # Run the pipeline
        result = run_pipeline(
            input_path=input_path,
            output_path=output_path,
            config={'normalize': True, 'add_features': True}
        )
        
        # Verify pipeline completed successfully
        assert result == True
        
        # Verify output file exists and has correct data
        assert os.path.exists(output_path)
        
        # Check output data
        output_data = pd.read_csv(output_path)
        assert len(output_data) == 5
        assert 'value_normalized' in output_data.columns
        
        # Clean up
        os.unlink(input_path)
        os.unlink(output_path)
```

### 5. Testing with Mocks and Fixtures

Create reusable test fixtures:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Fixtures for common test scenarios
@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    mock_db = Mock()
    
    # Define behavior for mock query execution
    def execute_query(query):
        if "user" in query.lower():
            return pd.DataFrame({
                'user_id': range(1, 6),
                'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
            })
        elif "transaction" in query.lower():
            return pd.DataFrame({
                'transaction_id': range(101, 111),
                'user_id': [1, 2, 3, 1, 2, 3, 4, 5, 4, 5],
                'amount': [10.5, 20.3, 15.7, 8.2, 5.1, 3.8, 12.4, 18.9, 7.6, 9.3]
            })
        else:
            return pd.DataFrame()
    
    mock_db.execute = execute_query
    return mock_db

@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    return {
        'extract': {
            'source': 'database',
            'query': 'SELECT * FROM transactions'
        },
        'transform': {
            'clean_nulls': True,
            'normalize': ['amount'],
            'features': ['day_of_week', 'is_weekend']
        },
        'load': {
            'destination': 'output.csv',
            'if_exists': 'replace'
        }
    }

# Test configuration loading
def test_config_loading(sample_config, tmp_path):
    # Write config to temporary file
    config_path = tmp_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(sample_config, f)
    
    # Test loading configuration
    loaded_config = load_pipeline_config(config_path)
    
    # Verify
    assert loaded_config == sample_config

# Test database extraction with mock
def test_database_extraction(mock_database):
    with patch('mypackage.pipeline.create_db_connection', return_value=mock_database):
        # Extract data
        data = extract_from_database(query="SELECT * FROM transactions")
        
        # Verify
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert 'transaction_id' in data.columns
        assert 'amount' in data.columns
```

### 6. Testing Error Handling

Verify how the pipeline handles errors:

```python
def test_extraction_error_handling(mocker):
    """Test how extraction handles errors."""
    # Mock database that raises an exception
    mock_db = mocker.patch('mypackage.pipeline.create_db_connection')
    mock_db.side_effect = ConnectionError("Database connection failed")
    
    # Test extraction with error handling
    with pytest.raises(ConnectionError) as excinfo:
        extract_from_database(query="SELECT * FROM test")
    
    # Verify error message
    assert "Database connection failed" in str(excinfo.value)

def test_transform_with_invalid_data():
    """Test transformation with invalid data."""
    # Create invalid data
    invalid_data = pd.DataFrame({
        'id': [1, 2, 3],
        'invalid_column': ['a', 'b', 'c']  # Missing expected column
    })
    
    # Test transformation with validation
    with pytest.raises(ValueError) as excinfo:
        transform_data(invalid_data)
    
    # Verify error message
    assert "Missing required column" in str(excinfo.value)

def test_pipeline_retry_mechanism(mocker):
    """Test that pipeline retries on temporary failures."""
    # Mock a function that fails on first call, succeeds on second
    mock_extract = mocker.patch('mypackage.pipeline.extract_data')
    
    # Set up side effect to fail first, then succeed
    mock_extract.side_effect = [
        ConnectionError("Temporary failure"),
        pd.DataFrame({'data': [1, 2, 3]})
    ]
    
    # Run pipeline with retry
    result = run_pipeline_with_retry(
        input_path="test_input",
        max_retries=3,
        retry_delay=0.1
    )
    
    # Verify pipeline completed after retry
    assert result is not None
    assert mock_extract.call_count == 2  # Called twice
```

### 7. Performance Testing

Test the performance of your pipeline:

```python
import time
import pytest
import pandas as pd
import numpy as np

@pytest.mark.slow
def test_pipeline_performance():
    """Test performance with larger dataset."""
    # Generate large test dataset
    large_df = pd.DataFrame({
        'id': range(100000),
        'value': np.random.normal(size=100000),
        'category': np.random.choice(['A', 'B', 'C'], size=100000)
    })
    
    # Measure performance
    start_time = time.time()
    result = process_large_dataset(large_df)
    execution_time = time.time() - start_time
    
    # Verify performance meets expectations
    assert execution_time < 5.0  # Should process within 5 seconds
    assert len(result) == len(large_df)

@pytest.mark.parametrize("data_size", [10, 100, 1000, 10000])
def test_scaling_behavior(data_size):
    """Test how processing time scales with data size."""
    # Generate test data
    df = pd.DataFrame({
        'id': range(data_size),
        'value': np.random.normal(size=data_size)
    })
    
    # Measure execution time
    start_time = time.time()
    process_data(df)
    execution_time = time.time() - start_time
    
    # Log results for analysis
    print(f"Size: {data_size}, Time: {execution_time:.4f}s")
    
    # We expect roughly linear scaling, but with some overhead
    # This is a simple check that execution doesn't explode unexpectedly
    if data_size > 100:
        assert execution_time < (data_size / 100) * 2  # Allow some margin
```

### 8. Testing Data Quality

Verify data quality at each pipeline stage:

```python
import pandas as pd
import numpy as np
import pytest
from mypackage.pipeline import process_data
from mypackage.validation import validate_schema, check_data_quality

def test_output_schema():
    """Test that output conforms to expected schema."""
    # Input data
    input_data = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30],
        'category': ['A', 'B', 'C']
    })
    
    # Expected schema
    expected_schema = {
        'id': 'int',
        'value': 'float',
        'category': 'object',
        'value_normalized': 'float',
        'category_encoded': 'int'
    }
    
    # Process data
    result = process_data(input_data)
    
    # Validate schema
    schema_validation = validate_schema(result, expected_schema)
    assert schema_validation['valid'] == True, schema_validation['errors']
    
    # Check column presence
    for column in expected_schema:
        assert column in result.columns

def test_data_quality_metrics():
    """Test data quality metrics on pipeline output."""
    # Generated test data with known characteristics
    input_data = pd.DataFrame({
        'id': range(1000),
        'value': np.random.normal(100, 15, size=1000),
        'category': np.random.choice(['A', 'B', 'C'], size=1000, p=[0.6, 0.3, 0.1])
    })
    
    # Process data
    output_data = process_data(input_data)
    
    # Check data quality
    quality_report = check_data_quality(output_data)
    
    # Verify quality metrics
    assert quality_report['row_count'] == 1000
    assert quality_report['null_count']['value'] == 0
    assert abs(quality_report['mean']['value'] - 100) < 2  # Expected mean with some margin
    assert abs(quality_report['std']['value'] - 15) < 2  # Expected std with some margin
    
    # Check category distribution (with tolerance)
    cat_dist = quality_report['category_distribution']
    assert 0.55 < cat_dist['A'] < 0.65  # Expected around 0.6
    assert 0.25 < cat_dist['B'] < 0.35  # Expected around 0.3
    assert 0.05 < cat_dist['C'] < 0.15  # Expected around 0.1
```

### 9. CI/CD Integration

Configure tests for continuous integration:

```python
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Tests that take longer to run
    performance: Performance tests

# .github/workflows/python-tests.yml
name: Python Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
        pip install -e .
    - name: Run unit tests
      run: |
        pytest tests/ -m "unit" -v
    - name: Run integration tests
      run: |
        pytest tests/ -m "integration" -v
    - name: Run performance tests
      run: |
        pytest tests/ -m "performance" -v
```

### 10. Best Practices for Data Pipeline Testing

1. **Test at multiple levels**:
   - Unit tests for individual functions
   - Integration tests for component interactions
   - End-to-end tests for complete pipeline

2. **Use appropriate test data**:
   - Small datasets for unit tests
   - Realistic datasets for integration tests
   - Edge cases to test error handling

3. **Parameterize tests**:
   - Test multiple scenarios efficiently
   - Test with different configurations

4. **Mock external dependencies**:
   - Databases, APIs, file systems
   - Focus tests on your code, not external systems

5. **Validate data quality**:
   - Test for schema compliance
   - Check statistical properties
   - Verify business rules

6. **Test performance**:
   - Set performance expectations
   - Monitor scaling behavior
   - Create performance regression tests

7. **Test failure modes**:
   - Network failures
   - Invalid input data
   - Missing permissions

8. **Document test coverage**:
   - Use tools like pytest-cov to track coverage
   - Ensure critical paths are well-tested

## 47. Explain mocking in Python testing. When and how would you use it?

**Key Concept**: Mocking replaces real dependencies with simulated objects to create isolated, controlled testing environments, enabling testing without relying on external systems.

### 1. What is Mocking and Why Use It

Mocking is creating simulated objects that mimic the behavior of real components:

- **Why use mocking**:
  - Isolate the code being tested
  - Remove external dependencies (databases, APIs, file systems)
  - Test specific scenarios or error conditions
  - Increase test speed and reliability
  - Test components that aren't fully developed yet

```python
# Without mocking: Tests depend on actual database
def test_get_user_without_mock():
    user = get_user_from_database(user_id=123)
    assert user.name == "John"  # Breaks if database changes

# With mocking: Tests are isolated from the database
def test_get_user_with_mock(mocker):
    # Mock the database function
    mock_db = mocker.patch('myapp.database.query')
    mock_db.return_value = {'id': 123, 'name': 'John', 'email': 'john@example.com'}
    
    # Now test function that uses database
    user = get_user_from_database(user_id=123)
    assert user.name == "John"  # Works regardless of actual database
```

### 2. Different Types of Test Doubles

```python
# 1. Dummy: Objects passed but never used
dummy_logger = None

# 2. Fake: Working implementations, but not suitable for production
class FakeDatabase:
    def __init__(self):
        self.data = {}
    
    def save(self, key, value):
        self.data[key] = value
    
    def get(self, key):
        return self.data.get(key)

# 3. Stubs: Provide canned answers
class UserServiceStub:
    def get_user(self, user_id):
        return {'id': user_id, 'name': 'Test User'}

# 4. Spies: Record calls made
class DatabaseSpy:
    def __init__(self):
        self.queries = []
    
    def execute(self, query):
        self.queries.append(query)
        return []
    
    def get_executed_queries(self):
        return self.queries

# 5. Mocks: Pre-programmed with expectations
# (Usually created with mocking frameworks)
```

### 3. Using the `unittest.mock` Library

```python
import unittest
from unittest.mock import Mock, MagicMock, patch
import requests
from myapp.user_service import get_user_profile

class TestUserService(unittest.TestCase):
    def test_get_user_profile_with_mock(self):
        # Create a mock for the requests.get function
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 123,
            'name': 'John Doe',
            'email': 'john@example.com'
        }
        
        # Replace requests.get with our mock
        with patch('requests.get', return_value=mock_response):
            # Call function that uses requests.get
            user = get_user_profile(user_id=123)
            
            # Assert on the result
            self.assertEqual(user['name'], 'John Doe')
    
    def test_get_user_profile_connection_error(self):
        # Make requests.get raise an exception when called
        with patch('requests.get', side_effect=requests.ConnectionError("API unavailable")):
            # Test error handling behavior
            with self.assertRaises(ValueError) as context:
                get_user_profile(user_id=123)
            
            self.assertIn("API connection failed", str(context.exception))

if __name__ == '__main__':
    unittest.main()
```

### 4. Using pytest-mock

```python
import pytest
import pandas as pd
from myapp.data_processor import process_csv_file

def test_process_csv_file(mocker):
    # Create a mock for pandas.read_csv
    mock_df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })
    mocker.patch('pandas.read_csv', return_value=mock_df)
    
    # Mock os.path.exists to return True
    mocker.patch('os.path.exists', return_value=True)
    
    # Call the function we're testing
    result = process_csv_file('dummy_file.csv')
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'id' in result.columns
    assert 'value_squared' in result.columns  # Assuming the function adds this column
    assert result['value_squared'].tolist() == [100, 400, 900]
```

### 5. Mocking Methods in Classes

```python
from unittest.mock import patch
import pytest
from myapp.database import Database
from myapp.user_repository import UserRepository

class TestUserRepository:
    def test_get_user(self, mocker):
        # Mock the Database class's get_connection method
        mock_connection = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        
        # Configure the cursor's fetchone method to return user data
        mock_cursor.fetchone.return_value = (1, "John Doe", "john@example.com")
        
        # Configure connection to return our cursor
        mock_connection.cursor.return_value = mock_cursor
        
        # Patch the Database.get_connection method
        with patch.object(Database, 'get_connection', return_value=mock_connection):
            # Create repository and call method
            repository = UserRepository()
            user = repository.get_user(1)
            
            # Assertions
            assert user.id == 1
            assert user.name == "John Doe"
            assert user.email == "john@example.com"
            
            # Verify SQL query was executed correctly
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "SELECT * FROM users WHERE id = ?" in call_args[0]
            assert call_args[1] == (1,)
```

### 6. Mocking in Data Engineering Pipelines

```python
import pytest
import pandas as pd
from myapp.etl_pipeline import extract, transform, load

class TestETLPipeline:
    def test_extract_from_api(self, mocker):
        # Mock API client
        mock_api = mocker.patch('myapp.etl_pipeline.APIClient')
        mock_api.return_value.get_data.return_value = [
            {'id': 1, 'name': 'Product A', 'price': 10.99},
            {'id': 2, 'name': 'Product B', 'price': 24.99}
        ]
        
        # Call extract function
        data = extract(source='api', api_key='dummy_key')
        
        # Verify API was called correctly
        mock_api.assert_called_once_with(api_key='dummy_key')
        mock_api.return_value.get_data.assert_called_once()
        
        # Verify extracted data
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert data['price'].sum() == 35.98
    
    def test_transform(self):
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2],
            'price': [10.99, 24.99]
        })
        
        # Mock database connection
        mock_db = mocker.patch('myapp.etl_pipeline.Database')
        mock_connection = mock_db.return_value.get_connection.return_value
        
        # Call load function
        result = load(data, destination='database')
        
        # Verify database was called correctly
        mock_db.assert_called_once()
        mock_connection.execute.assert_called()
        
        # Typically, we'd verify the SQL, but it depends on implementation
        assert result == True  # Assuming load returns True on success,
            'category': ['A', 'B']
        })
        
        # Call transform function
        result = transform(data)
        
        # Verify transformation
        assert 'price_normalized' in result.columns
        assert result['price_normalized'].max() <= 1.0
        assert result['price_normalized'].min() >= 0.0
    
    def test_load_to_database(self, mocker):
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2],
            'price': [10.99, 24.99]