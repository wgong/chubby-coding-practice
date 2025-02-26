import multiprocessing as mp
import time
import random
from datetime import datetime

def producer(queue, producer_id, num_items):
    """Producer function that places items into a shared queue."""
    for i in range(num_items):
        # Create an item to produce
        item = f"Producer {producer_id} - Item {i}"
        
        # Simulate variable production time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Place item in the queue
        queue.put(item)
        ts = str(datetime.now())
        print(f"[Producer {producer_id}] Produced: {item} \t ({ts})")
    
    # Indicate this producer is done
    queue.put(None)  # Signal value
    print(f"[Producer {producer_id}] Done producing.")

def consumer(queue, consumer_id):
    """Consumer function that processes items from a shared queue."""
    while True:
        # Get an item from the queue, waiting if necessary
        item = queue.get()
        
        # Check for Signal / sentinel value (end of production)
        if item is None:
            # Put the Signal / sentinel back for other consumers and exit
            queue.put(None)
            break
            
        # Process the item (simulate work)
        ts = str(datetime.now())
        print(f"[**Consumer** {consumer_id}] Processing: {item} \t ({ts})")
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
