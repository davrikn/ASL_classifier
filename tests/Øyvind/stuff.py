# Initializing a queue
queue = []
  
# Adding elements to the queue
queue.append('a')
queue.append('b')
queue.append('c')
  
print("Initial queue")
print(queue)
  
# Removing elements from the queue
print("\nElements dequeued from queue")
print(queue.pop(0))
queue.append('d')
print(queue.pop(0))
queue.append('e')

  
print("\nQueue after removing elements")
print(queue)