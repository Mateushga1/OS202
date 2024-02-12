from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def bucket_sort(arr):
    max_val = max(arr)
    min_val = min(arr)
    bucket_range = (max_val - min_val) / size
    
    # Assign elements to buckets
    buckets = [[] for _ in range(size)]
    for num in arr:
        index = int((num - min_val) // bucket_range)
        if index != size:
            buckets[index].append(num)
        else:
            buckets[size - 1].append(num)
    
    # Sort individual buckets
    for i in range(size):
        buckets[i].sort()
    
    # Gather all the buckets
    all_buckets = comm.gather(buckets, root=0)
    
    # Merge the buckets
    if rank == 0:
        sorted_arr = []
        for b in all_buckets:
            for bucket in b:
                sorted_arr.extend(bucket)
        sorted_arr.sort()
        return sorted_arr
    else:
        return None

if rank == 0:
    unsorted_array = np.random.randint(0, 100, 20)  # Generate random array
    print("Unsorted array:", unsorted_array)
    chunk_size = len(unsorted_array) // size
    chunks = np.array_split(unsorted_array, size)
else:
    chunks = None

# Scatter the chunks to all processes
chunk = comm.scatter(chunks, root=0)

# Perform bucket sort
sorted_chunk = bucket_sort(chunk)

# Gather sorted chunks
sorted_array = comm.gather(sorted_chunk, root=0)

if rank == 0:
    # Flatten the sorted array
    final_sorted_array = [item for sublist in sorted_array if sublist is not None for item in sublist]
    print("Sorted array:", final_sorted_array)
