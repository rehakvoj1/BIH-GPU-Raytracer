#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "linearprobing.h"
#include <cuda.h>
#include <iostream>
#include <thrust/execution_policy.h>


// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & ( kHashTableCapacity - 1 );
}


// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert( KeyValue* hashtable, const uint32_t* d_keys, unsigned int numkvs, uint32_t* size  )
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = d_keys[threadid];
       // uint32_t value = d_vals[threadid];
        uint32_t slot = hash(key);
        
        while (true)
        {
            uint32_t prev = atomicCAS(&(hashtable[slot].key), kEmpty, key);
            if (prev == kEmpty || prev == key)
            {
                if ( prev == kEmpty )
                {
                    atomicAdd(size, 1);
                }
                //hashtable[slot].value = value;
                return;
            }
    
            slot = (slot + 1) & (kHashTableCapacity-1);
        }
    }
}
 
void insert_hashtable(KeyValue* pHashTable, const uint32_t* d_keys, uint32_t num_kvs, uint32_t* size)
{
    // Copy the keyvalues to the GPU
    /*
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);
    */
    
    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, d_keys, (uint32_t)num_kvs, size );
    cudaDeviceSynchronize();
    /* cudaFree(device_kvs); */
}

// Lookup keys in the hashtable, and return the values
__global__ void gpu_hashtable_lookup(KeyValue* hashtable, KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
               // kvs[threadid].value = hashtable[slot].value;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
               // kvs[threadid].value = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void lookup_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_lookup << <gridsize, threadblocksize >> > (pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaFree(device_kvs);
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
__global__ void gpu_hashtable_delete(KeyValue* hashtable, const KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
               // hashtable[slot].value = kEmpty;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void delete_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_delete<< <gridsize, threadblocksize >> > (pHashTable, device_kvs, (uint32_t)num_kvs);

    
    cudaFree(device_kvs);
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__ void gpu_iterate_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t* kvs_size)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity) 
    {
        if (pHashTable[threadid].key != kEmpty) 
        {
            uint32_t value = pHashTable[threadid].key;
            if (value != kEmpty)
            {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[threadid];
            }
        }
    }
}

void iterate_hashtable(KeyValue* pHashTable, thrust::device_vector<KeyValue> & pOutArray)
{
    uint32_t* device_num_kvs;
    cudaMalloc(&device_num_kvs, sizeof(uint32_t));
    cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

    uint32_t num_kvs;
    cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    pOutArray.resize(num_kvs);
    thrust::copy_n(thrust::device, device_kvs, num_kvs, pOutArray.begin());
    
    cudaFree(device_kvs);
    cudaFree(device_num_kvs);

}

// Free the memory of the hashtable
void destroy_hashtable(KeyValue* pHashTable)
{
    cudaFree(pHashTable);
}
