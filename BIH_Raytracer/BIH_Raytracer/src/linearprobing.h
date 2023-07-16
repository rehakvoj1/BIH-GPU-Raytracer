#pragma once

#include "thrust/device_vector.h"

struct KeyValue
{
    uint32_t key;
    //uint32_t value;
};

const uint32_t kHashTableCapacity = 128 * 1024 * 1024;

const uint32_t kNumKeyValues = kHashTableCapacity / 2;

const uint32_t kEmpty = 0xffffffff;

__device__ uint32_t hash(uint32_t k);

void create_hashtable(KeyValue* pHashTable);

void insert_hashtable(KeyValue* hashtable, const uint32_t* d_keys, uint32_t num_kvs, uint32_t* size);

void lookup_hashtable(KeyValue* hashtable, KeyValue* kvs, uint32_t num_kvs);

void delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

void iterate_hashtable(KeyValue* hashtable, thrust::device_vector<KeyValue> & pOutArray);

void destroy_hashtable(KeyValue* hashtable);
