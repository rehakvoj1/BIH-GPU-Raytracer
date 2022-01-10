#pragma once

#include <iostream>
#include "Hitable.h"
#include "Triangle.h"
#include "Managed.cu"

class HitableList : public Managed {
public:
    HitableList();
    HitableList( const HitableList& list ); 
    ~HitableList();
    __device__ bool Hit( const Ray& r, float tmin, float tmax, HitRecord& rec ) const;
        
    HitableList& operator=( const HitableList& list );
    __host__ __device__ Triangle& operator[]( int pos ) const;

    Triangle* m_list;
    int m_listSize;

private:
    void _realloc( int len );
};

