#pragma once

#include "Hitable.h"

class HitableList : public Hitable {
public:
    __device__ HitableList();
    __host__ HitableList( Hitable** l, int n );
    __device__ virtual bool Hit( const Ray& r, float tmin, float tmax, HitRecord& rec ) const;
    
    Hitable** list;
    int list_size;
};

