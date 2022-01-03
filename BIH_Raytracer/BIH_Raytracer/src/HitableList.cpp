#include "HitableList.h"

__device__ HitableList::HitableList(): list(nullptr), list_size(0) 
{
}

__device__ HitableList::HitableList( Hitable** l, int n ) : list(l), list_size(n)
{
}

__device__ bool HitableList::Hit( const Ray& r, float t_min, float t_max, HitRecord& rec ) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for ( int i = 0; i < list_size; i++ ) {
        if ( list[i]->Hit( r, t_min, closest_so_far, temp_rec ) ) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}