#include "Triangle.h"

Triangle::Triangle( Vertex vert1, Vertex vert2, Vertex vert3 ) : v1( vert1 ), v2( vert2 ), v3( vert3 ) 
{
}

__device__ bool Triangle::Hit( const Ray& r, float tmin, float tmax, HitRecord& rec ) const {
    return true;
}
