#include "Triangle.h"

__device__ bool Triangle::Hit( const Ray& r, float tmin, float tmax, HitRecord& rec ) const {
    glm::vec3 v1v2 = v2.Position - v1.Position;
    glm::vec3 v1v3 = v3.Position - v1.Position;

    glm::vec3 pvec = glm::cross( r.Direction(), v1v3 );

    float det = glm::dot( v1v2, pvec );

    if ( det < 0.000001 )
        return false;

    float invDet = 1.0 / det;

    glm::vec3 tvec = r.Origin() - v1.Position;

    float u = dot( tvec, pvec ) * invDet;

    if ( u < 0 || u > 1 )
        return false;

    glm::vec3 qvec = glm::cross( tvec, v1v2 );

    float v = glm::dot( r.Direction(), qvec ) * invDet;

    if ( v < 0 || u + v > 1 )
        return false;

    return true;
}
