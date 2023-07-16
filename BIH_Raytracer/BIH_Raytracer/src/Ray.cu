#include "Ray.h"

__device__ Ray::Ray( const glm::vec3& a, const glm::vec3& b ) {
    A = a;
    B = b;
    invDir = { 1 / b.x, 1 / b.y, 1 / b.z };
    sign[0] = ( invDir.x < 0 );
    sign[1] = ( invDir.y < 0 );
    sign[2] = ( invDir.z < 0 );
}

__device__ glm::vec3 Ray::Origin() const {
    return A;
}

__device__ glm::vec3 Ray::Direction() const {
    return B;
}

__device__ glm::vec3 Ray::PointAtParameter( float t ) const {
    return A + t * B;
}
