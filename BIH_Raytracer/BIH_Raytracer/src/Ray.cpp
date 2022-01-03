#include "Ray.h"

Ray::Ray() {
}

__device__ Ray::Ray( const glm::vec3& a, const glm::vec3& b ) {
    A = a; 
    B = b;
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
