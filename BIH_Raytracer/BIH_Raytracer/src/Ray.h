#pragma once

#include "glad/glad.h"

#include "glm/glm.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


class Ray {
public:
    __device__ Ray();
    __device__ Ray( const glm::vec3& a, const glm::vec3& b );
    __device__ glm::vec3 Origin() const;
    __device__ glm::vec3 Direction() const;
    __device__ glm::vec3 PointAtParameter( float t ) const;
    

    glm::vec3 A;
    glm::vec3 B;
    glm::vec3 invDir;
    int sign[3];
};
