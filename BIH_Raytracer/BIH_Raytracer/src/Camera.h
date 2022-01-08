#pragma once
#include "Ray.h"

class Camera {
public:
	__host__ Camera( glm::vec3 origin, float aspectRatio );
	__device__ Ray GetRay( float u, float v );

	glm::vec3 m_origin;
	glm::vec3 m_lowerLeftCorner;
	glm::vec3 m_horizontal;
	glm::vec3 m_vertical;
};

