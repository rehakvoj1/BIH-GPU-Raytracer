#pragma once
#include "Ray.h"
#include "Managed.cu"

class Camera : public Managed {
public:
	 Camera( glm::vec3 origin, float aspectRatio );
	 Camera( const Camera& c );
	 __device__ Ray GetRay( float u, float v );

	 Camera& operator=( const Camera& c );
	 

	glm::vec3 m_origin;
	glm::vec3 m_lowerLeftCorner;
	glm::vec3 m_horizontal;
	glm::vec3 m_vertical;
};

