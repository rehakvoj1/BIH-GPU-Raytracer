#pragma once

#include "Ray.h"

struct HitRecord {
	float t;
	glm::vec3 p;
	glm::vec3 normal;
};

class Hitable {
public:
	__device__ virtual bool Hit( const Ray& r, float t_min, float t_max, HitRecord& rec ) const = 0;
};

