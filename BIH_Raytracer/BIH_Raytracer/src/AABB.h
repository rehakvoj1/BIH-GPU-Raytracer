#pragma once

#include "glm/glm.hpp"
#include "thrust/device_vector.h"

struct AABB {
	float3 lo;
	float3 hi;
	float3 centroid;
	float3 centroid_normalized;
};


struct AABBs {
	float3 sceneBBoxLo;
	float3 sceneBBoxHi;
	thrust::device_vector<float3> arrLo;
	thrust::device_vector<float3> arrHi;
	thrust::device_vector<float3> arrCenter;
	thrust::device_vector<float3> arrCenterNorm;
};