#pragma once

#include "Hitable.h"
#include "Mesh.h"

class Triangle {
public:
	 Triangle( Vertex v1, Vertex v2, Vertex v3 );
	__device__ bool Hit( const Ray& r, float tmin, float tmax, HitRecord& rec ) const;

	Vertex v1;
	Vertex v2;
	Vertex v3;
};

