#pragma once

#include "Hitable.h"
#include "Mesh.h"

class Triangle : public Hitable {
public:
	__device__ Triangle( Vertex v1, Vertex v2, Vertex v3 );
	__device__ virtual bool Hit( const Ray& r, float tmin, float tmax, HitRecord& rec ) const;

	Vertex v1;
	Vertex v2;
	Vertex v3;
};

