#pragma once
#include "glm/glm.hpp"
#include <thrust/device_vector.h>
#include "linearprobing.h"
#include "AABB.h"
#include "Mesh.h"
#include "Ray.h"

struct HitRecord {
	glm::vec3 pointOfIntersect;
	glm::vec3 normal;
	double t;
};

struct TreeInternalNode {
	float t_clipPlanes[2];
	int t_axis; // 0 == x; 1 == y; 2 == z
	bool isLeftLeaf;
	bool isRightLeaf;
	int children[2];
	int parent;
};

struct StackElement {
	TreeInternalNode* t_node;
	float t_tMin;
	float t_tMax;
};

struct TreeLeafNode {
	uint32_t idxFirst;
	int parent;
};

class Triangle {
public:
	//Triangle();
	//Triangle(Vertex v1, Vertex v2, Vertex v3);
	//__device__ bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;

};
/*
struct Tree : public Managed {
	
	~Tree()
	{
		cudaFree( d_mortonCodes );
		cudaFree( d_mortonCodesUnique );
		cudaFree( d_internalNodes );
		cudaFree( d_leafNodes );
		cudaFree( d_idxsToLeafNodes );
	}


	AABB t_sceneBbox;
	float3* t_trianglesBBoxes{ nullptr };
	
	uint32_t* d_mortonCodes{ nullptr };			   // length => t_triangleCnt 
	int* d_idxsToTriangles{ nullptr };			   // length => t_triangleCnt 
	int t_triangleCnt{ 0 };


	uint32_t* size;
	thrust::device_vector<KeyValue> d_unique;
	TreeInternalNode* d_internalNodes{ nullptr };  // length => t_leafNodesLen - 1
	uint32_t* d_mortonCodesUnique{ nullptr };      // length => t_leafNodesLen
	TreeLeafNode* d_leafNodes{ nullptr };		   // length => t_leafNodesLen
	uint32_t* d_idxsToLeafNodes{ nullptr };		   // length => t_leafNodesLen
	KeyValue* d_hashTable{ nullptr };
	int t_leafNodesLen{ 0 };

};*/