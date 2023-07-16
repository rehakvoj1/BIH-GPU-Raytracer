#pragma once
#include "AABB.h"
#include "Tree.cuh"
#include "Triangle.cuh"

class GPUArrayManager
{
public:
	GPUArrayManager() : m_trisSize{0}, m_uniqueMCSize{0}, m_traversalStacksSize{0}
	{}

	// allocate objects
	bool AllocateTris(int size);
	bool AllocateMortonCodes(int size);
	bool AllocateBIHTree(int size);
	bool AllocateTraversalStacks(int size);

	// object getters
	AABBs& GetBBoxArrays();
	thrust::device_vector<uint32_t>& GetMortonCodesVectorRef();
	thrust::device_vector<uint32_t>& GetUniqueMortonCodesVectorRef();
	thrust::device_vector<uint32_t>& GetTrisIndexes();
	thrust::device_vector<TreeInternalNode>& GetBIHTree();
	thrust::device_vector<int>& GetLeafParentIdxs();
	thrust::device_vector<uint32_t>& GetDuplicatesCnts();
	thrust::device_vector<int>& GetFirstIdxs();
	thrust::device_vector<Triangle>& GetTriangles();
	thrust::device_vector<StackElement>& GetTraversalStacks();
	// size getters
	int GetTrisSize();
	int GetBIHTreeSize();
	int GetTraversalStacksSize();

	void SetUniqueMCSize(int size);
	int  GetUniqueMCSize();


private:
	int m_trisSize;
	int m_uniqueMCSize;
	int m_traversalStacksSize;

	AABBs m_trisBBoxes;
	thrust::device_vector<Triangle> m_triangles;
	thrust::device_vector<uint32_t> m_uniqueMortonCodes;
	thrust::device_vector<uint32_t> m_mortonCodes;
	thrust::device_vector<uint32_t> m_trisIndexes;
	thrust::device_vector<TreeInternalNode> m_BIHTree;
	thrust::device_vector<int> m_leafParents;
	thrust::device_vector<uint32_t> m_duplicatesCnts;
	thrust::device_vector<int> m_firstIdxs;
	thrust::device_vector<StackElement> m_traversalStacks;
};

