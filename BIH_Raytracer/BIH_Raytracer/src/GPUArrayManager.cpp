#include "GPUArrayManager.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sequence.h"
#include "thrust/execution_policy.h"

bool GPUArrayManager::AllocateTris(int size)
{
    m_triangles.resize(size);
    if ( m_triangles.size() < size )
        return false;
    m_trisBBoxes.arrCenterNorm.resize(size);
    if ( m_trisBBoxes.arrCenterNorm.size() < size )
        return false;
    m_trisBBoxes.arrCenter.resize(size);
    if ( m_trisBBoxes.arrCenter.size() < size )
        return false;
    m_trisBBoxes.arrHi.resize(size);
    if ( m_trisBBoxes.arrHi.size() < size )
        return false;
    m_trisBBoxes.arrLo.resize(size);
    if ( m_trisBBoxes.arrLo.size() < size )
        return false;
    m_trisIndexes.resize(size);
    if ( m_trisIndexes.size() < size )
        return false;
    thrust::sequence(thrust::device, m_trisIndexes.begin(), m_trisIndexes.begin() + size);
    
    m_trisSize = size;
    return true;
}

bool GPUArrayManager::AllocateMortonCodes(int size)
{
    m_mortonCodes.resize(size);
    thrust::fill(thrust::device, m_mortonCodes.begin(), m_mortonCodes.end(), -1);
    if ( m_mortonCodes.size() < size )
        return false;

    m_uniqueMortonCodes.resize(size);
    thrust::fill(thrust::device, m_uniqueMortonCodes.begin(), m_uniqueMortonCodes.end(), -1);
    if ( m_uniqueMortonCodes.size() < size )
        return false;

    m_duplicatesCnts.resize(size);
    thrust::fill(thrust::device, m_duplicatesCnts.begin(), m_duplicatesCnts.end(), -1);
    if ( m_duplicatesCnts.size() < size )
        return false;

    m_firstIdxs.resize(size);
    thrust::fill(thrust::device, m_firstIdxs.begin(), m_firstIdxs.end(), -1);
    if ( m_firstIdxs.size() < size )
        return false;

    return true;
}

bool GPUArrayManager::AllocateBIHTree(int size)
{
    thrust::host_vector<int> h_leafParents;
    h_leafParents.resize(size + 1);
    
    for ( auto& element : h_leafParents ) {
        element = -1;
    }

    m_leafParents = h_leafParents;
    if ( m_leafParents.size() < size + 1 )
        return false;


    thrust::host_vector<TreeInternalNode> h_Tree;
    h_Tree.resize(size);

    for ( auto& node : h_Tree ) {
        node.parent = -1;
        node.children[0] = node.children[1] = -1;
        node.t_axis = -1;
        node.t_clipPlanes[0] = std::numeric_limits<float>::lowest();
        node.t_clipPlanes[1] = std::numeric_limits<float>::max();
        node.isLeaf[0] = false;
        node.isLeaf[1] = false;
        node.ID = -1;
        node.traversed = false;
    }
    m_BIHTree = h_Tree;
    if ( m_BIHTree.size() < size )
        return false;
    
    return true;
}

bool GPUArrayManager::AllocateTraversalStacks(int size)
{
    thrust::host_vector<StackElement> h_traversalStacks;
    h_traversalStacks.resize(size);

    for ( auto& element : h_traversalStacks) {
        element.t_node = nullptr;
        element.t_tMin = std::numeric_limits<float>::infinity();
        element.t_tMax = std::numeric_limits<float>::infinity();
    }

    m_traversalStacks = h_traversalStacks;
    if ( m_traversalStacks.size() < size )
        return false;
    m_traversalStacksSize = size;

    return true;
}

AABBs& GPUArrayManager::GetBBoxArrays()
{
    return m_trisBBoxes;
}

thrust::device_vector<uint32_t>& GPUArrayManager::GetMortonCodesVectorRef()
{
    return m_mortonCodes;
}

thrust::device_vector<uint32_t>& GPUArrayManager::GetUniqueMortonCodesVectorRef()
{
    return m_uniqueMortonCodes;
}

thrust::device_vector<uint32_t>& GPUArrayManager::GetTrisIndexes()
{
    return m_trisIndexes;
}

thrust::device_vector<TreeInternalNode>& GPUArrayManager::GetBIHTree()
{
    return m_BIHTree;
}

thrust::device_vector<int>& GPUArrayManager::GetLeafParentIdxs()
{
    return m_leafParents;
}

thrust::device_vector<uint32_t>& GPUArrayManager::GetDuplicatesCnts()
{
    return m_duplicatesCnts;
}

thrust::device_vector<int>& GPUArrayManager::GetFirstIdxs()
{
    return m_firstIdxs;
}

thrust::device_vector<Triangle>& GPUArrayManager::GetTriangles()
{
    return m_triangles;
}

thrust::device_vector<StackElement>& GPUArrayManager::GetTraversalStacks()
{
    return m_traversalStacks;
}

int GPUArrayManager::GetTrisSize()
{
    return m_trisSize;
}

int GPUArrayManager::GetBIHTreeSize()
{
    return GetTrisSize() - 1;
}

int GPUArrayManager::GetTraversalStacksSize()
{
    return m_traversalStacksSize;
}

void GPUArrayManager::SetUniqueMCSize(int size)
{
    m_uniqueMCSize = size;
}

int GPUArrayManager::GetUniqueMCSize()
{
    return m_uniqueMCSize;
}

void GPUArrayManager::ResetClipPlanes()
{
    thrust::host_vector<TreeInternalNode> h_BIH = GetBIHTree();
    for ( auto& node : h_BIH ) {
        node.t_clipPlanes[0] = std::numeric_limits<float>::lowest();
        node.t_clipPlanes[1] = std::numeric_limits<float>::max();
    }
    m_BIHTree = h_BIH;
}

void GPUArrayManager::ResetTrisIdxs()
{
    thrust::sequence(thrust::device, m_trisIndexes.begin(), m_trisIndexes.begin() + GetTrisSize());
}
