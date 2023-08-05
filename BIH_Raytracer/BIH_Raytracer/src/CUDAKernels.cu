#pragma once
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "Renderer.h"
#include "App.h"
#include "Constants.h"
#include "Camera.h"
#include <inttypes.h>
#include <float.h>
#include "math_constants.h"
#include <bitset>
#include "linearprobing.h"
#include "glm/glm.hpp"
#include "Tree.cuh"


__device__ bool RayTriangleIntersection(const Triangle& triangle, const Ray& ray, float& outT) {
    glm::vec3 v0v1 = triangle.v1 - triangle.v0;
    glm::vec3 v0v2 = triangle.v2 - triangle.v0;
    
    //printf("ray origin: [%.2f,%.2f,%.2f]\n", ray.Origin().x, ray.Origin().y, ray.Origin().z);
    //printf("ray dir: [%.2f,%.2f,%.2f]\n", ray.Direction().x, ray.Direction().y, ray.Direction().z);

    glm::vec3 pvec = glm::cross(ray.Direction(), v0v2);

    float det = glm::dot(v0v1,pvec);

    if ( det < 0.000001 )
        return false;

    float invDet = 1.0 / det;

    glm::vec3 tvec = ray.Origin() - triangle.v0;

    float u = dot(tvec,pvec) * invDet;

    if ( u < 0 || u > 1 )
        return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);

    float v = glm::dot(ray.Direction(), qvec) * invDet;

    if ( v < 0 || u + v > 1 )
        return false;

    outT = glm::dot(v0v2,qvec) * invDet;
    
    return true;
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ int signum( int val )
{
    return ( 0 < val ) - ( val < 0 );
}

// clamp x to range [a, b]
__device__ float clamp( float x, float a, float b ) {
    return max( a, min( b, x ) );
}

__device__ int clamp( int x, int a, int b ) {
    return max( a, min( b, x ) );
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt( float r, float g, float b ) {
    r = clamp( r, 0.0f, 255.0f );
    g = clamp( g, 0.0f, 255.0f );
    b = clamp( b, 0.0f, 255.0f );
    return ( int( b ) << 16 ) | ( int( g ) << 8 ) | int( r );
}

__device__ void PrintFloatInBinary( float f )
{
    int a[32] = { 0 }, i;

    union
    {
        float f; uint32_t i;
    } u;
    u.f = f;

    // Loop to calculate and store the binary format
    for ( i = 0; u.i > 0; i++ )
    {
        a[i] = u.i % 2;
        u.i = u.i / 2;
    }

    printf( "\nBinary Format:" );

    // Loop to print the binary format of given number
    for ( i = 31; i >= 0; i-- )
    {
        printf( "%d", a[i] );
    }
}

__device__ void PrintIntInBinary( int number )
{
    int a[32] = { 0 }, i;


    // Loop to calculate and store the binary format
    for ( i = 0; number > 0; i++ )
    {
        a[i] = number % 2;
        number = number / 2;
    }

    printf( "\nBinary Format:" );

    // Loop to print the binary format of given number
    for ( i = 31; i >= 0; i-- )
    {
        printf( "%d", a[i] );
    }
}

__device__ bool IntersectNode(TreeInternalNode* BIHTree,
                              const Ray& r,
                              int nodeIdx,
                              float& tMin,
                              float& tMax,
                              HitRecord& outRecord) 
{
    TreeInternalNode& currNode = BIHTree[nodeIdx];



    //if ( currNode.isLeftLeaf || currNode.isRightLeaf ) {
        // currNode.leftChild/rightChild <= index do pole FirstIdxs
        // pole duplicatesCnts mi da pocet trojuhelniku
        // ty proiteruju a zjistim nejmensi "t", normalu a pointOfIntersect
        // to strcim do HitRecord
        return true;
    //}
}

__device__ bool TraverseTriangles(  int BihSize,
                                    TreeInternalNode* BIHTree,
                                    int* firstIdxs,
                                    uint32_t* duplicatesCnts,
                                    Triangle* triangles,
                                    uint32_t* triangleIdxs,
                                    const Ray& r,
                                    HitRecord& outRecord)
{
    float t = FLT_MAX;
    for ( int i = 0; i < BihSize; i++ ) {
        if ( BIHTree[i].isLeaf[0] ) {
            int umcIdx = BIHTree[i].children[0];
            int firstIdx = firstIdxs[umcIdx];
            int duplicateCnt = duplicatesCnts[umcIdx];
            
            for ( int j = firstIdx; j < firstIdx + duplicateCnt; j++ ) {
                int triangleIdx = triangleIdxs[j];
                if ( RayTriangleIntersection(triangles[triangleIdx], r, t) ) {
                    if ( t > 0 && t < outRecord.t ) {
                        outRecord.t = t;
                        outRecord.triangleIdx = i;
                    }
                }
            }
        }

        if ( BIHTree[i].isLeaf[1] ) {
            int umcIdx = BIHTree[i].children[1];
            int firstIdx = firstIdxs[umcIdx];
            int duplicateCnt = duplicatesCnts[umcIdx];

            for ( int j = firstIdx; j < firstIdx + duplicateCnt; j++ ) {
                int triangleIdx = triangleIdxs[j];
                if ( RayTriangleIntersection(triangles[triangleIdx], r, t) ) {
                    if ( t > 0 && t < outRecord.t ) {
                        outRecord.t = t;
                        outRecord.triangleIdx = i;
                    }
                }
            }
        }

    }
    return outRecord.triangleIdx >= 0;
}



__device__ void FindNearestTriangle( int* firstIdxs,
                                      uint32_t* duplicatesCnts,
                                      Triangle* triangles,
                                      uint32_t* triangleIdxs,
                                      const Ray& r,
                                      int childIdx,
                                      HitRecord& outRecord ) 
{
    float t = FLT_MAX;
    for ( int i = firstIdxs[childIdx]; i < firstIdxs[childIdx] + duplicatesCnts[childIdx]; i++ ) {
        uint32_t triangleIdx = triangleIdxs[i];
        if ( RayTriangleIntersection(triangles[triangleIdx], r, t) ) {
            if ( t > 0 && t < outRecord.t ) {
                outRecord.t = t;
                outRecord.triangleIdx = i;
            }
        }
    }
}


__device__ bool TraverseTree(const Ray& r,
                             TreeInternalNode* BIHTree,
                             int* firstIdxs,
                             uint32_t* duplicatesCnts,
                             Triangle* triangles,
                             uint32_t* triangleIdxs,
                             float3 sceneBBoxLo,
                             float3 sceneBBoxHi,
                             HitRecord& outRecord) {

    glm::vec3 sceneBBox[2]{ { sceneBBoxLo.x, sceneBBoxLo.y, sceneBBoxLo.z }, { sceneBBoxHi.x, sceneBBoxHi.y, sceneBBoxHi.z } };
    float tymin, tymax, tzmin, tzmax;

    float tMin = ( sceneBBox[r.sign[0]].x - r.Origin().x ) * r.invDir.x;
    float tMax = ( sceneBBox[1 - r.sign[0]].x - r.Origin().x ) * r.invDir.x;
    tymin = ( sceneBBox[r.sign[1]].y - r.Origin().y ) * r.invDir.y;
    tymax = ( sceneBBox[1 - r.sign[1]].y - r.Origin().y ) * r.invDir.y;

    if ( ( tMin > tymax ) || ( tymin > tMax ) )
        return false;

    if ( tymin > tMin )
        tMin = tymin;
    if ( tymax < tMax )
        tMax = tymax;

    tzmin = ( sceneBBox[r.sign[2]].z - r.Origin().z ) * r.invDir.z;
    tzmax = ( sceneBBox[1 - r.sign[2]].z - r.Origin().z ) * r.invDir.z;

    if ( ( tMin > tzmax ) || ( tzmin > tMax ) )
        return false;

    if ( tzmin > tMin )
        tMin = tzmin;
    if ( tzmax < tMax )
        tMax = tzmax;


    
    TreeInternalNode* currNode = BIHTree;
    float t[2];
    int splitAxis = -1;
    float dir = INFINITY;
    float org = INFINITY;
    float invDir = INFINITY;
    int near = -1;
    int far = -1;
    
     
    StackElement stack[64];
    int stackIdx = 0;
    stack[stackIdx].t_node = nullptr;
    stackIdx++;
    while( currNode != nullptr ) {

        splitAxis = currNode->t_axis;
        dir = r.Direction()[splitAxis];
        org = r.Origin()[splitAxis];
        invDir = r.invDir[splitAxis];
        near = r.sign[splitAxis];
        far = 1 - near;
        t[0] = ( currNode->t_clipPlanes[0] - org ) * invDir;
        t[1] = ( currNode->t_clipPlanes[1] - org ) * invDir;

        
        bool tMinLessThanNear = ( tMin < t[near] );
        bool tMaxLessThanFar = ( tMax < t[far] );

        bool noIntersection = ( !tMinLessThanNear && tMaxLessThanFar );
        bool nearIntersection = ( tMinLessThanNear && tMaxLessThanFar );
        bool farIntersection = ( !tMinLessThanNear && !tMaxLessThanFar );
        bool bothIntersection = ( tMinLessThanNear && !tMaxLessThanFar );

        if ( currNode->isLeaf[near] || currNode->isLeaf[far] ) {
            if ( currNode->isLeaf[near] && !currNode->isLeaf[far] )
            {
                FindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[near], outRecord);
                currNode = &( BIHTree[currNode->children[far]] );
                tMin = t[far];
            }
            else if( currNode->isLeaf[far] && !currNode->isLeaf[near] )
            {
                FindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[far], outRecord);
                currNode = &( BIHTree[currNode->children[near]] );
                tMax = t[near];
            }
            else {
                FindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[near], outRecord);
                FindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[far], outRecord);
                stackIdx--;
                currNode = stack[stackIdx].t_node;
                tMin = stack[stackIdx].t_tMin;
                tMax = stack[stackIdx].t_tMax;

            }
        }
        else 
        {
            

            if ( noIntersection ) {
                stackIdx--;
                currNode = stack[stackIdx].t_node;
                tMin = stack[stackIdx].t_tMin;
                tMax = stack[stackIdx].t_tMax;
            }
            else
            {
                if ( bothIntersection ) {
                    stack[stackIdx].t_node = &( BIHTree[currNode->children[far]] );
                    stack[stackIdx].t_tMin = t[far];
                    stack[stackIdx].t_tMax = tMax;
                    stackIdx++;
                    currNode = &(BIHTree[currNode->children[near]]);
                    tMax = t[near];
                }
                else {
                    currNode = nearIntersection ? &( BIHTree[currNode->children[near]] ) : &( BIHTree[currNode->children[far]] );
                    tMin = nearIntersection ? tMin : t[far];
                    tMax = nearIntersection ? t[near] : tMax;
                }
            }
        }




    } 
    return ( outRecord.triangleIdx >= 0 );
}

__device__ glm::vec3 Color( const Ray& r, 
                           TreeInternalNode* BIHTree,
                           int* firstIdxs,
                           uint32_t* duplicatesCnts,
                           Triangle* triangles,
                           uint32_t* triangleIdxs,
                           float3 sceneBBoxLo, 
                           float3 sceneBBoxHi,
                           int trisSize,
                           int bihSize) {
    HitRecord rec;
    rec.triangleIdx = -1;
    rec.t = FLT_MAX;
    //if( TraverseTriangles(bihSize, BIHTree,firstIdxs,duplicatesCnts,triangles,triangleIdxs,r,rec) ){
    if ( TraverseTree(r, BIHTree, firstIdxs, duplicatesCnts, triangles, triangleIdxs, sceneBBoxLo, sceneBBoxHi, rec) ) {   
        return glm::vec3( 255.0f, 255.0f, 0.0f );
    } else {
        return glm::vec3( 20.0f, 20.0f, 40.0f );
    }
}

__global__ void cudaRender(unsigned int* g_odata,
                           Camera* cam,
                           curandState* rand_state,
                           TreeInternalNode* BIHTree,
                           int* firstIdxs,
                           uint32_t* duplicatesCnts,
                           Triangle* triangles,
                           uint32_t* triangleIdxs,
                           float3 sceneBBoxLo,
                           float3 sceneBBoxHi,
                           int trisSize,
                           int bihSize ) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ( ( i >= SCREEN_WIDTH ) || ( j >= SCREEN_HEIGHT ) ) {
        return;
    }

    int pixel_index = j * SCREEN_WIDTH + i;
    curandState local_rand_state = rand_state[pixel_index];
    glm::vec3 col( 0, 0, 0 );
    for ( int s = 0; s < RAYS_PER_PIXEL; s++ ) {
        float u = float( i + curand_uniform( &local_rand_state ) ) / float( SCREEN_WIDTH );
        float v = float( j + curand_uniform( &local_rand_state ) ) / float( SCREEN_HEIGHT );
        Ray r = cam->GetRay( u, v );
        col += Color( r, BIHTree, firstIdxs, duplicatesCnts, triangles, triangleIdxs, sceneBBoxLo, sceneBBoxHi, trisSize, bihSize );
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float( RAYS_PER_PIXEL );

    g_odata[j * SCREEN_WIDTH + i] = rgbToInt( col.r, col.g, col.b );
}

void Renderer::Launch_cudaRender( unsigned int* g_odata, GPUArrayManager& gpuArrayManager ) {
    TreeInternalNode* d_BIHTree = thrust::raw_pointer_cast(gpuArrayManager.GetBIHTree().data());
    int* d_firstIdxs = thrust::raw_pointer_cast(gpuArrayManager.GetFirstIdxs().data());
    uint32_t* d_duplicatesCnt = thrust::raw_pointer_cast(gpuArrayManager.GetDuplicatesCnts().data());
    Triangle* d_triangles = thrust::raw_pointer_cast(gpuArrayManager.GetTriangles().data());
    uint32_t* d_triangleIdxs = thrust::raw_pointer_cast(gpuArrayManager.GetTrisIndexes().data());
    int d_trisSize = gpuArrayManager.GetTrisSize();
    int d_bihSize = gpuArrayManager.GetBIHTreeSize();
    float3 sceneBBoxLo = gpuArrayManager.GetBBoxArrays().sceneBBoxLo;
    float3 sceneBBoxHi = gpuArrayManager.GetBBoxArrays().sceneBBoxHi;
    cudaRender <<<m_blocks, m_threads, 0 >>> ( g_odata, 
                                               d_camera, 
                                               d_rand_state, 
                                               d_BIHTree,
                                               d_firstIdxs,
                                               d_duplicatesCnt,
                                               d_triangles,
                                               d_triangleIdxs,
                                               sceneBBoxLo, 
                                               sceneBBoxHi,
                                               d_trisSize,
                                               d_bihSize );
}


__global__ void InitRandGPU( int max_x, int max_y, curandState* randState ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ( ( i >= SCREEN_WIDTH ) || ( j >= SCREEN_HEIGHT ) ) {
        return;
    }
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init( 1984, pixel_index, 0, &randState[pixel_index] );
}

void Renderer::Launch_cudaRandInit( curandState* rand_state ) {
    InitRandGPU <<<m_blocks, m_threads >>> ( SCREEN_WIDTH, SCREEN_HEIGHT, rand_state );
}


__global__ void PrepareUniqueMortonCodes( uint32_t* uniqueMCs, uint32_t* mortonCodes, int listSize ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( idx >= listSize ) {
        return;
    }

    uint32_t mc = mortonCodes[idx];
    uint32_t newMC = mc;
    newMC = newMC << 32;
    // the highest value of idx must be numerical_limits<uint32_t>::max()
    newMC += idx;
    uniqueMCs[idx] = newMC;
   

}

void Renderer::Launch_MakeMortonCodesUnique( GPUArrayManager& gpuArrayManager ) {
    int threads = 256;
    int trisSize = gpuArrayManager.GetTrisSize();
    int blocks = ( trisSize / threads ) + 1;
    uint32_t* d_uniqueMCs = thrust::raw_pointer_cast(gpuArrayManager.GetUniqueMortonCodesVectorRef().data());
    uint32_t* d_mortonCodes = thrust::raw_pointer_cast(gpuArrayManager.GetMortonCodesVectorRef().data());

    PrepareUniqueMortonCodes<<<blocks, threads >>> ( d_uniqueMCs,
                                                     d_mortonCodes,
                                                     trisSize );
    
}


__global__ void FindClipPlanes( TreeInternalNode* BIHTree, 
                                float3* arrLo, 
                                float3* arrHi, 
                                uint32_t* trisIdxs, 
                                int* d_leafParents,
                                uint32_t* duplicatesCnt,
                                int* firstIdxs,
                                int UMCSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( idx >= UMCSize ) {
        return;
    }
    
    int firstIdx = firstIdxs[idx];
    uint32_t numOfDuplicates = duplicatesCnt[idx];
    glm::vec3 bboxLoGLM = {arrLo[trisIdxs[firstIdx]].x, arrLo[trisIdxs[firstIdx]].y, arrLo[trisIdxs[firstIdx]].z };
    glm::vec3 bboxHiGLM = {arrHi[trisIdxs[firstIdx]].x, arrHi[trisIdxs[firstIdx]].y, arrHi[trisIdxs[firstIdx]].z };
    uint32_t bboxIdx = -1;
    int trisIdx = -1;
    for ( int i = firstIdx; i < firstIdx + numOfDuplicates; i++ )
    {
        bboxIdx = trisIdxs[i];
        float3 bboxLo = arrLo[bboxIdx];
        float3 bboxHi = arrHi[bboxIdx];

        bboxLoGLM.x = min(bboxLoGLM.x, bboxLo.x);
        bboxLoGLM.y = min(bboxLoGLM.y, bboxLo.y);
        bboxLoGLM.z = min(bboxLoGLM.z, bboxLo.z);
        bboxHiGLM.x = max(bboxHiGLM.x, bboxHi.x);
        bboxHiGLM.y = max(bboxHiGLM.y, bboxHi.y);
        bboxHiGLM.z = max(bboxHiGLM.z, bboxHi.z);
    }


    int axis = -1;
    int prev = idx;
    int parent = d_leafParents[idx]; // temporary change just to silence compiler -> probably nonsense and should be changed

    while ( parent != -1 ) {
        TreeInternalNode& currNode = BIHTree[parent];
        axis = currNode.t_axis;

        if( currNode.children[0] == prev )
            atomicMaxFloat(&( currNode.t_clipPlanes[0] ), bboxHiGLM[axis]);
        if( currNode.children[1] == prev )
            atomicMinFloat(&( currNode.t_clipPlanes[1] ), bboxLoGLM[axis]);
        
        prev = parent;
        parent = currNode.parent;
    }

}

void Renderer::Launch_FindClipPlanes(GPUArrayManager& gpuArrayManager) {
    int threads = 256;
    int UMCSize = gpuArrayManager.GetUniqueMCSize();
    int blocks = ( UMCSize / threads ) + 1;

    uint32_t* trisIdxs = thrust::raw_pointer_cast(gpuArrayManager.GetTrisIndexes().data());
    float3* arrLo = thrust::raw_pointer_cast(gpuArrayManager.GetBBoxArrays().arrLo.data());
    float3* arrHi = thrust::raw_pointer_cast(gpuArrayManager.GetBBoxArrays().arrHi.data());
    TreeInternalNode* d_BIHTree = thrust::raw_pointer_cast(gpuArrayManager.GetBIHTree().data());
    int* d_leafParents = thrust::raw_pointer_cast(gpuArrayManager.GetLeafParentIdxs().data());
    int* d_firstIdxs = thrust::raw_pointer_cast(gpuArrayManager.GetFirstIdxs().data());
    uint32_t* d_duplicatesCnt = thrust::raw_pointer_cast(gpuArrayManager.GetDuplicatesCnts().data());

    FindClipPlanes <<<blocks, threads >>>( d_BIHTree, arrLo, arrHi, trisIdxs, d_leafParents, d_duplicatesCnt, d_firstIdxs, UMCSize );
}

__global__ void GetClips( uint32_t mask, uint32_t refMask, int axis, int split, int nodesCnt, bool isLeftNodes )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( !isLeftNodes )
    {
        split++;
        nodesCnt--;
    }

    int nodeIdx = 0;
    while ( idx <= nodesCnt )
    {
        if ( isLeftNodes )
        {
            nodeIdx = split - idx;
        }
        else
        {
            nodeIdx = split + idx;
        }
        idx++;
    }
}

__global__ void BuildTree( uint32_t* uniqueMCs, TreeInternalNode* BIHTree, int* leafParents, int UMCSize)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx > UMCSize - 2 ) {
        return;
    }
   
    
    uint32_t currentNodeMortonCode = uniqueMCs[idx];
    uint32_t neighboursCommonPrefixes[2] = { -1, -1 }; // 0--->left neighbour ; 1--->right neighbour 
    
    // left neighbour
    if( idx )
    {
        uint32_t leftNeighbourMortonCode = uniqueMCs[idx - 1];
        neighboursCommonPrefixes[0] = __clz(currentNodeMortonCode ^ leftNeighbourMortonCode);
    }
   
    // right neighbour
    if ( idx < UMCSize - 1 )
    {
        uint32_t rightNeighbourMortonCode = uniqueMCs[idx + 1];
        neighboursCommonPrefixes[1] = __clz(currentNodeMortonCode ^ rightNeighbourMortonCode);
    }
    
    int expansionDirection = signum(neighboursCommonPrefixes[1] - neighboursCommonPrefixes[0]);  // "d" in Karras paper
    
    // find range which current node covers
    // first expand range that can overflow
    int lcpMin = neighboursCommonPrefixes[ 1 - ((expansionDirection+1)/2) ]; // lcp => Longest Common Prefix
    int lMax = 1;
    int lcpTmp = -2; // invalid value
    int lIdx = -1;
    do
    {
        lMax *= 2;
        lIdx = idx + (lMax * expansionDirection);
        if ( lIdx < 0 || lIdx > UMCSize - 1 )
            lcpTmp = -1;
        else
            lcpTmp = __clz(currentNodeMortonCode ^ uniqueMCs[lIdx]);
   
    } while ( lcpTmp > lcpMin );
    
    // then find last element with binary search
    int l = 0;
    int tmpIdx = -1;
    for ( int t = lMax / 2; t >= 1; t /= 2 )
    {
        tmpIdx = idx + ( ( l + t ) * expansionDirection );
        if ( tmpIdx < 0 || tmpIdx > UMCSize - 1 )
            lcpTmp = -1;
        else
            lcpTmp = __clz(currentNodeMortonCode ^ uniqueMCs[tmpIdx]);
   
        if ( lcpTmp > lcpMin )
        {
            l = l + t;
        }
    }
    int otherEnd = idx + ( l * expansionDirection );
    int lcpOfEndsOfInterval = __clz(currentNodeMortonCode ^ uniqueMCs[otherEnd]);
    
    // find split position using binary tree
    int s = 0;
    bool breakLoop = false;
    int t = l;
    while ( !breakLoop )
    {
        t = __float2int_ru(t / 2.0f);
        
        tmpIdx = idx + ( ( s + t ) * expansionDirection );
        if ( tmpIdx < 0 || tmpIdx > UMCSize - 1 )
            lcpTmp = -1;
        else
            lcpTmp = __clz(currentNodeMortonCode ^ uniqueMCs[tmpIdx]);
        
        if ( lcpTmp > lcpOfEndsOfInterval )
        {
            s = s + t;
        }
   
        if ( t == 1 )
            breakLoop = true;
    }
    
    int split = idx + ( s * expansionDirection ) + min(expansionDirection, 0);
    TreeInternalNode& currentInternalNode = BIHTree[idx];
    currentInternalNode.ID = idx;
    currentInternalNode.children[0] = split;
    currentInternalNode.children[1] = split + 1;
    
    currentInternalNode.isLeaf[0] = min(idx, otherEnd) == split;
    currentInternalNode.isLeaf[1] = max(idx, otherEnd) == (split + 1);
    
    if ( currentInternalNode.isLeaf[0] ) {
        leafParents[split] = idx; // just for silence compiler. Now i have no clue what should be there
    } 
    else
    {
        BIHTree[split].parent = idx;
    }

    if ( currentInternalNode.isLeaf[1] ) {
        leafParents[split + 1] = idx; // just for silence compiler. Now i have no clue what should be there
    }
    else
    {
        BIHTree[split + 1].parent = idx;
    }
    //split axis
    uint32_t leftChildMC = uniqueMCs[split];
    uint32_t rightChildMC = uniqueMCs[split + 1];
    const int lcpOfChildren = __clz(leftChildMC ^ rightChildMC);
    int axis = (lcpOfChildren + 1) % 3;
    currentInternalNode.t_axis = axis;

    // JUST FOR NOTE: __clz(code1 ^ code2) => najde common prefix - 
    // strecha je xor, takze u stejnych bitu vrati nulu a funkce __clz pocita po sobe jdouci nuly od nejvyznamejsiho bitu
}

void Renderer::Launch_BuildTree(GPUArrayManager& gpuArrayManager)
{
    int UMCSize = gpuArrayManager.GetUniqueMCSize();
    int threads = THREADS_X * THREADS_Y;
    int blocks = ( UMCSize / threads ) + 1;

    uint32_t* d_uniqueMCs = thrust::raw_pointer_cast(gpuArrayManager.GetUniqueMortonCodesVectorRef().data());
    TreeInternalNode* d_BIHTree = thrust::raw_pointer_cast(gpuArrayManager.GetBIHTree().data());
    int* d_leafParents = thrust::raw_pointer_cast(gpuArrayManager.GetLeafParentIdxs().data());

    BuildTree<<< blocks, threads >>>( d_uniqueMCs, d_BIHTree, d_leafParents, UMCSize );
}

/*__global__ void PrepareLeafNodes(HitableList& world, Tree& tree)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= world.m_listSize ) {
        return;
    }

    uint32_t mortonCodeOfTriangle = tree.d_mortonCodes[idx];
    uint32_t slot = hash(mortonCodeOfTriangle);
    uint32_t idxOfLeafNode = kEmpty;
    Triangle& triangle = world[idx];
    
    // search idx of Leaf Node in hash table
    while ( true )
    {
        uint32_t prev = tree.d_hashTable[slot].key;
        if ( prev == mortonCodeOfTriangle )
        {
            //idxOfLeafNode = tree.d_hashTable[slot].value;
            break;
        }

        slot = ( slot + 1 ) & ( kHashTableCapacity - 1 );
    }
    //-----------------------------------------------------

    auto& leafNode = tree.d_leafNodes[idxOfLeafNode];
    atomicMin(&(leafNode.idxL), idx);
    atomicMax(&(leafNode.idxH), idx);
}

void Renderer::Launch_PrepareLeafNodes( HitableList& world, Tree& tree ) 
{
    int threads = THREADS_X * THREADS_Y;
    int blocks = ( world.m_listSize / threads ) + 1;
    PrepareLeafNodes<<< blocks, threads >>>( world, tree );
}

__global__ void InitLeafNodes(Tree& tree, uint32_t min, uint32_t max)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= tree.t_leafNodesLen ) {
        return;
    }
    tree.d_leafNodes[idx].idxH = min;
    tree.d_leafNodes[idx].idxL = max;

}

void Renderer::Launch_InitLeafNodes(Tree& tree, uint32_t min, uint32_t max)
{
    int threads = THREADS_X * THREADS_Y;
    int blocks = ( tree.t_leafNodesLen / threads ) + 1;
    InitLeafNodes<<< blocks, threads>>>( tree, min, max );
}
*/