#include "Renderer.h"
#include <stdio.h>
#include "Constants.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "thrust/reduce.h"
#include "thrust/unique.h"
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include "thrust/device_ptr.h"
#include "thrust/iterator/constant_iterator.h"
#include "thrust/iterator/counting_iterator.h"
#include <algorithm>
#include <chrono>
#include <bitset>
#include <vector>
#include <deque>
#include <stack>
#include <iostream>
#include <fstream>
#include <random>

float clip(float n, float lower, float upper) {
	return std::max(lower, std::min(n, upper));
}

int CPUrgbToInt(float r, float g, float b) {
	r = clip(r, 0.0f, 255.0f);
	g = clip(g, 0.0f, 255.0f);
	b = clip(b, 0.0f, 255.0f);
	return ( int(b) << 16 ) | ( int(g) << 8 ) | int(r);
}


void BFS(TreeInternalNode* BIHTree ) {
	
	std::deque<TreeInternalNode> q;
	q.push_back(BIHTree[0]);

	int depth = 0;
	int popped = 0;
	while ( !q.empty() ) {
		int size = q.size();

		for ( int i = 0; i < size; i++ ) {
			auto curr = q.front();
			q.pop_front();
			popped++;

			if ( !curr.isLeaf[0] )
				q.push_back(BIHTree[curr.children[0]]);
			if ( !curr.isLeaf[1] )
				q.push_back(BIHTree[curr.children[1]]);
		}
		depth++;
	}

	std::cout << "Depth: " << depth << std::endl;
	std::cout << "popped: " << popped << std::endl;
}


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda( cudaError_t result, char const* const func, const char* const file, int const line ) {
	if ( result ) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>( result ) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit( 99 );
	}
}

__host__ Renderer::Renderer() :	d_rand_state(nullptr), 
								d_camera(nullptr), 
								m_quadTexture(0), 
								m_shaderProgram(0), 
								m_cudaTexResultRes( nullptr ), 
								m_cudaDestResource(nullptr),
								m_quadVAO(0), 
								m_quadVBO(0),
								m_quadEBO(0)
{
}

__host__ void Renderer::Init() {
	// create texture that will receive the result of CUDA
	CreateTextureDst();
	
	// load shader programs
	m_shaderProgram = CompileGLSLprogram();

	// calculate grid size
	m_threads = { THREADS_X, THREADS_Y, 1 };
	m_blocks = { SCREEN_WIDTH / THREADS_X + 1, SCREEN_HEIGHT / THREADS_Y + 1, 1 };

	// init Camera
	d_camera = new Camera( glm::vec3( 2.0, 0.0, -2.0 ), (float)SCREEN_WIDTH / SCREEN_HEIGHT );

	//cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 100'000);
	CreateCUDABuffers();
	InitQuad();
	InitRand();
}

struct KeyValueCmp {
	__host__ __device__
		bool operator()(const KeyValue& o1, const KeyValue& o2) {
		return o1.key < o2.key;
	}
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v)
{
	v = ( v * 0x00010001u ) & 0xFF0000FFu;
	v = ( v * 0x00000101u ) & 0x0F00F00Fu;
	v = ( v * 0x00000011u ) & 0xC30C30C3u;
	v = ( v * 0x00000005u ) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z)
{
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

struct morton_functor {

	__device__ uint32_t operator()(float3 center)
	{
		return morton3D(center.x, center.y, center.z);
	}

};

bool CPURayTriangleIntersection(const Triangle& triangle, const Ray& ray, float& outT) {
	glm::vec3 v0v1 = triangle.v1 - triangle.v0;
	glm::vec3 v0v2 = triangle.v2 - triangle.v0;

	//printf("ray origin: [%.2f,%.2f,%.2f]\n", ray.Origin().x, ray.Origin().y, ray.Origin().z);
	//printf("ray dir: [%.2f,%.2f,%.2f]\n", ray.Direction().x, ray.Direction().y, ray.Direction().z);

	glm::vec3 pvec = glm::cross(ray.Direction(), v0v2);

	float det = glm::dot(v0v1, pvec);

	if ( det < 0.000001 )
		return false;

	float invDet = 1.0 / det;

	glm::vec3 tvec = ray.Origin() - triangle.v0;

	float u = dot(tvec, pvec) * invDet;

	if ( u < 0 || u > 1 )
		return false;

	glm::vec3 qvec = glm::cross(tvec, v0v1);

	float v = glm::dot(ray.Direction(), qvec) * invDet;

	if ( v < 0 || u + v > 1 )
		return false;

	outT = glm::dot(v0v2, qvec) * invDet;

	return true;
}

void CPUFindNearestTriangle(int* firstIdxs,
						 uint32_t* duplicatesCnts,
						 Triangle* triangles,
						 uint32_t* triangleIdxs,
						 const Ray& r,
						 int childIdx,
						 HitRecord& outRecord)
{
	float t = FLT_MAX;
	for ( int i = firstIdxs[childIdx]; i < firstIdxs[childIdx] + duplicatesCnts[childIdx]; i++ ) {
		uint32_t triangleIdx = triangleIdxs[i];
		if ( CPURayTriangleIntersection(triangles[triangleIdx], r, t) ) {
			if ( t > 0 && t < outRecord.t ) {
				outRecord.t = t;
				outRecord.triangleIdx = i;
			}
		}
	}
}

bool CPUTraverseTree(const Ray& r,
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

	//std::cout << "tmin: " << tMin << std::endl;
	//std::cout << "tmax: " << tMax << std::endl;
	//std::cout << "==========================" << std::endl;
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
	while ( currNode != nullptr ) {

		if ( currNode->ID == 9 ) {
			currNode->traversed = true;
		}

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


		//std::cout << "clipPlane0: " << currNode->t_clipPlanes[0] << std::endl;
		//std::cout << "clipPlane1: " << currNode->t_clipPlanes[1] << std::endl;
		//std::cout << "t0: " << t[0] << std::endl;
		//std::cout << "t1: " << t[1] << std::endl;
		//std::cout << "noIntersection: " << (noIntersection ? "TRUE" : "FALSE") << std::endl;
		//std::cout << "nearIntersection: " << (nearIntersection ? "TRUE" : "FALSE") << std::endl;
		//std::cout << "farIntersection: " << (farIntersection ? "TRUE" : "FALSE") << std::endl;
		//std::cout << "bothIntersection: " << (bothIntersection ? "TRUE" : "FALSE") << std::endl;
		//std::cout << "----------------" << std::endl;


		if ( currNode->isLeaf[near] || currNode->isLeaf[far] ) {
			if ( currNode->isLeaf[near] && !currNode->isLeaf[far] )
			{
				CPUFindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[near], outRecord);
				currNode = &( BIHTree[currNode->children[far]] );
			}
			else if ( currNode->isLeaf[far] && !currNode->isLeaf[near] )
			{
				CPUFindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[far], outRecord);
				currNode = &( BIHTree[currNode->children[near]] );
			}
			else {
				CPUFindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[near], outRecord);
				CPUFindNearestTriangle(firstIdxs, duplicatesCnts, triangles, triangleIdxs, r, currNode->children[far], outRecord);
				stackIdx--;
				currNode = stack[stackIdx].t_node;
				tMin = stack[stackIdx].t_tMin;
				tMax = stack[stackIdx].t_tMax;

			}
			// break is temporary. Full algorithm -> compute T and pop from the stack
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
					stack[stackIdx].t_tMin = tMin;
					stack[stackIdx].t_tMax = tMax;
					stackIdx++;
					currNode = &( BIHTree[currNode->children[near]] );
				}
				else {
					currNode = nearIntersection ? &( BIHTree[currNode->children[near]] ) : &( BIHTree[currNode->children[far]] );
					tMin = nearIntersection ? tMin : t[far];
					tMax = nearIntersection ? t[near] : tMax;
				}
			}
		}




	}
	//printf("NO: %d\nONE: %d\nBOTH: %d\n-----\n", noCnt, oneCnt, bothCnt);
	if ( outRecord.triangleIdx == 0 ) printf("Triangle %d HIT.\n", outRecord.triangleIdx);
	return ( outRecord.triangleIdx >= 0 );
}


glm::vec3 CPUColor(const Ray& r,
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
	if ( CPUTraverseTree(r, BIHTree, firstIdxs, duplicatesCnts, triangles, triangleIdxs, sceneBBoxLo, sceneBBoxHi, rec) ) {
		return glm::vec3(255.0f, 255.0f, 0.0f);
	}
	else {
		return glm::vec3(20.0f, 20.0f, 40.0f);
	}
}

void DebugRender(GPUArrayManager& gpuArrayManager, Camera* cam, unsigned int* g_odata ) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	thrust::host_vector<TreeInternalNode> h_BIH = gpuArrayManager.GetBIHTree();
	thrust::host_vector<int> h_firstIdxs = gpuArrayManager.GetFirstIdxs();
	thrust::host_vector<uint32_t> h_duplCnts = gpuArrayManager.GetDuplicatesCnts();
	thrust::host_vector<Triangle> h_triangles = gpuArrayManager.GetTriangles();
	thrust::host_vector<uint32_t> h_trisIdxs = gpuArrayManager.GetTrisIndexes();
	float3 sceneBBoxLo = gpuArrayManager.GetBBoxArrays().sceneBBoxLo;
	float3 sceneBBoxHi = gpuArrayManager.GetBBoxArrays().sceneBBoxHi;
	int bihSize = gpuArrayManager.GetBIHTreeSize();
	int trisSize = gpuArrayManager.GetTrisSize();


//	std::cout << "before pixel iterating" << std::endl;
	for ( int i = 0; i < SCREEN_WIDTH; i++ ) {
		for ( int j = 0; j < SCREEN_HEIGHT; j++ ) {
			int pixel_index = j * SCREEN_WIDTH + i;
			glm::vec3 col(0, 0, 0);
			for ( int s = 0; s < RAYS_PER_PIXEL; s++ ) {
				float u = float(i + dis(gen)) / float(SCREEN_WIDTH);
				float v = float(j + dis(gen)) / float(SCREEN_HEIGHT);
				//std::cout << "==========================" << std::endl;
				Ray r = cam->GetRay(u, v);
				//std::cout << "Ray:" << pixel_index << "-" << s << std::endl;
				col += CPUColor(r, h_BIH.data(), h_firstIdxs.data(), h_duplCnts.data(), h_triangles.data(), h_trisIdxs.data(), sceneBBoxLo, sceneBBoxHi, trisSize, bihSize);
			}
			col /= float(RAYS_PER_PIXEL);

	//		std::cout << "Before godata" << std::endl;
			g_odata[j * SCREEN_WIDTH + i] = CPUrgbToInt(col.r, col.g, col.b); // CUDAMALLOCMANAGED - ZMENIT ZPATKY NA CUDAMALLOC
	//		std::cout << "After godata" << std::endl;
		}
	}

	gpuArrayManager.GetBIHTree() = h_BIH;
}

bool clipPlanesFound = false;
__host__ void Renderer::Render(GPUArrayManager& gpuArrayManager) {
	//using std::chrono::high_resolution_clock;
	//using std::chrono::duration_cast;
	//using std::chrono::duration;
	//using std::chrono::microseconds;
	//int trisSize = gpuArrayManager.GetTrisSize();
	//auto t1 = high_resolution_clock::now();
	thrust::transform(thrust::device,
					  gpuArrayManager.GetBBoxArrays().arrCenterNorm.begin(),
					  gpuArrayManager.GetBBoxArrays().arrCenterNorm.begin() + gpuArrayManager.GetTrisSize(),
					  gpuArrayManager.GetMortonCodesVectorRef().begin(),
					  morton_functor());
	
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
	//auto t2 = high_resolution_clock::now();

	/* Getting number of milliseconds as a double. */
	//duration<double, std::micro> us_double = t2 - t1;
	//std::cout << us_double.count() << "us\n";

	gpuArrayManager.ResetTrisIdxs();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::stable_sort_by_key(thrust::device,
						gpuArrayManager.GetMortonCodesVectorRef().begin(),
						gpuArrayManager.GetMortonCodesVectorRef().begin() + gpuArrayManager.GetTrisSize(),
						gpuArrayManager.GetTrisIndexes().begin()
	);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	auto newEnd = thrust::reduce_by_key(thrust::device,
						  gpuArrayManager.GetMortonCodesVectorRef().data(),
						  gpuArrayManager.GetMortonCodesVectorRef().data() + gpuArrayManager.GetTrisSize(),
						  thrust::make_constant_iterator(1),
						  gpuArrayManager.GetUniqueMortonCodesVectorRef().data(),
						  gpuArrayManager.GetDuplicatesCnts().data()
	);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	gpuArrayManager.SetUniqueMCSize(newEnd.first - gpuArrayManager.GetUniqueMortonCodesVectorRef().data());
	
	//if ( !gpuArrayManager.GetTraversalStacksSize() ) {
	//	if ( !gpuArrayManager.AllocateTraversalStacks(SCREEN_WIDTH * SCREEN_HEIGHT * ( std::log2(gpuArrayManager.GetUniqueMCSize()) + 1 )) );
	//	std::cout << "Failed to allocate Traversal stacks on GPU." << std::endl;
	//}

	thrust::unique_by_key_copy(thrust::device,
							   gpuArrayManager.GetMortonCodesVectorRef().data(),
							   gpuArrayManager.GetMortonCodesVectorRef().data() + gpuArrayManager.GetTrisSize(),
							   thrust::make_counting_iterator(0),
							   gpuArrayManager.GetUniqueMortonCodesVectorRef().data(),
							   gpuArrayManager.GetFirstIdxs().data()
	);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	
	//
	//for ( int i = 181; i < 188; i++ ) {
	//	std::cout << "MC: " << h_MCs[i] << std::endl;
	//}
	//std::cout << "-------" << std::endl;

	//for ( int i = 0; i < gpuArrayManager.GetUniqueMCSize(); i++ ) {
	//	if ( h_duplCnts[i] > 1 )
	//	{
	//		std::cout << "cnts: " << h_duplCnts[i] << std::endl;
	//		std::cout << "firstIdx: " << h_firstIdxs[i] << std::endl;
	//		std::cout << "mortoncode: " << h_UMCs[i] << std::endl;
	//		std::cout << "----------" << std::endl;
	//	}
	//}

	//PrintMCs(gpuArrayManager);

	Launch_BuildTree( gpuArrayManager );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
	
	

	Launch_FindClipPlanes(gpuArrayManager);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	//thrust::host_vector<uint32_t> h_trisIdxs = gpuArrayManager.GetTrisIndexes();
	//thrust::host_vector<float3> h_centers = gpuArrayManager.GetBBoxArrays().arrCenter;
	//thrust::host_vector<uint32_t> h_UMCs = gpuArrayManager.GetUniqueMortonCodesVectorRef();
	//
	//thrust::host_vector<uint32_t> h_duplCnts = gpuArrayManager.GetDuplicatesCnts();
	//thrust::host_vector<int> h_firstIdxs = gpuArrayManager.GetFirstIdxs();
	//thrust::host_vector<uint32_t> h_MCs = gpuArrayManager.GetMortonCodesVectorRef();
	//auto h_leafParents = gpuArrayManager.GetLeafParentIdxs();
	//int BIHTreeSize = gpuArrayManager.GetBIHTreeSize();
	//thrust::host_vector<float3> arrLo = gpuArrayManager.GetBBoxArrays().arrLo;
	//thrust::host_vector<float3> arrHi = gpuArrayManager.GetBBoxArrays().arrHi;

	// print UMCs
	//for ( int i = 0; i < BIHTreeSize + 1; i++ ) {
	//	std::cout << h_UMCs[i] << std::endl;
	//}

	//for ( int i = 0; i < BIHTreeSize + 1; i++ ) {
	//	int trisIdx = h_trisIdxs[i];
	//	std::cout << "Tris: " << trisIdx << std::endl;
	//	std::cout << "BBoxLo: [" << arrLo[trisIdx].x << ", " << arrLo[trisIdx].y << ", " << arrLo[trisIdx].z << "]" << std::endl;
	//	std::cout << "BBoxHi: [" << arrHi[trisIdx].x << ", " << arrHi[trisIdx].y << ", " << arrHi[trisIdx].z << "]" << std::endl;
	//	std::cout << "------------" << std::endl;
	//}
	//
	//std::cout << "==================" << std::endl;

	//int UMCsCnt = 0;
	//int BIHCnt = 0;
	//
	//for ( auto& el : h_BIH ) {
	//	if ( el.children[0] == -1 && el.children[1] == -1 && el.parent == -1 ) {
	//		break;
	//	}
	//	else {
	//		BIHCnt++;
	//	}
	//}
	//
	//for ( auto& el : h_UMCs ) {
	//	if ( el == -1 ) {
	//		break;
	//	}
	//	else {
	//		UMCsCnt++;
	//	}
	//}

	//std::cout << "BIH size: " << BIHCnt << std::endl;
	//std::cout << "Unique MCs: " << UMCsCnt << std::endl;
	//BFS(h_BIH.data());

	//for ( int i = 0; i < trisSize; i++ ) {
	//	int idx = h_trisIdxs[i];
	//	
	//	std::cout << "[" << h_centers[idx].x << "; " << h_centers[idx].y << "; " << h_centers[idx].z << "]" << std::endl;
	//	std::cout << std::bitset<64>(h_MCs[i]) << std::endl;
	//	std::cout << "----------------------------" << std::endl;
	//}
	
	//for ( int i = 0; i < BIHTreeSize + 1; i++ ) {
	//		
	//	int firstIdx = h_firstIdxs[i];
	//	uint32_t numOfDuplicates = h_duplCnts[i];
	//	glm::vec3 bboxLoGLM = { arrLo[h_trisIdxs[firstIdx]].x, arrLo[h_trisIdxs[firstIdx]].y, arrLo[h_trisIdxs[firstIdx]].z };
	//	glm::vec3 bboxHiGLM = { arrHi[h_trisIdxs[firstIdx]].x, arrHi[h_trisIdxs[firstIdx]].y, arrHi[h_trisIdxs[firstIdx]].z };
	//	uint32_t bboxIdx = -1;
	//	int trisIdx = -1;
	//	for ( int j = firstIdx; j < firstIdx + numOfDuplicates; j++ )
	//	{
	//		bboxIdx = h_trisIdxs[j];
	//		float3 bboxLo = arrLo[bboxIdx];
	//		float3 bboxHi = arrHi[bboxIdx];
	//
	//		bboxLoGLM.x = std::min(bboxLoGLM.x, bboxLo.x);
	//		bboxLoGLM.y = std::min(bboxLoGLM.y, bboxLo.y);
	//		bboxLoGLM.z = std::min(bboxLoGLM.z, bboxLo.z);
	//		bboxHiGLM.x = std::max(bboxHiGLM.x, bboxHi.x);
	//		bboxHiGLM.y = std::max(bboxHiGLM.y, bboxHi.y);
	//		bboxHiGLM.z = std::max(bboxHiGLM.z, bboxHi.z);
	//	}
	//
	//
	//	int axis = -1;
	//	int prev = i;
	//	int parent = h_leafParents[i]; // temporary change just to silence compiler -> probably nonsense and should be changed
	//
	//	while ( parent != -1 ) {
	//		TreeInternalNode& currNode = h_BIH[parent];
	//		axis = currNode.t_axis;
	//
	//		if ( currNode.children[0] == prev )
	//			currNode.t_clipPlanes[0] = std::max( currNode.t_clipPlanes[0] , bboxHiGLM[axis]);
	//		if ( currNode.children[1] == prev )
	//			currNode.t_clipPlanes[1] = std::min( currNode.t_clipPlanes[1] , bboxLoGLM[axis]);
	//
	//		prev = parent;
	//		parent = currNode.parent;
	//	}
	//}
	//
	//gpuArrayManager.GetBIHTree() = h_BIH;

	
	
	
	
	
	//DebugRender(gpuArrayManager,d_camera, m_cudaDestResource);
	//thrust::host_vector<TreeInternalNode> h_BIH = gpuArrayManager.GetBIHTree();

	/*-----PRINT BIH TREE -----*/
	//for ( int i = 0; i < gpuArrayManager.GetBIHTreeSize(); i++ )
	//{
	//	auto currNode = h_BIH[i];
	//	if ( !currNode.traversed ) {
	//		std::cout << "Node ID: " << currNode.ID << std::endl;
	//	}
	//	
	//	std::cout << "NODE " << i << std::endl;
	//	std::cout << "parent: " << currNode.parent << std::endl;
	//	std::cout << "leftChild:  " << currNode.children[0] << std::endl;
	//	std::cout << "rightChild: " << currNode.children[1] << std::endl;
	//	std::cout << "axis: " << currNode.t_axis << std::endl;
	//	std::cout << "isLeftLeaf: " << ( currNode.isLeaf[0] ? "TRUE" : "FALSE" ) << std::endl;
	//	std::cout << "isRightLeaf: " << ( currNode.isLeaf[1] ? "TRUE" : "FALSE" ) << std::endl;
	//	std::cout << "clipPlaneLEFT: " << currNode.t_clipPlanes[0] << std::endl;
	//	std::cout << "clipPlaneRIGHT: " << currNode.t_clipPlanes[1] << std::endl;
	//	std::cout << std::endl;
	//	
	//}
	//std::cout << "===================" << std::endl;
	Launch_cudaRender( m_cudaDestResource, gpuArrayManager );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );

	//gpuArrayManager.ResetClipPlanes();
	
	cudaArray* texture_ptr;
	checkCudaErrors( cudaGraphicsMapResources( 1, &m_cudaTexResultRes, 0 ) );
	checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(
		&texture_ptr, m_cudaTexResultRes, 0, 0 ) );
	
	unsigned int num_texels = SCREEN_WIDTH * SCREEN_HEIGHT;
	unsigned int num_values = num_texels * 4;
	unsigned int size_tex_data = sizeof( GLubyte ) * num_values;
	checkCudaErrors( cudaMemcpy2DToArray( texture_ptr, 0, 0, m_cudaDestResource, 
										  size_tex_data/SCREEN_HEIGHT, size_tex_data/SCREEN_HEIGHT, SCREEN_HEIGHT, 
										  cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_cudaTexResultRes, 0 ) );
	
	glClearColor( 0.2f, 0.3f, 0.3f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT );
	
	// bind Texture
	glBindTexture( GL_TEXTURE_2D, m_quadTexture );
	glEnable( GL_TEXTURE_2D );
	glDisable( GL_DEPTH_TEST );
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
	
	// render container
	glUseProgram( m_shaderProgram );
	glBindVertexArray( m_quadVAO );
	glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
	checkCudaErrors(cudaDeviceSynchronize());
	//cudaMemset(tree.d_hashTable, 0xff, sizeof(KeyValue) * kHashTableCapacity);
}


__host__ GLuint Renderer::CompileGLSLprogram() {
	GLuint v, f, p = 0;

	p = glCreateProgram();

	if ( m_texVertexShader ) {
		v = glCreateShader( GL_VERTEX_SHADER );
		glShaderSource( v, 1, &m_texVertexShader, NULL );
		glCompileShader( v );

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv( v, GL_COMPILE_STATUS, &compiled );

		if ( !compiled ) {
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog( v, 256, NULL, temp );
			printf( "Vtx Compile failed:\n%s\n", temp );
			//#endif
			glDeleteShader( v );
			return 0;
		} else {
			glAttachShader( p, v );
		}
	}

	if ( m_texFragmentShader ) {
		f = glCreateShader( GL_FRAGMENT_SHADER );
		glShaderSource( f, 1, &m_texFragmentShader, NULL );
		glCompileShader( f );

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv( f, GL_COMPILE_STATUS, &compiled );

		if ( !compiled ) {
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog( f, 256, NULL, temp );
			printf( "frag Compile failed:\n%s\n", temp );
			//#endif
			glDeleteShader( f );
			return 0;
		} else {
			glAttachShader( p, f );
		}
	}

	glLinkProgram( p );

	int infologLength = 0;
	int charsWritten = 0;

	glGetProgramiv( p, GL_INFO_LOG_LENGTH, (GLint*)&infologLength );

	if ( infologLength > 0 ) {
		char* infoLog = new char[ infologLength ];
		glGetProgramInfoLog( p, infologLength, (GLsizei*)&charsWritten, infoLog );
		printf( "Shader compilation error: %s\n", infoLog );
		delete [] infoLog;
	}

	return p;
}

__host__ void Renderer::CreateTextureDst() {
	// create a texture
	glGenTextures( 1, &m_quadTexture );
	glBindTexture( GL_TEXTURE_2D, m_quadTexture );

	// set basic parameters
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, SCREEN_WIDTH, SCREEN_HEIGHT, 0,
				  GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL );

	
	//// register this texture with CUDA
	checkCudaErrors( cudaGraphicsGLRegisterImage(
		&m_cudaTexResultRes, m_quadTexture, GL_TEXTURE_2D,
		cudaGraphicsMapFlagsWriteDiscard ) );
}

__host__ void Renderer::CreateCUDABuffers() {
	// set up vertex data parameter
	unsigned int num_texels = SCREEN_WIDTH * SCREEN_HEIGHT;
	unsigned int num_values = num_texels * 4;
	unsigned int size_tex_data = sizeof( GLubyte ) * num_values;
	checkCudaErrors( cudaMallocManaged( (void**)&m_cudaDestResource, size_tex_data ) );
}

__host__ void Renderer::InitQuad() {
	glGenVertexArrays( 1, &m_quadVAO );
	glGenBuffers( 1, &m_quadVBO );
	glGenBuffers( 1, &m_quadEBO );

	glBindVertexArray( m_quadVAO );

	glBindBuffer( GL_ARRAY_BUFFER, m_quadVBO );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );

	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_quadEBO );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( indices ), indices, GL_STATIC_DRAW );

	// position attribute
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (void*)0 );
	glEnableVertexAttribArray( 0 );
	// texture coord attribute
	glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (void*)( 3 * sizeof( float ) ) );
	glEnableVertexAttribArray( 1 );
}

__host__ void Renderer::InitRand() {
	unsigned int num_texels = SCREEN_WIDTH * SCREEN_HEIGHT;
	checkCudaErrors( cudaMalloc( (void**)&d_rand_state, num_texels * sizeof( curandState ) ) );
	Launch_cudaRandInit(d_rand_state);
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
}


