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
#include <chrono>
#include <bitset>
#include <vector>
#include <deque>
#include <stack>



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

			if ( !curr.isLeftLeaf )
				q.push_back(BIHTree[curr.children[0]]);
			if ( !curr.isRightLeaf )
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
	d_camera = new Camera( glm::vec3( 0.0, 0.0, -5.0 ), (float)SCREEN_WIDTH / SCREEN_HEIGHT );

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

__host__ void Renderer::Render(GPUArrayManager& gpuArrayManager) {
	//using std::chrono::high_resolution_clock;
	//using std::chrono::duration_cast;
	//using std::chrono::duration;
	//using std::chrono::microseconds;
	int trisSize = gpuArrayManager.GetTrisSize();
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


	thrust::host_vector<uint32_t> h_trisIdxs = gpuArrayManager.GetTrisIndexes();
	thrust::host_vector<float3> h_centers = gpuArrayManager.GetBBoxArrays().arrCenter;
	thrust::host_vector<uint32_t> h_UMCs = gpuArrayManager.GetUniqueMortonCodesVectorRef();
	thrust::host_vector<TreeInternalNode> h_BIH = gpuArrayManager.GetBIHTree();
	thrust::host_vector<uint32_t> h_duplCnts = gpuArrayManager.GetDuplicatesCnts();
	thrust::host_vector<int> h_firstIdxs = gpuArrayManager.GetFirstIdxs();
	thrust::host_vector<uint32_t> h_MCs = gpuArrayManager.GetMortonCodesVectorRef();

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
	

	/*-----PRINT BIH TREE -----*/
	//for ( int i = 0; i < gpuArrayManager.GetBIHTreeSize(); i++ )
	//{
	//	auto currNode = h_BIH[i];
	//	std::cout << "NODE " << i << std::endl;
	//	std::cout << "parent: " << currNode.parent << std::endl;
	//	std::cout << "leftChild: " << currNode.children[0] << std::endl;
	//	std::cout << "isLeftLeaf: " << (currNode.isLeftLeaf ? "TRUE" : "FALSE") << std::endl;
	//	std::cout << "rightChild: " << currNode.children[1] << std::endl;
	//	std::cout << "isRightLeaf: " << ( currNode.isRightLeaf ? "TRUE" : "FALSE" ) << std::endl;
	//	std::cout << "clipPlaneLEFT: " << currNode.t_clipPlanes[0] << std::endl;
	//	std::cout << "clipPlaneRIGHT: " << currNode.t_clipPlanes[1] << std::endl;
	//	std::cout << "axis: " << currNode.t_axis << std::endl;
	//	std::cout << std::endl;
	//}

	Launch_cudaRender( m_cudaDestResource, gpuArrayManager );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
	
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
	checkCudaErrors( cudaMalloc( (void**)&m_cudaDestResource, size_tex_data ) );
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


