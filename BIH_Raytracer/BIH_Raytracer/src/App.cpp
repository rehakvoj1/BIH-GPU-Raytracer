#include "App.h"
#include "Model.h"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/device_reference.h>
#include <thrust/sequence.h>
#include "linearprobing.h"
#include <cmath>


App::App() {

}

App::~App() {
//	delete m_tree;
	glfwTerminate();
}


GLFWwindow* App::NewWindow( int width, int height, std::string title ) {
	
	std::cout << "Initializing GLFW..." << std::endl;
	if ( !glfwInit() ) {
		std::cout << "glfw init failed!" << std::endl;
	}
	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

	std::cout << "Creating window..." << std::endl;
	GLFWwindow* window = glfwCreateWindow( width, height, title.c_str(), NULL, NULL );

	if ( !window ) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return nullptr;
	}

	m_window = std::make_unique<Window>( window, width, height, title );
	
	std::cout << "Initializing GLAD..." << std::endl;
	if ( !gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress ) ) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		glfwTerminate();
		return nullptr;
	}

	std::cout << "Initializing renderer..." << std::endl;
	m_renderer = std::make_unique<Renderer>();
	m_renderer->Init();

	std::cout << "Initializing GPU array manager..." << std::endl;
	m_GPUArrayManager = std::make_unique<GPUArrayManager>();

	return m_window->GetWindow();
}

void App::LoadModels(const std::string meshFile) {
	
	std::vector<Model> models;
	models.emplace_back( meshFile.c_str() );
	
	int triangleCnt = 0;
	for ( auto& model : models ) {
		for ( int i = 0; i < model.meshes.size(); ++i ) {
			triangleCnt += (model.meshes[i].indices.size() / 3);
		}
	}

	std::cout << "Loaded " << triangleCnt << " triangles..." << std::endl;

	if ( !m_GPUArrayManager->AllocateTris(triangleCnt) )
		std::cout << "Failed to allocate Bounding box arrays on GPU." << std::endl;
	if ( !m_GPUArrayManager->AllocateMortonCodes(triangleCnt) )
		std::cout << "Failed to allocate Morton codes array on GPU." << std::endl;
	if( !m_GPUArrayManager->AllocateBIHTree(triangleCnt - 1) )
		std::cout << "Failed to allocate BIH Tree array on GPU." << std::endl;
	
	/*size_t stackLimit;
	cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize);
	std::cout << "Cuda stack limit: " << stackLimit << std::endl;
	
	auto res = cudaDeviceSetLimit(cudaLimitStackSize, 4096);
	if( res == cudaErrorMemoryAllocation )
		std::cout << "FAILED to increase stack limit." << std::endl;*/


	AABBs& trisBBoxes = m_GPUArrayManager->GetBBoxArrays();
	thrust::device_vector<Triangle>& d_triangles = m_GPUArrayManager->GetTriangles();
	thrust::host_vector<float3> h_arrLo(triangleCnt);
	thrust::host_vector<float3> h_arrHi(triangleCnt);
	thrust::host_vector<float3> h_arrCenter(triangleCnt);
	thrust::host_vector<float3> h_arrCenterNorm(triangleCnt);
	thrust::host_vector<Triangle> h_triangles(triangleCnt);

	glm::vec3 randPos = models[0].meshes[0].vertices[0].Position;
	AABB sceneBBox;
	sceneBBox.lo = { randPos.x, randPos.y, randPos.z };
	sceneBBox.hi = { randPos.x, randPos.y, randPos.z };

	int backwardIter = triangleCnt;
	int idx = 0;
	for ( auto& model : models ) {
		for ( auto& mesh : model.meshes ) {
			for ( int i = 0; i < mesh.indices.size(); i += 3 ) {
				Vertex v1 = mesh.vertices[ mesh.indices[i] ];
				Vertex v2 = mesh.vertices[ mesh.indices[i + 1] ];
				Vertex v3 = mesh.vertices[ mesh.indices[i + 2] ];
				idx = triangleCnt - backwardIter;
				
				h_triangles[idx].v0 = v1.Position;
				h_triangles[idx].v1 = v2.Position;
				h_triangles[idx].v2 = v3.Position;

				// triangle BBoxes
				auto minmaxX = std::minmax({ v1.Position.x, v2.Position.x, v3.Position.x });
				auto minmaxY = std::minmax({ v1.Position.y, v2.Position.y, v3.Position.y });
				auto minmaxZ = std::minmax({ v1.Position.z, v2.Position.z, v3.Position.z });
				h_arrLo[idx] = { minmaxX.first, minmaxY.first, minmaxZ.first };
				h_arrHi[idx] = { minmaxX.second, minmaxY.second, minmaxZ.second };
				h_arrCenter[idx] = { ( minmaxX.first + minmaxX.second ) / 2.0f,
									 ( minmaxY.first + minmaxY.second ) / 2.0f,
									 ( minmaxZ.first + minmaxZ.second ) / 2.0f,
								   };

				auto sceneMinmaxX = std::minmax({ minmaxX.first, minmaxX.second, sceneBBox.lo.x, sceneBBox.hi.x });
				auto sceneMinmaxY = std::minmax({ minmaxY.first, minmaxY.second, sceneBBox.lo.y, sceneBBox.hi.y });
				auto sceneMinmaxZ = std::minmax({ minmaxZ.first, minmaxZ.second, sceneBBox.lo.z, sceneBBox.hi.z });
				sceneBBox.lo = { sceneMinmaxX.first, sceneMinmaxY.first, sceneMinmaxZ.first };
				sceneBBox.hi = { sceneMinmaxX.second, sceneMinmaxY.second, sceneMinmaxZ.second };

				backwardIter--;
			}
		}
	}

	for ( int i = 0; i < triangleCnt; i++ )
	{
		float CURRminusMINX = h_arrCenter[i].x - sceneBBox.lo.x;
		float CURRminusMINY = h_arrCenter[i].y - sceneBBox.lo.y;
		float CURRminusMINZ = h_arrCenter[i].z - sceneBBox.lo.z;

		float MAXminusMINX = sceneBBox.hi.x - sceneBBox.lo.x;
		float MAXminusMINY = sceneBBox.hi.y - sceneBBox.lo.y;
		float MAXminusMINZ = sceneBBox.hi.z - sceneBBox.lo.z;
		h_arrCenterNorm[i].x = CURRminusMINX / MAXminusMINX;
		h_arrCenterNorm[i].y = CURRminusMINY / MAXminusMINY;
		h_arrCenterNorm[i].z = CURRminusMINZ / MAXminusMINZ;
	}

	d_triangles = h_triangles;
	trisBBoxes.arrCenter = h_arrCenter;
	trisBBoxes.arrCenterNorm = h_arrCenterNorm;
	trisBBoxes.arrHi = h_arrHi;
	trisBBoxes.arrLo = h_arrLo;
	trisBBoxes.sceneBBoxLo = sceneBBox.lo;
	trisBBoxes.sceneBBoxHi = sceneBBox.hi;
	std::cout << "scene bbox lo: [" << sceneBBox.lo.x << ", " << sceneBBox.lo.y << ", " << sceneBBox.lo.z << "]" << std::endl;
	std::cout << "scene bbox hi: [" << sceneBBox.hi.x << ", " << sceneBBox.hi.y << ", " << sceneBBox.hi.z << "]" << std::endl;
}


void App::Run() {
	float deltaTime=0.0;
	float lastFrame=0.0;

	while ( !glfwWindowShouldClose( m_window->GetWindow() ) ) {
		
		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		m_window->ShowFPS( 1.0f / deltaTime );

		
		m_renderer->Render( *m_GPUArrayManager );
		glfwSwapBuffers( m_window->GetWindow() );
		glfwPollEvents();
	}
}


