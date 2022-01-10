#include "App.h"
#include "Model.h"
#include <iostream>
#include <filesystem>
#include <cmath>


App::App() {
}

App::~App() {
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
	
	return m_window->GetWindow();
}

void App::LoadModels() {
	
	std::vector<Model> models;
	models.emplace_back( "resources/sponza/sponza.obj" );
	
	int triangleCnt = 0;
	for ( auto& model : models ) {
		for ( int i = 0; i < model.meshes.size(); ++i ) {
			triangleCnt += (model.meshes[i].indices.size() / 3);
		}
	}

	m_deviceWorld = new HitableList;
	m_deviceWorld->m_listSize = triangleCnt;
	cudaMallocManaged( &( m_deviceWorld->m_list ), triangleCnt * sizeof( Triangle ) );

	
	int triangleIdx = triangleCnt;
	for ( auto& model : models ) {
		for ( auto& mesh : model.meshes ) {
			for ( int i = 0; i < mesh.indices.size(); i += 3 ) {
				Vertex v1 = mesh.vertices[ mesh.indices[i] ];
				Vertex v2 = mesh.vertices[ mesh.indices[i + 1] ];
				Vertex v3 = mesh.vertices[ mesh.indices[i + 2] ];
				Triangle t( v1, v2, v3 );
				memcpy( m_deviceWorld->m_list + ( triangleCnt - triangleIdx ), &t, sizeof( Triangle ) );
				triangleIdx--;
			}
		}
	}
}


void App::Run() {
	float deltaTime=0.0;
	float lastFrame=0.0;

	while ( !glfwWindowShouldClose( m_window->GetWindow() ) ) {
		
		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		m_window->ShowFPS( 1.0f / deltaTime );

		m_renderer->Render(*m_deviceWorld);
		glfwSwapBuffers( m_window->GetWindow() );
		glfwPollEvents();
	}
}


