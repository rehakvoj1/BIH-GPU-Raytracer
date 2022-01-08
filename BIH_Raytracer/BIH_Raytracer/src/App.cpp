#include "App.h"
#include "Model.h"
#include <iostream>
#include <filesystem>
#include "Triangle.h"

App::App() : d_world(nullptr) {
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
	
	std::vector<Triangle> faces;
	for ( auto& model : models ) {
		for ( auto& mesh : model.meshes ) {
			for ( unsigned int i = 0; i < mesh.indices.size(); i += 3 ) {
				Vertex v1 = mesh.vertices[mesh.indices[i    ]];
				Vertex v2 = mesh.vertices[mesh.indices[i + 1]];
				Vertex v3 = mesh.vertices[mesh.indices[i + 2]];
				faces.emplace_back( v1, v2, v3 );
			}
		}
	}

	Triangle* triangleFaces = nullptr;
	cudaMalloc( &triangleFaces, faces.size() * sizeof( Triangle ) );
	cudaMemcpy( triangleFaces, faces.data(), faces.size() * sizeof( Triangle ), cudaMemcpyHostToDevice );
	Hitable* pHitable = static_cast<Hitable*>( triangleFaces );
	d_world = new HitableList( &pHitable, faces.size() );
}


void App::Run() {
	float deltaTime=0.0;
	float lastFrame=0.0;

	while ( !glfwWindowShouldClose( m_window->GetWindow() ) ) {
		
		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		m_window->ShowFPS( 1.0f / deltaTime );

		m_renderer->Render();
		glfwSwapBuffers( m_window->GetWindow() );
		glfwPollEvents();
	}
}


