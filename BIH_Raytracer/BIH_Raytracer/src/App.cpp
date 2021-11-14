#include "App.h"

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


void App::Run() {
	while ( !glfwWindowShouldClose( m_window->GetWindow() ) ) {
		
		m_renderer->Render();
		glfwSwapBuffers( m_window->GetWindow() );
		glfwPollEvents();
	}
}


