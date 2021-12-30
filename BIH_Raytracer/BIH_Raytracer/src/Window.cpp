#include "Window.h"


//---------------- CALLBACKS -------------------//
void Window::framebuffer_size_callback( GLFWwindow* window, int width, int height ) {
	glViewport( 0, 0, width, height );
}

void Window::processInput() {
}

void Window::key_callback( GLFWwindow* window, int key, int scancode, int action, int mods ) {
	if ( key == GLFW_KEY_ESCAPE && action == GLFW_PRESS ) {
		glfwSetWindowShouldClose( window, true );
	}
}

void scroll_callback( GLFWwindow* window, double xoffset, double yoffset ) {
}

void Window::mouse_callback( GLFWwindow* window, double xpos, double ypos ) {
}
//-----------------------------------------------------------------------------//




Window::Window( GLFWwindow* window, int width, int height, std::string title ) : m_width(width), 
																				 m_height(height),
																				 m_title(title) {

	m_wnd.reset( window );

	glfwMakeContextCurrent( m_wnd.get() );
	glfwSetFramebufferSizeCallback( m_wnd.get(), Window::framebuffer_size_callback );
	glfwSetKeyCallback( m_wnd.get(), key_callback );
	glfwSetCursorPosCallback( m_wnd.get(), mouse_callback );
	glfwSetScrollCallback( m_wnd.get(), scroll_callback );
	glfwSetInputMode( m_wnd.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED );
	
}

GLFWwindow* Window::GetWindow() {
    return m_wnd.get();
}

void Window::ShowFPS( float fps ) {
	glfwSetWindowTitle( m_wnd.get(), ( m_title + "     FPS: " + std::to_string( fps ) ).c_str() );
}