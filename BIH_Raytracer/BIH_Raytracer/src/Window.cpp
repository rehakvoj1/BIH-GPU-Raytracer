#include "Window.h"


//---------------- CALLBACKS -------------------//
void Window::framebuffer_size_callback( GLFWwindow* window, int width, int height ) {
	glViewport( 0, 0, width, height );
}

void Window::processInput() {
}

void Window::key_callback( GLFWwindow* window, int key, int scancode, int action, int mods ) {
}

void scroll_callback( GLFWwindow* window, double xoffset, double yoffset ) {
}

void Window::mouse_callback( GLFWwindow* window, double xpos, double ypos ) {
}
//-----------------------------------------------------------------------------//




Window::Window( GLFWwindow* window ) {

	m_wnd.reset( window );

	glfwMakeContextCurrent( m_wnd.get() );
	glfwSetFramebufferSizeCallback( m_wnd.get(), Window::framebuffer_size_callback );
	glfwSetKeyCallback( m_wnd.get(), key_callback );
	glfwSetCursorPosCallback( m_wnd.get(), mouse_callback );
	glfwSetScrollCallback( m_wnd.get(), scroll_callback );
	glfwSetInputMode( m_wnd.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED );
}

GLFWwindow* Window::GetWindow() {
    return nullptr;
}
