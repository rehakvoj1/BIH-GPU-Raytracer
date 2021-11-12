#pragma once

// opengl includes
#include "glad/glad.h"
#include "GLFW/glfw3.h"

// cuda includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <string>
#include <memory>





struct glfwDeleter {
	void operator()( GLFWwindow* wnd ) {
		std::cout << "Destroying GLFW Window Context" << std::endl;
		glfwDestroyWindow( wnd );
	}
};


class Window {
public:
	Window() = delete;
	Window( GLFWwindow* window );
	GLFWwindow* GetWindow();

	void processInput();
	// callbacks
	void static framebuffer_size_callback( GLFWwindow* window, int width, int height );
	void static key_callback( GLFWwindow* window, int key, int scancode, int action, int mods );
	void static mouse_callback( GLFWwindow* window, double xpos, double ypos );
	// ----------------------- //

private:
	std::unique_ptr<GLFWwindow, glfwDeleter> m_wnd;
};

