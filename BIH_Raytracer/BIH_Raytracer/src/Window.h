#pragma once

// opengl includes
#include "glad/glad.h"
#include "GLFW/glfw3.h"



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
	Window( GLFWwindow* window, int width, int height, std::string title );
	GLFWwindow* GetWindow();

	void processInput();
	// callbacks
	void static framebuffer_size_callback( GLFWwindow* window, int width, int height );
	void static key_callback( GLFWwindow* window, int key, int scancode, int action, int mods );
	void static mouse_callback( GLFWwindow* window, double xpos, double ypos );
	// ----------------------- //

private:
	std::unique_ptr<GLFWwindow, glfwDeleter> m_wnd;
	int m_width;
	int m_height;
	std::string m_title;
};

