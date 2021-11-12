#pragma once

#include "Window.h"

class App {
public:
	App();
	

	// -------- WINDOW ------- //
	GLFWwindow* NewWindow( int width, int height, std::string title );
	void Run();

private:
	std::unique_ptr<Window> m_window;
};

