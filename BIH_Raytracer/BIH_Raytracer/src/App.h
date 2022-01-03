#pragma once

#include "Window.h"
#include "Renderer.h"
#include "HitableList.h"

class App {
public:
	App();
	~App();
	

	// -------- WINDOW ------- //
	GLFWwindow* NewWindow( int width, int height, std::string title );
	void LoadModels();
	void Run();

private:
	std::unique_ptr<Window> m_window;
	std::unique_ptr<Renderer> m_renderer;
	Hitable* d_world;
};

