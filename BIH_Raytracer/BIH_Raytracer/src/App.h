#pragma once

#include "Window.h"
#include "Renderer.h"
#include "Triangle.cuh"
#include "Tree.cuh"
#include "GPUArrayManager.h"

#include <thrust/device_ptr.h>

class App {
public:
	App();
	~App();
	

	// -------- WINDOW ------- //
	GLFWwindow* NewWindow( int width, int height, std::string title );
	void LoadModels(const std::string meshFile);
	void Run();

private:
	std::unique_ptr<Window> m_window;
	std::unique_ptr<Renderer> m_renderer;
	std::unique_ptr<GPUArrayManager> m_GPUArrayManager;
};

