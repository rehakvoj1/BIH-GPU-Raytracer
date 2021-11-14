#pragma once

#include "glad/glad.h"
#include "Constants.h"

// cuda includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>





class Renderer {
public:
	Renderer();
    void Init();
    void Render();

private:
    GLuint CompileGLSLprogram();
    void CreateTextureDst();
    void CreateCUDABuffers();
    void InitQuad();

private:
    GLuint m_quadTexture;
    GLuint m_shaderProgram;
    cudaGraphicsResource* m_cudaTexResultRes;
    unsigned int* m_cudaDestResource;
    unsigned int m_quadVAO;
    unsigned int m_quadVBO;
    unsigned int m_quadEBO;

private:
	const char* m_texVertexShader = 
        "#version 460 core\n"
        "\n"
        "layout (location = 0) in vec3 inVertexPos;\n"
        "layout (location = 1) in vec2 inTexCoord;\n"
        "\n"
        "out vec2 TexCoord;\n"
        "\n"
        "void main(void)\n"
        "{\n"
        "	gl_Position = vec4(inVertexPos,1.0f);\n"
        "	TexCoord = inTexCoord;\n"
        "}\n";

    const char* m_texFragmentShader = 
        "#version 130\n"
        "uniform usampler2D texImage;\n"
        "in vec2 TexCoord;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   vec4 c = texture(texImage, TexCoord);\n"
        "   //vec4 c = vec4(0.5,0.5,0.5,1.0);\n"
        "	color = c/255.0;\n"
        "}\n";

    float vertices[20] = {
        // positions         // texture coords
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f  // top left 
    };
    unsigned int indices[6] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
};

