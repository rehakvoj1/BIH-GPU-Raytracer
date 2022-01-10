#pragma once

#include "glad/glad.h"
#include "Constants.h"
#include "Camera.h"
#include "HitableList.h"

// cuda includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "curand_kernel.h"
#include <iostream>





class Renderer {
public:
	__host__ Renderer();
    __host__ void Init();
    __host__ void Render(HitableList& world);

private:
    void Launch_cudaRender( unsigned int* g_odata, HitableList& world );
    void Launch_cudaRandInit(curandState * rand_state);
    __host__ GLuint CompileGLSLprogram();
    __host__ void CreateTextureDst();
    __host__ void CreateCUDABuffers();
    __host__ void InitQuad();
    __host__ void InitRand();
    
private:
    curandState* d_rand_state;
    Camera* d_camera;
    GLuint m_quadTexture;
    GLuint m_shaderProgram;
    cudaGraphicsResource* m_cudaTexResultRes;
    unsigned int* m_cudaDestResource;
    unsigned int m_quadVAO;
    unsigned int m_quadVBO;
    unsigned int m_quadEBO;

    dim3 m_blocks;
    dim3 m_threads;

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

