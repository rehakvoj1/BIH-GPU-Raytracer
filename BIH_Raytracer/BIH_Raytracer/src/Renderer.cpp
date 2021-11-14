#include "Renderer.h"
#include <stdio.h>
#include "Constants.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
extern "C" void launch_cudaProcess( dim3 grid, dim3 block, int sbytes, unsigned int* g_odata, int imgw );

void check_cuda( cudaError_t result, char const* const func, const char* const file, int const line ) {
	if ( result ) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>( result ) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit( 99 );
	}
}

Renderer::Renderer() : m_quadTexture(0), m_shaderProgram(0), m_cudaTexResultRes( nullptr ), m_cudaDestResource(nullptr),
					   m_quadVAO(0), m_quadVBO(0), m_quadEBO(0)
{
}

void Renderer::Init() {
	// create texture that will receive the result of CUDA
	CreateTextureDst();
	
	// load shader programs
	m_shaderProgram = CompileGLSLprogram();

	CreateCUDABuffers();
	InitQuad();
}

void Renderer::Render() {
	// calculate grid size
	dim3 block( 16, 16, 1 );
	// dim3 block(16, 16, 1);
	dim3 grid( SCREEN_WIDTH / block.x, SCREEN_HEIGHT / block.y, 1 );

	launch_cudaProcess( grid, block, 0, m_cudaDestResource, SCREEN_WIDTH );

	cudaArray* texture_ptr;
	checkCudaErrors( cudaGraphicsMapResources( 1, &m_cudaTexResultRes, 0 ) );
	checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(
		&texture_ptr, m_cudaTexResultRes, 0, 0 ) );

	unsigned int num_texels = SCREEN_WIDTH * SCREEN_HEIGHT;
	unsigned int num_values = num_texels * 4;
	unsigned int size_tex_data = sizeof( GLubyte ) * num_values;
	checkCudaErrors( cudaMemcpy2DToArray( texture_ptr, 0, 0, m_cudaDestResource, 
										  size_tex_data/SCREEN_HEIGHT, size_tex_data/SCREEN_HEIGHT, SCREEN_HEIGHT, 
										  cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_cudaTexResultRes, 0 ) );

	glClearColor( 0.2f, 0.3f, 0.3f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT );

	// bind Texture
	glBindTexture( GL_TEXTURE_2D, m_quadTexture );
	glEnable( GL_TEXTURE_2D );
	glDisable( GL_DEPTH_TEST );
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

	// render container
	glUseProgram( m_shaderProgram );
	glBindVertexArray( m_quadVAO );
	glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );
}


GLuint Renderer::CompileGLSLprogram() {
	GLuint v, f, p = 0;

	p = glCreateProgram();

	if ( m_texVertexShader ) {
		v = glCreateShader( GL_VERTEX_SHADER );
		glShaderSource( v, 1, &m_texVertexShader, NULL );
		glCompileShader( v );

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv( v, GL_COMPILE_STATUS, &compiled );

		if ( !compiled ) {
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog( v, 256, NULL, temp );
			printf( "Vtx Compile failed:\n%s\n", temp );
			//#endif
			glDeleteShader( v );
			return 0;
		} else {
			glAttachShader( p, v );
		}
	}

	if ( m_texFragmentShader ) {
		f = glCreateShader( GL_FRAGMENT_SHADER );
		glShaderSource( f, 1, &m_texFragmentShader, NULL );
		glCompileShader( f );

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv( f, GL_COMPILE_STATUS, &compiled );

		if ( !compiled ) {
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog( f, 256, NULL, temp );
			printf( "frag Compile failed:\n%s\n", temp );
			//#endif
			glDeleteShader( f );
			return 0;
		} else {
			glAttachShader( p, f );
		}
	}

	glLinkProgram( p );

	int infologLength = 0;
	int charsWritten = 0;

	glGetProgramiv( p, GL_INFO_LOG_LENGTH, (GLint*)&infologLength );

	if ( infologLength > 0 ) {
		char* infoLog = new char[ infologLength ];
		glGetProgramInfoLog( p, infologLength, (GLsizei*)&charsWritten, infoLog );
		printf( "Shader compilation error: %s\n", infoLog );
		delete [] infoLog;
	}

	return p;
}

void Renderer::CreateTextureDst() {
	// create a texture
	glGenTextures( 1, &m_quadTexture );
	glBindTexture( GL_TEXTURE_2D, m_quadTexture );

	// set basic parameters
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, SCREEN_WIDTH, SCREEN_HEIGHT, 0,
				  GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL );

	
	//// register this texture with CUDA
	checkCudaErrors( cudaGraphicsGLRegisterImage(
		&m_cudaTexResultRes, m_quadTexture, GL_TEXTURE_2D,
		cudaGraphicsMapFlagsWriteDiscard ) );
}

void Renderer::CreateCUDABuffers() {
	// set up vertex data parameter
	unsigned int num_texels = SCREEN_WIDTH * SCREEN_HEIGHT;
	unsigned int num_values = num_texels * 4;
	unsigned int size_tex_data = sizeof( GLubyte ) * num_values;
	checkCudaErrors( cudaMalloc( (void**)&m_cudaDestResource, size_tex_data ) );
}

void Renderer::InitQuad() {
	glGenVertexArrays( 1, &m_quadVAO );
	glGenBuffers( 1, &m_quadVBO );
	glGenBuffers( 1, &m_quadEBO );

	glBindVertexArray( m_quadVAO );

	glBindBuffer( GL_ARRAY_BUFFER, m_quadVBO );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );

	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_quadEBO );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( indices ), indices, GL_STATIC_DRAW );

	// position attribute
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (void*)0 );
	glEnableVertexAttribArray( 0 );
	// texture coord attribute
	glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof( float ), (void*)( 3 * sizeof( float ) ) );
	glEnableVertexAttribArray( 1 );
}
