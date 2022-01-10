#pragma once
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "Renderer.h"
#include "HitableList.h"
#include "Constants.h"
#include "Camera.h"

// clamp x to range [a, b]
__device__ float clamp( float x, float a, float b ) {
    return max( a, min( b, x ) );
}

__device__ int clamp( int x, int a, int b ) {
    return max( a, min( b, x ) );
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt( float r, float g, float b ) {
    r = clamp( r, 0.0f, 255.0f );
    g = clamp( g, 0.0f, 255.0f );
    b = clamp( b, 0.0f, 255.0f );
    return ( int( b ) << 16 ) | ( int( g ) << 8 ) | int( r );
}

__device__ glm::vec3 Color( const Ray& r, curandState* local_rand_state, HitableList& world ) {
    HitRecord rec;
    if ( world.Hit( r, 0.0f, FLT_MAX, rec ) ) { 
        return glm::vec3( 255.0f, 255.0f, 0.0f );
    } else {
        return glm::vec3( 20.0f, 20.0f, 40.0f );
    }
}

__global__ void cudaRender( unsigned int* g_odata, Camera* cam, curandState* rand_state, HitableList& world ) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ( ( i >= SCREEN_WIDTH ) || ( j >= SCREEN_HEIGHT ) ) {
        return;
    }

    int pixel_index = j * SCREEN_WIDTH + i;
    curandState local_rand_state = rand_state[pixel_index];
    glm::vec3 col( 0, 0, 0 );
    for ( int s = 0; s < RAYS_PER_PIXEL; s++ ) {
        float u = float( i + curand_uniform( &local_rand_state ) ) / float( SCREEN_WIDTH );
        float v = float( j + curand_uniform( &local_rand_state ) ) / float( SCREEN_HEIGHT );
        Ray r = cam->GetRay( u, v );
        col += Color( r, &local_rand_state, world );
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float( RAYS_PER_PIXEL );

    g_odata[j * SCREEN_WIDTH + i] = rgbToInt( col.r, col.g, col.b );
}

void Renderer::Launch_cudaRender( unsigned int* g_odata, HitableList& world) {
    cudaRender <<<m_blocks, m_threads, 0 >>> ( g_odata, d_camera, d_rand_state, world );
}


__global__ void InitRandGPU( int max_x, int max_y, curandState* randState ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ( ( i >= SCREEN_WIDTH ) || ( j >= SCREEN_HEIGHT ) ) {
        return;
    }
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init( 1984, pixel_index, 0, &randState[pixel_index] );
}

void Renderer::Launch_cudaRandInit( curandState* rand_state ) {
    InitRandGPU <<<m_blocks, m_threads >>> ( SCREEN_WIDTH, SCREEN_HEIGHT, rand_state );
}