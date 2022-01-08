#pragma once
#include "cuda_runtime.h"
#include "Renderer.h"

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

__global__ void cudaRender( unsigned int* g_odata, int imgw ) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    if ( ( x >= SCREEN_WIDTH ) || ( y >= SCREEN_HEIGHT ) ) {
        return;
    }

    uchar4 c4 = make_uchar4( ( x & 0x20 ) ? 100 : 0, 0, ( y & 0x20 ) ? 100 : 0, 0 );
    g_odata[y * imgw + x] = rgbToInt( c4.z, c4.y, c4.x );
}

void Renderer::Launch_cudaRender( dim3 grid, dim3 block, int sbytes,
                                    unsigned int* g_odata, int imgw ) {
    cudaRender <<<grid, block, sbytes >>> ( g_odata, imgw );
}


__global__ void InitRandGPU( int max_x, int max_y, curandState* randState ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ( ( i >= max_x ) || ( j >= max_y ) ) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init( 1984, pixel_index, 0, &randState[pixel_index] );
}

void Renderer::Launch_cudaRandInit( curandState* rand_state ) {
    InitRandGPU <<<m_blocks, m_threads >>> ( SCREEN_WIDTH, SCREEN_HEIGHT, rand_state );
}