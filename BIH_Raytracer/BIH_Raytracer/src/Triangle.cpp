#include "Triangle.cuh"
#include "Renderer.h"
#include <format>
#include <thrust/host_vector.h>



void Renderer::PrintMCs(GPUArrayManager& gpuArrayManager)
{
	thrust::host_vector<uint32_t> h_uniques = gpuArrayManager.GetUniqueMortonCodesVectorRef();
	for ( int i = 0; i < gpuArrayManager.GetTrisSize(); i++ )
	{
		std::cout << std::format("{:064b}", h_uniques[i]) << std::endl;
	}
}