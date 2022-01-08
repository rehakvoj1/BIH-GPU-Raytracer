#include "Camera.h"

__host__ Camera::Camera( glm::vec3 origin, float aspectRatio ) : m_origin( origin ),
                                                        m_lowerLeftCorner( glm::vec3(-2.0,-1.0,-1.0) ),
                                                        m_vertical( glm::vec3(0.0,2.0,0.0) ),
                                                        m_horizontal( glm::vec3(aspectRatio*2.0,0.0,0.0) )
{
}

__device__ Ray Camera::GetRay( float u, float v ) {
    return Ray( m_origin, m_lowerLeftCorner + u * m_horizontal + v * m_vertical - m_origin);
}
