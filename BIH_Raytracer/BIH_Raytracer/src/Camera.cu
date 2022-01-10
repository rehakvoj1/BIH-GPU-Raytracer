#include "Camera.h"
#include "Ray.h"


Camera::Camera( glm::vec3 origin, float aspectRatio ) : m_origin( origin ),
                                                        m_lowerLeftCorner( glm::vec3( origin.x - 2.0, origin.y -1.0, origin.z - 1.0 ) ),
                                                        m_vertical( glm::vec3( 0.0, 2.0, 0.0 ) ),
                                                        m_horizontal( glm::vec3( aspectRatio * 2.0, 0.0, 0.0 ) ) {
}

Camera::Camera( const Camera& c ) {
    m_origin = c.m_origin;
    m_lowerLeftCorner = c.m_lowerLeftCorner;
    m_vertical = c.m_vertical;
    m_horizontal = c.m_horizontal;
}

__device__ Ray Camera::GetRay( float u, float v ) {
    return Ray( m_origin, m_lowerLeftCorner + u * m_horizontal + v * m_vertical - m_origin );
}

Camera& Camera::operator=( const Camera& c ) {
    m_origin = c.m_origin;
    m_lowerLeftCorner = c.m_lowerLeftCorner;
    m_vertical = c.m_vertical;
    m_horizontal = c.m_horizontal;

    return *this;
}


