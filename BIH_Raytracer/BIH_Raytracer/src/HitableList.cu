#include "HitableList.h"

HitableList::HitableList() : m_list( nullptr ), m_listSize( 0 ) {
}

HitableList::HitableList( const HitableList& list ) : m_listSize( 0 ), m_list( nullptr ) {
    _realloc( list.m_listSize );
    memcpy( m_list, list.m_list, list.m_listSize * sizeof( Triangle ) );
}

HitableList::~HitableList() {
    cudaFree( m_list );
}

__device__ bool HitableList::Hit( const Ray& r, float t_min, float t_max, HitRecord& rec ) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for ( int i = 0; i < m_listSize; i++ ) {
        if ( m_list[i].Hit( r, t_min, closest_so_far, temp_rec ) ) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

HitableList& HitableList::operator=( const HitableList& list ) {
    _realloc( list.m_listSize );
    memcpy( m_list, list.m_list, list.m_listSize * sizeof( Triangle ) );
    return *this;
}

__host__ __device__ Triangle& HitableList::operator[]( int pos ) const {
    return m_list[pos];
}

void HitableList::_realloc( int len ) {
    cudaFree( m_list );
    m_listSize = len;
    cudaMallocManaged( &m_list, m_listSize + 1 );
}
