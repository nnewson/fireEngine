#include <fire_engine/graphics/geometry.hpp>

#include <fire_engine/render/resources.hpp>

namespace fire_engine
{

void Geometry::load(Resources& resources)
{
    vertexBuffer_ = storageVertices_ ? resources.createStorageVertexBuffer(vertices_)
                                     : resources.createVertexBuffer(vertices_);
    indexBuffer_ = resources.createIndexBuffer(indices_);
}

} // namespace fire_engine
