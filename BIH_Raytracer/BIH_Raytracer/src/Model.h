#pragma once
#include <vector>
#include <string>

#include "Mesh.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

class Model {
public:
	std::vector<Texture> textures_loaded;
	std::vector<Mesh> meshes;
	std::string directory;

	Model( std::string path );

private:
	void LoadModel( std::string const& path );
	void ProcessNode( aiNode* node, const aiScene* scene );
	Mesh ProcessMesh( aiMesh* mesh, const aiScene* scene );
	std::vector<Texture> LoadMaterialTextures( aiMaterial* mat, aiTextureType type, std::string typeName );
	unsigned int TextureFromFile( const char* path, const std::string& directory );
};

