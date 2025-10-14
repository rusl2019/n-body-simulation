#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include "../include/particle.h"

class Graphics {
public:
    Graphics(int width, int height, int numParticles);
    ~Graphics();

    void init();
    void render(const std::vector<Particle>& particles, const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPos);

private:
    int screenWidth, screenHeight, numParticles;
    unsigned int instanceShaderProgram, simpleShaderProgram;
    unsigned int sphereVAO, sphereEBO;
    unsigned int instanceMatrixVBO, instanceColorVBO;
    GLsizei sphereIndexCount;

    std::string readShaderFile(const std::string& filePath);
    unsigned int createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
};

#endif // GRAPHICS_H
