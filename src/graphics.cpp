#include "../include/graphics.h"
#include "../include/sphere.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

Graphics::Graphics(int width, int height, int numParticles) 
    : screenWidth(width), screenHeight(height), numParticles(numParticles) {}

Graphics::~Graphics() {
    // Proper cleanup should be added here
}

std::string Graphics::readShaderFile(const std::string& filePath) {
    std::ifstream shaderFile(filePath);
    if (!shaderFile.is_open()) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << filePath << std::endl;
    }
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    shaderFile.close();
    return shaderStream.str();
}

unsigned int Graphics::createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath) {
    std::string vertexCode = readShaderFile(vertexPath);
    std::string fragmentCode = readShaderFile(fragmentPath);
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    unsigned int vertex, fragment;
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);

    unsigned int ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return ID;
}

void Graphics::init() {
    instanceShaderProgram = createShaderProgram("shaders/instance.vert", "shaders/instance.frag");
    simpleShaderProgram = createShaderProgram("shaders/simple.vert", "shaders/simple.frag");

    std::vector<glm::vec3> sphereVertices;
    std::vector<glm::vec3> sphereNormals;
    std::vector<unsigned int> sphereIndices;
    generateSphere(sphereVertices, sphereNormals, sphereIndices, 0.25f, 12, 6);
    sphereIndexCount = sphereIndices.size();

    unsigned int sphereVBO, normalsVBO;
    glGenVertexArrays(1, &sphereVAO);
    glBindVertexArray(sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(glm::vec3), &sphereVertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &normalsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, normalsVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereNormals.size() * sizeof(glm::vec3), &sphereNormals[0], GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(1);
    glGenBuffers(1, &sphereEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int), &sphereIndices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &instanceMatrixVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceMatrixVBO);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &instanceColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceColorVBO);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

    glBindVertexArray(sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceMatrixVBO);
    for(int i = 0; i < 4; i++) {
        glEnableVertexAttribArray(2 + i);
        glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
        glVertexAttribDivisor(2 + i, 1);
    }
    glBindBuffer(GL_ARRAY_BUFFER, instanceColorVBO);
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glVertexAttribDivisor(6, 1);
    glBindVertexArray(0);
}

void Graphics::render(const std::vector<Particle>& particles, const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPos, const glm::vec3& centerColor) {
    std::vector<glm::mat4> modelMatrices(numParticles);
    std::vector<glm::vec3> instanceColors(numParticles);
    glm::vec3 colorHot(1.0f, 0.3f, 0.1f);
    glm::vec3 colorCool(0.8f, 0.9f, 1.0f);
    float maxDist = 80.0f;
    for(int i = 0; i < numParticles; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, particles[i].position);
        modelMatrices[i] = model;
        float dist = glm::length(particles[i].position);
        float blendFactor = glm::clamp(dist / maxDist, 0.0f, 1.0f);
        instanceColors[i] = glm::mix(colorHot, colorCool, blendFactor);
    }
    glBindBuffer(GL_ARRAY_BUFFER, instanceMatrixVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(glm::mat4), &modelMatrices[0]);
    glBindBuffer(GL_ARRAY_BUFFER, instanceColorVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(glm::vec3), &instanceColors[0]);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(instanceShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(instanceShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(instanceShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniform3f(glGetUniformLocation(instanceShaderProgram, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
    glUniform3f(glGetUniformLocation(instanceShaderProgram, "lightPos"), 0.0f, 0.0f, 0.0f);
    glBindVertexArray(sphereVAO);
    glDrawElementsInstanced(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0, numParticles);
    
    glUseProgram(simpleShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(simpleShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(simpleShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniform3f(glGetUniformLocation(simpleShaderProgram, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
    glUniform3f(glGetUniformLocation(simpleShaderProgram, "lightPos"), 0.0f, 0.0f, 0.0f);
    
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(8.0f));
    glUniformMatrix4fv(glGetUniformLocation(simpleShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(glGetUniformLocation(simpleShaderProgram, "objectColor"), centerColor.r, centerColor.g, centerColor.b);
    
    glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
