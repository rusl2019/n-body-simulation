#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define PI 3.14159265359

// --- CUDA Error Checking Utility ---
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        exit(99);
    }
}

// --- Function Prototypes ---
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window);
void generateSphere(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned int>& indices, float radius, int sectorCount, int stackCount);

// --- Particle Structure ---
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float mass;
};

// --- CUDA KERNEL (Unchanged) ---
__global__ void updateParticles(Particle* particles, int numParticles, Particle blackHole, float G, float dt, float softeningFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    glm::vec3 pos = particles[i].position;
    glm::vec3 vel = particles[i].velocity;
    float mass = particles[i].mass;
    glm::vec3 r_bh = blackHole.position - pos;
    float dist_sq_bh = r_bh.x * r_bh.x + r_bh.y * r_bh.y + r_bh.z * r_bh.z;
    glm::vec3 force_bh = glm::vec3(0.0f);
    if (dist_sq_bh > 0.25f) {
        float dist_bh = sqrtf(dist_sq_bh);
        force_bh = (r_bh / dist_bh) * (G * blackHole.mass * mass) / dist_sq_bh;
    }
    glm::vec3 force_others = glm::vec3(0.0f);
    for (int j = 0; j < numParticles; j++) {
        if (i == j) continue;
        glm::vec3 r_ij = particles[j].position - pos;
        float dist_sq_ij = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        float inv_dist = rsqrtf(dist_sq_ij + softeningFactor * softeningFactor);
        float forceMag = (G * particles[j].mass * mass) * (inv_dist * inv_dist * inv_dist);
        force_others += r_ij * forceMag;
    }
    glm::vec3 total_force = force_bh + force_others;
    vel += (total_force / mass) * dt;
    pos += vel * dt;
    particles[i].position = pos;
    particles[i].velocity = vel;
}

// --- SHADERS FOR INSTANCED PARTICLES ---
const char* instanceVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in mat4 aInstanceMatrix;
    layout (location = 6) in vec3 aInstanceColor;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;
    out vec3 Normal;
    out vec3 InstanceColor;

    void main() {
        FragPos = vec3(aInstanceMatrix * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(aInstanceMatrix))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
        InstanceColor = aInstanceColor;
    }
)";

const char* instanceFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;
    in vec3 InstanceColor;

    uniform vec3 lightPos;
    uniform vec3 viewPos;

    void main() {
        // Ambient
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * InstanceColor;

        // Diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * InstanceColor;

        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

        vec3 result = ambient + diffuse + specular;
        FragColor = vec4(result, 1.0);
    }
)";

// --- (!!!) NEW: SHADERS FOR SINGLE BLACK HOLE OBJECT (!!!) ---
const char* simpleVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;
    out vec3 Normal;

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const char* simpleFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;

    uniform vec3 objectColor;
    uniform vec3 lightPos;
    uniform vec3 viewPos;

    void main() {
        // Using the same lighting model as the instanced shader
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * objectColor;
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * objectColor;
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
        vec3 result = ambient + diffuse + specular;
        FragColor = vec4(result, 1.0);
    }
)";


// --- Global Variables (for C++ part) ---
const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;
glm::vec3 cameraPos   = glm::vec3(0.0f, 60.0f, 150.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.4f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);
bool firstMouse = true;
float yaw = -90.0f, pitch = -25.0f;
float lastX = SCREEN_WIDTH / 2.0f, lastY = SCREEN_HEIGHT / 2.0f;
float deltaTime = 0.0f, lastFrame = 0.0f;

// --- Main Simulation Logic (CPU) ---
int main() {
    // --- Standard OpenGL/GLFW Initialization ---
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "CUDA Accelerated N-Body Sphere Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glewInit();
    glEnable(GL_DEPTH_TEST);

    // --- Build and compile instance shader program ---
    unsigned int instanceShaderProgram;
    {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &instanceVertexShaderSource, NULL); glCompileShader(vertexShader);
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &instanceFragmentShaderSource, NULL); glCompileShader(fragmentShader);
        instanceShaderProgram = glCreateProgram();
        glAttachShader(instanceShaderProgram, vertexShader); glAttachShader(instanceShaderProgram, fragmentShader); glLinkProgram(instanceShaderProgram);
        glDeleteShader(vertexShader); glDeleteShader(fragmentShader);
    }

    // --- Build and compile simple shader program for the black hole ---
    unsigned int simpleShaderProgram;
    {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &simpleVertexShaderSource, NULL); glCompileShader(vertexShader);
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &simpleFragmentShaderSource, NULL); glCompileShader(fragmentShader);
        simpleShaderProgram = glCreateProgram();
        glAttachShader(simpleShaderProgram, vertexShader); glAttachShader(simpleShaderProgram, fragmentShader); glLinkProgram(simpleShaderProgram);
        glDeleteShader(vertexShader); glDeleteShader(fragmentShader);
    }
    
    // --- SETUP FOR INSTANCED SPHERE MESH (Unchanged) ---
    std::vector<glm::vec3> sphereVertices;
    std::vector<glm::vec3> sphereNormals;
    std::vector<unsigned int> sphereIndices;
    generateSphere(sphereVertices, sphereNormals, sphereIndices, 0.25f, 12, 6);

    unsigned int sphereVAO, sphereVBO, sphereEBO, normalsVBO;
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

    // --- Simulation Setup (CPU) ---
    Particle blackHole;
    blackHole.position = glm::vec3(0.0f);
    blackHole.mass = 80000.0f;
    int numParticles = 16384;
    const float G = 0.05f;
    const float instabilityFactor = 0.85f;
    std::vector<Particle> particles_host(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        float phi = glm::radians((float)(rand() % 36000) / 100.0f);
        float cos_theta = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float theta = acos(cos_theta);
        float radius = 40.0f + (rand() % 2000) / 100.0f;
        particles_host[i].position.x = radius * sin(theta) * cos(phi);
        particles_host[i].position.y = radius * cos(theta);
        particles_host[i].position.z = radius * sin(theta) * sin(phi);
        float perfectOrbitSpeed = std::sqrt((G * blackHole.mass) / radius);
        glm::vec3 tangent = glm::cross(particles_host[i].position, glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::length(tangent) < 0.01f) {
            tangent = glm::cross(particles_host[i].position, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        tangent = glm::normalize(tangent);
        particles_host[i].velocity = tangent * perfectOrbitSpeed * instabilityFactor;
        particles_host[i].mass = (float)(rand() % 10 + 1);
    }
    const float dt = 0.016f;
    const float softeningFactor = 0.1f;

    // --- CUDA SETUP ---
    Particle* particles_device;
    checkCudaErrors(cudaMalloc((void**)&particles_device, numParticles * sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(particles_device, particles_host.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice));

    // --- VBOs for instance data (matrices and colors) ---
    unsigned int instanceMatrixVBO, instanceColorVBO;
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

    // --- Render Loop ---
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);

        // --- PHYSICS UPDATE ON GPU ---
        int threadsPerBlock = 256;
        int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(particles_device, numParticles, blackHole, G, dt, softeningFactor);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(particles_host.data(), particles_device, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost));
        
        // --- Particle Recycling ---
        for (int i = 0; i < numParticles; ++i) {
            if (glm::length(particles_host[i].position) < 1.0f) {
                float phi = glm::radians((float)(rand() % 36000) / 100.0f);
                float cos_theta = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                float theta = acos(cos_theta);
                float radius = 60.0f + (rand() % 2000) / 100.0f;
                particles_host[i].position.x = radius * sin(theta) * cos(phi);
                particles_host[i].position.y = radius * cos(theta);
                particles_host[i].position.z = radius * sin(theta) * sin(phi);
                float perfectOrbitSpeed = std::sqrt((G * blackHole.mass) / radius);
                glm::vec3 tangent = glm::cross(particles_host[i].position, glm::vec3(0.0f, 1.0f, 0.0f));
                if (glm::length(tangent) < 0.01f) { tangent = glm::cross(particles_host[i].position, glm::vec3(1.0f, 0.0f, 0.0f)); }
                tangent = glm::normalize(tangent);
                particles_host[i].velocity = tangent * perfectOrbitSpeed * instabilityFactor;
                checkCudaErrors(cudaMemcpy(&particles_device[i], &particles_host[i], sizeof(Particle), cudaMemcpyHostToDevice));
            }
        }

        // --- Update Instance Data for Rendering ---
        std::vector<glm::mat4> modelMatrices(numParticles);
        std::vector<glm::vec3> instanceColors(numParticles);
        glm::vec3 colorHot(1.0f, 0.3f, 0.1f);
        glm::vec3 colorCool(0.8f, 0.9f, 1.0f);
        float maxDist = 80.0f;
        for(int i = 0; i < numParticles; i++) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, particles_host[i].position);
            modelMatrices[i] = model;
            float dist = glm::length(particles_host[i].position);
            float blendFactor = glm::clamp(dist / maxDist, 0.0f, 1.0f);
            instanceColors[i] = glm::mix(colorHot, colorCool, blendFactor);
        }
        glBindBuffer(GL_ARRAY_BUFFER, instanceMatrixVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(glm::mat4), &modelMatrices[0]);
        glBindBuffer(GL_ARRAY_BUFFER, instanceColorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * sizeof(glm::vec3), &instanceColors[0]);

        // --- Rendering ---
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- Render Particles (Instanced) ---
        glUseProgram(instanceShaderProgram);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glUniformMatrix4fv(glGetUniformLocation(instanceShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(instanceShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniform3f(glGetUniformLocation(instanceShaderProgram, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
        glUniform3f(glGetUniformLocation(instanceShaderProgram, "lightPos"), 0.0f, 0.0f, 0.0f);
        glBindVertexArray(sphereVAO);
        glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, 0, numParticles);
        
        // --- (!!!) NEW: Render Black Hole (Single Object) (!!!) ---
        glUseProgram(simpleShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(simpleShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(simpleShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniform3f(glGetUniformLocation(simpleShaderProgram, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
        glUniform3f(glGetUniformLocation(simpleShaderProgram, "lightPos"), 0.0f, 0.0f, 0.0f);
        
        // Set model matrix and color for the black hole
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(8.0f)); // Make it much larger than the particles
        glUniformMatrix4fv(glGetUniformLocation(simpleShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(simpleShaderProgram, "objectColor"), 0.1f, 0.0f, 0.15f); // Dark purple
        
        glDrawElements(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --- Cleanup ---
    checkCudaErrors(cudaFree(particles_device));
    glDeleteVertexArrays(1, &sphereVAO);
    // ... delete other buffers ...
    glfwTerminate();
    return 0;
}

// --- Function Implementations ---
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
    float cameraSpeed = 40.0f * deltaTime; // Increased speed
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) cameraPos += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) cameraPos -= cameraSpeed * cameraUp;
}
void framebuffer_size_callback(GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX, yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    float sensitivity = 0.1f;
    xoffset *= sensitivity; yoffset *= sensitivity;
    yaw += xoffset; pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f; if (pitch < -89.0f) pitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

// --- Sphere Generation Function (Unchanged) ---
void generateSphere(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned int>& indices, float radius, int sectorCount, int stackCount) {
    float x, y, z, xy;
    float nx, ny, nz, lengthInv = 1.0f / radius;

    float sectorStep = 2 * PI / sectorCount;
    float stackStep = PI / stackCount;
    float sectorAngle, stackAngle;

    for(int i = 0; i <= stackCount; ++i) {
        stackAngle = PI / 2 - i * stackStep;
        xy = radius * cosf(stackAngle);
        z = radius * sinf(stackAngle);

        for(int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;
            x = xy * cosf(sectorAngle);
            y = xy * sinf(sectorAngle);
            vertices.push_back(glm::vec3(x, y, z));
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            normals.push_back(glm::vec3(nx, ny, nz));
        }
    }

    int k1, k2;
    for(int i = 0; i < stackCount; ++i) {
        k1 = i * (sectorCount + 1);
        k2 = k1 + sectorCount + 1;

        for(int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            if(i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }
            if(i != (stackCount-1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }
}
