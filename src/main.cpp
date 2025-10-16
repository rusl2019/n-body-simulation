#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <stdexcept>

#include "../include/graphics.h"
#include "../include/simulation.h"
#include "../include/camera.h"

// --- Global Variables & Constants ---
const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;
int NUM_PARTICLES = 16384;
glm::vec3 centerColor(0.1f, 0.0f, 0.15f);

Camera camera(glm::vec3(0.0f, 60.0f, 150.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -25.0f);
float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
bool firstMouse = true;

double deltaTime = 0.0;
double lastFrame = 0.0;

// --- Function Prototypes ---
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") {
            if (i + 1 < argc) {
                try {
                    NUM_PARTICLES = std::stoi(argv[++i]);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Invalid number for -n argument." << std::endl;
                    return -1;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Number for -n argument is out of range." << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "-n option requires one argument." << std::endl;
                return -1;
            }
        } else if (arg == "-c") {
            if (i + 3 < argc) {
                try {
                    centerColor.r = std::stof(argv[++i]);
                    centerColor.g = std::stof(argv[++i]);
                    centerColor.b = std::stof(argv[++i]);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Invalid number for -c argument." << std::endl;
                    return -1;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Number for -c argument is out of range." << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "-c option requires three float arguments (R G B)." << std::endl;
                return -1;
            }
        }
    }

    // --- GLFW/GLEW Initialization ---
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "CUDA N-Body Simulation", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    // --- Initialize Systems ---
    Graphics graphics(SCREEN_WIDTH, SCREEN_HEIGHT, NUM_PARTICLES);
    graphics.init();

    Simulation simulation(NUM_PARTICLES);
    simulation.init();

    // --- Render Loop ---
    double lastTime = glfwGetTime();
    int nbFrames = 0;
    lastFrame = lastTime;
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        deltaTime = currentTime - lastFrame;
        lastFrame = currentTime;

        nbFrames++;
        if ( currentTime - lastTime >= 1.0 ){
            char title[256];
            sprintf(title, "CUDA N-Body Simulation - %d FPS", nbFrames);
            glfwSetWindowTitle(window, title);
            nbFrames = 0;
            lastTime += 1.0;
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        camera.ProcessKeyboard(window, (float)deltaTime);

        simulation.update(0.016f); // Fixed timestep

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 500.0f);
        glm::mat4 view = camera.GetViewMatrix();
        graphics.render(simulation.getHostParticles(), view, projection, camera.Position, centerColor);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}