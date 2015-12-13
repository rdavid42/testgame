
#ifndef CORE_HPP
# define CORE_HPP

# include "Mat4.hpp"
# include "Mat4Stack.hpp"
# include "Utils.hpp"
# include "Bmp.hpp"

class Core
{
public:
	/* glfw */
	GLFWwindow				*window;
	int						windowWidth;
	int						windowHeight;

	/* shaders */
	GLuint					vertexShader;
	GLuint					fragmentShader;
	GLuint					program;

	/* matrices */
	Mat4Stack<float>		ms;
	Mat4<float>				projMatrix;
	Mat4<float>				viewMatrix;

	/* camera */
	Vec3<float>				cameraPos;
	Vec3<float>				cameraLookAt;

	/* Locations */
	GLuint					projLoc;
	GLuint					viewLoc;
	GLuint					objLoc;
	GLuint					positionLoc;
	GLuint					colorLoc;

	GLuint					triangleVao;
	GLuint					triangleVbo;

	std::ostringstream		oss_ticks;

	Core(void);
	~Core(void);

	/* core */
	int						init(void);
	void					update(void);
	void					render(void);
	void					loop(void);

	/* textures */
	GLuint					loadTexture(char const *filename);

	/* matrices */ 
	void					setViewMatrix(Mat4<float> &view, Vec3<float> const &dir,
										Vec3<float> const &right, Vec3<float> const &up);
	void					setCamera(Mat4<float> &view, Vec3<float> const &pos, Vec3<float> const &lookAt);
	void					buildProjectionMatrix(Mat4<float> &proj, float const &fov,
												float const &near, float const &far);

	/* shaders */
	void					getLocations(void);
	int						compileShader(GLuint shader, char const *filename);
	GLuint					loadShader(GLenum type, char const *filename);
	int						loadShaders(void);
	int						linkProgram(GLuint &p);
	void					deleteShaders(void);
	int						initShaders(void);

	/* tests */
	void					initTriangle(void);

	Core &					operator=(Core const &rhs);

private:
	Core(Core const &src);
};

#endif
