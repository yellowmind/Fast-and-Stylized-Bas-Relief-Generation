#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

#include "GLee.h"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <ANN/ANN.h>

#include <cv.h>
//#include <cvaux.h>
#include <highgui.h>

#include "swgl.h"
#include "math3d.h"
#include "glm.h"

//#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <set>
#include <ctime>

#include <omp.h>
#include "CVector3.h"
#include "PDSampling.h"

using namespace std;
using namespace cv;

#ifndef bool
#define bool int
#define false 0
#define true 1
#endif

#define HistogramBins  10000
#define ahe_m 32
int interval = 22,linewidth = 3;
#define SIDE 29
typedef pair<int, float> P;
typedef pair<int, int> Node;

float maxf(float a, float b)
{
    return (a < b) ? b : a;
} 
float minf(float a, float b)
{
    return (a < b) ? a : b;
}
inline float round( float d )
{
	return floor( d + 0.5 );
}
inline double log2(double x)
{
      static const double xxx = 1.0/log(2.0);
      return log(x)*xxx;
}

#ifndef M_PI
#define M_PI 3.14159
#endif

//using namespace cv;

int 	winWidth0, winHeight0, winWidth, winHeight;
int windowHeight = 552, horizontalSplit = 3;
int Window0, Window;
int boundary =20;
int disp=0;

float 	angle = 0.0, axis[3], trans[3];
bool 	trackingMouse = false;
bool 	redrawContinue = false;
bool    trackballMove = false;
GLdouble TRACKM[16]={1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
GLfloat m[16]={1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

GLdouble ClipNear = 0.1, ClipFar = 25;

GLdouble DEBUG_M[16];

GLdouble Angle1=0, Angle2=0;
GLint TICK=0;

GLMmodel *MODEL;
CVector3 farPoint, far2Point, nearPoint;

GLint scalingFactor = 16;
GLdouble MODELSCALE = 1.75;
GLdouble LIGHTP = 15;

bool scene=true, relief1=false, relief2=false , mesh1=false, mesh2=false, profile0=false, profile1=false, profile2=false, map1=true, map2=true, time1=false, time2=false;
float dynamicRange;
GLdouble angleX = 0,angleY = 0,angleZ = 0,scale =1;
GLdouble reliefAngleX = 0, reliefAngleY = 0, reliefAngleZ = 0;
GLdouble outputHeight = (windowHeight - boundary*2)/10240.0;//0.05;

GLfloat lightPos0[] = { -15.f, 15.0f, 15.0, 1.0f };
GLfloat lightPos1[] = { -25.f, 15.0f, outputHeight*81.25 + 0.4375, 1.0f };
GLfloat lightPos11[] = { 25.f, -15.0f, /*3131/*/(outputHeight*81.25 + 0.4375), 1.0f };
GLfloat lightPos2[] = { -27.f, 15.0f, outputHeight*81.25 + 0.4375, 1.0f };
GLfloat lightPos21[] = { 27.f, -15.0f, /*31*31/*/(outputHeight*81.25 + 0.4375), 1.0f };

GLfloat threshold = 0.001;
vector< vector<GLfloat> > heightList, laplaceList;
vector<GLfloat> maxList;
const int pyrLevel = 3;
float alpha[5] = {1, 1, 1, 1, 1};
float beta[5] = {0, 0, 0, 0, 0};

vector<bool> bgMask;
vector<GLfloat> outlineMask;
vector< vector<GLfloat > > heightPyr(pyrLevel);
CvMat **imgPyr;
IplImage *img0, *colorImg, *depthImg, *sample, *edge, *histImage1, *histImage2, *histImage3, *ImageL, *ImageLPrime, *GradientA, *carve, *carve2;
//vector<GLfloat> height;
vector<GLfloat> compressedH, referenceHeight, sceneProfile, reliefProfile;


//sampling density
const int n=4;
//float thetaS[n+1] = {1, 0.00001, 0.000001, 0.0000001, 0};
float thetaS[n+1] = {1, 0.00001, 0.000001, 0.0000001, 0};
float o = (interval - linewidth) * 1/3;
//float radius[n] = {o, o + 3, o + 5, o + 7};
float radius[n] = {30, 33, 35, 37};

GLdouble *pThreadRelief = NULL;
GLdouble *pThreadNormal = NULL;
GLdouble *pThreadEqualizeRelief = NULL;
GLdouble *pThreadEqualizeNormal = NULL;
GLint vertCount = 1;

int DRAWTYPE = 1;// 0:hw1, 1:hw2, 2:Gouraud shading, 3: Phong Shading
//int ReliefType = 1;// 0:no processing, 1:bilateral filtering,
int method = 3, reference = 2;//; ref1: gradient correction, ref2: original histogram, ref3: base histogram
//int method = 3, reference = 0;//; ref1: gradient correction, ref2: original histogram, ref3: base histogram
int numNonZero;
float lookat[9] = {0, 0, 4, 0, 0, 0, 0, 1, 0};
float perspective[4] = {60, 1, 0.1, 10};
int frame=0;
float samplingRatio = 0.02;
GLdouble projection[16], modelview[16], inverse[16];
GLint viewport[4];
/*----------------------------------------------------------------------*/
/*
** Draw the wireflame cube.
*/
GLfloat vertices[][3] = {
    {-1.0,-1.0,-1.0},{1.0,-1.0,-1.0}, {1.0,1.0,-1.0}, {-1.0,1.0,-1.0}, 
    {-1.0,-1.0,1.0}, {1.0,-1.0,1.0}, {1.0,1.0,1.0}, {-1.0,1.0,1.0}
};

GLfloat colors[][3] = {
    {0.0,0.0,0.0},{1.0,0.0,0.0}, {1.0,1.0,0.0}, {0.0,1.0,0.0}, 
    {0.0,0.0,1.0}, {1.0,0.0,1.0}, {1.0,1.0,1.0}, {0.0,1.0,1.0}
};


inline void SwglTri(GLdouble x1, GLdouble y1, GLdouble z1, 
			 GLdouble x2, GLdouble y2, GLdouble z2, 
			 GLdouble x3, GLdouble y3, GLdouble z3,
			 GLdouble nx1=1, GLdouble ny1=0, GLdouble nz1=0, 
			 GLdouble nx2=1, GLdouble ny2=0, GLdouble nz2=0, 
			 GLdouble nx3=1, GLdouble ny3=0, GLdouble nz3=0,
			 GLdouble r1=1, GLdouble g1=1, GLdouble b1=1,
			 GLdouble r2=1, GLdouble g2=1, GLdouble b2=1,
			 GLdouble r3=1, GLdouble g3=1, GLdouble b3=1)
{
	//copy to homogenous coordinate
	GLdouble h1[4]={x1, y1, z1, 1.0};
	GLdouble h2[4]={x2, y2, z2, 1.0};
	GLdouble h3[4]={x3, y3, z3, 1.0};

	//window coordinate
	GLdouble w1[4]={x1, y1, 0, 1.0}; 
	GLdouble w2[4]={x2, y2, 0, 1.0};
	GLdouble w3[4]={x3, y3, 0, 1.0};

	//implement the opengl pipeline here
	swTransformation(h1, w1);
	swTransformation(h2, w2);
	swTransformation(h3, w3);

	switch(DRAWTYPE) {
		case 0:
		{
			//copy to homogenous coordinate
			GLdouble h1[4]={x1, y1, z1, 1.0};
			GLdouble h2[4]={x2, y2, z2, 1.0};
			GLdouble h3[4]={x3, y3, z3, 1.0};

			//window coordinate
			GLdouble w1[4]={x1, y1, 0, 1.0}; 
			GLdouble w2[4]={x2, y2, 0, 1.0};
			GLdouble w3[4]={x3, y3, 0, 1.0};

			//implement the opengl pipeline here
			swTransformation(h1, w1);
			swTransformation(h2, w2);
			swTransformation(h3, w3);

			writepixel(w1[0], w1[1], r1, g1, b1);
			writepixel(w2[0], w2[1], r2, g2, b2);
			writepixel(w3[0], w3[1], r3, g3, b3);
		}
		break;

		case 1:
		{
			//copy to homogenous coordinate
			GLdouble h1[4]={x1, y1, z1, 1.0};
			GLdouble h2[4]={x2, y2, z2, 1.0};
			GLdouble h3[4]={x3, y3, z3, 1.0};

			//window coordinate
			GLdouble w1[4]={x1, y1, 0, 1.0}; 
			GLdouble w2[4]={x2, y2, 0, 1.0};
			GLdouble w3[4]={x3, y3, 0, 1.0};

			//implement the opengl pipeline here
			swTransformation(h1, w1);
			swTransformation(h2, w2);
			swTransformation(h3, w3);

			swTriangle(w1[0], w1[1], w1[2],
					   w2[0], w2[1], w2[2],
					   w3[0], w3[1], w3[2],
					   r1, g1, b1);
		}
		break;

		case 2:
		{
			swTriangleG(x1, y1, z1,
						x2, y2, z2,
						x3, y3, z3,
						nx1, ny1, nz1,
						nx2, ny2, nz2,
						nx3, ny3, nz3,
						r1, g1, b1,
						r2, g2, b2,
						r3, g3, b3);
		}
		break;

		case 3:
		{
			swTriangleP(x1, y1, z1,
						x2, y2, z2,
						x3, y3, z3,
						nx1, ny1, nz1,
						nx2, ny2, nz2,
						nx3, ny3, nz3,
						r1, g1, b1,
						r2, g2, b2,
						r3, g3, b3);
		}
		break;

	}

}

void SwglTri(int index1, int index2, int index3)
{
	SwglTri( vertices[index1][0], vertices[index1][1], vertices[index1][2],
		     vertices[index2][0], vertices[index2][1], vertices[index2][2],
			 vertices[index3][0], vertices[index3][1], vertices[index3][2]);
}


void SwglLine(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2)
{
	//copy to homogenous coordinate
	GLdouble h1[4]={x1, y1, z1, 1.0};
	GLdouble h2[4]={x2, y2, z2, 1.0};
	
	GLdouble w1[4]={x1, y1, 0, 1.0}; //window coordinate
	GLdouble w2[4]={x2, y2, 0, 1.0};

	//implement the opengl pipeline here
	swTransformation(h1, w1);
	swTransformation(h2, w2);
	
	////draw the 2D line
	//glBegin(GL_LINES);
	//	//glColor3fv(colors[index1]);
	//	glVertex2f(w1[0], w1[1]);
	//	//glColor3fv(colors[index2]);
	//	glVertex2f(w2[0], w2[1]);
	//glEnd();

	//implement 
	switch(DRAWTYPE) {
		case 0:
		{
			writepixel(w1[0], w1[1], 1, 0, 0);
			writepixel(w2[0], w2[1], 0, 1, 0);
		}
		break;

		case 1: case 2: case 3:
		{
			GLdouble col[4];
			glGetDoublev(GL_CURRENT_COLOR, col);
			BresenhamLine(w1[0], w1[1], w2[0], w2[1], col[0], col[1], col[2]);
		}
		break;
	}
}

void SwglLine(int index1, int index2)
{
	SwglLine(vertices[index1][0], vertices[index1][1], vertices[index1][2],
		     vertices[index2][0], vertices[index2][1], vertices[index2][2]);
}


void SolidQuad(int a, int b, int c, int d, bool USINGOPENGL)
{
	if(USINGOPENGL) {
		glBegin(GL_TRIANGLES);
			glVertex3fv(vertices[a]);
			glVertex3fv(vertices[b]);
			glVertex3fv(vertices[c]);

			glVertex3fv(vertices[c]);
			glVertex3fv(vertices[d]);
			glVertex3fv(vertices[a]);
		glEnd();
	} else {
		SwglTri(a, b, c);
		SwglTri(c, d, a);
	}
}

void swSolidCube(void)
{
    // map vertices to faces */
    SolidQuad(1,0,3,2, false);
    SolidQuad(3,7,6,2, false);
    SolidQuad(7,3,0,4, false);
    SolidQuad(2,6,5,1, false);
    SolidQuad(4,5,6,7, false);
    SolidQuad(5,4,0,1, false);
}

void glSolidCube(void)
{
    // map vertices to faces */
    SolidQuad(1,0,3,2, true);
    SolidQuad(3,7,6,2, true);
    SolidQuad(7,3,0,4, true);
    SolidQuad(2,6,5,1, true);
    SolidQuad(4,5,6,7, true);
    SolidQuad(5,4,0,1, true);
}



void OpenglLine(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2)
{
	glBegin(GL_LINES);
		glVertex3f(x1, y1, z1);
		glVertex3f(x2, y2, z2);
	glEnd();
}

void OpenglLine(int index1, int index2)
{
	OpenglLine(vertices[index1][0], vertices[index1][1], vertices[index1][2],
		       vertices[index2][0], vertices[index2][1], vertices[index2][2]);
}

void WireQuad(int a, int b, int c , int d, bool USINGOPENGL)
{
	if(USINGOPENGL) {
		OpenglLine(a, b);
		OpenglLine(b, c);
		OpenglLine(c, d);
		OpenglLine(d, a);
	} else {
		SwglLine(a, b);
		SwglLine(b, c);
		SwglLine(c, d);
		SwglLine(d, a);
	}
}

void swWireCube(void)
{
    // map vertices to faces */
    WireQuad(1,0,3,2, false);
    WireQuad(3,7,6,2, false);
    WireQuad(7,3,0,4, false);
    WireQuad(2,6,5,1, false);
    WireQuad(4,5,6,7, false);
    WireQuad(5,4,0,1, false);
}

void glWireCube(void)
{
    // map vertices to faces */
    WireQuad(1,0,3,2, true);
    WireQuad(3,7,6,2, true);
    WireQuad(7,3,0,4, true);
    WireQuad(2,6,5,1, true);
    WireQuad(4,5,6,7, true);
    WireQuad(5,4,0,1, true);
}

void polygon(int a, int b, int c , int d, int face)
{
    /* draw a polygon via list of vertices */
    glBegin(GL_POLYGON);
  	glColor3fv(colors[a]);
  	glVertex3fv(vertices[a]);
  	glColor3fv(colors[b]);
  	glVertex3fv(vertices[b]);
  	glColor3fv(colors[c]);
  	glVertex3fv(vertices[c]);
  	glColor3fv(colors[d]);
  	glVertex3fv(vertices[d]);
    glEnd();
}

void colorcube(void)
{

    /* map vertices to faces */
    polygon(1,0,3,2,0);
    polygon(3,7,6,2,1);
    polygon(7,3,0,4,2);
    polygon(2,6,5,1,3);
    polygon(4,5,6,7,4);
    polygon(5,4,0,1,5);
}

GLvoid swglmDraw(GLMmodel* model)
{ 	         
	GLfloat *n1, *n2, *n3; 
	GLfloat *v1, *v2, *v3;

	//get current color
	GLdouble col[4];
	glGetDoublev(GL_CURRENT_COLOR, col);

    for (unsigned int i = 0; i < model->numtriangles; i++) {
        GLMtriangle* triangle = &(model->triangles[i]);

        n1 = &model->normals[3 * triangle->nindices[0]];
        v1 = &model->vertices[3 * triangle->vindices[0]];
        n2 = &model->normals[3 * triangle->nindices[1]];
        v2 = &model->vertices[3 * triangle->vindices[1]];        
		n3 = &model->normals[3 * triangle->nindices[2]];
        v3 = &model->vertices[3 * triangle->vindices[2]];

		SwglTri(v1[0], v1[1], v1[2],
			    v2[0], v2[1], v2[2],
				v3[0], v3[1], v3[2],
				n1[0], n1[1], n1[2],
			    n2[0], n2[1], n2[2],
				n3[0], n3[1], n3[2],
				col[0], col[1], col[2],
			    col[0], col[1], col[2],
				col[0], col[1], col[2]);   
    }

}

void ReduceToUnit(double vector[3])					// Reduces A Normal Vector (3 Coordinates)
{									// To A Unit Normal Vector With A Length Of One.
	double length;							// Holds Unit Length
	// Calculates The Length Of The Vector
	length = (double)sqrt((vector[0]*vector[0]) + (vector[1]*vector[1]) + (vector[2]*vector[2]));

	if(length == 0.0f)						// Prevents Divide By 0 Error By Providing
		length = 1.0f;						// An Acceptable Value For Vectors To Close To 0.

	vector[0] /= length;						// Dividing Each Element By
	vector[1] /= length;						// The Length Results In A
	vector[2] /= length;						// Unit Normal Vector.
}

void setNormal(double v[3][3], double out[3])				// Calculates Normal For A Quad Using 3 Points
{
	double v1[3],v2[3];						// Vector 1 (x,y,z) & Vector 2 (x,y,z)
	static const int x = 0;						// Define X Coord
	static const int y = 1;						// Define Y Coord
	static const int z = 2;						// Define Z Coord

	// Finds The Vector Between 2 Points By Subtracting
	// The x,y,z Coordinates From One Point To Another.

	// Calculate The Vector From Point 1 To Point 0
	v1[x] = v[0][x] - v[1][x];					// Vector 1.x=Vertex[0].x-Vertex[1].x
	v1[y] = v[0][y] - v[1][y];					// Vector 1.y=Vertex[0].y-Vertex[1].y
	v1[z] = v[0][z] - v[1][z];					// Vector 1.z=Vertex[0].y-Vertex[1].z
	// Calculate The Vector From Point 2 To Point 1
	v2[x] = v[1][x] - v[2][x];					// Vector 2.x=Vertex[0].x-Vertex[1].x
	v2[y] = v[1][y] - v[2][y];					// Vector 2.y=Vertex[0].y-Vertex[1].y
	v2[z] = v[1][z] - v[2][z];					// Vector 2.z=Vertex[0].z-Vertex[1].z
	// Compute The Cross Product To Give Us A Surface Normal
	out[x] = v1[y]*v2[z] - v1[z]*v2[y];				// Cross Product For Y - Z
	out[y] = v1[z]*v2[x] - v1[x]*v2[z];				// Cross Product For X - Z
	out[z] = v1[x]*v2[y] - v1[y]*v2[x];				// Cross Product For X - Y

	ReduceToUnit(out);						// Normalize The Vectors
}

template <class T>
void Relief2Image(vector<T> src, IplImage *dst, int height=0)
{
	if(height)
	{

	}
	else
	{
		height = sqrt( (float) src.size() );
	}
	int width = src.size()/ height;

	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			cvSetReal2D( dst, height - 1 - j, i, (double) src[ i*height + j ] );
		}
	}
}

//void Relief2Image(vector<GLfloat> src, IplImage *dst)
//{
//	int width = sqrt( (float) src.size() );
//	int height = sqrt( (float) src.size() );
//
//	
//	for(int i=0; i < width; i++)
//	{
//		for(int j=0; j < height; j++)
//		{
//			cvSetReal2D( dst, height - 1 - j, i, (double) src.at( i*height + j ) );
//		}
//	}
//}

template <class T>
void Image2Relief(IplImage *src, vector<T> &dst)
{
	int width = cvGetSize(src).width;
	int height = cvGetSize(src).height;

	dst.clear();
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			dst.push_back( (T)cvGetReal2D( src, height - 1 - j, i) );
		}
	}
}

//void Image2Relief(IplImage *src, vector<GLdouble> &dst)
//{
//	int width = cvGetSize(src).width;
//	int height = cvGetSize(src).height;
//
//	dst.clear();
//	
//	for(int i=0; i < width; i++)
//	{
//		for(int j=0; j < height; j++)
//		{
//			dst.push_back( cvGetReal2D( src, height - 1 - j, i) );
//		}
//	}
//}

void subSample(vector<GLfloat> array1, vector<GLfloat> &array2, int height =0)
{
	/*int width = sqrt( (float) array1.size() );
	int height = sqrt( (float) array1.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) array1.size() );
	}

	int width;
	if(height)
	{
		width = array1.size()/height;
	}
	else
	{
		width = 0;
	}
	
	GLfloat sum;
	for(int i=0; i < width-1; i+=2)
	{
		for(int j=0; j < height-1; j+=2)
		{
			sum = array1.at( i*height + j ) + array1.at( i*height + j+1 ) + array1.at( (i+1)*height + j ) + array1.at( (i+1)*height + j+1 );
			array2.push_back(sum/4);		
		}
	}
}

void upSample(vector<GLfloat> array1, vector<GLfloat> &array2, int height =0)
{
	/*int width = sqrt( (float) array1.size() );
	int height = sqrt( (float) array1.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) array1.size() );
	}

	int width;
	if(height)
	{
		width = array1.size()/height;
	}
	else
	{
		width = 0;
	}

	float A,B,C,D;
	int x, y, index;
	float x_ratio = ( (float)(width-1) ) / (width*2 );
	float y_ratio = ( (float)(height-1) ) / (height*2);
	float x_diff, y_diff;
	
	for(int i=0; i < width*2; i++)
	{
		for(int j=0; j < height*2; j++)
		{
			x = (int)(x_ratio * i) ;
            y = (int)(y_ratio * j) ;
			x_diff = (x_ratio * i) - x ;
            y_diff = (y_ratio * j) - y ;
            index = y + x*height ;
			
			A = array1.at( index );
            B = array1.at( index+height );
            C = array1.at( index+1 );
            D = array1.at( index+height+1 );

			array2.push_back( A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) + C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff) );
		}

		/*for(int j=0; j<height; j++)
		{
			array2.push_back( array1.at( i*height + j ) );	
			array2.push_back( array1.at( i*height + j ) );		
		}*/

		/*for(int j=0; j<height; j++)
		{
			array2.push_back( array1.at( i*height + j ) );	
			array2.push_back( array1.at( i*height + j ) );		
		}*/
	}
}

void reliefSub(vector<GLfloat> &src1, vector<GLfloat> &src2, vector<GLfloat> &dst, int height=0)
{
	/*int width = sqrt( (float) src1.size() );
	int  height = sqrt( (float) src1.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) src1.size() );
	}
	int width = src1.size()/height;
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			int adress = i*height + j;
			dst.push_back( src1.at( adress ) - src2.at( adress ) );	
		}
	}
}

void reliefAdd(vector<GLfloat> &src1, vector<GLfloat> &src2, int height=0)
{
	/*int width = sqrt( (float) src1.size() );
	int  height = sqrt( (float) src1.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) src1.size() );
	}
	int width = src1.size()/height;
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			int adress = i*height + j;
			src1[adress] = src1.at( adress ) + src2.at( adress );	
		}
	}
}

void reliefAdd(vector<GLfloat> &src1, vector<GLfloat> &src2, vector<GLfloat> &dst, int height=0)
{
	/*int width = sqrt( (float) src1.size() );
	int  height = sqrt( (float) src1.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) src1.size() );
	}
	int width = src1.size()/height;
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			int adress = i*height + j;
			dst.push_back( src1.at( adress ) + src2.at( adress ) );	
		}
	}
}

void remapping(vector<GLfloat> &dst, GLfloat g, GLfloat threshold)	//range compression
{
	for(int k=0; k < dst.size(); k++)
	{
		dst[k] = minf( maxf( dst.at(k), g - threshold), g + threshold );
	}
}

GLfloat sign(GLfloat value)
{
	if(value > 0) return 1;
	else if(value < 0) return -1;
	else return 0;
}

double compress(double x, double alpha=0.5)
{
	//if(x == 0)	return 0;
	
	double c;
	c = log10(1+alpha*x) / alpha;
	return c;
}

void compress(IplImage *src, float alpha=0.5)
{
	//IplImage *sign = cvCreateImage( cvGetSize(src), IPL_DEPTH_64F, 1);

	int width =src->width;
	int height = src->height;

	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			double value = cvGetReal2D( src, j, i);
			if( value == 0)
			{
				//cvSetReal2D( src, j, i, 0 );
			}
			else if( value >0 )
			{
				//cvSetReal2D( sign, j, i, 1 );
				cvSetReal2D( src, j, i, log10( value*alpha + 1 ) / alpha );
			}
			else
			{
				//cvSetReal2D( sign, j, i, -1 );
				cvSetReal2D( src, j, i, -log10( -value*alpha + 1 ) / alpha );
			}
		}
	}

	/*int absMask = 0x7fffffff;
	cvAndS(&src, cvRealScalar(*(float*)&absMask), &src, 0);*/
	//cvConvertScale(src, src, alpha, 1);
	//cvLog(src, src);
	/*cvConvertScale(src, src, 1/alpha, 0);*/

	//cvMul(src, sign, src, 1/alpha);
}

double compress(double x, double alpha, int n)
{
	double c;
	for(int i=0; i<n; i++)
	{
		c = log( 1 + alpha*abs(x) ) / alpha;
		if(c < 0) break;
		x = c;
	}
	return c;
}

GLfloat fdetail(GLfloat delta, GLfloat alpha)
{
	return pow(delta, alpha);
}

GLfloat fdetail(GLfloat delta, GLfloat alpha, GLfloat maxIntensity)
{
	if(delta == 0)
	{
		return 0;
	}
	
	float tau = delta / maxIntensity / 0.01;	
	if(tau <= 1)
	{
		tau = 0;
	}
	else if(tau > 2)
	{
		tau = 1;
	}
	else
	{
		tau -= 1;
	}
	return tau*pow(delta, alpha) + (1-tau)*delta;
}

GLfloat fedge(GLfloat delta, GLfloat beta)
{
	return delta*beta;
}

//GLfloat fedge(GLfloat delta, GLfloat beta)
//{
//	return compress( delta, beta );
//}

//GLfloat fedge(GLfloat delta, GLfloat beta)
//{
//	return compress( delta+threshold, beta ) - compress( threshold, beta );
//}


void remapping(vector<GLfloat> &dst, GLfloat g, GLfloat threshold, GLfloat alpha, GLfloat beta=0.25)	//general remapping
{
	
	for(int k=0; k < dst.size(); k++)
	{
		if( abs( dst[k] - g ) <= threshold )	//detail cases
		{
			dst[k] = g + sign( dst.at(k) -g ) * threshold * fdetail( abs(dst[k] - g) / threshold, alpha, threshold);
		}
		else		//edge cases
		{
			dst[k] = g + sign( dst.at(k) - g ) * (fedge( abs(dst[k] - g) - threshold, beta ) + threshold);
		}
	}
}

void updateLaplace(vector<GLfloat> src1, vector<GLfloat> src2, vector<GLfloat*> dst, int height=0)
{
	/*int width = sqrt( (float) src2.size() );
	int  height = sqrt( (float) src2.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) src2.size() );
	}

	int width;
	if(height)
	{
		width = src2.size()/height;
	}
	else
	{
		width = 0;
	}
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			int adress = i*height + j;
			if( *dst.at(adress) == 0 )
			{
				*dst.at(adress) =  (float) ( src1.at( adress ) - src2.at( adress ) );
			}
		}
	}
}

void laplacianFilter(vector<GLfloat> src, vector<GLfloat> &dst, float alpha, float beta, int height=0, int aperture=7)
{
	if(height)
	{}
	else
	{
		height = sqrt( (float) src.size() );
	}
	int width = src.size() / height;

	dst.clear();
	//vector<GLfloat> *m_src;
	//vector<GLfloat> laplace;
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			dst.push_back( 0 );
		}
	}

	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			/*if(bgMask[i*height+j])
			{
				continue;
			}*/
			
			//dst.push_back( src1.at( adress ) - src2.at( adress ) );
			vector<GLfloat> sub;
			vector<GLfloat*> subLaplace;

			int ext = (aperture-1) / 2;
			if(i >= ext  && j >= ext && i+ext+1 < width && j+ext+1 < height)
			{
				for(int p=-ext; p <=ext+1; p++)
				{
					for(int q=-ext; q <=ext+1; q++)
					{
						//window[ aperture * (p+ext)  +  q+ext ] = cvGetReal2D(src, i+p, j+q);
						sub.push_back(  src.at( (i+p)*height + j+q )  );
						subLaplace.push_back(  &dst.at( (i+p)*height + j+q )  );
					}
				}
			}

			/*else	//boundary issues
			{
				int extU = -ext,extD = ext,extL = -ext,extR = ext;
				if( ext > j)		extU = -j;
				if( ext > i)		extL = -i;
				if( j+ext >= height)		extD = height - 1 - j;
				if( i+ext >= width)		extR =  width - 1 - i;

				//int window[ (extR - extL + 1) * (extD - extU +1) ];
				
				for(int p=extL; p <= extR; p++)
				{
					for(int q=extU; q <= extD; q++)
					{
						//window[ (extR-extL+1) * (p-extU)  +  q-extL ] = cvGetReal2D(src, i+p, j+q);
						sub.push_back(  src.at( (i+p)*height + j+q )  );
						subLaplace.push_back(  &dst.at( (i+p)*height + j+q )  );
					}
				}
			}*/
			//Remapping
			float g = src.at( i*height + j );
			remapping(sub, g, ::threshold, alpha, beta);
			/*for(int k=0; k < sub.size(); k++)
			{			
				sub[k] = minf( maxf( sub.at(k), g - threshold), g + threshold );
			}*/

			vector<GLfloat>  sub2, upSub;
			/*IplImage *img0, *img1, *imgUp;
			
			int width = sqrt( (float)sub.size() );
			img0 = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);	
			Relief2Image(sub, img0);
			
			img1 = cvCreateImage( cvSize( ( width+1)/2, (height+1)/2 ), IPL_DEPTH_32F, 1);
			imgUp = cvCreateImage( cvSize( width,  height), IPL_DEPTH_32F, 1);
			cvPyrDown(img0, img1);*/
			/*for(int i=0; i < img1->width; i++)
			{
				for(int j=0; j < img1->height; j++)
				{
					double value = cvGetReal2D( img1, j, i);
				}
			}*/
			
			/*cvPyrUp(img1, imgUp);
			Image2Relief(imgUp, upSub);*/
			
			
			subSample( sub, sub2 );
			upSample( sub2, upSub );

			
			
			//Image2Relief(imgUp, upSub);
			updateLaplace( sub, upSub, subLaplace);

			/*subSample( sub2, sub3 );
			upSample( sub3, upSub );
			reliefSub( sub2, upSub, subLaplace2);*/
		}
	}
}

void bgFilter(vector<float> &src, vector<bool> mask, int height=0)
{
	/*int width = sqrt( (float) src.size() );
	int height = sqrt( (float) src.size() );*/
	if(height)
	{}
	else
	{
		height = sqrt( (float) src.size() );
	}
	int width = src.size()/height;
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( mask.at(i*height + j) && src.at(i*height + j) != 0 )
			{
				src.at(i*height + j) = 0;
			}
		}
	}
}

void bgFilter(IplImage *src, IplImage  *mask)
{	
	int width =src->width;
	int height = src->height;

	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( cvGetReal2D( mask, j, i) && cvGetReal2D( src, j, i) )
			{
				cvSetReal2D( src, j, i, 0 );
			}
		}
	}
}

void extractOutline(vector<float> src, vector<float> &dst, int height=0, int boolean=0)
{
	dst.clear();
	for(int i=0; i < src.size(); i++)
	{
		dst.push_back(0);
	}	
	
	if(height)
	{
	}
	else
	{
		height = sqrt( (float) src.size() );
	}
	int width = src.size() / height;
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( src.at(i*height + j) != 0)
			{
				//not avoiding boundary issues
				if(	src.at( i*height + (j+1) ) == 0 ||
					src.at( (i+1)*height + j ) == 0 ||
					src.at( i*height + (j-1) ) == 0 ||
					src.at( (i-1)*height + j ) == 0 )
				{
					if(boolean)
					{
						dst.at( i*height + j ) = 1;
					}
					else
					{
						dst.at( i*height + j ) = src.at(i*height + j);
					}
				}
			}
		}
	}
}

void extractOutline(IplImage *src, IplImage *dst, bool boolean=0)
{
	cvSetZero(dst);
	
	int width = src->width;
	int height = src->height;
	
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( cvGetReal2D( src, j, i)  != 0)
			{
				//not avoiding boundary issues
				int up=0,down=0,left=0,right=0;
				if( i-1 > 0 )
				{
					right=1;
				}
				if( i+1 < width )
				{
					left=1;
				}
				if( j-1 > 0 )
				{
					down=1;
				}
				if( j+1 < height )
				{
					up=1;
				}

				if(	cvGetReal2D( src, j+up, i ) == 0 ||
					cvGetReal2D( src, j, i+left ) == 0 ||
					cvGetReal2D( src, j-down, i) == 0 ||
					cvGetReal2D( src, j, i-right ) == 0 )
				{
					if(boolean)
					{
						cvSetReal2D( dst, j, i, 1 );
					}
					else
					{
						cvSetReal2D( dst, j, i, cvGetReal2D( src, j, i) );
					}
				}
			}
		}
	}
}


void recordMax(vector<GLfloat> v)
{
	vector<GLfloat>::iterator first = v.begin();
	vector<GLfloat>::iterator last = v.end();
	maxList.push_back( *max_element ( first, last ) );
}

void BuildRelief(vector<GLfloat> &height, GLdouble *pThreadRelief, GLdouble *pThreadNormal)
{
	for(int i=0; i < /*height.size() / (winHeight - boundary*2)*/winWidth/horizontalSplit - boundary*2 - 1; i++)
		{
			for(int j=0; j < winHeight - boundary*2 -1; j++)
			{
				int position = i*(winHeight-boundary*2) + j;
				
				GLdouble normal[3];
				GLdouble v1[3][3] =  { {i, j, height.at(position)}, {i+1, j, height.at(position+winHeight - boundary*2)}, { i, j+1, height.at(position+1)} };
				
				//glBegin(GL_TRIANGLES);
					setNormal( v1, normal );
					glNormal3dv(normal);
					memcpy( ( pThreadNormal + ( i*(winHeight - boundary*2)*2 + j*2 ) * 3 ), normal, sizeof(GLdouble)*3 );
					/*glVertex3d( i, j, height.at(position) );
					glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );
					glVertex3d( i, j+1, height.at(position+1) );*/
					memcpy( ( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 ), v1[0], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 ), v1[1], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 ), v1[2], sizeof(GLdouble)*3 );
					
				GLdouble v2[3][3] = { {i, j+1, height.at(position+1)}, {i+1, j, height.at(position+winHeight - boundary*2)}, {i+1, j+1, height.at(position+winHeight - boundary*2 + 1)} };
					setNormal( v2, normal);
					glNormal3dv(normal);
					memcpy( ( pThreadNormal + ( i*(winHeight - boundary*2)*2 + j*2 ) * 3 + 3), normal, sizeof(GLdouble)*3 );
					/*glVertex3d( i, j+1, height.at(position+1)  );
					glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );				
					glVertex3d( i+1, j+1, height.at(position+winHeight - boundary*2 + 1) );*/
					memcpy( ( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 ), v2[0], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 ), v2[1], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3 ), v2[2], sizeof(GLdouble)*3 );
				//glEnd();
				//glColor3f(1.0, 0.0, 0.0);
			}
		}
}

void BuildRelief(vector<GLfloat> *height, GLdouble *pThreadRelief, GLdouble *pThreadNormal)
{
	for(int i=0; i < height->size() / (winHeight - boundary*2) - 1; i++)
		{
			for(int j=0; j < winHeight - boundary*2 -1; j++)
			{
				int position = i*(winHeight-boundary*2) + j;
				
				GLdouble normal[3];
				GLdouble v1[3][3] =  { {i, j, height->at(position)}, {i+1, j, height->at(position+winHeight - boundary*2)}, { i, j+1, height->at(position+1)} };
				
				//glBegin(GL_TRIANGLES);
					setNormal( v1, normal );
					glNormal3dv(normal);
					memcpy( ( pThreadNormal + ( i*(winHeight - boundary*2)*2 + j*2 ) * 3 ), normal, sizeof(GLdouble)*3 );
					/*glVertex3d( i, j, height.at(position) );
					glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );
					glVertex3d( i, j+1, height.at(position+1) );*/
					memcpy( ( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 ), v1[0], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 ), v1[1], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 ), v1[2], sizeof(GLdouble)*3 );
					
				GLdouble v2[3][3] = { {i, j+1, height->at(position+1)}, {i+1, j, height->at(position+winHeight - boundary*2)}, {i+1, j+1, height->at(position+winHeight - boundary*2 + 1)} };
					setNormal( v2, normal);
					glNormal3dv(normal);
					memcpy( ( pThreadNormal + ( i*(winHeight - boundary*2)*2 + j*2 ) * 3 + 3), normal, sizeof(GLdouble)*3 );
					/*glVertex3d( i, j+1, height.at(position+1)  );
					glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );				
					glVertex3d( i+1, j+1, height.at(position+winHeight - boundary*2 + 1) );*/
					memcpy( ( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 ), v2[0], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 ), v2[1], sizeof(GLdouble)*3 );
					memcpy( ( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3 ), v2[2], sizeof(GLdouble)*3 );
				//glEnd();
				//glColor3f(1.0, 0.0, 0.0);
			}
		}
}

inline void DrawProfile(const vector<GLfloat> &src, IplImage *dst, float ratio=0.3)
{
	for(int i=0; i < src.size()-1; i++)
	{
		cvLine( dst, cvPoint( i, dst->height * ratio * (1 - src[i]) + 1 ), cvPoint( i+1, dst->height *  ratio * (1 - src[i+1]) +1 ), cvScalarAll(0) );
	}
}

//1st Laplace
void firstLaplace(void)
{	
	
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/

	glPushMatrix();
		//glPolygonMode(GL_FRONT,GL_LINE);
		//multiple trackball matrix
		glMultMatrixd(TRACKM);
		
		
		GLfloat maxDepth=0,minDepth=1;
		if(relief1)
		{
			disp++;
			for(int i=0; i < pyrLevel; i++)
			{
				heightPyr[i].clear();
			}
			bgMask.clear();
			
			colorImg = cvCreateImage( cvSize(winWidth/3 - 2*boundary, winHeight - 2*boundary),  IPL_DEPTH_8U, 3);
			depthImg = cvCreateImage( cvSize(winWidth/3 - 2*boundary, winHeight - 2*boundary),  IPL_DEPTH_8U, 3);
			IplImage *bImg = cvCreateImage( cvGetSize(colorImg),  IPL_DEPTH_8U, 1);
			IplImage *gImg = cvCreateImage( cvGetSize(colorImg),  IPL_DEPTH_8U, 1);
			IplImage *rImg = cvCreateImage( cvGetSize(colorImg),  IPL_DEPTH_8U, 1);
			IplImage *depthImg1 = cvCreateImage( cvGetSize(depthImg),  IPL_DEPTH_8U, 1);
			IplImage *depthImg2 = cvCreateImage( cvGetSize(depthImg),  IPL_DEPTH_8U, 1);
			IplImage *depthImg3 = cvCreateImage( cvGetSize(depthImg),  IPL_DEPTH_8U, 1);

			int size = winWidth/3*winHeight;

			GLubyte *red = new GLubyte[size];
			GLubyte *green = new GLubyte[size];
			GLubyte *blue = new GLubyte[size];
			glReadPixels(0, 0, winWidth/3, winHeight, GL_RED, GL_UNSIGNED_BYTE, red);
			glReadPixels(0, 0, winWidth/3, winHeight, GL_GREEN, GL_UNSIGNED_BYTE, green);
			glReadPixels(0, 0, winWidth/3, winHeight, GL_BLUE, GL_UNSIGNED_BYTE, blue);

			for(int i=boundary; i<winWidth/3 - boundary; i++)
			{
				for(int j=boundary; j<winHeight - boundary; j++)
				{					
					cvSetReal2D( bImg, winHeight - 1 - j - boundary, i - boundary, blue[ j*winWidth/3 + i ]);
					cvSetReal2D( gImg, winHeight - 1 - j - boundary, i - boundary, green[ j*winWidth/3 + i ]);
					cvSetReal2D( rImg, winHeight - 1 - j - boundary, i - boundary, red[ j*winWidth/3 + i ]);
				}
			}

			cvMerge(bImg, gImg, rImg, 0, colorImg);

			float *depthmap = new float[size];
			glReadPixels(0+frame, 0+frame, winWidth/3-frame, winHeight-frame, GL_DEPTH_COMPONENT, GL_FLOAT, depthmap);

			for(int i=boundary; i<winWidth/3 - boundary; i++)
			{
				for(int j=boundary; j<winHeight - boundary; j++)
				{
					
					cvSetReal2D( bImg, winHeight - 1 - j - boundary, i - boundary, blue[ j*winWidth/3 + i ]);
					cvSetReal2D( gImg, winHeight - 1 - j - boundary, i - boundary, green[ j*winWidth/3 + i ]);
					cvSetReal2D( rImg, winHeight - 1 - j - boundary, i - boundary, red[ j*winWidth/3 + i ]);

					GLfloat depth = depthmap[ j*winWidth/3 + i ];

					//glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

					if( maxDepth < depth && depth !=1)
					{
						maxDepth = depth;
					}
					if( minDepth > depth)
					{
						minDepth = depth;
					}
					heightPyr[0].push_back(depth);
					bgMask.push_back(0);
				}
			}

			cvMerge(bImg, gImg, rImg, 0, colorImg);

			delete [] depthmap;

			//dynamicRange = (1-minDepth)/(1-maxDepth);

			int enhance = 50;
			double maxDepthPrime = compress(maxDepth*enhance, 5, 5);
			double minDepthPrime = compress(minDepth*enhance, 5, 5);

			for(int i=0; i<heightPyr[0].size(); i++)
			{			
				if ( heightPyr[0].at(i) == 1 )	//infinite point
				{
					heightPyr[0].at(i) = 0;
					bgMask.at(i) = 1;
				}
				else
				{
					/*height.at(i) = compress(height.at(i)*enhance, 5, 5);
					height.at(i) = maxDepthPrime - height.at(i);
					//height.at(i) = maxDepth - height.at(i);	//transfer depth to height
					height.at(i) /= (maxDepthPrime - minDepthPrime );	//normalize to [0,1]*/

					heightPyr[0].at(i) = maxDepth - heightPyr[0].at(i);
					//height.at(i) = maxDepth - height.at(i);	//transfer depth to height
					heightPyr[0].at(i) /= (maxDepth - minDepth);	//normalize to [0,1]
				}
			}

			//extractOutlineo(vector<bool> src, vector<bool> dst);

			heightList.clear();
			vector<GLfloat> height2, upHeight, laplace;
			heightList.push_back(heightPyr[0]);
			IplImage *img1, *imgUP, *imgLa;

			//int width = sqrt( (float) heightPyr[0].size() );
			//int height = sqrt( (float) height[0].size() );
			int height = winHeight - boundary*2;
			int width = heightPyr[0].size() / height;
			img0 = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);	
			Relief2Image(heightPyr[0], img0, winHeight - boundary*2);

			GLfloat divisor = 1.0/256;
			for(int i=boundary; i<winWidth/3 - boundary; i++)
			{
				for(int j=boundary; j<winHeight - boundary; j++)
				{
					
					GLfloat depth = cvGetReal2D( img0, winHeight - 1 - j - boundary, i - boundary);
					
				
						int a = (int)(depth/divisor);
						int b = (int)(fmod(depth, divisor)/divisor/divisor);
						int c = (int)(fmod(fmod(depth, divisor), divisor*divisor)/divisor/divisor/divisor);
						cvSetReal2D( depthImg3, winHeight - 1 - j - boundary, i - boundary, (int)(depth/divisor) );
						cvSetReal2D( depthImg2, winHeight - 1 - j - boundary, i - boundary, (int)(fmod(depth, divisor)/divisor/divisor) );
						cvSetReal2D( depthImg1, winHeight - 1 - j - boundary, i - boundary, (int)(fmod(fmod(depth, divisor), divisor*divisor)/divisor/divisor/divisor) );
					
					
				}
			}
			cvMerge(depthImg1, depthImg2, depthImg3, 0, depthImg);

			imgPyr = cvCreatePyramid(img0, pyrLevel, 0.5);
			for(int i=0; i < pyrLevel-1; i++)
			{
				subSample( heightPyr[i], heightPyr[i+1] );
			}
			//Relief2Image(height[1], img1);
			imgUP = cvCreateImage( cvSize( imgPyr[1]->width*2,  imgPyr[1]->height*2), IPL_DEPTH_32F, 1);
			imgLa = cvCreateImage( cvSize( imgPyr[1]->width*2,  imgPyr[1]->height*2), IPL_DEPTH_32F, 1);
			/*cvPyrUp(img[1], imgUP);
			cvSub(img0, imgUP, imgLa);

			Image2Relief(imgLa, laplace);*/
			
			//subSample(height, height2);
			//heightList.push_back(height2);
			//upSample(height2, upHeight);
			//reliefSub(height, upHeight, laplace);

			laplacianFilter(heightPyr[0], laplace, alpha[0], beta[0], winHeight - boundary*2);
			laplaceList.clear();
			laplaceList.push_back(laplace);

			maxList.clear();
			recordMax(laplace);
			//for(int i=0; i<height.size(); i++)
			//{
			//	if ( height.at(i) == 1 )	//infinite point
			//	{
			//		height.at(i) = 0;
			//	}
			//	else
			//	{
			//		height.at(i) = ( height.at(i) - minDepth ) / (maxDepth - minDepth) ;	//normalize to [0,1]
			//		height.at(i) = 1 - height.at(i);	//transfer depth to height
			//	}
			//}
			//BuildRelief(laplace, pThreadRelief, pThreadNormal);
			//mesh1 = true;
			//relief1 = false;
			profile0 = true;
		}

		if( winHeight || winWidth )
		{
			int height = winHeight - boundary*2;
			int width = winWidth/3 - boundary*2;
			if( profile0 )
			{			
				sceneProfile.clear();
				
				for(int i=0; i < width - 1; i++)
				{			
					/*if( winHeight%2 == 0 )
					{
						sceneProfile.push_back( (heightPyr[0][(winHeight-1)/2 + i*height] + heightPyr[0][(winHeight+1)/2 + i*height]) / 2 );
					}
					else
					{
						sceneProfile.push_back( heightPyr[0][winHeight/2 + i*height] );
					}*/
					sceneProfile.push_back( ( heightPyr[0][ floor( (height-1)/2.0 ) + i*height ] + heightPyr[0][ ceil( (height-1)/2.0 ) + i*height ] ) / 2 );
				}

				//for(int i=0; i < height - 1; i++)
				//{			
				//	/*if( winHeight%2 == 0 )
				//	{
				//		sceneProfile.push_back( (heightPyr[0][(winHeight-1)/2 + i*height] + heightPyr[0][(winHeight+1)/2 + i*height]) / 2 );
				//	}
				//	else
				//	{
				//		sceneProfile.push_back( heightPyr[0][winHeight/2 + i*height] );
				//	}*/
				//	sceneProfile.push_back( ( heightPyr[0][ floor( (width-1)*3/4.0 )*height + i ] + heightPyr[0][ ceil( (width-1)*3/4.0 )*height + i ] ) / 2 );
				//}
				

				IplImage *profileImg = cvCreateImage( cvSize( width, height ), IPL_DEPTH_8U, 1);
				cvSet( profileImg, cvScalar(208) );
				DrawProfile(sceneProfile, profileImg, 0.9);
				cvNamedWindow("Scene Profile", 1);
				cvShowImage("Scene Profile", profileImg);

				profile0 = false;
			}
		}
		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
		glColor3f(0.6, 0.6, 0.6);

		glTranslated(-0.2, 0, 0);
		
		glRotatef(reliefAngleX, 1, 0, 0);
		glRotatef(reliefAngleY, 0, 1, 0);
		glRotatef(reliefAngleZ, 0, 0, 1);

		glScalef(0.01*scale, 0.01*scale, 1);

		glTranslated(-2/0.01, -2/0.01, 0.4);

		glScalef(1, 1, outputHeight);
		
		if( winHeight || winWidth )
		{
			int height = winHeight - boundary*2;
			int width = winWidth/3 - boundary*2;
			
			if(mesh1)
			{			
					/*if( profile0 )
					{			
						sceneProfile.clear();
						
						for(int i=0; i < width - 1; i++)
						{			
							sceneProfile.push_back( (heightPyr[0][199 + i*height] + heightPyr[0][200 + i*height]) / 2 );
							
							for(int j=0; j < height - 1; j++)
							{
								glBegin(GL_TRIANGLES);
								glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 ) *3);
								glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 );
								glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
								glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 );
								glEnd();
								
								glBegin(GL_TRIANGLES);
								glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 )  *3 + 3);
								glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 );
								glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
								glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3);
								glEnd();
							}
						}

						IplImage *profileImg = cvCreateImage( cvSize( width, height ), IPL_DEPTH_8U, 1);
						DrawProfile(sceneProfile, profileImg, 0.9);
						cvNamedWindow("Scene Profile", 1);
						cvShowImage("Scene Profile", profileImg);

						profile0 = false;
					}*/

					/*else
					{
						for(int i=0; i < width - 1; i++)
						{		
							for(int j=0; j < height - 1; j++)
							{
								glBegin(GL_TRIANGLES);
								glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 ) *3);
								glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 );
								glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
								glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 );
								glEnd();
								
								glBegin(GL_TRIANGLES);
								glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 )  *3 + 3);
								glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 );
								glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
								glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3);
								glEnd();
							}
						}
					}*/
			}
		}
		
		//for(int i=0; i < height.size() / (winHeight - boundary*2) - 1; i++)
		//{
		//	for(int j=0; j < winHeight - boundary*2 -1; j++)
		//	{
		//		int position = i*(winHeight-boundary*2) + j;
		//		if( height.at(position) >= 0 )
		//		{
		//			i++;
		//			i--;
		//		}
		//		if( height.at(position) >= 0.5 )
		//		{
		//			glColor3f(1.0, 1.0, 1.0);
		//		}
		//		
		//		GLdouble normal[3];
		//		GLdouble v1[3][3] =  { {i, j, height.at(position)}, {i+1, j, height.at(position+winHeight - boundary*2)}, { i, j+1, height.at(position+1)} };
		//		
		//		glBegin(GL_TRIANGLES);
		//			setNormal( v1, normal );
		//			glNormal3dv(normal);
		//			glVertex3d( i, j, height.at(position) );
		//			glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );
		//			glVertex3d( i, j+1, height.at(position+1) );
		//			
		//		GLdouble v2[3][3] = { {i, j+1, height.at(position+1)}, {i+1, j, height.at(position+winHeight - boundary*2)}, {i+1, j+1, height.at(position+winHeight - boundary*2 + 1)} };
		//			setNormal( v2, normal);
		//			glNormal3dv(normal);
		//			glVertex3d( i, j+1, height.at(position+1)  );
		//			glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );				
		//			glVertex3d( i+1, j+1, height.at(position+winHeight - boundary*2 + 1) );
		//		glEnd();
		//		//glColor3f(1.0, 0.0, 0.0);
		//	}
		//}
		//swglmDraw(MODEL);
		
	glPopMatrix();

	glPushMatrix();
		//glTranslated(0, 2, 0);
		glMultMatrixd(TRACKM);

		
	glPopMatrix();
}

void partition2(void)
{
	glDisable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT3);
	glDisable(GL_LIGHT2);
	glDisable(GL_LIGHT4);
	
	glViewport(0, 0, winWidth, winHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
	glOrtho(0, winWidth, 0, winHeight, -2.0, 2.0);
    
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(winWidth/3 + boundary, 0 + boundary, winWidth/3 - 2*boundary, winHeight - 2*boundary);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	//swOrtho(-2.0, 2.0, -2.0, 2.0, -3.0, 3.0);
	//swFrustum(-2.0, 2.0, -2.0, 2.0, -3.0, 3.0);
	gluPerspective(60, (GLfloat)(winWidth/horizontalSplit)/winHeight, 0.1, 300); 

    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0);

	glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
	glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);
	glLightfv(GL_LIGHT3, GL_POSITION, lightPos11);
	
	
	/*glPushMatrix();
		glTranslated(-10,15,0);
		glutSolidSphere(1,8,8);
	glPopMatrix();*/

	
	//world coordinate
	glColor3f(1, 0, 0);
}

void partition3(void)
{
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);
	glDisable(GL_LIGHT4);
	glEnable(GL_LIGHT2);
	glEnable(GL_LIGHT3);
	//Do not change, setting a basic transformation
	glViewport(0, 0, winWidth, winHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	//glOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
	glOrtho(0, winWidth, 0, winHeight, -2.0, 2.0);
    
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//glColor3f(1, 0, 0);
	//OpenglLine(winWidth/2, 0, 0, winWidth, winHeight, 0);


	//
	//replace the opengl function in openglPath() to sotfgl
    //


	//swClearZbuffer();



	//view transform
	glViewport(winWidth*2/3 + boundary, 0 + boundary, winWidth/3 - 2*boundary, winHeight - 2*boundary);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
	//swOrtho(-2.0, 2.0, -2.0, 2.0, -3.0, 3.0);
	//swFrustum(-2.0, 2.0, -2.0, 2.0, -3.0, 3.0);
	gluPerspective(60, (GLfloat)(winWidth/horizontalSplit)/winHeight, 0.1, 300); 

    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0);

	glLightfv(GL_LIGHT2, GL_POSITION, lightPos2);
	glLightfv(GL_LIGHT4, GL_POSITION, lightPos21);
	/*glPushMatrix();
		glTranslated(-10,15,0);
		glutSolidSphere(1,8,8);
	glPopMatrix();*/

	
	//world coordinate
	glColor3f(1, 0, 0);
}

double distanceWeight(int x, int y, int u, int v,int m)
{
	double d = sqrt( (double) (x-u)*(x-u) + (y-v)*(y-v) );
	return pow(M_E, -d*d/(2*m) );
}

double distanceWeight(double d,int m)
{
	return pow(M_E, -d*d/(2*m) );
}

void gradientWeight(vector<GLfloat> &src, int ext, int srcHeight=0)
{
	if(srcHeight)
	{}
	else
	{
		srcHeight = sqrt( (float)src.size() );
	}
	int srcWidth = src.size() / srcHeight;
	vector<GLfloat> dst;

	for(int i=0; i< srcHeight; i++)
	{
		for(int j=0; j< srcHeight; j++)
		{
			dst.push_back(0);
		}
	}
	
	for(int i=0; i< srcHeight; i++)
	{
		for(int j=0; j< srcHeight; j++)
		{
				if(i >= ext  && j >= ext && i+ext < srcHeight && j+ext < srcHeight)
				{
					for(int p=-ext; p <=ext; p++)
					{
						for(int q=-ext; q <=ext; q++)
						{
							dst [ i*srcHeight + j ] += src [ (i+p)*srcHeight + j+q ] * distanceWeight(i, j, i+p, j+q, ext);
						}
					}
				}

				else	//boundary issues
				{
					int extU = -ext,extD = ext,extL = -ext,extR = ext;
					if( ext > j)		extL = -j;
					if( ext > i)		extU = -i;
					if( j+ext >= srcHeight)		extR = srcHeight - 1 - j;
					if( i+ext >= srcHeight)		extD = srcHeight - 1 - i;

					//int window[ (extR - extL + 1) * (extD - extU +1) ];
					
					for(int p=extU; p <= extD; p++)
					{
						for(int q=extL; q <= extR; q++)
						{
							dst [ i*srcHeight + j ] += src [ (i+p)*srcHeight + j+q ] * distanceWeight(i, j, i+p, j+q, ext);
						}
					}
				}
		
		}
	}

	src.clear();
	for(int i=0;i < dst.size(); i++)
	{
		src.push_back( dst[i] );
	}
}

struct setcomp {
	bool operator() (const P& lhs, const P& rhs) const
	{
		if( lhs.second != rhs.second )
			return lhs.second > rhs.second;
		else
			return lhs.first > rhs.first;
	}
};

bool vectorcomp(const P& lhs, const P& rhs)
{
	if( lhs.second != rhs.second )
		return lhs.second > rhs.second;
	else
		return lhs.first > rhs.first;
}

template <class T>
void sortAddress(T src[], int dst[], const int size)
{	
	/*set<P, setcomp> s;

	for(int i = 1; i < size; i++)
	{
		s.insert( P(i, src[i]) );
	}
	
	set<P, setcomp>::iterator iter;
	int i = 1;
    for (iter = s.begin(); iter != s.end(); ++iter)
    {
		dst[i] = (*iter).first;
		i++;
    }*/

	vector<P> v;

	for(int i = 1; i < size; i++)
	{
		v.push_back( P(i, src[i]) );
	}
	sort(v.begin(), v.end(), vectorcomp);

	vector<P>::iterator iter;
	int i = 1;
    for (iter = v.begin(); iter != v.end(); ++iter)
    {
		dst[i] = (*iter).first;
		i++;
    }

}

//void sortAddress(double src[], int dst[], const int size)
//{	
//	set<P, setcomp> s;
//
//	for(int i = 1; i < size; i++)
//	{
//		s.insert( P(i, src[i]) );
//	}
//	
//	set<P, setcomp>::iterator iter;
//	int i = 1;
//    for (iter = s.begin(); iter != s.end(); ++iter)
//    {
//		dst[i] = (*iter).first;
//		i++;
//    }
//
//}

void bucketsort(float src[], int dst[], int n, int k = HistogramBins/100)
{
	//vector< set<P, setcomp> > bucket;

	//bucket.resize(k);

	//float min = src[1], max = 0;
	//for(int i=1; i<n; i++)
	//{
	//	if( max < src[i] )	max = src[i];
	//	if( min > src[i] )	min = src[i];
	//}

	//for(int i=1; i<n; i++)
	//{
	//	/*P element;
	//	element.first = i;
	//	element.second = src[i];*/

	//	bucket[ k - 1 - (int)( (src[i]-min) / (max-min) * (k-1) ) ].insert( P(i, src[i]) );
	//}
	//
	//for(int i=1, j=0; i < n && j < k; j++)
	//{
	//	set<P, setcomp>::iterator iter;
	//	
	//	for (iter = bucket[j].begin(); iter != bucket[j].end(); ++iter)
	//	{
	//		dst[i] = (*iter).first;
	//		i++;
	//	}		
	//}

	vector< vector<P> > bucket;

	bucket.resize(k);

	float min = src[1], max = 0;
	for(int i=1; i<n; i++)
	{
		if( max < src[i] )	max = src[i];
		if( min > src[i] )	min = src[i];
	}

	for(int i=1; i<n; i++)
	{
		bucket[ k - 1 - (int)( (src[i]-min) / (max-min) * (k-1) ) ].push_back( P(i, src[i]) );
	}
	

	//static int isHist;
	//if( isHist < 1920 )
	//{
	//	int hist[HistogramBins], histMax=0;
	//	for(int i=0; i < k; i++)
	//	{
	//		hist[i] = bucket[i].size();
	//		if(hist[i] > histMax)	histMax = hist[i];
	//	}	
	//	
	////int bin_w = cvRound((double)histImage->width/k);

	//	for(int i=0; i < k; i++)
	//	{
	//		int value = 255 - (double)(hist[i]*255)/histMax;
	//		cvSetReal2D(histImage, i, isHist, 255 - (double)(hist[i]*255)/histMax);
	//		/*cvRectangle( histImage, cvPoint(i*bin_w, histImage->height),
	//		   cvPoint((i+1)*bin_w, 
	//			histImage->height*( 1 - hist[i]/histMax ) ),
	//		   cvScalarAll(0), -1, 8, 0 );*/
	//	}
	//	
	//}
	//else if( isHist == 1920)
	//{
	//	cvNamedWindow("bucket histogram", 1);
	//	cvShowImage("bucket histogram", histImage);
	//}
	//isHist++;
	
	for(int i=1, j=0; i < n && j < k; j++)
	{
		vector<P>::iterator iter;

		sort(bucket[j].begin(), bucket[j].end(), vectorcomp);
		for (iter = bucket[j].begin(); iter != bucket[j].end(); ++iter)
		{
			dst[i] = (*iter).first;
			i++;
		}		
	}
}

void clipping(float h[], float cumulative, float l)
{
	l = cumulative*l / HistogramBins;
	
	int i;
	for(i = 1; i < HistogramBins+1; i++)
	{
		if( h[i] > l )
		{
			h[i] = l;
		}
	}
	
}

void redistribute(float h[], float cumulative, float l, float ratio=1)
{
	l = cumulative*l / HistogramBins;

	int sort[ HistogramBins+1 ];
	
	//bucketsort(h, sort, HistogramBins+1);
	sortAddress(h, sort, HistogramBins+1);

	

	int i;
	float S = 0;
	for(i = 1; i < HistogramBins+1; i++)
	{
		if( h[ sort[i] ] >= l )
		{
			S += (h[ sort[i] ] - l) * ratio;
		}
		else break;
	}
	if( i - 1 >= 1 )
	{
		int q = i - 1;
		for(i = q; i < HistogramBins ; i++)
		{
			if( S / ( HistogramBins - i ) > l - h[ sort[i+1] ] )
			{
				S +=  (h[ sort[i+1] ] - l) * ratio;
			}
			else	break;
		}

		int j;
		//#pragma omp parallel for
		for(j = 1; j <= i; j++)
		{
			h[ sort[j] ] = l;
		}
		//#pragma omp parallel for
		for(j = i+1; j < HistogramBins+1; j++)
		{
			h[ sort[j] ] += S / (HistogramBins - i) ;
		}
	}
}

const int				k				= 4;			// number of nearest neighbors
int				dim				= 2;			// dimension
double			eps				= 0;			// error bound
int				maxPts			= 50000;			// maximum number of data points
istream*		dataIn			= NULL;			// input for data points
istream*		queryIn			= NULL;			// input for query points

bool readPt(istream &in, ANNpoint p)			// read point (false on EOF)
{
	for (int i = 0; i < dim; i++) {
		if(!(in >> p[i])) return false;
	}
	return true;
}

void printPt(ostream &out, ANNpoint p)			// print point
{
	out << "(" << p[0];
	for (int i = 1; i < dim; i++) {
		out << ", " << p[i];
	}
	out << ")\n";
}

double dist(ANNpoint p1, ANNpoint p2)
{
	return sqrt( pow(p1[1] - p2[1], 2) + pow(p1[0] - p2[0], 2) );
}

ANNpoint midpoint(ANNpoint p1, ANNpoint p2)
{
	ANNpoint p;
	p[0] = (p1[0] + p2[0]) / 2;
	p[1] = (p1[1] + p2[1]) / 2;
	return p;
}

void setSector(ANNpoint origin, ANNpoint sample, int &dst)
{
	int x = origin[1];
	int y = origin[0];

	if(sample[0] > y)
	{
		if(sample[1] > x)
		{
			dst = 1;
		}
		else if(sample[1] == x)
		{
			dst = 2;
		}
		else
		{
			dst = 3;
		}
	}
	else if(sample[0] == y)
	{
		if(sample[1] > x)
		{
			dst = 0;
		}
		else
		{
			dst = 4;
		}
	}
	else
	{
		if(sample[1] > x)
		{
			dst = 7;
		}
		else if(sample[1] == x)
		{
			dst = 6;
		}
		else
		{
			dst = 5;
		}
	}
}

bool compareSector(int sector1, int sector2)
{
	if( sector1 == sector2)	return true;	//the same
	if( (sector1+1) % 8 == sector2)	return true;	//clockwise rotation adjacence
	if( (sector1-1) % 8 == sector2)	return true;	//counterclockwise rotation adjacence
	return false;	
}

void knninterpolation(IplImage *src, IplImage *dst)
{
	string dataS,queryS;
	int width = src->width;
	int height = src->height;

	char buffer [10];
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( !bgMask[i*height + height - 1 - j] )
			{
				if( cvGetReal2D(src, j, i) )
				{
					dataS += itoa(j, buffer, 10);
					dataS += " ";
					dataS += itoa(i, buffer, 10);
					dataS += "\n";
				}
				else
				{
					queryS += itoa(j, buffer, 10);
					queryS += " ";
					queryS += itoa(i, buffer, 10);
					queryS += "\n";
				}
			}
		}
	}

	istringstream dataIs(dataS);
	dataIn = &dataIs;
	istringstream queryIs(queryS);
	queryIn = &queryIs;

	ANNpointArray		dataPts;				// data points
	ANNpoint			queryPt;				// query point
	ANNidxArray			nnIdx;					// near neighbor indices
	ANNdistArray		dists;					// near neighbor distances
	ANNkd_tree*			kdTree;					// search structure

	//getArgs(argc, argv);						// read command-line arguments

	int					nPts;					// actual number of data points
	queryPt = annAllocPt(dim);					// allocate query point
	dataPts = annAllocPts(maxPts, dim);			// allocate data points
	nnIdx = new ANNidx[k];						// allocate near neigh indices
	dists = new ANNdist[k];						// allocate near neighbor dists

	nPts = 0;									// read data points

	//cout << "Data Points:\n";
	while (nPts < maxPts && readPt(*dataIn, dataPts[nPts])) {
		//printPt(cout, dataPts[nPts]);
		nPts++;
	}

	kdTree = new ANNkd_tree(					// build search structure
					dataPts,					// the data points
					nPts,						// number of points
					dim);						// dimension of space

	while (readPt(*queryIn, queryPt)) {			// read query points
		cout << "Query point: ";				// echo query point
		printPt(cout, queryPt);

		kdTree->annkSearch(						// search
				queryPt,						// query point
				k,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				eps);							// error bound

		//cout << "\tNN:\tIndex\tDistance\n";
		//for (int i = 0; i < k; i++) {			// print summary
		//	dists[i] = sqrt(dists[i]);			// unsquare distance
		//	cout << "\t" << i << "\t" << dataPts[ nnIdx[i] ][0] << "\t" << dists[i] << "\n";
		//}

		double sum=0,wsum=0;
		for (int i = 0; i < k; i++)
		{
			dists[i] = sqrt(dists[i]);
			//cout << "\t" << i << "\t" << nnIdx[i] <<  "\t" << dataPts[ nnIdx[i] ][0] << "\t" << dataPts[ nnIdx[i] ][1] << "\t" << dists[i] << "\n";
			sum += 1.0/dists[i]*cvGetReal2D( src, dataPts[ nnIdx[i] ][0], dataPts[ nnIdx[i] ][1] );
			wsum += 1.0/dists[i];
		}

		cvSetReal2D( dst, queryPt[0], queryPt[1], 1.0/wsum*sum);
	}

	delete [] nnIdx;							// clean things up
	delete [] dists;
	delete kdTree;
	annClose();									// done with ANN
}

void knninterpolation(const vector<GLfloat> &src, vector<GLfloat> &dst, vector<GLfloat> &carve, int m=0)
{
	string dataS,queryS;
	int height = winHeight - boundary*2;
	int width = src.size() / height;

	char buffer [10];
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( !bgMask[ i*height + j ] )
			{
				if( cvGetReal2D(sample, height - 1 - j, i) )
				{
					dataS += itoa(j, buffer, 10);
					dataS += " ";
					dataS += itoa(i, buffer, 10);
					dataS += "\n";
				}
				else /*if( !cvGetReal2D(edge, height - 1 - j, i) )*/
				{
					queryS += itoa(j, buffer, 10);
					queryS += " ";
					queryS += itoa(i, buffer, 10);
					queryS += "\n";
				}
			}
		}
	}

	istringstream dataIs(dataS);
	dataIn = &dataIs;
	istringstream queryIs(queryS);
	queryIn = &queryIs;

	int					nPts;					// actual number of data points
	ANNpointArray		dataPts;				// data points
	ANNpoint			queryPt;				// query point
	ANNidxArray			nnIdx;					// near neighbor indices
	ANNdistArray		dists;					// near neighbor distances
	ANNkd_tree*			kdTree;					// search structure

	//getArgs(argc, argv);						// read command-line arguments

	queryPt = annAllocPt(dim);					// allocate query point
	maxPts = pow( (float)winHeight - boundary*2, 2);
	dataPts = annAllocPts(maxPts, dim);			// allocate data points
	nnIdx = new ANNidx[k];						// allocate near neigh indices
	dists = new ANNdist[k];						// allocate near neighbor dists

	nPts = 0;									// read data points
	

	IplImage *type = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *r = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *g = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *b = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	cvCopy(sample, r);
	cvCopy(sample, g);
	cvCopy(sample, b);

	//cout << "Data Points:\n";
	while (nPts < maxPts && readPt(*dataIn, dataPts[nPts])) {
		//printPt(cout, dataPts[nPts]);
		nPts++;
	}

	kdTree = new ANNkd_tree(					// build search structure
					dataPts,					// the data points
					nPts,						// number of points
					dim);						// dimension of space

	while (readPt(*queryIn, queryPt)) {			// read query points
		//cout << "Query point: ";				// echo query point
		//printPt(cout, queryPt);

		kdTree->annkSearch(						// search
				queryPt,						// query point
				k,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				eps);							// error bound

		//cout << "\tNN:\tIndex\tDistance\n";
		//for (int i = 0; i < k; i++) {			// print summary
		//	dists[i] = sqrt(dists[i]);			// unsquare distance
		//	cout << "\t" << i << "\t" << dataPts[ nnIdx[i] ][0] << "\t" << dists[i] << "\n";
		//}

		//cout << "\tNN:\tIndex\tDistance\n";

		float thetaD = 2*radius[0] + 1;

		double distMax=0, distPrimeMax=0;
		for(int i=0; i<4; i++)
		{
			if(dists[i] > distPrimeMax)
			{
				distMax = dists[i];
			}
			
			for(int j=i+1; j<4;j++)
			{
				double distance = dist(dataPts[ nnIdx[i] ], dataPts[ nnIdx[j] ]);
				if( distance > distPrimeMax )
				{
					distPrimeMax = distance;
				}
			}
		}

		distMax = sqrt(distMax);
		distPrimeMax = sqrt(distPrimeMax);

		float alpha = 0.01;
		if( distMax < thetaD )	//on  the edge
		{
			cvSetReal2D( r, height - 1 - queryPt[0], queryPt[1], 255);

			double distMin=dists[1];
			for (int i = 0; i < k; i++)
			{
				dists[i] = pow( dists[i], 0.5);
				if(distMin > dists[i]) distMin = dists[i];
			}
			
			carve[ queryPt[1]*height + queryPt[0] ] = 1.0/alpha*log10(1+maxf( distMin-radius[3]*1.5, 0)*alpha);

			for (int i = 0; i < k; i++)
			{
				dists[i] = pow( dists[i], 3);
			}
		}
		else if( distPrimeMax < thetaD )	//close to an edge
		{
			cvSetReal2D( g, height - 1 - queryPt[0], queryPt[1], 255);
			
			int sector[k];
			for(int i=0; i < k; i++)
			{				
				setSector(queryPt, dataPts[ nnIdx[i] ], sector[i]);
			}
			
			int sort[k];
			sortAddress(dists, sort, k);

			for(int i=k-1; i > 0; i--)//start from the smallest to the biggest
			{
				
				int sector1 = sector[ sort[i] ];
				int x1 = dataPts[ nnIdx[ sort[i] ] ][1];
				int y1 = dataPts[ nnIdx[ sort[i] ] ][0];				
				
				for(int j=i-1; j>0; j--)
				{
					
					int sector2 = sector[ sort[j] ];
					if( compareSector(sector1, sector2) )
					{
						int x2 = dataPts[ nnIdx[ sort[j] ] ][1];
						int y2 = dataPts[ nnIdx[ sort[j] ] ][0];
						if( abs( src[	x1*height + y1 ] - src[	x2*height + y2 ] ) > ::threshold )	//thetaV
						{
							dists[ sort[j] ]  *= 50;	//reduced by factor f
						}					
					}
					
				}
				
			}

			for (int i = 0; i < k; i++)
			{
				dists[i] = pow( dists[i], 0.5); 
			}		

			carve[ queryPt[1]*height + queryPt[0] ] = 1.0/alpha*log10(1+maxf( dists[k-1]-radius[3]*1.5, 0)*alpha);
		}
		else	//far from edges
		{
			for (int i = 0; i < k; i++)
			{
				dists[i] = pow( dists[i], 0.5); 
			}		
		}


		double sum=0,wsum=0;
		for (int i = 0; i < k; i++)
		{
			//dists[i] = sqrt(dists[i]);
			//cout << "\t" << i << "\t" << nnIdx[i] <<  "\t" << dataPts[ nnIdx[i] ][0] << "\t" << dataPts[ nnIdx[i] ][1] << "\t" << dists[i] << "\n";
			if( m )
			{
				double weight = distanceWeight(dists[i], radius[n-1]);
				sum += weight * src[	dataPts[ nnIdx[i] ][1]*height +
														dataPts[ nnIdx[i] ][0]					];
				wsum += weight;
			}
			else
			{
				double weight = 1.0/dists[i];
				sum += weight * src[	dataPts[ nnIdx[i] ][1]*height +
															dataPts[ nnIdx[i] ][0]					];
				wsum += weight;
			}
		}

		dst[ queryPt[1]*height + queryPt[0] ] = 1.0/wsum*sum;
	}

	cvMerge(b, g, r, 0, type);
	cvNamedWindow("interpolation type", 1);
	cvShowImage("interpolation type", type);

	delete [] nnIdx;							// clean things up
	delete [] dists;
	delete kdTree;
	annClose();									// done with ANN
}

void knninterpolation(const vector<GLfloat> &src, vector<GLfloat> &carve, int m=0)
{
	string dataS,queryS;
	int height = winHeight - boundary*2;
	int width = carve.size() / height;

	char buffer [10];
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( !bgMask[ i*height + j ] )
			{
				if( cvGetReal2D(sample, height - 1 - j, i) != 0 )
				{
					dataS += itoa(j, buffer, 10);
					dataS += " ";
					dataS += itoa(i, buffer, 10);
					dataS += "\n";
				}
				else /*if( !cvGetReal2D(edge, height - 1 - j, i) )*/
				{
					queryS += itoa(j, buffer, 10);
					queryS += " ";
					queryS += itoa(i, buffer, 10);
					queryS += "\n";
				}
			}
		}
	}

	istringstream dataIs(dataS);
	dataIn = &dataIs;
	istringstream queryIs(queryS);
	queryIn = &queryIs;

	int					nPts;					// actual number of data points
	ANNpointArray		dataPts;				// data points
	ANNpoint			queryPt;				// query point
	ANNidxArray			nnIdx;					// near neighbor indices
	ANNdistArray		dists;					// near neighbor distances
	ANNkd_tree*			kdTree;					// search structure

	//getArgs(argc, argv);						// read command-line arguments

	queryPt = annAllocPt(dim);					// allocate query point
	maxPts = pow( (float)winHeight - boundary*2, 2);
	dataPts = annAllocPts(maxPts, dim);			// allocate data points
	nnIdx = new ANNidx[k];						// allocate near neigh indices
	dists = new ANNdist[k];						// allocate near neighbor dists

	nPts = 0;									// read data points
	

	IplImage *type = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *r = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *g = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *b = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	cvCopy(sample, r);
	cvCopy(sample, g);
	cvCopy(sample, b);

	//cout << "Data Points:\n";
	while (nPts < maxPts && readPt(*dataIn, dataPts[nPts])) {
		//printPt(cout, dataPts[nPts]);
		nPts++;
	}
	cout << "Points: " << nPts << endl;

	kdTree = new ANNkd_tree(					// build search structure
					dataPts,					// the data points
					nPts,						// number of points
					dim);						// dimension of space

	float Min=10000, Max=0;

	while (readPt(*queryIn, queryPt)) {			// read query points
		//cout << "Query point: ";				// echo query point
		//printPt(cout, queryPt);

		kdTree->annkSearch(						// search
				queryPt,						// query point
				k,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				eps);							// error bound

		//cout << "\tNN:\tIndex\tDistance\n";
		//for (int i = 0; i < k; i++) {			// print summary
		//	dists[i] = sqrt(dists[i]);			// unsquare distance
		//	cout << "\t" << i << "\t" << dataPts[ nnIdx[i] ][0] << "\t" << dists[i] << "\n";
		//}

		//cout << "\tNN:\tIndex\tDistance\n";
		

		float thetaD = 2*radius[0] + 1;

		double distMax=0, distPrimeMax=0;
		for(int i=0; i<4; i++)
		{
			if(dists[i] > distPrimeMax)
			{
				distMax = dists[i];
			}
			
			for(int j=i+1; j<4;j++)
			{
				double distance = dist(dataPts[ nnIdx[i] ], dataPts[ nnIdx[j] ]);
				if( distance > distPrimeMax )
				{
					distPrimeMax = distance;
				}
			}
		}

		distMax = sqrt(distMax);
		distPrimeMax = sqrt(distPrimeMax);

		float alpha = 10;
		if( distMax < thetaD )	//on  the edge
		{
			cvSetReal2D( r, height - 1 - queryPt[0], queryPt[1], 255);

			/*double distMin=dists[1];
			for (int i = 0; i < k; i++)
			{
				dists[i] = pow( dists[i], 0.5);
				if(distMin > dists[i]) distMin = dists[i];
			}
			
			carve[ queryPt[1]*height + queryPt[0] ] = 1.0/alpha*log10(1+maxf( distMin-radius[3]*1.5, 0)*alpha);*/

		}
		else if( distPrimeMax < thetaD )	//close to an edge
		{
			cvSetReal2D( g, height - 1 - queryPt[0], queryPt[1], 255);
			
			//int sector[k];
			//for(int i=0; i < k; i++)
			//{				
			//	setSector(queryPt, dataPts[ nnIdx[i] ], sector[i]);
			//}
			//
			//int sort[k];
			//sortAddress(dists, sort, k);

			//for(int i=k-1; i > 0; i--)//start from the smallest to the biggest
			//{
			//	
			//	int sector1 = sector[ sort[i] ];
			//	int x1 = dataPts[ nnIdx[ sort[i] ] ][1];
			//	int y1 = dataPts[ nnIdx[ sort[i] ] ][0];				
			//	
			//	for(int j=i-1; j>0; j--)
			//	{
			//		
			//		int sector2 = sector[ sort[j] ];
			//		if( compareSector(sector1, sector2) )
			//		{
			//			int x2 = dataPts[ nnIdx[ sort[j] ] ][1];
			//			int y2 = dataPts[ nnIdx[ sort[j] ] ][0];
			//			if( abs( src[	x1*height + y1 ] - src[	x2*height + y2 ] ) > ::threshold )	//thetaV
			//			{
			//				dists[ sort[j] ]  *= 50;	//reduced by factor f
			//			}					
			//		}
			//		
			//	}

			//	carve[ queryPt[1]*height + queryPt[0] ] = 1.0/alpha*log10(1+maxf( distMin-radius[3]*1.5, 0)*alpha);
			//	
			//}

			
		}
		else	//far from edges
		{
			for (int i = 0; i < k; i++)
			{
				dists[i] = 0; 
			}		
		}

		double distMin=dists[0], distMin2;
		for (int i = 0; i < k; i++)
		{
			dists[i] = pow( dists[i], 0.5);
			if(distMin > dists[i])
			{	
				distMin2 = distMin;
				distMin = dists[i];
			}
		}

		/*int sort[k];
		sortAddress(dists, sort, k);*/
			
		/*if( cvGetReal2D(ImageL, height - 1 - queryPt[0], queryPt[1]) )
		{*/
			carve[ queryPt[1]*height + queryPt[0] ] = 1.0/alpha*log10(1.0+maxf( (float)distMin-/*190*/radius[0]*1, 0)*alpha);

			
			if(distMin > Max) Max = distMin;
			else if(Min > distMin) Min = distMin;
			
			//ANNpoint p = annAllocPt(dim);/* = midpoint(dataPts[ nnIdx[0] ], dataPts[ nnIdx[1] ]);*/
			//p[0] = (dataPts[ nnIdx[0] ][0] + dataPts[ nnIdx[1] ][0]) / 2;
			//p[1] = (dataPts[ nnIdx[0] ][1] + dataPts[ nnIdx[1] ][1]) / 2;
			//double distance = dist(queryPt, p);
			//double distanceMax = dist(dataPts[ nnIdx[0] ], p);
			//double carveCurrent = 1.0/alpha*log10(1+maxf( distanceMax - distance -radius[0]*0.5, 0)*alpha);
			//carve[ queryPt[1]*height + queryPt[0] ] = carveCurrent;
		/*}*/

	}

	cout << "Min: " << Min << " Max: " << Max << endl;

	cvMerge(b, g, r, 0, type);
	cvNamedWindow("interpolation type", 1);
	cvShowImage("interpolation type", type);

	delete [] nnIdx;							// clean things up
	delete [] dists;
	delete kdTree;
	annClose();									// done with ANN
}

void setPathMask(vector< pair<int, int> > &mask, int extent=5)
{
	for(int i=-extent; i <= extent; i++)
	{
		for(int j=-extent; j <= extent; j++)
		{
			if( i==0 && j==0 ) continue;
			if( gcd( abs(i), abs(j) ) == 1 )
			{
				mask.push_back( pair<int, int>(i, j) );
			}
		}
	}
}

float length(Node p1, Node p2)
{
	return sqrt( pow((float)p1.first - p2.first, 2) + pow((float)p1.second - p2.second, 2) );
}

float costSlope(Node p1, Node p2, int extent=5)
{
	int height = winHeight - boundary*2;
	float h1 = compressedH/*heightPyr[0]*/[ p1.first*height + p1.second ];
	float h2 = compressedH/*heightPyr[0]*/[ p2.first*height + p2.second ];
	float cost = abs( (h1 - h2)*height ) / length(p1, p2);
	//float cost = abs( (h1 - h2)*height ) / sqrt( pow( length(p1, p2), 2) + pow( (h1 - h2)*ahe_m/*height*/, 2) );
	//float cost = 1 - length(p1, p2) / sqrt( pow( length(p1, p2), 2) + pow( (h1 - h2)*height/**ahe_m*sqrt(2.0)*/, 2) );
	cout << "Cost: " << cost << endl;
	return cost;
}

float costCurvature(Node p1, Node p2, Node p3)
{
	float x1, y1, x2, y2;
	x1 = p2.first - p1.first;
	y1 = p2.second - p1.second;
	x2 = p3.first - p2.first;
	y2 = p3.second - p2.second;
	float theta;
	theta = abs( atan2(y1, x1) - atan2(y2, x2) );
	if( theta > M_PI)
	{
		theta = 2*M_PI - theta;
	}
	theta = M_PI - theta;
	return theta / M_PI;//normalize to [0,1]
}

float cost(vector< Node > path, int extent=5, float weight=0.5)
{
	float slope=0, curvature=0;
	int i;
	for(i=0; i < path.size()-2; i++)
	{
		slope += costSlope(path[i+1], path[i+2], extent);
		curvature += costCurvature(path[i], path[i+1], path[i+2]);
	}
	//slope += costSlope(path[i], path[i+1]);

	return (1-weight)*slope + weight*curvature;
}

void setMinOfPathMask(vector< Node > pathMask, vector< Node > &path, Node &p, int extent=5, float weight=0.5)
{
	int height = winHeight - boundary*2;
	
	float costMin=5/*, lengthMin=costList.size()*/;
	vector < pair<int, float> > costList;
	int index=-1;
	for(int i=0; i < pathMask.size(); i++)
	{
		int x = path[ path.size()-2 ].first + pathMask[i].first;//
		int y = path[ path.size()-2 ].second + pathMask[i].second;//

		if( x >= 0 && y >= 0 && x < height && y < height && !bgMask[ x*height + y ])
		{
			p = Node(x, y);
			vector< Node > newpath /*= vector< Node >()*/;
			/*newpath.push_back( path[ path.size() - 2 ] );
			newpath.push_back( p );*/
			if( path.size() < 3 )
			{
				newpath.assign( path.end()-2, path.end() );
			}
			else
			{
				newpath.assign( path.end()-3, path.end() );
			}
			newpath.insert( newpath.end()-1, p );

			float costCurrent = cost(newpath, extent, weight);
			//cout << "costCurrent: " << costCurrent << endl;
			if( costCurrent <= costMin )
			{
				costList.push_back( pair<int, float>(i, costCurrent) );
				costMin = costCurrent;

				/*lengthCurrent = length( path[ path.size()-2 ], p );
				if( lengthCurrent < lengthMin )
				{
					lengthMin = lengthCurrent;
					index = i;
				}*/
			}
		}

	}
	cout << "Cost Min: " << costMin << endl;

	float lengthMin=costList.size();
	for(int i=0; i < costList.size(); i++)
	{
		if( costList[i].second == costMin )
		{
			int x = path[ path.size()-2 ].first + pathMask[ costList[i].first ].first;
			int y = path[ path.size()-2 ].second + pathMask[ costList[i].first ].second;
			
			float lengthCurrent = length( path[ path.size()-1 ], Node(x, y) );
			if( lengthCurrent < lengthMin )
			{
				lengthMin = lengthCurrent;
				index = i;
			}
		}
	}

	if( index != -1)
	{
		int x = path[ path.size()-2 ].first + pathMask[index].first;
		int y = path[ path.size()-2 ].second + pathMask[index].second;
		p = Node(x, y);
		path.insert( path.end()-1, p );
	}
}

void printPath(const vector< Node > &path)
{
	for(int i=0; i<path.size(); i++)
	{
		cout <<'(' << path[i].first <<", " << path[i].second <<')' << '\t' ;
	}
	cout << endl;
}

void findPath(vector< Node > &path, float zpower=0.5, int extent=5)
{
	vector< Node > pathMask;
	setPathMask(pathMask, extent);
	cout << "Mask Size: " << pathMask.size() << endl;

	float costMax;
	costMax = cost(path, extent, zpower);
	vector< Node > initialPath = path;

	Node p;
	setMinOfPathMask(pathMask, path, p, extent, zpower);
	vector< Node > prePath, newPath;

	if( path.size() >=  3 )
	{
		printPath( path );
		prePath.assign( path.end()-3, path.end() );
		prePath.erase( prePath.end()-2 );

		newPath.assign( path.end()-3, path.end() );

		/*if( cost(newPath, zpower) > cost(prePath, zpower) )
		{
			path.erase( path.end()-2 );
			printPath( path );
		}
		else
		{*/
			for(int i=path.size(); p.first != path[ path.size()-1 ].first && p.second != path[ path.size()-1 ].second; i++)
			{		
				if( path.size() != i ) break;
				
				setMinOfPathMask(pathMask, path, p);	

				if( path.size() >=  4 )
				{					
					prePath.assign( path.end()-4, path.end() );
					prePath.erase( prePath.end()-2 );
					//prePath.insert( prePath.end()-1, p );

					newPath.assign( path.end()-4, path.end() );
					//newPath.erase( newPath.end()-3 );
					//newPath[ newPath.size() - 2 ] = p;

					/*if( cost(newPath, zpower) > cost(prePath, zpower) )
					{
						path.erase( path.end()-2 );
						break;
					}*/		
					
				}

				//printPath( path );

			}
		/*}*/
	}

	/*if( cost(path) > costMax )
	{
		path = initialPath;
	}*/
}

void generateCarvePath(const vector<GLfloat> &src)
{
	string dataS,queryS;
	int height = winHeight - boundary*2;
	int width = src.size() / height;

	char buffer [10];
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( !bgMask[ i*height + j ] )
			{
				if( cvGetReal2D(sample, height - 1 - j, i) != 0 )
				{
					dataS += itoa(j, buffer, 10);
					dataS += " ";
					dataS += itoa(i, buffer, 10);
					dataS += "\n";

					queryS += itoa(j, buffer, 10);
					queryS += " ";
					queryS += itoa(i, buffer, 10);
					queryS += "\n";
				}
			}
		}
	}

	istringstream dataIs(dataS);
	dataIn = &dataIs;
	istringstream queryIs(queryS);
	queryIn = &queryIs;

	int					nPts;					// actual number of data points
	ANNpointArray		dataPts;				// data points
	ANNpoint			queryPt;				// query point
	ANNidxArray			nnIdx;					// near neighbor indices
	ANNdistArray		dists;					// near neighbor distances
	ANNkd_tree*			kdTree;					// search structure

	//getArgs(argc, argv);						// read command-line arguments

	queryPt = annAllocPt(dim);					// allocate query point
	maxPts = width * height;
	dataPts = annAllocPts(maxPts, dim);			// allocate data points
	nnIdx = new ANNidx[k];						// allocate near neigh indices
	dists = new ANNdist[k];						// allocate near neighbor dists

	nPts = 0;									// read data points
	

	IplImage *pathImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *r = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *g = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *b = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	cvCopy(sample, r);
	cvCopy(sample, g);
	cvCopy(sample, b);

	//cout << "Data Points:\n";
	while (nPts < maxPts && readPt(*dataIn, dataPts[nPts])) {
		//printPt(cout, dataPts[nPts]);
		nPts++;
	}
	cout << "Points: " << nPts << endl;

	vector< pair< Node, Node > > connectedList;
	
	kdTree = new ANNkd_tree(					// build search structure
					dataPts,					// the data points
					nPts,						// number of points
					dim);						// dimension of space

	while (readPt(*queryIn, queryPt)) {			// read query points
		//cout << "Query point: ";				// echo query point
		//printPt(cout, queryPt);

		kdTree->annkSearch(						// search
				queryPt,						// query point
				2,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				eps);							// error bound

		//cout << "\tNN:\tIndex\tDistance\n";
		//for (int i = 0; i < k; i++) {			// print summary
		//	dists[i] = sqrt(dists[i]);			// unsquare distance
		//	cout << "\t" << i << "\t" << dataPts[ nnIdx[i] ][0] << "\t" << dists[i] << "\n";
		//}

		//cout << "\tNN:\tIndex\tDistance\n";
		Node initial = Node( queryPt[1], queryPt[0] );
		Node final = Node( dataPts[ nnIdx[1] ][1], dataPts[ nnIdx[1] ][0] );

		bool connected = false;
		for(int i=0; i < connectedList.size(); i++)
		{
			if( final == connectedList[i].first || initial == connectedList[i].second || initial == connectedList[i].first || final == connectedList[i].second)
			{
				connected = true;
				break;
			}
		}	

		if( !connected )
		{
			vector< Node> path;
			path.push_back( initial );
			path.push_back( final );
			findPath( path, 0 );

			for(int i=0; i < path.size()-1; i++)
			{
				//cvSetReal2D( r, height - 1 - path[i].second, path[i].first, 255);
				cvLine( r, cvPoint( path[i].first, height - 1 - path[i].second ), cvPoint( path[i+1].first, height - 1 - path[i+1].second ), cvScalarAll(255) );
			}

			connectedList.push_back( pair< Node, Node >(initial, final) );
		}	

	}

	/*cvMerge(b, g, r, 0, pathImg);
	cvNamedWindow("Carve Path", 1);
	cvShowImage("Carve Path", pathImg);*/

	delete [] nnIdx;							// clean things up
	delete [] dists;
	delete kdTree;
	annClose();									// done with ANN
}

void nextNeighbor( IplImage *src, IplImage *length, int x, int y, int offset, float depth, float beta, float spread, float gamma)
{	
	int height = src->height;
	int width = src->width;

	double before = cvGetReal2D( src, height - 1 - y, x);

	int min = 256;
	int i=x, j=y;
	for(int m= -1; m <= 1; m++)
	{
		for(int n = -1; n <= 1; n++)
		{
			if( m!=0 || n!=0 ) 
			{
				if( height - 1 - (y+n) >= 0 &&  (y+n) > -1 && x+m >= 0 && x+m < width) 
				{
					double carveDepth = cvGetReal2D( carve, height - 1 - (y+n), x+m );
					if( carveDepth /*< 1 && carveDepth > 0*/ ) continue;
					
					double red = cvGetReal2D( src, height - 1 - (y+n), x+m );
					
					if( before == 1 )
					{
						if( red >= before && red < min )
						{
							if( red == 1)
							{
								if( min != 256 ) continue;
							}
							else
							{
								min = red;
							}
							i = x+m;
							j = y+n;
						}
					}
					else
					{
						if( red >= before && red < min )
						{
							if( red == before )
							{
								if( min != 256 ) continue;
							}
							else
							{
								min = red;
							}
							i = x+m;
							j = y+n;
						}
					}
				}
			}

		}
	}

	if( i != x || j != y )
	{
		double depthRatio = depth / compress( cvGetReal2D( length, height - 1 - j, i ), beta );
		cvSetReal2D( carve, height - 1 - j, i, compress( offset, beta )*depthRatio );

		double spreadRatio = spread / compress( cvGetReal2D( length, height - 1 - j, i ), gamma );
		double segment = cvGetReal2D( length, height - 1 - j, i ) * compress( offset, gamma )*spreadRatio;
		double angle = cvGetReal2D( GradientA, height - 1 - j, i );

		double depthSpreadRatio = depth / compress( spread*cvGetReal2D( length, height - 1 - j, i ), beta );
		x = i;
		y = height - 1 - j;
		for(int k=segment; k > 0 ; k--)
		{
			//if( cvGetReal2D( carve, y, x) ) continue;
			
			CvPoint p1 = cvPoint( x, y );
			x += cos( angle );
			y += sin( angle );		


			
			/*if( cvGetReal2D( carve, y, x) == 0.0 )
			{*/
				cvSetReal2D( carve2, y, x, compress( k, beta )*depthSpreadRatio );
			//}
			//cvLine( carve, p1, cvPoint( x, y ), cvScalarAll(1) );					
		}

		x = i;
		y = height - 1 - j;
		for(int k=segment; k > 0 ; k--)
		{
			//if( cvGetReal2D( carve, y, x) ) continue;
			
			CvPoint p1 = cvPoint( x, y );
			x -= cos( angle );
			y -= sin( angle );		

			cvSetReal2D( carve2, y, x, compress( k, beta )*depthSpreadRatio );
			//cvLine( carve, p1, cvPoint( x, y ), cvScalarAll(1) );					
		}
		//cvSetReal2D( carve, height - 1 - j, i, compress( offset, beta )*depthRatio );

		if( min != 255 )
		{		
			//cout << min << endl;
			nextNeighbor( src, length, i, j, offset+1, depth, beta, spread, gamma);
		}
	}

	
	/*else
	{
		double depthRatio = depth / compress( cvGetReal2D( length, height - 1 - j, i ), beta );
		cvSetReal2D( carve, height - 1 - j, i, compress( offset, beta )*depthRatio );
	}*/
		
}

void generateCarve( IplImage *angleImg, float weight=0.5, float depth=0.1, float beta=5, float spread=0.33, float gamma=0.5 )
{
	int height = angleImg->height;
	int width = angleImg->width;

	carve = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
	carve2 = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
	IplImage *pathImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *r = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *g = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage *b = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
	cvCopy(sample, r);
	cvCopy(sample, g);
	cvCopy(sample, b);


	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( !bgMask[ i*height + j ] )
			{
				double brightness = cvGetReal2D(sample, height - 1 - j, i);
				if( brightness )
				{
					//cvSetReal2D(r, height - 1 - j, i, 255);
					
					double multiple = abs( angle / (M_PI/4) );

					int length = round( 0.25 * sqrt( (float)height*width ) * samplingRatio + 10.0 * (1 - brightness/255) );
					float x = i;
					float y = height - 1 - j;
					float xT = i;
					float yT = height - 1 - j;

					vector< pair<CvPoint, CvPoint> > segmentList;
					int k;
					for(k=length; k > 0 ; k--)
					{
						CvPoint p1 = cvPoint( x, y );
						CvPoint p2 = cvPoint( xT, yT );
						double angle = cvGetReal2D(angleImg, y, x);
						if(angle == 100) break;
						double angleT = cvGetReal2D(angleImg, yT, xT);
						x += cos( angle ) - sin( angle ) * M_PI/2 * sin( weight * sqrt( pow(x-i, 2) + pow(y-j, 2) ) );
						y += sin( angle ) - cos( angle ) * M_PI/2 * sin( weight * sqrt( pow(x-i, 2) + pow(y-j, 2) ) );
						xT += sin( angle );
						yT += cos( angle );

						/*if(  cvGetReal2D( r, y, x) == 0 )
						{*/
							segmentList.push_back( pair<CvPoint, CvPoint>(p1, cvPoint( x, y )) );
							if( k != 1)
							{
								cvLine( r, p1, cvPoint( x, y ), cvScalarAll(254 - length + k) );					
							}
							else
							{
								cvLine( r, p1, cvPoint( x, y ), cvScalarAll(1) );
							}
							
							//cvLine( g, p2, cvPoint( xT, yT ), cvScalarAll(255) );
						/*}
						else
						{
							break;
						}*/
					}

					for(int s=0; s < segmentList.size(); s++)
					{
						cvLine( g, segmentList[s].first, segmentList[s].second, cvScalarAll(length - k) );
					}
					segmentList.clear();

					x = i;
					y = height - 1 - j;
					xT = i;
					yT = height - 1 - j;
					
					for(k=length; k > 0 ; k--)
					{
						CvPoint p1 = cvPoint( x, y );
						CvPoint p2 = cvPoint( xT, yT );
						double angle = cvGetReal2D(angleImg, y, x);
						if(angle == 100) break;
						double angleT = cvGetReal2D(angleImg, yT, xT);
						x -= cos( angle ) - sin( angle ) * M_PI/2 * sin( weight * sqrt( pow(x-i, 2) + pow(y-j, 2) ) );
						y -= sin( angle ) - cos( angle ) * M_PI/2 * sin( weight * sqrt( pow(x-i, 2) + pow(y-j, 2) ) );
						xT -= sin( angle );
						yT -= cos( angle );

						/*if(  cvGetReal2D( r, y, x) == 0 )
						{*/
							segmentList.push_back( pair<CvPoint, CvPoint>(p1, cvPoint( x, y )) );
							if( k != 1)
							{
								cvLine( r, p1, cvPoint( x, y ), cvScalarAll(255 - length + k) );					
							}
							else
							{
								cvLine( r, p1, cvPoint( x, y ), cvScalarAll(1) );
							}
							//cvLine( g, p1, cvPoint( x, y ), cvScalarAll(length) );
							//cvLine( g, p2, cvPoint( xT, yT ), cvScalarAll(255) );
						/*}
						else
						{
							break;
						}*/
					}

					for(int s=0; s < segmentList.size(); s++)
					{
						cvLine( g, segmentList[s].first, segmentList[s].second, cvScalarAll(length - k) );
					}

					IplImage *bgImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_8U, 1);
					Relief2Image( bgMask, bgImg);
					bgFilter( r, bgImg );
				}
			}
		}
	}

	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			if( cvGetReal2D( r, height - 1 - j, i) )
			{
			
				int end=0;
				int x,y;
				for(int m= -1; m <= 1 && end <=1 ; m++)
				{
					for(int n = -1; n <= 1 && end <=1 ; n++)
					{
						if( m!=0 || n!=0 ) 
						{						
							if( cvGetReal2D( r, height - 1 - (j+m), i+n) )
							{
								end++;	
								if( end == 2 )
								{
									if( x == i+n )
									{
										if( abs( y - (j+m) ) == 1 )
										{
											end--;
										}
									}
									else if( y == j+m )
									{
										if( abs( x - (i+n) ) == 1 )
										{
											end--;
										}
									}
								}

								x = i+n;
								y = j+m;
							}				
						}

					}
				}

				if( end == 1 )
				{
					float length = cvGetReal2D( g, height - 1 - j, i );
					if(  round( length ) != 0.0 )
					{
						double depthRatio = depth / compress( length, beta );
						cvSetReal2D( carve, height - 1 - j, i, compress( 1, beta )*depthRatio );
					
						nextNeighbor( r, g, i, j, 2, depth, beta, spread, gamma );
					}
					/*double before = cvGetReal2D( r, height - 1 - j, i);
					int min=255;
					int height = src->height;

					int i, j;
					for(int m= -1; m <= 1; m++)
					{
						for(int n = -1; n <= 1; n++)
						{
							if( m==0 && n==0 ) break;

							double red = cvGetReal2D( r, height - 1 - (y+n), x+m );
							if( red > before && red < min )
							{
								min = red;
								i = x+m;
								j = y+n;
							}
						}
					}
					cvSetReal2D( carve, height - 1 - j, i, compress( length, beta )*depthRatio );
					nextNeighbor( src, i, j, length+1);*/				
					
				}
			}

		}
	}

	cvMax( carve, carve2, carve);

	double min, max;
	cvMinMaxLoc(carve, &min, &max);
	double scale = 255.0 / (max - min);   
	double shift = -min * scale;  
	IplImage *show =  cvCreateImage( cvGetSize(carve),  IPL_DEPTH_8U, 1);
	cvConvertScaleAbs(carve, show, scale, shift);	

	cvNamedWindow("Carve Depth", 1);
	cvShowImage("Carve Depth", show);

	cvMerge(b, g, r, 0, pathImg);
	cvNamedWindow("Carve Path", 1);
	cvShowImage("Carve Path", pathImg);
}

void equalizeHist(const vector<GLfloat> &src, vector<GLfloat> &dst, IplImage *sample, IplImage *gradient=NULL, int aperture=33, int srcHeight=0)
{	
	if(srcHeight)
	{}
	else
	{
		srcHeight = sqrt((float)src.size());
	}
	int srcWidth = src.size() / srcHeight;
	
	vector<GLfloat> weight;
	if( gradient != NULL)
	{		
		if( srcHeight != gradient->height || srcWidth != gradient->width ) return;

		compress(gradient);
		Image2Relief(gradient, weight);
	}

	int ext = (aperture-1) / 2;
	/*if( gradient != NULL)
	{
		gradientWeight(weight, ext, srcHeight);
	}*/
	
	/***** Equalization on Sample & Edge*****/
	for(int i=0; i< srcWidth; i++)
	{
		//#pragma omp parallel for private(hist)
		for(int j=0; j< srcHeight; j++)
		{
			if( cvGetReal2D(sample, srcHeight - 1 - j, i) || cvGetReal2D(edge, srcHeight - 1 - j, i) )
			{
			
				float hist[ HistogramBins+1 ];
				for(int k=0; k<= HistogramBins; k++)
				{
					hist[k] =0;
				}
				
				if( src[ i*srcHeight + j ] == 0 )
				{
					dst[ i*srcHeight + j ] = 0;
				}

				else
				{
					int extU = -ext,extD = ext,extL = -ext,extR = ext;
					
						if( ext > j)		extL = -j;
						if( ext > i)		extU = -i;
						if( j+ext >= srcHeight)		extR = srcHeight - 1 - j;
						if( i+ext >= srcHeight)		extD = srcHeight - 1 - i;					

					for(int p=extU; p <= extD; p++)
					{
							//#pragma omp parallel for
							for(int q=extL; q <= extR; q++)
							{
								if( gradient == NULL)
								{
									hist [ (int) (src.at( (i+p)*srcHeight + j+q)  *HistogramBins) ] += 1;
								}
								else
								{
									hist [ (int) (src.at( (i+p)*srcHeight + j+q)  *HistogramBins) ] += weight[ (i+p)*srcHeight + j+q ] * distanceWeight(i, j, i+p, j+q, ext);;
								}
							}
					}

					float cumulative =  0;
					for(int bin=1; bin < HistogramBins+1; bin++)
					{
						cumulative += hist[bin];
					}
					
					/*float test[6] = {0, 50, 100, 200, 150, 100};
					redistribute(test, 600, 1);*/
					//clipping(hist, cumulative, scalingFactor*HistogramBins/10000);
					redistribute(hist, cumulative, scalingFactor*HistogramBins/10000);

					float sum[ HistogramBins+1 ];
			
					
					for(int bin=0; bin < src[ i*srcHeight + j] * HistogramBins + 1; bin++)
					{
						if( bin <= 1 )
						{
							sum[bin] = 0;
							//sum[bin] =  hist[bin] * 1000 /  number;
							/*cout << "Sum " << sum[i] << std::endl;
							sum2[i] =  cvGetReal1D(lHist1->bins, i);*/
						}
						else if( bin == 2)
						{
							sum[bin] = sum[bin-1] + ( hist[2] - hist[1]) /* *HistogramBins*/ / cumulative;
						}
						else
						{
							sum[bin] = sum[bin-1] + hist[bin] /* *HistogramBins*/  / cumulative;
							//sum2[i] = sum2[i-1] + cvGetReal1D(lHist1->bins, i);
						}
					}
					
					dst[ i*srcHeight + j ] = sum [ (int) (src[ i*srcHeight + j ]  * HistogramBins) ] ;
				}

				//dst.clear();
				/*if(src.at( i*srcHeight + j) == 0)
				{
					dst.push_back(0);
				}
				else
				{
					dst.push_back( sum [ (int) (src.at( i*srcHeight + j)  *1000) ]  );
				}*/
			}

		}
	}
	/***** Equalization on Sample *****/

	vector<GLfloat> carve;
	carve.resize(src.size(), 0);
	//knninterpolation(dst, dst, carve);

	//GLfloat carveMax = 0;
	//for(int i=0; i< srcHeight; i++)
	//{
	//	for(int j=0; j< srcHeight; j++)
	//	{
	//		//if( !bgMask[ i*height + j ] )
	//		//{
	//			if( carve[ i*srcHeight + j ] > carveMax ) carveMax = carve[ i*srcHeight + j ];
	//		//}
	//	}
	//}

	//for(int i=0; i< srcHeight; i++)
	//{
	//	for(int j=0; j< srcHeight; j++)
	//	{
	//		if( !bgMask[ i*srcHeight + j ] )
	//		{
	//			dst[ i*srcHeight + j ] -= carve[ i*srcHeight + j ] *0.5 / carveMax;
	//		}
	//	}
	//}
	
}

void equalizeHist(const vector<GLfloat> &src, vector<GLfloat> &dst, int spacing = 1, IplImage *mask=NULL, IplImage *gradient=NULL, int aperture=33, int srcHeight=0, bool ratioOn=true)
{	
	if(srcHeight)
	{}
	else
	{
		srcHeight = sqrt((float)src.size());
	}
	int srcWidth = src.size() / srcHeight;

	vector<GLfloat> weight;
	if( gradient != NULL)
	{		
		compress(gradient);
		Image2Relief(gradient, weight);
	}

	
	/*if( gradient != NULL)
	{
		gradientWeight(weight, ext, srcWidth);
	}*/

	/*IplImage *subedge = cvCreateImage( cvSize(edge->width, edge->height), IPL_DEPTH_8U, 1);
	cvCopy(edge, subedge);

	for(int i=1; i<spacing; i*=2)
	{
		IplImage *temp = cvCreateImage( cvSize(subedge->width/2, subedge->height/2), IPL_DEPTH_8U, 1);
		cvPyrDown(subedge, temp);
		subedge = cvCreateImage( cvSize(subedge->width/2, subedge->height/2), IPL_DEPTH_8U, 1);
		cvCopy(temp, subedge);
	}*/

	int ext = (aperture-1) / 2;
	int extU, extD, extL, extR;
	
	for(int i=spacing/2; i< srcWidth; i+=spacing)
	{
		//#pragma omp parallel for private(hist)
		for(int j=spacing/2; j< srcHeight; j+=spacing)
		{
			
			if(mask == NULL)
			{
				mask = cvCreateImage( cvSize(srcWidth, srcHeight), IPL_DEPTH_8U, 1);
				cvSet(mask, cvScalar(1));
			}
			
			if( cvGetReal2D(mask, srcHeight - 1 - j, i) )
			{
			
				float hist[ HistogramBins+1 ];
				for(int k=0; k<= HistogramBins; k++)
				{
					hist[k] =0;
				}
				
				//extractOutline(src, outlineMask, srcHeight, 1);
				bool distribute = true, broken = false;
				
				if( src[ i*srcHeight + j ] == 0 )
				{
					dst[ i*srcHeight + j ] = 0;
				}

				else
				{
					int countAll = 0, countOutline = 0, countFG = 0;
					//for( ext=(aperture-1) / 2; ext >=16; ext /= 2 )
					//{
					//
						extU = -ext; extD = ext; extL = -ext; extR = ext;
						
							if( ext > j)		extL = -j;
							if( ext > i)		extU = -i;
							if( j+ext >= srcHeight)		extR = srcHeight - 1 - j;
							if( i+ext >= srcHeight)		extD = srcHeight - 1 - i;					

					//	for(int p=extU; p <= extD && !broken; p++)
					//	{
					//		//#pragma omp parallel for
					//		for(int q=extL; q <= extR; q++)
					//		{
					//			if( outlineMask[ (i+p)*srcHeight + j+q ] )
					//			{									
					//				if(ext <= 16)
					//				{
					//					distribute = false;
					//				}
					//				broken = true;
					//				break;
					//			}

					//		}
					//	}
					//}

					for(int p=extU; p <= extD; p++)
					{
							//#pragma omp parallel for
							for(int q=extL; q <= extR; q++)
							{
								countAll++;
								if( src[ (i+p)*srcHeight + j+q ] )
								{
									countFG++;
									/*if( outlineMask[ (i+p)*srcHeight + j+q ] )
									{
										countOutline++;
									}*/
									
									if( gradient == NULL)
									{
										hist [ (int) (src.at( (i+p)*srcHeight + j+q)  *HistogramBins) ] += 1;
									}
									else
									{
										hist [ (int) (src.at( (i+p)*srcHeight + j+q)  *HistogramBins) ] += weight[ (i+p)*srcHeight + j+q ] * distanceWeight(i, j, i+p, j+q, ext);;
									}
								}
							}
					}

					float cumulative =  0;
					for(int bin=1; bin < HistogramBins+1; bin++)
					{
						cumulative += hist[bin];
					}


					static int isHist;
					int histMax=0;
					if( isHist == 1920 )
					{						
						histImage1 = cvCreateImage( cvSize(500, 500), IPL_DEPTH_8U, 1);
						histImage2 = cvCreateImage( cvSize(500, 500), IPL_DEPTH_8U, 1);
						cvSet( histImage1, cvScalar(208) );
						cvSet( histImage2, cvScalar(208) );

						for(int i=0; i < HistogramBins; i++)
						{
							//hist[i] = bucket[i].size();
							if(hist[i] > histMax)	histMax = hist[i];
						}	
						
						int bin_w = cvRound((double)histImage1->width/HistogramBins);

						for(int i=0; i < HistogramBins; i++)
						{
							//int value = 255 - (double)(hist[i]*255)/histMax;
							//cvSetReal2D(histImage, i, isHist, 255 - (double)(hist[i]*255)/histMax);
							cvRectangle( histImage1, cvPoint(i*bin_w, histImage1->height),
																					cvPoint((i+1)*bin_w, 
																					histImage1->height*( 1 - hist[i]/histMax*0.9 ) ),
												cvScalarAll(0), -1, 8, 0 );
						}

						cvNamedWindow("histogram before", 1);
						cvShowImage("histogram before", histImage1);
						
					}
					
					/*float test[6] = {0, 50, 100, 200, 150, 100};
					redistribute(test, 600, 1);*/
					
					if( ratioOn )
					{
						float ratio = (float) countFG / countAll;
						redistribute(hist, cumulative, scalingFactor, ratio);
						//clipping(hist, cumulative, scalingFactor*HistogramBins/10000);
					}
					else
					{
						redistribute(hist, cumulative, scalingFactor);
						//clipping(hist, cumulative, scalingFactor*HistogramBins/10000);
					}

					if( isHist == 1920 )
					{
						
						int bin_w = cvRound((double)histImage2->width/HistogramBins);

						for(int i=0; i < HistogramBins; i++)
						{
							//int value = 255 - (double)(hist[i]*255)/histMax;
							//cvSetReal2D(histImage, i, isHist, 255 - (double)(hist[i]*255)/histMax);
							cvRectangle( histImage2, cvPoint(i*bin_w, histImage2->height),
																				   cvPoint((i+1)*bin_w, 
																				   histImage2->height*( 1 - hist[i]/histMax*0.9) ) ,
												cvScalarAll(0), -1, 8, 0 );
						}

						cvNamedWindow("histogram after", 1);
						cvShowImage("histogram after", histImage2);
						
					}
					isHist++;

					float sum[ HistogramBins+1 ];
			
					
					for(int bin=1; bin < src[ i*srcHeight + j] * HistogramBins + 1; bin++)
					{
						if( bin <= 1 )
						{
							sum[bin] = 0;
							//sum[bin] =  hist[bin] * 1000 /  number;
							/*cout << "Sum " << sum[i] << std::endl;
							sum2[i] =  cvGetReal1D(lHist1->bins, i);*/
						}
						else if( bin == 2)
						{
							sum[bin] = sum[bin-1] + ( hist[2] - hist[1]) /* *HistogramBins*/ / cumulative;
						}
						else
						{
							sum[bin] = sum[bin-1] + hist[bin] /* *HistogramBins*/  / cumulative;
							//sum2[i] = sum2[i-1] + cvGetReal1D(lHist1->bins, i);
						}
					}
					
					dst[ i*srcHeight + j ] = sum [ (int) (src[ i*srcHeight + j ]  * HistogramBins) ] ;
				}

				//dst.clear();
				/*if(src.at( i*srcHeight + j) == 0)
				{
					dst.push_back(0);
				}
				else
				{
					dst.push_back( sum [ (int) (src.at( i*srcHeight + j)  *1000) ]  );
				}*/
			}

		}
	}

	if( spacing !=1 )
	{
	
		IplImage *srcImg = cvCreateImage( cvSize(gradient->width/spacing, gradient->height/spacing), IPL_DEPTH_32F, 1);

		for(int i=0; i< srcWidth/spacing; i++)
		{
			for(int j=0; j< srcHeight/spacing; j++)
			{
				cvSetReal2D(srcImg, srcHeight/spacing - 1 - j, i, dst[ (int)((i+0.5)*spacing)*srcHeight + (int)((j+0.5)*spacing) ] );
			}
		}

		/*****	interpolation	*****/
		for(int i=1; i <= (int) log2( (double)spacing ); i++)
		{
			IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
			cvPyrUp(srcImg, dstImg);
			srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
			cvCopy(dstImg, srcImg);
		}
		Image2Relief(srcImg, dst);
		/*****	interpolation	*****/
	}
	
	//if( isHist == 1920 )
	//{
	//	
	//	int bin_w = cvRound((double)histImage3->width/HistogramBins);

	//	for(int i=0; i < HistogramBins; i++)
	//	{
	//		//int value = 255 - (double)(hist[i]*255)/histMax;
	//		//cvSetReal2D(histImage, i, isHist, 255 - (double)(hist[i]*255)/histMax);
	//		cvRectangle( histImage3, cvPoint(i*bin_w, histImage3->height),
	//															   cvPoint((i+1)*bin_w, 
	//															   histImage3->height*( 1 - hist[i]/histMax*0.9) ) ,
	//							cvScalarAll(0), -1, 8, 0 );
	//	}

	//	cvNamedWindow("histogram final", 1);
	//	cvShowImage("histogram final", histImage3);
	//	
	//}
	//isHist++;
	
	/*for(int i=0; i< srcHeight; i++)
	{
		for(int j=0; j< srcHeight; j++)
		{		
			if(src.at( i*srcHeight + j) == 0)
			{
				dst.push_back(0);
			}
			else
			{
				dst.push_back( sum [ (int) (src.at( i*srcHeight + j)  *HistogramBins) ]  );
			}
		}
	}*/
		
}

void vectorAdd(const vector<GLfloat> &src1, const vector<GLfloat> &src2, vector<GLfloat> &dst)
{
	if( src1.size() != src2.size() || src2.size() != dst.size() || dst.size() != src1.size() ) return;
	else
	{
		for(int i=0; i < src1.size(); i++)
		{
			dst[i] = src1[i] + src2[i];
		}
	}
}

//void vectorAdd(vector<GLfloat> *src1, vector<GLfloat> *src2, vector<GLfloat> *dst)
//{
//	if( src1->size() != src2->size() || src2->size() != dst->size() || dst->size() != src1->size() ) return;
//	else
//	{
//		for(int i=0; i < src1->size(); i++)
//		{
//			dst->at(i) = src1->at(i) + src2->at(i);
//		}
//	}
//}

void vectorScale(const vector<GLfloat> &src, vector<GLfloat> &dst, float scale)
{
	if( src.size() != dst.size() ) return;
	for(int i=0; i < src.size(); i++)
	{
			dst[i] = src[i] * scale;
	}
}

//void vectorScale(vector<GLfloat> *src, vector<GLfloat> *dst, float scale)
//{
//	if( src->size() != dst->size() ) return;
//	for(int i=0; i < src->size(); i++)
//	{
//			dst->at(i) = src->at(i) * scale;
//	}
//}

void vectorReplace(const vector<GLfloat> &src, const vector<GLfloat> &replace, vector<GLfloat> &dst)
{
	if( src.size() != replace.size() || replace.size() != dst.size() || dst.size() != src.size() ) return;
	else
	{
		for(int i=0; i < src.size(); i++)
		{
			if( replace[i] )
			{
				dst[i] = replace[i];
			}
			else
			{
				dst[i] = src[i];
			}
		}
	}
}

void DrawRelief(const vector<GLfloat> &src, GLdouble *pThreadRelief, GLdouble *pThreadNormal)
{
		glColor3f(0.6, 0.6, 0.6);
		
		//glTranslated(-0.2, 0, 0);
		
		glRotatef(reliefAngleX, 1, 0, 0);
		glRotatef(reliefAngleY, 0, 1, 0);
		glRotatef(reliefAngleZ, 0, 0, 1);
		
		glScalef(0.01*scale, 0.01*scale, 1);

		glTranslated( -(winHeight/2-boundary), -(winWidth/horizontalSplit/2-boundary), 0.55/*sqrt( (float)(3*winHeight*winHeight - winWidth/horizontalSplit*winWidth/horizontalSplit)/4 )*/ );
		
		glScalef(1, 1, outputHeight);

		if( winHeight || winWidth )
		{
			int height = winHeight - boundary*2;
			int width = src.size() / height;
			/*if(mesh)
			{	*/
				if( profile1 || profile2 )
				{
					reliefProfile.clear();
					for(int i=0; i < width - 1; i++)
					{			
						
						/*if( winHeight%2 == 0 )
						{
							reliefProfile.push_back( (src[(height-1)/2 + i*height] + src[(height+1)/2 + i*height]) / 2 );
						}
						else
						{
							reliefProfile.push_back( src[height/2 + i*height] );
						}*/
						reliefProfile.push_back( ( src[ floor( (height-1)/2.0 ) + i*height ] + src[ ceil( (height-1)/2.0 ) + i*height ] ) / 2 );
						//reliefProfile.push_back( ( src[ floor( (width-1)*3/4.0 )*height + i ] + src[ ceil( (width-1)*3/4.0 )*height + i ] ) / 2 );

						for(int j=0; j < height - 1; j++)
						{
							glBegin(GL_TRIANGLES);
							glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 ) *3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 );
							glEnd();
							
							glBegin(GL_TRIANGLES);
							glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 )  *3 + 3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3);
							glEnd();
						}		
					}

					char *name;
					static int id;
					IplImage *profileImg = cvCreateImage( cvSize( width, height ), IPL_DEPTH_8U, 1);
					cvSet( profileImg, cvScalar(208) );
					DrawProfile(reliefProfile, profileImg, 0.9);
					int c = _snprintf( NULL, 0, "Relief Profile %d", id+1 );
					name = new char[ c + 1 ];
					_snprintf(name, c+1, "Relief Profile %d", id+1);
					id++;
					cvNamedWindow(name, 1);
					cvShowImage(name, profileImg);

					profile1 = false;
					profile2 = false;
				}
				
				else
				{
					for(int i=0; i < width - 1; i++)
					{			
						for(int j=0; j < height - 1; j++)
						{
							glBegin(GL_TRIANGLES);
							glNormal3dv( pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 ) *3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 );
							glEnd();
							
							glBegin(GL_TRIANGLES);
							glNormal3dv( pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 )  *3 + 3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3);
							glEnd();
						}		
					}
				}
			//}
		}
		
		//for(int i=0; i < height.size() / (winHeight - boundary*2) - 1; i++)
		//{
		//	for(int j=0; j < winHeight - boundary*2 -1; j++)
		//	{
		//		int position = i*(winHeight-boundary*2) + j;
		//		if( height.at(position) >= 0 )
		//		{
		//			i++;
		//			i--;
		//		}
		//		if( height.at(position) >= 0.5 )
		//		{
		//			glColor3f(1.0, 1.0, 1.0);
		//		}
		//		
		//		GLdouble normal[3];
		//		GLdouble v1[3][3] =  { {i, j, height.at(position)}, {i+1, j, height.at(position+winHeight - boundary*2)}, { i, j+1, height.at(position+1)} };
		//		
		//		glBegin(GL_TRIANGLES);
		//			setNormal( v1, normal );
		//			glNormal3dv(normal);
		//			glVertex3d( i, j, height.at(position) );
		//			glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );
		//			glVertex3d( i, j+1, height.at(position+1) );
		//			
		//		GLdouble v2[3][3] = { {i, j+1, height.at(position+1)}, {i+1, j, height.at(position+winHeight - boundary*2)}, {i+1, j+1, height.at(position+winHeight - boundary*2 + 1)} };
		//			setNormal( v2, normal);
		//			glNormal3dv(normal);
		//			glVertex3d( i, j+1, height.at(position+1)  );
		//			glVertex3d( i+1, j, height.at(position+winHeight - boundary*2) );				
		//			glVertex3d( i+1, j+1, height.at(position+winHeight - boundary*2 + 1) );
		//		glEnd();
		//		//glColor3f(1.0, 0.0, 0.0);
		//	}
		//}
		//swglmDraw(MODEL);

	glPushMatrix();
		glTranslated(0, 2, 0);
		glMultMatrixd(TRACKM);

		
	glPopMatrix();
}

void DrawRelief(vector<GLfloat> *src, GLdouble *pThreadRelief, GLdouble *pThreadNormal)
{
		glColor3f(0.6, 0.6, 0.6);
		
		//glTranslated(-0.2, 0, 0);
		
		glRotatef(reliefAngleX, 1, 0, 0);
		glRotatef(reliefAngleY, 0, 1, 0);
		glRotatef(reliefAngleZ, 0, 0, 1);
		
		glScalef(0.01*scale, 0.01*scale, 1);

		glTranslated(-2/0.01, -2/0.01, 0.4);
		
		glScalef(1, 1, outputHeight);

		if( winHeight || winWidth )
		{
			int height = winHeight - boundary*2;
			int width = src->size() / height;
			/*if(mesh)
			{	*/
				if( profile1 || profile2 )
				{
					reliefProfile.clear();
					for(int i=0; i < width - 1; i++)
					{			
						reliefProfile.push_back( ( src->at( floor( (height-1)/2.0 ) + i*height ) + src->at( ceil( (height-1)/2.0 ) + i*height ) ) / 2 );
						
						for(int j=0; j < height - 1; j++)
						{
							glBegin(GL_TRIANGLES);
							glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 ) *3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 );
							glEnd();
							
							glBegin(GL_TRIANGLES);
							glNormal3dv(pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 )  *3 + 3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3);
							glEnd();
						}		
					}

					char *name;
					static int id;
					IplImage *profileImg = cvCreateImage( cvSize( width, height*2 ), IPL_DEPTH_8U, 1);
					cvSet( profileImg, cvScalar(208) );
					DrawProfile(reliefProfile, profileImg);
					int c = _snprintf( NULL, 0, "Relief Profile %d", id+1 );
					name = new char[ c + 1 ];
					_snprintf(name, c+1, "Relief Profile %d", id+1);
					id++;
					cvNamedWindow(name, 1);
					cvShowImage(name, profileImg);

					profile1 = false;
					profile2 = false;
				}
				
				else
				{
					for(int i=0; i < width - 1; i++)
					{			
						for(int j=0; j < height - 1; j++)
						{
							glBegin(GL_TRIANGLES);
							glNormal3dv( pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 ) *3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + (j+1) ) * 3 );
							glEnd();
							
							glBegin(GL_TRIANGLES);
							glNormal3dv( pThreadNormal + ( ( i*(winHeight - boundary*2)  +  j)*2 )  *3 + 3);
							glVertex3dv( pThreadRelief + ( i*(winHeight - boundary*2) + j+1 ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + j ) * 3 );
							glVertex3dv( pThreadRelief + ( (i+1)*(winHeight - boundary*2) + (j+1) ) * 3);
							glEnd();
						}		
					}
				}
			//}
		}

	glPushMatrix();
		glTranslated(0, 2, 0);
		glMultMatrixd(TRACKM);

		
	glPopMatrix();
}

//Bas-Relief
void BilateralDetailBase(void)
{	
	
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/

	//glPushMatrix();
		//glPolygonMode(GL_FRONT,GL_LINE);
		//multiple trackball matrix
		//glMultMatrixd(TRACKM);
		
		
		//GLfloat maxDepth=0,minDepth=1;
		
		//	disp++;
		//	height.clear();
		//	
		//	for(int i=boundary; i<winWidth*0.5 - boundary; i++)
		//	{
		//		for(int j=boundary; j<winHeight - boundary; j++)
		//		{
		//			GLfloat depth;
		//			glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

		//			if( maxDepth < depth && depth !=1)
		//			{
		//				maxDepth = depth;
		//			}
		//			if( minDepth > depth)
		//			{
		//				minDepth = depth;
		//			}
		//			height.push_back(depth);
		//			
		//		}
		//	}
		//	int enhance = 50;
		//	double maxDepthPrime = compress(maxDepth*enhance, 5, 5);
		//	double minDepthPrime = compress(minDepth*enhance, 5, 5);

		//	for(int i=0; i<height.size(); i++)
		//	{			
		//		if ( height.at(i) == 1 )	//infinite point
		//		{
		//			height.at(i) = 0;
		//		}
		//		else
		//		{
		//			height.at(i) = compress(height.at(i)*enhance, 5, 5);
		//			height.at(i) = maxDepthPrime - height.at(i);
		//			//height.at(i) = maxDepth - height.at(i);	//transfer depth to height
		//			height.at(i) /= (maxDepthPrime - minDepthPrime );	//normalize to [0,1]
		//		}
		//	}
			//for(int i=0; i<height.size(); i++)
			//{
			//	if ( height.at(i) == 1 )	//infinite point
			//	{
			//		height.at(i) = 0;
			//	}
			//	else
			//	{
			//		height.at(i) = ( height.at(i) - minDepth ) / (maxDepth - minDepth) ;	//normalize to [0,1]
			//		height.at(i) = 1 - height.at(i);	//transfer depth to height
			//	}
			//}
			
			//equalizeHist(height[0], referenceHeight);

			/*float maxHeight=0, minHeight=1000;
			for(int i=0; i < referenceHeight.at(0).size(); i++)
			{
				float height = referenceHeight.at(i);
				if( maxHeight < height )
				{
					maxHeight = height;
				}
				if( minHeight > height && height !=0 )
				{
					minHeight = height;
				}
			}
			float range = maxHeight - minHeight;
			for(int i=0; i < referenceHeight.size(); i++)
			{
				referenceHeight.at(i) /= 1000;
			}*/

			//BuildRelief(referenceHeight, pThreadEqualizeRelief, pThreadEqualizeNormal);
			
			//heightList.push_back(height);
			vector<GLfloat> height2, upHeight/*, laplace[pyrLevel - 1]*/;
			IplImage *img, *img2, *imgGa, *imgLa, *bgImg, *base_img, *detail_img;
			/*subSample(heightList.at(0), height2);
			heightList.push_back(height2);*/
			//upSample(height2, upHeight);
			//reliefSub(heightList.at(0), upHeight, laplace);

			/*laplacianFilter(height[1], *laplace);
			laplaceList.push_back(*laplace);
			laplacianFilter(height[2], *laplace);
			laplaceList.push_back(*laplace);*/
			IplImage *tempImg;
			//for(int i=1;i< pyrLevel - 1;i++)
			//{
			//	/*tempImg = cvCreateImage( cvSize(imgPyr[i]->width, imgPyr[i]->height), IPL_DEPTH_32F, 1);
			//	cvGetImage(imgPyr[i], tempImg);
			//	Image2Relief(tempImg, height[i]);
			//	laplacianFilter(height[i], laplace);
			//	laplaceList.push_back(laplace);*/
			//	laplacianFilter(heightPyr[i], laplace[i]);
			//	laplaceList.push_back(laplace[i]);
			//	recordMax(laplace[i]);
			//}
			
			int height = (winHeight - boundary*2) / pow(2.0, pyrLevel - 1);
			int width = heightPyr[pyrLevel - 1].size() / height;
			//thresholding
			/*for(int i=0; i < width; i++)
			{
				for(int j=0; j < height; j++)
				{
					int adress = i*height + j;
					if( heightPyr[pyrLevel - 1][adress] > threshold )
					{
						heightPyr[pyrLevel - 1][adress] = 0;
					}
				}
			}*/

			//collapse the pyramid
			vector<GLfloat> outline;
			/*for(int i=0;i < height[pyrLevel - 1].size(); i++)
			{
				compressedH->push_back( height[pyrLevel - 1].at(i) );
			}*/
			
			imgGa = cvCreateImage( cvSize( width*2,  width*2), IPL_DEPTH_32F, 1);
			imgLa = cvCreateImage( cvSize( width*2,  width*2), IPL_DEPTH_32F, 1);
			img = cvCreateImage( cvSize( width*2,  width*2), IPL_DEPTH_32F, 1);

			
			//cvGetImage(imgPyr[pyrLevel - 1], tempImg);

			/*****	gaussian layer	*****/
			bgImg = cvCreateImage( cvGetSize(imgPyr[0]), IPL_DEPTH_32F, 1);
			Relief2Image( bgMask, bgImg, winHeight - boundary*2);
			
			for(int i=0; i < pyrLevel-1; i++)
			{
				tempImg = cvCreateImage( cvSize(bgImg->width/2, bgImg->height/2), IPL_DEPTH_32F, 1);
				cvPyrDown(bgImg, tempImg);
				bgImg = cvCreateImage( cvGetSize(tempImg), IPL_DEPTH_32F, 1);
			}
			cvCopy(tempImg, bgImg);

			tempImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
			Relief2Image(	heightPyr.at( pyrLevel - 1), tempImg, (winHeight - boundary*2)/pow(2.0, pyrLevel - 1) );

			//bgFilter( tempImg, bgImg);
			cvPyrUp( tempImg, imgGa );
			
			/*****	bilateral filter	*****/
			//base_img = cvCreateImage( cvGetSize( imgGa ), IPL_DEPTH_8U, 1);
			//detail_img = cvCreateImage( cvGetSize( imgGa ), IPL_DEPTH_8U, 1);

			////cvConvertScaleAbs(img2, base_img, 255, 0);
			//cvConvertScaleAbs(imgGa, detail_img, 255, 0);
			//cvSmooth(detail_img, base_img, CV_BILATERAL, 0, 0, 0.95, 11);
			//cvSub(detail_img, base_img, detail_img);
			//
			//for(int i=0; i < cvGetSize( imgGa ).width; i++)
			//{
			//	for(int j=0; j < cvGetSize( imgGa ).height; j++)
			//	{
			//		cvSetReal2D( imgGa, j, i, cvGetReal2D(detail_img, j, i) / 255 );
			//	}
			//}
			/*****	bilateral filter	*****/

			tempImg = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);	
			
			height = imgPyr[pyrLevel - 2]->height;
			width = imgPyr[pyrLevel - 2]->width;
			for(int i=pyrLevel - 2; i > 0; i--)
			{
				width *= 2;
				height *= 2;
				//img = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				img2 = cvCreateImage( cvSize( width,  height ), IPL_DEPTH_32F, 1);
				cvSetZero(img2);

				cvPyrUp(imgGa, img2);	//upsample the smallest laplace
				
				imgGa = cvCreateImage( cvSize( width,  height ), IPL_DEPTH_32F, 1);
				cvCopy(img2, imgGa);
			}
			

			/*****	gaussian  layer	*****/

			/*****	laplacian layer	*****/
			Relief2Image(laplaceList.at( pyrLevel - 2), imgLa, (winHeight - boundary*2)/pow(2.0, pyrLevel - 2) );
			
			//cvAdd(imgGa, imgLa, img);
			cvCopy(imgLa, img);
			for(int i=pyrLevel - 2; i > 0; i--)
			{
				width = imgPyr[i]->width;
				height = imgPyr[i]->height;
				//img = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				img2 = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
				cvSetZero(img2);
				imgLa = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);

				cvPyrUp(img, img2);	//upsample the smallest laplace
				Relief2Image(laplaceList.at(i-1), imgLa, (winHeight - boundary*2)/pow(2.0, i-1));

				img = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
				cvAdd(img2, imgLa, img);
				//cvCopy(img2, img);
				/*Image2Relief(img2, compressedH);
				reliefAdd( *compressedH, laplaceList.at(i) );*/
				img2 = cvCreateImage( cvGetSize( img ), IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(img, img2, 128, 128);
				char a[15];
				sprintf(a,"%d", i);
				cvNamedWindow(a, 1);
				cvShowImage(a, img2);

			}
			/*****	laplacian layer	*****/

			Image2Relief(imgGa, compressedH);
			bgFilter(compressedH, bgMask, winHeight - boundary*2);

			/*for(int i=0; i < compressedH.size(); i++)
			{
				outline.push_back(0);
			}*/
			extractOutline(compressedH, outline, winHeight - boundary*2);
			
			vector<GLfloat>::iterator first = outline.begin();
			vector<GLfloat>::iterator last = outline.end();
			outline.erase( remove(first, last, 0), outline.end() );
			first = outline.begin();
			last = outline.end();
			GLfloat outline_min = *min_element ( first, last);
			GLfloat outline_max = *max_element ( first, last);
			nth_element ( first, first + outline.size()/2, last );
			GLfloat outline_med = outline[ outline.size()/2 ];
			
			//Relief2Image(compressedH, img2);
			/*****	bilateral filter	*****/
			//if( ReliefType == 1 )
			//{
			//	base_img = cvCreateImage( cvGetSize( imgGa ), IPL_DEPTH_8U, 1);
			//	detail_img = cvCreateImage( cvGetSize( imgGa ), IPL_DEPTH_8U, 1);

			//	//cvConvertScaleAbs(img2, base_img, 255, 0);
			//	cvConvertScaleAbs(imgGa, detail_img, 255, 0);
			//	cvSmooth(detail_img, base_img, CV_BILATERAL, 0, 0, (outline_max+outline_min)/2*255, 25);
			//	cvSub(detail_img, base_img, detail_img);
			//	
			//	//img2 = cvCreateImage( cvGetSize( img ), IPL_DEPTH_32F, 1);
			//	for(int i=0; i < cvGetSize( imgGa ).width; i++)
			//	{
			//		for(int j=0; j < cvGetSize( imgGa ).height; j++)
			//		{
			//			cvSetReal2D( imgGa, j, i, cvGetReal2D(detail_img, j, i) / 255 );
			//		}
			//	}
			//}
			/*****	bilateral filter	*****/
			Relief2Image(heightPyr[0], imgGa, winHeight - boundary*2);
			/*****	linear compression	*****/
			/*double minVal,maxVal;
			
			cvMinMaxLoc(imgGa, &minVal, &maxVal);
			double val;
			for(int i=0; i < width; i++)
			{
				for(int j=0; j < height; j++)
				{					
					val = cvGetReal2D( imgGa, j, i) * outputHeight / maxVal;
					if( val )
					{
						cvSetReal2D( imgGa, j, i, val );
					}
				}
			}*/
			/*****	linear compression	*****/

			
			/*****	bilateral filter	*****/
			base_img = cvCreateImage( cvGetSize( imgGa ), IPL_DEPTH_8U, 1);
			detail_img = cvCreateImage( cvGetSize( imgGa ), IPL_DEPTH_8U, 1);

			cvConvertScaleAbs(imgGa, base_img, 255, 0);
			cvConvertScaleAbs(imgGa, detail_img, 255, 0);
			cvSmooth(detail_img, base_img, CV_BILATERAL, 0, 0, outline_med*255, 25);
			cvSub(detail_img, base_img, detail_img);
			
			for(int i=0; i < cvGetSize( imgGa ).width; i++)
			{
				for(int j=0; j < cvGetSize( imgGa ).height; j++)
				{
					cvSetReal2D( imgGa, j, i, cvGetReal2D(detail_img, j, i) / 255 );
				}
			}

			/*for(int i=0; i < cvGetSize( img0 ).width; i++)
			{
				for(int j=0; j < cvGetSize( img0 ).height; j++)
				{
					cvSetReal2D( img1, j, i, cvGetReal2D(base_img, j, i) / 255 );
				}
			}*/
			/*****	bilateral filter	*****/
			/*****	downsample	*****/
			for(int i=1; i < pyrLevel; i++)
			{
				int width = imgPyr[i]->width;
				//img = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				img2 = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				cvSetZero(img2);

				cvPyrDown(imgGa, img2);	
				
				imgGa = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				cvCopy(img2, imgGa);
			}
			/*****	downsample	*****/
			/*****	upsample	*****/
			for(int i=pyrLevel - 2; i >= 0; i--)
			{
				int width = imgPyr[i]->width;
				//img = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				img2 = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				cvSetZero(img2);

				cvPyrUp(imgGa, img2);
				
				imgGa = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				cvCopy(img2, imgGa);
			}
			/*****	upsample	*****/
			cvAdd(img, imgGa, img);
			
			Image2Relief(img, compressedH);
			bgFilter( compressedH, bgMask, winHeight - boundary*2 );
			
			first = compressedH.begin();
			last = compressedH.end();
			GLfloat max = *max_element ( first, last );
			cout << "max: " << max << endl;
			
			vector<GLfloat> gaussianH;
			Image2Relief(imgGa, gaussianH);
			bgFilter( gaussianH, bgMask, winHeight - boundary*2);
			recordMax( gaussianH );
			for(int i=0;i< compressedH.size(); i++)
			{
				compressedH.at(i) /= max; //normalize
			}
			for(int i=0;i < maxList.size(); i++)
			{
				cout << "level[" << i << "]: " << maxList[i]/max << endl;
			}
			//upHeight.clear();
			/*	height2.clear();
			upSample(heightList.at(1), height2);

			reliefAdd(height2, laplace, upHeight);
			height2.clear();
			upSample(upHeight, height2);*/
			//upHeight.clear();
			//reliefAdd(height2, laplaceList.at(0), upHeight);

			BuildRelief(compressedH, pThreadRelief, pThreadNormal);
			
			mesh1 = true;
			relief1 = false;
			profile2 = true;
	
		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
}

void linearCompressedBase(void)
{
	
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/

	//glPushMatrix();
		//glPolygonMode(GL_FRONT,GL_LINE);
		//multiple trackball matrix
		//glMultMatrixd(TRACKM);
		
		
		//GLfloat maxDepth=0,minDepth=1;
		
			//vector<GLfloat> laplace[pyrLevel - 1];
			IplImage *img, *img2, *imgGa, *imgLa, *bgImg, *base_img, *detail_img;
			
			IplImage *tempImg;
			/*for(int i=1;i< pyrLevel - 1;i++)
			{
				laplacianFilter(heightPyr[i], laplace[i]);
				laplaceList.push_back(laplace[i]);
				recordMax(laplace[i]);
			}*/
			
			/*int width = sqrt( (float) heightPyr[pyrLevel - 1].size() );
			int height = sqrt( (float) heightPyr[pyrLevel - 1].size() );*/
			int height = (winHeight - boundary*2) / pow(2.0, pyrLevel - 1);
			int width = heightPyr[pyrLevel - 1].size() / height;

			//collapse the pyramid
			vector<GLfloat> outline;
			/*for(int i=0;i < height[pyrLevel - 1].size(); i++)
			{
				compressedH->push_back( height[pyrLevel - 1].at(i) );
			}*/
			
			imgGa = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
			imgLa = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
			img = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);

			
			//cvGetImage(imgPyr[pyrLevel - 1], tempImg);

			/*****	gaussian layer	*****/
			bgImg = cvCreateImage( cvGetSize(imgPyr[0]), IPL_DEPTH_32F, 1);
			Relief2Image( bgMask, bgImg, winHeight - boundary*2);
			
			for(int i=0; i < pyrLevel-1; i++)
			{
				tempImg = cvCreateImage( cvSize(bgImg->width/2, bgImg->height/2), IPL_DEPTH_32F, 1);
				cvPyrDown(bgImg, tempImg);
				bgImg = cvCreateImage( cvGetSize(tempImg), IPL_DEPTH_32F, 1);
			}
			cvCopy(tempImg, bgImg);

			tempImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
			Relief2Image(	heightPyr.at( pyrLevel - 1), tempImg, (winHeight - boundary*2)/pow(2.0, pyrLevel - 1));

			//bgFilter( tempImg, bgImg);
			cvPyrUp( tempImg, imgGa );

			tempImg = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);	
			
			width = imgPyr[pyrLevel - 2]->width;
			for(int i=pyrLevel - 2; i > 0; i--)
			{
				width *= 2;
				//img = cvCreateImage( cvSize( width,  height ), IPL_DEPTH_32F, 1);
				img2 = cvCreateImage( cvSize( width,  height ), IPL_DEPTH_32F, 1);
				cvSetZero(img2);

				cvPyrUp(imgGa, img2);	//upsample the smallest laplace
				
				imgGa = cvCreateImage( cvSize( width,  height ), IPL_DEPTH_32F, 1);
				cvCopy(img2, imgGa);
			}
			

			/*****	gaussian  layer	*****/

			/*****	laplacian layer	*****/
			Relief2Image(laplaceList.at( pyrLevel - 2), imgLa, (winHeight - boundary*2)/pow(2.0, pyrLevel - 2));
			
			//cvAdd(imgGa, imgLa, img);
			cvCopy(imgLa, img);
			for(int i=pyrLevel - 2; i > 0; i--)
			{
				width = imgPyr[i]->width;
				height = imgPyr[i]->height;
				//img = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
				img2 = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
				cvSetZero(img2);
				imgLa = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);

				cvPyrUp(img, img2);	//upsample the smallest laplace
				Relief2Image(laplaceList.at(i-1), imgLa, (winHeight - boundary*2)/pow(2.0, i-1));

				img = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
				cvAdd(img2, imgLa, img);
				//cvCopy(img2, img);
				/*Image2Relief(img2, compressedH);
				reliefAdd( *compressedH, laplaceList.at(i) );*/
				img2 = cvCreateImage( cvGetSize( img ), IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(img, img2, 128, 128);
				char a[15];
				sprintf(a,"%d", i);
				cvNamedWindow(a, 1);
				cvShowImage(a, img2);

			}
			/*****	laplacian layer	*****/

			Image2Relief(imgGa, compressedH);
			bgFilter( compressedH, bgMask, winHeight - boundary*2 );

			for(int i=0; i < compressedH.size(); i++)
			{
				outline.push_back(0);
			}
			extractOutline(compressedH, outline, winHeight - boundary*2);
			
			vector<GLfloat>::iterator first = outline.begin();
			vector<GLfloat>::iterator last = outline.end();
			outline.erase( remove(first, last, 0), outline.end() );
			first = outline.begin();
			last = outline.end();
			GLfloat outline_min = *min_element ( first, last);
			GLfloat outline_max = *max_element ( first, last);
			nth_element ( first, first + outline.size()/2, last );
			GLfloat outline_med = outline[ outline.size()/2 ];

			double minVal,maxVal;
			Relief2Image(heightPyr[0], imgGa, winHeight - boundary*2);
			cvMinMaxLoc(imgGa, &minVal, &maxVal);
			double val;
			for(int i=0; i < width; i++)
			{
				for(int j=0; j < height; j++)
				{					
					val = cvGetReal2D( imgGa, j, i) * outputHeight / maxVal;
					if( val )
					{
						cvSetReal2D( imgGa, j, i, val );
					}
				}
			}
			cvAdd(img, imgGa, img);
			
			Image2Relief(img, compressedH);
			bgFilter( compressedH, bgMask, winHeight - boundary*2 );
			
			first = compressedH.begin();
			last = compressedH.end();
			GLfloat max = *max_element ( first, last );
			cout << "max: " << max << endl;
			
			vector<GLfloat> gaussianH;
			Image2Relief(imgGa, gaussianH);
			bgFilter( gaussianH, bgMask, winHeight - boundary*2 );
			recordMax( gaussianH );
			for(int i=0; i< compressedH.size(); i++)
			{
				compressedH.at(i) /= max; //normalize
			}
			for(int i=0; i < maxList.size(); i++)
			{
				cout << "level[" << i << "]: " << maxList[i]/max << endl;
			}

			BuildRelief(compressedH, pThreadRelief, pThreadNormal);
			
			mesh1 = true;
			relief1 = false;
			profile2 = true;
		
		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
}

//void histogramBase(const vector<GLfloat> &src, IplImage *gradientX, IplImage *gradientY, int level=0, float weight=0.5)
//{	
//	
//	/*OpenglLine(0, 0, 0, 3, 0, 0);
//	glColor3f(0, 1, 0);
//	OpenglLine(0, 0, 0, 0, 3, 0);
//	glColor3f(0, 0, 1);
//	OpenglLine(0, 0, 0, 0, 0, 3);*/ 
//		
//		
//		//GLfloat maxDepth=0,minDepth=1;
//		float maxHeight=0, minHeight=HistogramBins;
//		/*if(relief2)
//		{*/
//			IplImage *gradX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
//			IplImage *gradY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
//			cvConvertScale(gradientX, gradX, 1, 0);
//			cvConvertScale(gradientY, gradY, 1, 0);
//			cvPow(gradX, gradX, 2);
//			cvPow(gradY, gradY, 2);
//
//			IplImage *gradient = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
//			cvAdd(gradX, gradY, gradient);
//			cvPow(gradient, gradient, 0.5);		
//
//			IplImage *outlineImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
//			extractOutline(img0, outlineImg, 1);
//			bgFilter(gradient, outlineImg);
//
//			IplImage *bgImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 1);
//			Relief2Image( bgMask, bgImg, winHeight - boundary*2);
//			bgFilter(gradient, bgImg);
//			
//			int size = src.size();
//			vector<GLfloat> AHEHeight;
//			AHEHeight.resize(size);
//			compressedH.resize(size);
//			for(int i=0; i < size; i++)
//			{
//				compressedH[i] = 0;
//			}
//
//			int n=1;
//			//#pragma omp parallel for private(AHEHeight)
//			/*for(int k=1; k <= n; k++)
//			{
//				equalizeHist(sampleHeight, AHEHeight, gradient, pow(2.0, k-1) * 8*2/pow(2.0, level) + 1);
//				vectorAdd(referenceHeight, AHEHeight, referenceHeight);
//				AHEHeight.clear();
//			}
//			Relief2Image(sampleHeight, srcImg);
//
//			for(int i=1;i <= level;i++)
//			{
//				IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
//				cvPyrUp(srcImg, dstImg);
//				srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
//				cvCopy(dstImg, srcImg);
//			}
//			Image2Relief(srcImg, referenceHeight);*/
//
//			int height = (winHeight - boundary*2) /*/ pow(2.0, pyrLevel - 1)*/;
//			int width = heightPyr[0].size() / height;
//
//			IplImage *white = cvCreateImage( cvSize( width,  height), IPL_DEPTH_8U, 1);
//			cvSet(white, cvScalar(255));
//			
//			for(int k=1; k <= n; k++)
//			{
//				equalizeHist(src, AHEHeight, white, gradient, pow(2.0, k-1) * ahe_m *2 + 1, height);
//				vectorAdd(compressedH, AHEHeight, compressedH);
//				AHEHeight.clear();
//			}
//			vectorScale(compressedH, compressedH, 1.0/n);
//			
//			if(weight != 1)
//			{
//				IplImage *img, *img2, *imgGa;
//				
//				int divisor = pow(2.0, pyrLevel - 2);
//				img = cvCreateImage( cvSize( width/divisor,  height/divisor), IPL_DEPTH_32F, 1);
//				/*****	laplacian layer	*****/			
//				IplImage *imgLa = cvCreateImage( cvSize( width/divisor,  height/divisor), IPL_DEPTH_32F, 1);	
//				Relief2Image(laplaceList.at( pyrLevel - 2), imgLa, (winHeight - boundary*2)/divisor);
//				
//				//cvAdd(imgGa, imgLa, img);
//				cvCopy(imgLa, img);
//				for(int i=pyrLevel - 2; i > 0; i--)
//				{
//					width = imgPyr[i]->width;
//					height = imgPyr[i]->width;
//					//img = cvCreateImage( cvSize( width,  width), IPL_DEPTH_32F, 1);
//					img2 = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
//					cvSetZero(img2);
//					imgLa = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
//
//					cvPyrUp(img, img2);	//upsample the smallest laplace
//					Relief2Image(laplaceList.at(i-1), imgLa, (winHeight - boundary*2)/pow(2.0, i-1));
//
//					img = cvCreateImage( cvSize( width*2,  height*2), IPL_DEPTH_32F, 1);
//					cvAdd(img2, imgLa, img);
//					//cvCopy(img2, img);
//					/*Image2Relief(img2, compressedH);
//					reliefAdd( *compressedH, laplaceList.at(i) );*/
//					img2 = cvCreateImage( cvGetSize( img ), IPL_DEPTH_8U, 1);
//					cvConvertScaleAbs(img, img2, 128, 128);
//					char a[15];
//					sprintf(a,"%d", i);
//					cvNamedWindow(a, 1);
//					cvShowImage(a, img2);
//				}
//				/*****	laplacian layer	*****/
//
//				imgGa = cvCreateImage( cvGetSize( img ), IPL_DEPTH_32F, 1);
//				Relief2Image(compressedH, imgGa, winHeight - boundary*2);
//
//				double baseMin, baseMax, detailMin, detailMax;
//				cvMinMaxLoc(imgGa, &baseMin, &baseMax);
//				cvMinMaxLoc(img, &detailMin, &detailMax);
//				cvConvertScale(imgGa, imgGa, weight*(baseMax+detailMax) / baseMax );
//				cvConvertScale(img, img, (1 - weight)*(baseMax+detailMax) / detailMax );
//
//				cvAdd(img, imgGa, img);
//				Image2Relief(img, compressedH);
//			}
//
//			bgFilter(compressedH, bgMask, winHeight - boundary*2);
//
//			for(int i=0; i < compressedH.size(); i++)
//			{
//				float height = compressedH[i];
//				if( maxHeight < height )
//				{
//					maxHeight = height;
//				}
//				if( minHeight > height && height !=0 )
//				{
//					minHeight = height;
//				}
//			}
//			float range = maxHeight - minHeight;
//			for(int i=0; i < compressedH.size(); i++)
//			{
//				compressedH[i] /= range;
//			}
//
//			BuildRelief(compressedH, pThreadRelief, pThreadNormal);
//			
//			relief1 = false;
//			mesh1 = true;
//			profile1 = true;
//		//}
//		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
//}

void histogramBase(const vector<GLfloat> &src, IplImage *weightX, IplImage *weightY, int level=0, IplImage *mask=NULL)
{	
	
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/ 
		
		
		//GLfloat maxDepth=0,minDepth=1;
		float maxHeight=0, minHeight=HistogramBins;
		/*if(relief2)
		{*/
			IplImage *wX = cvCreateImage( cvGetSize(weightX), IPL_DEPTH_64F, 1);
			IplImage *wY = cvCreateImage( cvGetSize(weightY), IPL_DEPTH_64F, 1);
			cvConvertScale(weightX, wX, 1, 0);
			cvConvertScale(weightY, wY, 1, 0);
			cvPow(wX, wX, 2);
			cvPow(wY, wY, 2);

			IplImage *weight = cvCreateImage( cvGetSize(weightX), IPL_DEPTH_64F, 1);
			cvAdd(wX, wY, weight);
			cvPow(weight, weight, 0.5);

			IplImage *img = cvCreateImage( cvSize(img0->width, img0->height), IPL_DEPTH_32F, 1);
			cvCopy(img0, img);
			/*for(int i=1;i <=level; i++)
			{
				IplImage *dstImg = cvCreateImage( cvSize(img->width/2, img->height/2), IPL_DEPTH_32F, 1);
				cvPyrDown(img, dstImg);
				img = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
				cvCopy(dstImg, img);
			}*/

			IplImage *outlineImg = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1);
			extractOutline(img, outlineImg, 1);
			bgFilter(weight, outlineImg);

			IplImage *bgImg = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1);
			IplImage *zeroImg = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1);
			cvSetZero(zeroImg);
			cvCmp(img, zeroImg, bgImg, CV_CMP_EQ);
			bgFilter(weight, bgImg);

		//	disp++;
		//	height.clear();
		//	
		//	for(int i=boundary; i<winWidth*0.5 - boundary; i++)
		//	{
		//		for(int j=boundary; j<winHeight - boundary; j++)
		//		{
		//			GLfloat depth;
		//			glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

		//			if( maxDepth < depth && depth !=1)
		//			{
		//				maxDepth = depth;
		//			}
		//			if( minDepth > depth)
		//			{
		//				minDepth = depth;
		//			}
		//			height.push_back(depth);
		//			
		//		}
		//	}
		//	int enhance = 50;
		//	double maxDepthPrime = compress(maxDepth*enhance, 5, 5);
		//	double minDepthPrime = compress(minDepth*enhance, 5, 5);

		//	for(int i=0; i<height.size(); i++)
		//	{			
		//		if ( height.at(i) == 1 )	//infinite point
		//		{
		//			height.at(i) = 0;
		//		}
		//		else
		//		{
		//			height.at(i) = compress(height.at(i)*enhance, 5, 5);
		//			height.at(i) = maxDepthPrime - height.at(i);
		//			//height.at(i) = maxDepth - height.at(i);	//transfer depth to height
		//			height.at(i) /= (maxDepthPrime - minDepthPrime );	//normalize to [0,1]
		//		}
		//	}
			//for(int i=0; i<height.size(); i++)
			//{
			//	if ( height.at(i) == 1 )	//infinite point
			//	{
			//		height.at(i) = 0;
			//	}
			//	else
			//	{
			//		height.at(i) = ( height.at(i) - minDepth ) / (maxDepth - minDepth) ;	//normalize to [0,1]
			//		height.at(i) = 1 - height.at(i);	//transfer depth to height
			//	}
			//}

			/*IplImage *srcImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
			cvCopy(img0, srcImg);

			int level = 2;
			for(int i=1;i <= level;i++)
			{
				IplImage *dstImg = cvCreateImage( cvSize(srcImg->width/2, srcImg->height/2), IPL_DEPTH_32F, 1);
				cvPyrDown(srcImg, dstImg);
				srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
				cvCopy(dstImg, srcImg);
			}
			vector<GLfloat> sampleHeight;
			Image2Relief(srcImg, sampleHeight);

			IplImage *Img =  cvCreateImage( cvGetSize(srcImg),  IPL_DEPTH_8U, 1);
			cvConvertScaleAbs(srcImg, Img, 255, 0);

			IplImage *sampleX =  cvCreateImage( cvGetSize(srcImg),  IPL_DEPTH_16S, 1);
			IplImage *sampleY =  cvCreateImage( cvGetSize(srcImg),  IPL_DEPTH_16S, 1);
			cvSobel( Img, sampleX, 1, 0);
			cvSobel( Img, sampleY, 0, 1);

			IplImage *gradX = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_64F, 1);
			IplImage *gradY = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_64F, 1);
			cvConvertScale(sampleX, gradX, 1, 0);
			cvConvertScale(sampleY, gradY, 1, 0);
			cvPow(gradX, gradX, 2);
			cvPow(gradY, gradY, 2);

			IplImage *gradient = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_64F, 1);
			cvAdd(gradX, gradY, gradient);
			cvPow(gradient, gradient, 0.5);

			IplImage *outlineImg = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_32F, 1);
			IplImage *bgImg = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_32F, 1);

			extractOutline(srcImg, outlineImg, 1);
			bgFilter(gradient, outlineImg);

			int height = srcImg->height;
			for(int i=0; i < srcImg->width; i++)
			{
				for(int j=0; j < height; j++)
				{
					if( sampleHeight[ i*height + j ] )
					{
						cvSetReal2D( bgImg, height - 1 - j, i, 1 );
					}
					else
					{
						cvSetReal2D( bgImg, height - 1 - j, i, 0 );
					}
				}
			}
			bgFilter(gradient, bgImg);*/
			
			int size = src.size();
			int height = (winHeight - boundary*2)/*/pow(2.0, level)*/;
			int width = size/height;

			vector<GLfloat> AHEHeight;
			AHEHeight.resize(size);
			compressedH.resize(size);
			for(int i=0; i < size; i++)
			{
				compressedH[i] = 0;
			}

			int n=4;
			//#pragma omp parallel for private(AHEHeight)
			/*for(int k=1; k <= n; k++)
			{
				equalizeHist(sampleHeight, AHEHeight, gradient, pow(2.0, k-1) * 8*2/pow(2.0, level) + 1);
				vectorAdd(referenceHeight, AHEHeight, referenceHeight);
				AHEHeight.clear();
			}
			Relief2Image(sampleHeight, srcImg);

			for(int i=1;i <= level;i++)
			{
				IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
				cvPyrUp(srcImg, dstImg);
				srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
				cvCopy(dstImg, srcImg);
			}
			Image2Relief(srcImg, referenceHeight);*/
			

			if(level)
			{
				//int side = sqrt( (float)size );
				
				
				for(int k=1; k <= n; k++)
				{
					equalizeHist(src, AHEHeight, pow(2.0, level), NULL, weight, pow(2.0, k-1) * ahe_m*2 + 1, height, 0);
					vectorAdd(compressedH, AHEHeight, compressedH);
					AHEHeight.clear();
					AHEHeight.resize(size, 0);
				}
				vectorScale(compressedH, compressedH, 1.0/n);
				
				
				/*IplImage *srcImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
				Relief2Image(referenceHeight, srcImg, height);

				for(int i=1;i <=level; i++)
				{
					IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
					cvPyrUp(srcImg, dstImg);
					srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
					cvCopy(dstImg, srcImg);
				}*/

				vector<GLfloat> knn, carve;
				
				/*Image2Relief(srcImg, referenceHeight);*/
				Image2Relief(sample, knn);
				IplImage *sampleL = cvCreateImage( cvGetSize(sample), IPL_DEPTH_8U, 1);
				cvConvertScale(ImageL, sampleL, -1, 255);

				//cvOr(sample, sampleL, sample);
				
				cvNamedWindow("Sample & Line", 1);
				cvShowImage("Sample & Line", sample);
			
				carve.resize(compressedH.size(), 0);
				//knn.resize(compressedH.size(), 0);
				//knninterpolation( heightPyr[0], carve);

				height = winHeight - boundary*2;
				width = heightPyr[0].size() / height;
				//GLfloat carveMax = 0;
				//for(int i=0; i< width; i++)
				//{
				//	for(int j=0; j< height; j++)
				//	{
				//		//if( !bgMask[ i*height + j ] )
				//		//{
				//			if( carve[ i*height + j ] > carveMax ) carveMax = carve[ i*height + j ];
				//		//}
				//	}
				//}

				//if(carveMax)
				//{
				//	for(int i=0; i< width; i++)
				//	{
				//		for(int j=0; j< height; j++)
				//		{
				//			if( !bgMask[ i*height + j ] )
				//			{
				//				compressedH[ i*height + j ] -= carve[ i*height + j ] *0.05 / carveMax;
				//			}
				//		}
				//	}
				//}

			}
			else
			{
					for(int k=1; k <= n; k++)
					{
						equalizeHist(src, AHEHeight, 1, 0 /*, pow(2.0, pyrLevel-1)*/, weight, pow(2.0, k-1) * ahe_m*2 + 1, winHeight - boundary*2, 0);
						vectorAdd(compressedH, AHEHeight, compressedH);
						AHEHeight.clear();
						AHEHeight.resize(size, 0);
					}
					vectorScale(compressedH, compressedH, 1.0/n);
			}

			bgFilter(compressedH, bgMask, winHeight - boundary*2);

			for(int i=0; i < compressedH.size(); i++)
			{
				float height = compressedH[i];
				if( maxHeight < height )
				{
					maxHeight = height;
				}
				if( minHeight > height && height !=0 )
				{
					minHeight = height;
				}
			}
			float range = maxHeight - minHeight;
			for(int i=0; i < compressedH.size(); i++)
			{
				compressedH[i] /= range/*/= HistogramBins*/;
			}

			/*for(int i=0; i < width; i++)
			{
				for(int j=0; j < height; j++)
				{
					float carveDepth = cvGetReal2D( carve, height - 1 - j, i );
					
					compressedH[ i*height + j ] -= carveDepth;
				}
			}*/

			//generateCarvePath(compressedH);

			


			BuildRelief(compressedH, pThreadRelief, pThreadNormal);
			
			relief1 = false;
			mesh1 = true;
			profile1 = true;
		//}
		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
}

void reliefHistogram(const vector<GLfloat> &src, IplImage *weightX, IplImage *weightY, int level=0, IplImage *mask=NULL)
{	
	
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/ 
		
		
		//GLfloat maxDepth=0,minDepth=1;
		float maxHeight=0, minHeight=HistogramBins;
		/*if(relief2)
		{*/
			IplImage *wX = cvCreateImage( cvGetSize(weightX), IPL_DEPTH_64F, 1);
			IplImage *wY = cvCreateImage( cvGetSize(weightY), IPL_DEPTH_64F, 1);
			cvConvertScale(weightX, wX, 1, 0);
			cvConvertScale(weightY, wY, 1, 0);
			cvPow(wX, wX, 2);
			cvPow(wY, wY, 2);

			IplImage *weight = cvCreateImage( cvGetSize(weightX), IPL_DEPTH_64F, 1);
			cvAdd(wX, wY, weight);
			cvPow(weight, weight, 0.5);

			IplImage *img = cvCreateImage( cvSize(img0->width, img0->height), IPL_DEPTH_32F, 1);
			cvCopy(img0, img);
			/*for(int i=1;i <=level; i++)
			{
				IplImage *dstImg = cvCreateImage( cvSize(img->width/2, img->height/2), IPL_DEPTH_32F, 1);
				cvPyrDown(img, dstImg);
				img = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
				cvCopy(dstImg, img);
			}*/

			IplImage *outlineImg = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1);
			extractOutline(img, outlineImg, 1);
			bgFilter(weight, outlineImg);

			IplImage *bgImg = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1);
			IplImage *zeroImg = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1);
			cvSetZero(zeroImg);
			cvCmp(img, zeroImg, bgImg, CV_CMP_EQ);
			bgFilter(weight, bgImg);

		//	disp++;
		//	height.clear();
		//	
		//	for(int i=boundary; i<winWidth*0.5 - boundary; i++)
		//	{
		//		for(int j=boundary; j<winHeight - boundary; j++)
		//		{
		//			GLfloat depth;
		//			glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

		//			if( maxDepth < depth && depth !=1)
		//			{
		//				maxDepth = depth;
		//			}
		//			if( minDepth > depth)
		//			{
		//				minDepth = depth;
		//			}
		//			height.push_back(depth);
		//			
		//		}
		//	}
		//	int enhance = 50;
		//	double maxDepthPrime = compress(maxDepth*enhance, 5, 5);
		//	double minDepthPrime = compress(minDepth*enhance, 5, 5);

		//	for(int i=0; i<height.size(); i++)
		//	{			
		//		if ( height.at(i) == 1 )	//infinite point
		//		{
		//			height.at(i) = 0;
		//		}
		//		else
		//		{
		//			height.at(i) = compress(height.at(i)*enhance, 5, 5);
		//			height.at(i) = maxDepthPrime - height.at(i);
		//			//height.at(i) = maxDepth - height.at(i);	//transfer depth to height
		//			height.at(i) /= (maxDepthPrime - minDepthPrime );	//normalize to [0,1]
		//		}
		//	}
			//for(int i=0; i<height.size(); i++)
			//{
			//	if ( height.at(i) == 1 )	//infinite point
			//	{
			//		height.at(i) = 0;
			//	}
			//	else
			//	{
			//		height.at(i) = ( height.at(i) - minDepth ) / (maxDepth - minDepth) ;	//normalize to [0,1]
			//		height.at(i) = 1 - height.at(i);	//transfer depth to height
			//	}
			//}

			/*IplImage *srcImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
			cvCopy(img0, srcImg);

			int level = 2;
			for(int i=1;i <= level;i++)
			{
				IplImage *dstImg = cvCreateImage( cvSize(srcImg->width/2, srcImg->height/2), IPL_DEPTH_32F, 1);
				cvPyrDown(srcImg, dstImg);
				srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
				cvCopy(dstImg, srcImg);
			}
			vector<GLfloat> sampleHeight;
			Image2Relief(srcImg, sampleHeight);

			IplImage *Img =  cvCreateImage( cvGetSize(srcImg),  IPL_DEPTH_8U, 1);
			cvConvertScaleAbs(srcImg, Img, 255, 0);

			IplImage *sampleX =  cvCreateImage( cvGetSize(srcImg),  IPL_DEPTH_16S, 1);
			IplImage *sampleY =  cvCreateImage( cvGetSize(srcImg),  IPL_DEPTH_16S, 1);
			cvSobel( Img, sampleX, 1, 0);
			cvSobel( Img, sampleY, 0, 1);

			IplImage *gradX = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_64F, 1);
			IplImage *gradY = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_64F, 1);
			cvConvertScale(sampleX, gradX, 1, 0);
			cvConvertScale(sampleY, gradY, 1, 0);
			cvPow(gradX, gradX, 2);
			cvPow(gradY, gradY, 2);

			IplImage *gradient = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_64F, 1);
			cvAdd(gradX, gradY, gradient);
			cvPow(gradient, gradient, 0.5);

			IplImage *outlineImg = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_32F, 1);
			IplImage *bgImg = cvCreateImage( cvGetSize(srcImg), IPL_DEPTH_32F, 1);

			extractOutline(srcImg, outlineImg, 1);
			bgFilter(gradient, outlineImg);

			int height = srcImg->height;
			for(int i=0; i < srcImg->width; i++)
			{
				for(int j=0; j < height; j++)
				{
					if( sampleHeight[ i*height + j ] )
					{
						cvSetReal2D( bgImg, height - 1 - j, i, 1 );
					}
					else
					{
						cvSetReal2D( bgImg, height - 1 - j, i, 0 );
					}
				}
			}
			bgFilter(gradient, bgImg);*/
			
			int size = src.size();
			vector<GLfloat> AHEHeight;
			AHEHeight.resize(size);
			referenceHeight.resize(size);
			for(int i=0; i < size; i++)
			{
				referenceHeight[i] = 0;
			}

			const int n=4;
			//#pragma omp parallel for private(AHEHeight)
			/*for(int k=1; k <= n; k++)
			{
				equalizeHist(sampleHeight, AHEHeight, gradient, pow(2.0, k-1) * 8*2/pow(2.0, level) + 1);
				vectorAdd(referenceHeight, AHEHeight, referenceHeight);
				AHEHeight.clear();
			}
			Relief2Image(sampleHeight, srcImg);

			for(int i=1;i <= level;i++)
			{
				IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
				cvPyrUp(srcImg, dstImg);
				srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
				cvCopy(dstImg, srcImg);
			}
			Image2Relief(srcImg, referenceHeight);*/
			

			if(level)
			{
				//int side = sqrt( (float)size );
				int height = (winHeight - boundary*2)/*/pow(2.0, level)*/;
				int width = size/height;
				
				for(int k=1; k <= n; k++)
				{
					equalizeHist(src, AHEHeight, pow(2.0, level+k-1), NULL, weight, pow(2.0, k-1) * ahe_m*2 + 1, height);
					vectorAdd(referenceHeight, AHEHeight, referenceHeight);
					AHEHeight.clear();
					AHEHeight.resize(size, 0);
				}
				vectorScale(referenceHeight, referenceHeight, 1.0/n);
				
				
				/*IplImage *srcImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
				Relief2Image(referenceHeight, srcImg, height);

				for(int i=1;i <=level; i++)
				{
					IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
					cvPyrUp(srcImg, dstImg);
					srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
					cvCopy(dstImg, srcImg);
				}*/

				vector<GLfloat> knn, carve;
				
				/*Image2Relief(srcImg, referenceHeight);*/
				Image2Relief(sample, knn);
			
				carve.resize(referenceHeight.size(), 0);
				knninterpolation( heightPyr[0], carve);

				height = winHeight - boundary*2;
				width = heightPyr[0].size() / height;
				GLfloat carveMax = 0;
				for(int i=0; i< width; i++)
				{
					for(int j=0; j< height; j++)
					{
						//if( !bgMask[ i*height + j ] )
						//{
							if( carve[ i*height + j ] > carveMax ) carveMax = carve[ i*height + j ];
						//}
					}
				}

				if(carveMax)
				{
					for(int i=0; i< width; i++)
					{
						for(int j=0; j< height; j++)
						{
							if( !bgMask[ i*height + j ] )
							{
								referenceHeight[ i*height + j ] -= carve[ i*height + j ] *0.25 / carveMax;
							}
						}
					}
				}

			}
			else
			{
				float w[n] = { 1, 1, /**/1/**/, 1};
					
				for(int k=1; k <= n; k++)
				{
					equalizeHist(src, AHEHeight, /*1*/pow(2.0, k-1), 0 /*, pow(2.0, pyrLevel-1)*/, weight, pow(2.0, k-1) * ahe_m*2 + 1, winHeight - boundary*2, 1);
					vectorScale( AHEHeight, AHEHeight, w[ k-1] );
					vectorAdd(referenceHeight, AHEHeight, referenceHeight);
					AHEHeight.clear();
					AHEHeight.resize(size, 0);
				}
				vectorScale(referenceHeight, referenceHeight, 1.0/n);
			}

			bgFilter(referenceHeight, bgMask, winHeight - boundary*2);

			for(int i=0; i < referenceHeight.size(); i++)
			{
				float height = referenceHeight[i];
				if( maxHeight < height )
				{
					maxHeight = height;
				}
				if( minHeight > height && height !=0 )
				{
					minHeight = height;
				}
			}
			float range = maxHeight - minHeight;
			for(int i=0; i < referenceHeight.size(); i++)
			{
				referenceHeight[i] /= range/*/= HistogramBins*/;
			}

			BuildRelief(referenceHeight, pThreadEqualizeRelief, pThreadEqualizeNormal);
			
			relief2 = false;
			mesh2 = true;
			profile2 = true;
		//}
		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
}

void correct(IplImage *GradX, IplImage *GradY, double w=1.8, double e=0.001)
{
	double max_err, err;
	int nthreads, tid, i, chunk;
	chunk = 100;

	int m = cvGetSize(GradX).width;
	int n = cvGetSize(GradX).height;
	do {
		max_err = 0;

		//#pragma omp parallel for /*shared(GradX,GradY,nthreads,chunk)*/
		for(int count=0; count<(n-1)*(m-1); count++)
		{
					int i = count/(m-1);
					int j = count%(m-1);
					//cout << "c1" << std::endl;
					if( !(bgMask[ n-2-i + j*n ] && bgMask[ n-2-i + (j+1)*n ] && bgMask[ n-2-i + 1 + j*n ] && bgMask[ n-2-i + 1 + (j+1)*n ]) )
					{
						err = cvGetReal2D(GradX, i, j) + cvGetReal2D(GradY, i, j+1) - cvGetReal2D(GradY, i, j) - cvGetReal2D(GradX, i+1, j);
						if( abs(err) > max_err )
						{
							max_err = abs(err);
						}

						double s = 0.25 * err * w;
						//cout << "c2" << std::endl;
						cvSetReal2D( GradX, i, j, -s + cvGetReal2D(GradX, i, j) );
						cvSetReal2D( GradY, i, j+1, -s + cvGetReal2D(GradY, i, j+1) );
						cvSetReal2D( GradY, i, j, s + cvGetReal2D(GradY, i, j) );
						cvSetReal2D( GradX, i+1, j, s + cvGetReal2D(GradX, i+1, j) );
					}
					
				
		}
	} while(max_err >= e);
}

void intergrate(IplImage *GradX, IplImage *GradY, IplImage *dst)
{
	cvSetReal2D(dst, 0, 0, 0);
	
	int m = cvGetSize(dst).width;
	int n = cvGetSize(dst).height;
	double value;
	for(int i=0; i<n; i++)
	{
		//cout << "i1" << std::endl;
		if(i > 0)
		{
			if( bgMask[n - 1 -i] )
			{
				cvSetReal2D(dst, i, 0, 0);
			}
			else
			{
				value = cvGetReal2D(dst, i-1, 0) + cvGetReal2D(GradY, i-1, 0) ;
				//cout << value << std::endl;
				cvSetReal2D( dst, i, 0, cvGetReal2D(dst, i-1, 0) + cvGetReal2D(GradY, i-1, 0) );
			}
		}
		//cout << "i2" << std::endl;
		for(int j=1; j<m; j++)
		{
			if( bgMask[ n - 1 -i + j*n ] )
			{
				cvSetReal2D(dst, i, j, 0);
			}
			else
			{
				value = cvGetReal2D(dst, i, j-1) + cvGetReal2D(GradX, i, j-1);
				//cout << value << std::endl;
				cvSetReal2D( dst, i, j, cvGetReal2D(dst, i, j-1) + cvGetReal2D(GradX, i, j-1) );
			}
		}
	}
}

void next(int l, int  i, int j, vector<GLfloat> x, vector<GLfloat> y, vector<vector <GLfloat>> &sum, vector<GLfloat> &dst)
{
	if( bgMask[j] == 0 && ( outlineMask[j] > l || outlineMask[j] == 0 ) )
	{
		if( outlineMask[j] == 0 )
		{
			outlineMask[j] = l+1;
		}

		if(  j == i+1 || j == i-1 )
		{
			if( j < i )	dst[j] = dst[i] + y[i];
			else			dst[j] = dst[i] - y[i];
		}
		else
		{
			if( j > i )	dst[j] = dst[i] + x[i];
			else			dst[j] = dst[i] - x[i];
		}

		sum[j][0] += dst[j];
		sum[j][1] += 1;
	}
}

void intergrate(vector<GLfloat> x, vector<GLfloat> y, vector<GLfloat> &dst)
{
	int height = winHeight - boundary*2;
	int width = x.size() / height;
	
	extractOutline( heightPyr[0], outlineMask, height, 1);

	dst.clear();
	vector<vector <GLfloat>> sum;
	vector<GLfloat> element;
	element.push_back(0);
	element.push_back(0);
	for(int i=0; i< width*height; i++)
	{
		dst.push_back(0);
		sum.push_back(element);
	}
	
	bool first = true;
	int start, end;
	for(int i=0; i < width*height; i++)
	{
		if( bgMask[i] == 0 && first) 
		{
			start = i;
			first = false;
		}
		if( bgMask[i] == 0 ) end=i;
	}

	for(int l=1; l < width+height-2; l++)
	{
		for(int i=start; i <= end; i++)
		{
			if( outlineMask[i] == l )
			{
				if( i%height - 1 >= 0 ) next(l, i, i-1, x, y, sum, dst);
				if( i%height + 1 < height ) next(l, i, i+1, x, y, sum, dst);
				if( i-height >= 0 ) next(l, i, i-height, x, y, sum, dst);
				if( i/height + 1 < width) next(l, i, i+height, x, y, sum, dst);
			}
		}
	}

	for(int i=start; i <= end; i++)
	{
		if( bgMask[i] == 0 && outlineMask[i] != 1 )
		{
			dst[i] = sum[i][0] / sum[i][1];
		}
	}
}

void heightInterpolation(int l, int i, int j, int aperture, vector<GLfloat> &dst)
{
	/*int width = sqrt( (float) heightPyr[0].size() );
	int height = sqrt( (float) .size() );*/
	int height = winHeight - boundary*2;
	int width = heightPyr[0].size() / height;

	if( dst[ i*height + j ] == 0 && bgMask[ i*height + j ] == 0 )
	{
		outlineMask[ i*height + j ] = l + 1;
		
		int ext = (aperture-1) / 2;
		int extU,extD,extL,extR;
		
			extU = -ext;
			extD = ext;
			extL = -ext;
			extR = ext;
		
			if( ext > j)		extU = -j;
			if( ext > i)		extL = -i;
			if( j+ext >= height)		extD = height - 1 - j;
			if( i+ext >= width)		extR =  width - 1 - i;

		float sum=0, dsum=0;
		for(int p=extL; p <= extR; p++)
		{
			for(int q=extU; q <= extD; q++)
			{
				int layer = outlineMask[ (i+p)*height + j+q ];
				if( layer > 0 && layer < l+1 )
				{
					sum += ( /*compress(*/ heightPyr[0][ i*height + j ] - heightPyr[0][ (i+p)*height + j+q ] /*)*/ + dst[ (i+p)*height + j+q ] ) / sqrt( (float) p*p + q*q );
					dsum += 1 / sqrt( (float) p*p + q*q );
				}
			}
		}

		dsum = 1/dsum;
		dst[ i*height + j ] = sum*dsum;
	}
}

void heightInterpolation(int l, int i, int j, int aperture, vector<GLfloat> *dst)
{
	/*int width = sqrt( (float) heightPyr[0].size() );
	int height = sqrt( (float) heightPyr[0].size() );*/
	int height = winHeight - boundary*2;
	int width = heightPyr[0].size() / height;

	if( dst->at( i*height + j ) == 0 && bgMask[ i*height + j ] == 0 )
	{
		outlineMask[ i*height + j ] = l + 1;
		
		int ext = (aperture-1) / 2;
		int extU,extD,extL,extR;
		
			extU = -ext;
			extD = ext;
			extL = -ext;
			extR = ext;
		
			if( ext > j)		extU = -j;
			if( ext > i)		extL = -i;
			if( j+ext >= height)		extD = height - 1 - j;
			if( i+ext >= width)		extR =  width - 1 - i;

		float sum=0, dsum=0;
		for(int p=extL; p <= extR; p++)
		{
			for(int q=extU; q <= extD; q++)
			{
				int layer = outlineMask[ (i+p)*height + j+q ];
				if( layer > 0 && layer < l+1 )
				{
					sum += ( /*compress(*/ heightPyr[0][ i*height + j ] - heightPyr[0][ (i+p)*height + j+q ] /*)*/ + dst->at( (i+p)*height + j+q ) ) / sqrt( (float) p*p + q*q );
					dsum += 1 / sqrt( (float) p*p + q*q );
				}
			}
		}

		dsum = 1/dsum;
		dst->at( i*height + j ) = sum*dsum;
	}
}

void intergrate(vector<GLfloat> &dst)
{
	/*int width = sqrt( (float) heightPyr[0].size() );
	int height = sqrt( (float) heightPyr[0].size() );*/
	int height = winHeight - boundary*2;
	int width = heightPyr[0].size() / height;
	
	extractOutline( heightPyr[0], outlineMask, height, 1);

	dst.clear();
	for(int i=0; i< width*height; i++)
	{
		dst.push_back(0);
	}
	
	bool first = true;
	int start, end;
	for(int i=0; i < width*height; i++)
	{
		if( bgMask[i] == 0 && first) 
		{
			start = i;
			first = false;
		}
		if( bgMask[i] == 0 ) end=i;
	}

	for(int l=1; l < width+height-2; l++)
	{
		for(int i=start; i <= end; i++)
		{
			if( outlineMask[i] == l )
			{
				if( i%height - 1 >= 0 ) heightInterpolation(l, i/height, i%height - 1, SIDE, dst);
				if( i%height + 1 < height ) heightInterpolation(l, i/height, i%height + 1, SIDE, dst);
				if( i-height >= 0 ) heightInterpolation(l, i/height - 1, i%height, SIDE, dst);
				if( i/height + 1 < width) heightInterpolation(l, i/height + 1, i%height, SIDE, dst);
			}
		}
	}
}

void intergrate(vector<GLfloat> *dst)
{
	/*int width = sqrt( (float) heightPyr[0].size() );
	int height = sqrt( (float) heightPyr[0].size() );*/
	int height = winHeight - boundary*2;
	int width = heightPyr[0].size() / height;
	
	extractOutline( heightPyr[0], outlineMask, height, 1);

	dst->clear();
	for(int i=0; i< width*height; i++)
	{
		dst->push_back(0);
	}
	
	bool first = true;
	int start, end;
	for(int i=0; i < width*height; i++)
	{
		if( bgMask[i] == 0 && first) 
		{
			start = i;
			first = false;
		}
		if( bgMask[i] == 0 ) end=i;
	}

	for(int l=1; l < width+height-2; l++)
	{
		for(int i=start; i <= end; i++)
		{
			if( outlineMask[i] == l )
			{
				if( i%height - 1 >= 0 ) heightInterpolation(l, i/height, i%height - 1, SIDE, dst);
				if( i%height + 1 < height ) heightInterpolation(l, i/height, i%height + 1, SIDE, dst);
				if( i-height >= 0 ) heightInterpolation(l, i/height - 1, i%height, SIDE, dst);
				if( i/height + 1 < width) heightInterpolation(l, i/height + 1, i%height, SIDE, dst);
			}
		}
	}
}

void setGaussianKernel( vector<double> &a, int aperture)
{
    double sigma = 0.3*(aperture/2 - 1) + 0.8;
    
    double mean = aperture/2;
    double sum = 0;
    for (int x = 0; x < aperture; x++)
    {
        for (int y = 0; y < aperture; y++)
        {
            a[ x*aperture + y ] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
									/ (2 * M_PI * sigma * sigma);
            sum += a[ x*aperture + y ];
        }
    }

    //normalize
    for(int x = 0; x < aperture; x++)
    {
        for (int y = 0; y < aperture; y++)
        {
            a[ x*aperture + y ] /= sum;
        }
    }
}

//void gaussianFilter(vector<GLfloat> src, vector<GLfloat> &dst)
//{
//	int width = sqrt( (float) src.size() );
//	int  height = sqrt( (float) src.size() );
//
//	dst.clear();
//	for(int i=0; i < width; i++)
//	{
//		for(int j=0; j < height; j++)
//		{
//			dst.push_back( 0 );
//		}
//	}
//
//	vector<double> kernel;
//	vector< vector<double> > kernelList;
//	vector<GLfloat>::iterator first = outlineMask.begin();
//	vector<GLfloat>::iterator last = outlineMask.end();
//	GLint max = *max_element ( first, last );
//	for(int i=2; i <= max; i++)
//	{
//		int aperture = i*2 - 1;
//		for(int j=0; j<aperture*aperture; j++)
//		{
//			kernel.push_back( 0 );
//		}
//		setGaussianKernel( kernel,  aperture );
//		kernelList.push_back( kernel );
//		kernel.clear();
//	}
//	
//
//	for(int i=0; i < width; i++)
//	{
//		for(int j=0; j < height; j++)
//		{
//			if( outlineMask[ i*height + j ]  >= 2 )
//			{
//				int ext = outlineMask[ i*height + j ] - 1;
//				//int ext = (aperture-1) / 2;
//				if(i >= ext  && j >= ext && i+ext < width && j+ext < height)
//				{
//					for(int p=-ext; p <=ext; p++)
//					{
//						for(int q=-ext; q <=ext; q++)
//						{
//							dst[ i*height + j ] += src.at( (i+p)*height + j+q ) * kernelList[ outlineMask[ i*height + j ] - 2 ][ (p+ext)*(ext*2+1) + q+ext ];
//						}
//					}
//				}
//
//				else	//boundary issues
//				{
//					int extU = -ext,extD = ext,extL = -ext,extR = ext;
//					if( ext > j)		extU = -j;
//					if( ext > i)		extL = -i;
//					if( j+ext >= height)		extD = height - 1 - j;
//					if( i+ext >= width)		extR =  width - 1 - i;
//					
//					for(int p=extL; p <= extR; p++)
//					{
//						for(int q=extU; q <= extD; q++)
//						{
//							dst[ i*height + j ] += src.at( (i+p)*height + j+q ) * kernelList[ outlineMask[ i*height + j ] - 2 ][ (p+ext)*(ext*2+1) + q+ext ];
//						}
//					}
//				}
//
//			}
//		}
//	}
//}

void gradientCorrection(IplImage *gradientX, IplImage *gradientY)
{	
		//vector<GLfloat> h;
		/*if(relief2)
		{*/

			/*int width = sqrt( (float) heightPyr[0].size() );
			IplImage *img= cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
			Relief2Image(heightPyr[0], img);

			IplImage *Image =  cvCreateImage( cvGetSize(img),  IPL_DEPTH_8U, 1);
			cvConvertScaleAbs(img, Image, 255, 0);

			IplImage *gradientX =  cvCreateImage( cvGetSize(img),  IPL_DEPTH_16S, 1);
			IplImage *gradientY =  cvCreateImage( cvGetSize(img),  IPL_DEPTH_16S, 1);
			cvSobel( Image, gradientX, 1, 0);
			cvSobel( Image, gradientY, 0, 1);*/

			IplImage *gradX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
			IplImage *gradY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
			cvConvertScale(gradientX, gradX, 1, 0);
			cvConvertScale(gradientY, gradY, 1, 0);
			//Image2Relief(gradX, h);
			compress(gradX);
			compress(gradY);

			IplImage *intergration = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);

			
			//Image2Relief(gradX, h);
			//correct(gradX, gradY, 1.8);
			//Image2Relief(gradX, h);
			//intergrate(gradX, gradY, intergration);
			//Image2Relief(intergration, h);

			/*vector<GLfloat> x, y;
			Image2Relief(gradX, x);
			Image2Relief(gradX, y);
			intergrate(x, y, referenceHeight);*/
			intergrate(referenceHeight);

			IplImage *height = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_32F, 1);
			IplImage *bgImg = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_32F, 1);
			IplImage *zeroImg = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_32F, 1);
			IplImage *fgImg = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_8U, 1);
			//cvConvertScale(intergration, height, 1/255, 0);
			
			//Image2Relief(intergration, referenceHeight);
			/*Relief2Image( referenceHeight, height );
			Relief2Image( bgMask, bgImg );
			cvSetZero(zeroImg);
			cvCmp(bgImg, zeroImg, fgImg, CV_CMP_EQ);
			height->maskROI = fgImg;
			cvSmooth(height, height, CV_GAUSSIAN, 55, 55);
			 
			Image2Relief(height, referenceHeight);*/

			//bgFilter(referenceHeight, bgMask, winHeight - boundary*2);
			//gaussianFilter(referenceHeight, referenceHeight);

			vector<GLfloat>::iterator first = referenceHeight.begin();
			vector<GLfloat>::iterator last = referenceHeight.end();
			GLfloat max = *max_element ( first, last );
			for(int i=0; i< referenceHeight.size(); i++)
			{
				referenceHeight[i] /= max; //normalize
			}

			BuildRelief(referenceHeight, pThreadEqualizeRelief, pThreadEqualizeNormal);
			
			relief2 = false;
			mesh2 = true;
			profile2 = true;
		/*}*/
		//swScaled(MODELSCALE, MODELSCALE, MODELSCALE);
}

void displayline(void)
{
	glViewport(0, 0, winWidth/3, winHeight);
	glDisable(GL_LIGHTING);
	/*glDisable(GL_CULL_FACE);
	glClear(GL_DEPTH_BUFFER_BIT);*/

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glOrtho(0, winWidth/3, winHeight, 0, -1, 1);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor3f(1, 0, 0);

	glBegin(GL_LINES);
		glVertex2f(-5, -5);
		glVertex2f(winWidth/3, winHeight);
	glEnd();
}

//3D Scene
void openglPath(void)
{
    //view transform
	glViewport(0, 0, winWidth/3, winHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	//glOrtho(-2.0, 2.0, -2.0, 2.0, -3.0, 25.0);
	//glFrustum(-2.0, 2.0, -2.0, 2.0, -3.0, 3.0);
	//gluPerspective(60, (GLfloat)(winWidth/3)/winHeight, 0.1, 25); 
	gluPerspective(perspective[0], (GLfloat)(winWidth/3)/winHeight, perspective[2], perspective[3]); 
	glGetDoublev(GL_PROJECTION_MATRIX, DEBUG_M);


    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0);
	glGetDoublev(GL_MODELVIEW_MATRIX, DEBUG_M);

	lightPos0[0] = -LIGHTP;
	lightPos0[1] = LIGHTP;
	lightPos0[2] = LIGHTP;
	
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

	

	glTranslated(0, 0, 0);
	//world coordinate
	glColor3f(1, 0, 0);
	
	/*glPushMatrix();
		glTranslated(0,0,0);
		glutSolidSphere(1.5,9,9);
	glPopMatrix();*/
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/
	glPushMatrix();
		/*gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0);

		glBegin(GL_QUADS);
			
			glVertex3f(-2, 2, farPoint.n[2]);
			glVertex3f(-2, -2, farPoint.n[2]);
			glVertex3f(2, -2, farPoint.n[2]);
			glVertex3f(2, 2, farPoint.n[2]);

		glEnd();*/
	glPopMatrix();
	
	glPushMatrix();
		//multiple trackball matrix
		/*glRotated(angleX,1, 0, 0);
		glRotated(angleY,0, 1, 0);
		glRotated(angleZ,0, 0, 1);

		glMultMatrixd(TRACKM);

		glScaled(MODELSCALE, MODELSCALE, MODELSCALE);*/
		glMultMatrixf(m);
		glColor3f(1.0, 1.0, 1.0);
		glmDraw(MODEL, GLM_SMOOTH);//GLM_FLAT
		//glutSolidSphere(1, 20, 20);
	glPopMatrix();

	glPushMatrix();
		glTranslated(0, 2, 0);
		glMultMatrixd(TRACKM);

		/*glBegin(GL_TRIANGLES);
			glNormal3f(0, 0, 1);
			glColor3f(1, 0, 0);
			glVertex3f(-1, 0, 0);

			glColor3f(0, 1, 0);
			glVertex3f(1, 0, 0);

			glColor3f(0, 0, 1);
			glVertex3f(0, 1, 0);
		glEnd();		*/
	glPopMatrix();

}

/*----------------------------------------------------------------------*/
/* 
** These functions implement a simple trackball-like motion control.
*/

float lastPos[3] = {0.0F, 0.0F, 0.0F};
int curx, cury;
int startX, startY;

void trackball_ptov(int x, int y, int width, int height, float v[3])
{
    float d, a;

    /* project x,y onto a hemi-sphere centered within width, height */
    v[0] = (2.0F*x - width) / width;
    v[1] = (height - 2.0F*y) / height;
    d = (float) sqrt(v[0]*v[0] + v[1]*v[1]);
    v[2] = (float) cos((M_PI/2.0F) * ((d < 1.0F) ? d : 1.0F));
    a = 1.0F / (float) sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= a;
    v[1] *= a;
    v[2] *= a;
}


void mouseMotion0(int x, int y)
{
    float curPos[3], dx, dy, dz;

    trackball_ptov(x, y, winWidth, winHeight, curPos);
	if(trackingMouse)
	{
		dx = curPos[0] - lastPos[0];
		dy = curPos[1] - lastPos[1];
		dz = curPos[2] - lastPos[2];

		if (dx || dy || dz) {
			angle = 90.0F * sqrt(dx*dx + dy*dy + dz*dz);

			axis[0] = lastPos[1]*curPos[2] - lastPos[2]*curPos[1];
			axis[1] = lastPos[2]*curPos[0] - lastPos[0]*curPos[2];
			axis[2] = lastPos[0]*curPos[1] - lastPos[1]*curPos[0];

			lastPos[0] = curPos[0];
			lastPos[1] = curPos[1];
			lastPos[2] = curPos[2];
		}
	} 
    glutPostRedisplay();
}

void mouseMotion(int x, int y)
{
    float curPos[3], dx, dy, dz;

	if(trackingMouse)
	{
		dx = curPos[0] - lastPos[0];
		dy = curPos[1] - lastPos[1];
		dz = curPos[2] - lastPos[2];

		if (dx || dy || dz) {
			angle = 90.0F * sqrt(dx*dx + dy*dy + dz*dz);

			axis[0] = lastPos[1]*curPos[2] - lastPos[2]*curPos[1];
			axis[1] = lastPos[2]*curPos[0] - lastPos[0]*curPos[2];
			axis[2] = lastPos[0]*curPos[1] - lastPos[1]*curPos[0];

			lastPos[0] = curPos[0];
			lastPos[1] = curPos[1];
			lastPos[2] = curPos[2];
		}
	} 
    glutPostRedisplay();
}

void startMotion0(int x, int y)
{
    trackingMouse = true;
    redrawContinue = false;
    startX = x; startY = y;
    curx = x; cury = y;
    trackball_ptov(x, y, winWidth, winHeight, lastPos);
	trackballMove=true;
}

void stopMotion0(int x, int y)
{
	trackingMouse = false;

    if (startX != x || startY != y) {
		redrawContinue = true;
    } else {
		angle = 0.0F;
		redrawContinue = false;
		trackballMove = false;
    }
}

/*----------------------------------------------------------------------*/

CVector3 GetOGLPos(GLfloat pos[3])
{
  /*  GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];*/
    GLdouble winX, winY, winZ;
    GLdouble posX, posY, posZ;
 
 
    winX = (double)pos[0];
    winY = (double)pos[1];
    winZ = (double)pos[2];
 
    gluUnProject( winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);
 
    return CVector3(posX, posY, posZ);
}

void displayfont()
{
    //Font
	char mss01[30]="Clipping Scene";
	//sprintf(mss, "Score %d", Gamescore);
	glColor3f(1.0, 0.0, 0.0);  //set font color
	void * font = GLUT_BITMAP_9_BY_15;

	glWindowPos2i(10, winHeight-15);    //set font start position
	for(unsigned int i=0; i<strlen(mss01); i++) {
		glutBitmapCharacter(font, mss01[i]);
	}

	char mss02[30];
	sprintf(mss02, "%f", dynamicRange);

	glWindowPos2i(10+winWidth/6, winHeight-15);    //set font start position
	for(unsigned int i=0; i<strlen(mss02); i++) {
		glutBitmapCharacter(font, mss02[i]);
	}

	char mss11[30]="Relief";
	
	glWindowPos2i(10+(winWidth/3), winHeight-15);    //set font start position
	for(unsigned int i=0; i<strlen(mss11); i++) {
		glutBitmapCharacter(font, mss11[i]);
	}
	
	char mss12[30]="Computing...";
	if(relief1)
	{
		glColor3f(0.0, 1.0, 0.0);
		glWindowPos2i(10+(winWidth/3), winHeight-35);    //set font start position
		for(unsigned int i=0; i<strlen(mss12); i++) {
			glutBitmapCharacter(font, mss12[i]);
		}
	}
	
	char mss21[30]="Referred Relief";
	
	glWindowPos2i(10+(winWidth*2/3), winHeight-15);    //set font start position
	for(unsigned int i=0; i<strlen(mss21); i++) {
		glutBitmapCharacter(font, mss21[i]);
	}
	//char mss3[30];
	//sprintf(mss3,"%f",scale);
	//glWindowPos2i(10+(winWidth/2), winHeight-60);    //set font start position
	//for(unsigned int i=0; i<strlen(mss1); i++) {
	//	glutBitmapCharacter(font, mss3[i]);
	//}
}

void identity(GLdouble m[16])
{
    m[0+4*0] = 1; m[0+4*1] = 0; m[0+4*2] = 0; m[0+4*3] = 0;
    m[1+4*0] = 0; m[1+4*1] = 1; m[1+4*2] = 0; m[1+4*3] = 0;
    m[2+4*0] = 0; m[2+4*1] = 0; m[2+4*2] = 1; m[2+4*3] = 0;
    m[3+4*0] = 0; m[3+4*1] = 0; m[3+4*2] = 0; m[3+4*3] = 1;
}

GLboolean invert(GLdouble src[16], GLdouble inverse[16])
{
    double t;
    int i, j, k, swap;
    GLdouble tmp[4][4];

    identity(inverse);

    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    tmp[i][j] = src[i*4+j];
	}
    }

    for (i = 0; i < 4; i++) {
        /* look for largest element in column. */
        swap = i;
        for (j = i + 1; j < 4; j++) {
            if (fabs(tmp[j][i]) > fabs(tmp[i][i])) {
                swap = j;
            }
        }

        if (swap != i) {
            /* swap rows. */
            for (k = 0; k < 4; k++) {
                t = tmp[i][k];
                tmp[i][k] = tmp[swap][k];
                tmp[swap][k] = t;

                t = inverse[i*4+k];
                inverse[i*4+k] = inverse[swap*4+k];
                inverse[swap*4+k] = t;
            }
        }

        if (tmp[i][i] == 0) {
            /* no non-zero pivot.  the matrix is singular, which
	       shouldn't happen.  This means the user gave us a bad
	       matrix. */
            return GL_FALSE;
        }

        t = tmp[i][i];
        for (k = 0; k < 4; k++) {
            tmp[i][k] /= t;
            inverse[i*4+k] /= t;
        }
        for (j = 0; j < 4; j++) {
            if (j != i) {
                t = tmp[j][i];
                for (k = 0; k < 4; k++) {
                    tmp[j][k] -= tmp[i][k]*t;
                    inverse[j*4+k] -= inverse[i*4+k]*t;
                }
            }
        }
    }
    return GL_TRUE;
}

float normalize(float* v)
{
    float length;

    length = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] /= length;
    v[1] /= length;
    v[2] /= length;

    return length;
}

void model_transform(void)
{
	glRotated(angleX,1, 0, 0);
	glRotated(angleY,0, 1, 0);
	glRotated(angleZ,0, 0, 1);

	glMultMatrixd(TRACKM);

	glScaled(MODELSCALE, MODELSCALE, MODELSCALE);
}

void drawaxes(void)
{
    glColor3ub(255, 0, 0);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0.0, 0.0);
    glVertex3f(0.75, 0.25, 0.0);
    glVertex3f(0.75, -0.25, 0.0);
    glVertex3f(1.0, 0.0, 0.0);
    glVertex3f(0.75, 0.0, 0.25);
    glVertex3f(0.75, 0.0, -0.25);
    glVertex3f(1.0, 0.0, 0.0);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 1.0, 0.0);
    glVertex3f(0.0, 0.75, 0.25);
    glVertex3f(0.0, 0.75, -0.25);
    glVertex3f(0.0, 1.0, 0.0);
    glVertex3f(0.25, 0.75, 0.0);
    glVertex3f(-0.25, 0.75, 0.0);
    glVertex3f(0.0, 1.0, 0.0);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 1.0);
    glVertex3f(0.25, 0.0, 0.75);
    glVertex3f(-0.25, 0.0, 0.75);
    glVertex3f(0.0, 0.0, 1.0);
    glVertex3f(0.0, 0.25, 0.75);
    glVertex3f(0.0, -0.25, 0.75);
    glVertex3f(0.0, 0.0, 1.0);
    glEnd();

    glColor3ub(255, 255, 0);
    glRasterPos3f(1.1, 0.0, 0.0);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
    glRasterPos3f(0.0, 1.1, 0.0);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
    glRasterPos3f(0.0, 0.0, 1.1);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');
}

void world_display(void)
{
	double length;
    float l[3];

    l[0] = lookat[3] - lookat[0]; 
    l[1] = lookat[4] - lookat[1]; 
    l[2] = lookat[5] - lookat[2];
    length = normalize(l);
	
	invert(modelview, inverse);
	
	glEnable(GL_LIGHTING);
	glPushMatrix();
			glMultMatrixd(inverse);
			glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
	glPopMatrix();

	glPushMatrix();
		model_transform();
		glColor3f(1.0, 1.0, 1.0);
		glmDraw(MODEL, GLM_SMOOTH);//GLM_FLAT
	glPopMatrix();
	glDisable(GL_LIGHTING);

	glPushMatrix();

		glMultMatrixd(inverse);
		/* draw the axis and eye vector */
		glPushMatrix();
			glColor3ub(0, 0, 255);
			glBegin(GL_LINE_STRIP);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(0.0, 0.0, -1.0*length);
			glVertex3f(0.1, 0.0, -0.9*length);
			glVertex3f(-0.1, 0.0, -0.9*length);
			glVertex3f(0.0, 0.0, -1.0*length);
			glVertex3f(0.0, 0.1, -0.9*length);
			glVertex3f(0.0, -0.1, -0.9*length);
			glVertex3f(0.0, 0.0, -1.0*length);
			glEnd();
			glColor3ub(255, 255, 0);
			glRasterPos3f(0.0, 0.0, -1.1*length);
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'e');
			glColor3ub(255, 0, 0);
			glScalef(0.4, 0.4, 0.4);
			drawaxes();
		glPopMatrix();

		invert(projection, inverse);
		glMultMatrixd(inverse);

		/* draw the viewing frustum */
		glColor3f(0.2, 0.2, 0.2);
		glBegin(GL_QUADS);
		glVertex3i(1, 1, 1);
		glVertex3i(-1, 1, 1);
		glVertex3i(-1, -1, 1);
		glVertex3i(1, -1, 1);
		glEnd();

		glColor3ub(128, 196, 128);
		glBegin(GL_LINES);
		glVertex3i(1, 1, -1);
		glVertex3i(1, 1, 1);
		glVertex3i(-1, 1, -1);
		glVertex3i(-1, 1, 1);
		glVertex3i(-1, -1, -1);
		glVertex3i(-1, -1, 1);
		glVertex3i(1, -1, -1);
		glVertex3i(1, -1, 1);
		glEnd();

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4f(0.2, 0.2, 0.4, 0.5);
		glBegin(GL_QUADS);
		glVertex3i(1, 1, -1);
		glVertex3i(-1, 1, -1);
		glVertex3i(-1, -1, -1);
		glVertex3i(1, -1, -1);
		glEnd();
		glDisable(GL_BLEND);

    glPopMatrix();
}

bool GetClipDistance() 
{
	float maxDepthPoint[3], minDepthPoint[3], max2DepthPoint[3];
	maxDepthPoint[0] = 0, maxDepthPoint[1] = 0, maxDepthPoint[2] = 0;
	minDepthPoint[2] = 1;

	/*if( true )
	{*/
		/*glGetFloatv(GL_MODELVIEW_MATRIX, m);*/
		glFlush();

		float *depthmap = new float[winWidth0/2*winHeight0];
		glReadPixels(winWidth0/2 , 0, winWidth0/2, winHeight0, GL_DEPTH_COMPONENT, GL_FLOAT, depthmap);

		for(int i=boundary; i<winWidth0/2 - boundary; i++)
		{
			for(int j=boundary; j<winHeight0 - boundary; j++)
			{
				GLfloat depth = depthmap[j*winWidth0/2+i] ;
				//glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

				if( maxDepthPoint[2] < depth && depth !=1)
				{
					max2DepthPoint[0] = maxDepthPoint[0];
					max2DepthPoint[1] = maxDepthPoint[1];
					max2DepthPoint[2] = maxDepthPoint[2];
					
					maxDepthPoint[0] = i + winWidth0/2;
					maxDepthPoint[1] = j;
					maxDepthPoint[2] = depth;
				}
				if( minDepthPoint[2] > depth)
				{	
					minDepthPoint[0] = i + winWidth0/2;
					minDepthPoint[1] = j;
					minDepthPoint[2] = depth;
				}

			}
		}
	

		farPoint = GetOGLPos(maxDepthPoint);
		far2Point = GetOGLPos(max2DepthPoint);
		nearPoint = GetOGLPos(minDepthPoint);
		printf("N=%f, F=%f\n", minDepthPoint[2], maxDepthPoint[2]);
		printf("near (%f, %f, %f); far (%f, %f, %f)\n", nearPoint.n[0], nearPoint.n[1], nearPoint.n[2],
			farPoint.n[0], farPoint.n[1], farPoint.n[2]);
		CVector3 temp =  farPoint - nearPoint;

		perspective[2] = abs(lookat[2] - nearPoint.n[2] - 0.01);//-0.1*temp.Magnitude();
		perspective[3] = abs(lookat[2] - farPoint.n[2]);//+0.1*temp.Magnitude();;
		printf("Near=%f, Far=%f\n\n", perspective[2], perspective[3]);
		
		dynamicRange = (nearPoint.n[2] - farPoint.n[2]) / (far2Point.n[2] - farPoint.n[2]);

		//scene = false;
		delete [] depthmap;
	/*}*/

	return true;
}

//bool GetClipDistance() 
//{
//	float maxDepthPoint[3], minDepthPoint[3], max2DepthPoint[3];
//	maxDepthPoint[0] = 0, maxDepthPoint[1] = 0, maxDepthPoint[2] = 0;
//	minDepthPoint[2] = 1;
//
//	if( true )
//	{
//		
//		glFlush();
//		//glGetFloatv(GL_MODELVIEW_MATRIX, m);
//
//		float *depthmap = new float[winWidth0*winHeight0];
//		glReadPixels(0, 0, winWidth0, winHeight0, GL_DEPTH_COMPONENT, GL_FLOAT, depthmap);
//
//		for(int i=boundary; i<winWidth0 - boundary; i++)
//		{
//			for(int j=boundary; j<winHeight0 - boundary; j++)
//			{
//				GLfloat depth = depthmap[j*winWidth0+i] ;
//				//glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
//
//				if( maxDepthPoint[2] < depth && depth !=1)
//				{
//					max2DepthPoint[0] = maxDepthPoint[0];
//					max2DepthPoint[1] = maxDepthPoint[1];
//					max2DepthPoint[2] = maxDepthPoint[2];
//					
//					maxDepthPoint[0] = i;
//					maxDepthPoint[1] = j;
//					maxDepthPoint[2] = depth;
//				}
//				if( minDepthPoint[2] > depth)
//				{	
//					minDepthPoint[0] = i;
//					minDepthPoint[1] = j;
//					minDepthPoint[2] = depth;
//				}
//
//			}
//		}
//	
//
//		farPoint = GetOGLPos(maxDepthPoint);
//		far2Point = GetOGLPos(max2DepthPoint);
//		nearPoint = GetOGLPos(minDepthPoint);
//		printf("N=%f, F=%f\n", minDepthPoint[2], maxDepthPoint[2]);
//		printf("near (%f, %f, %f); far (%f, %f, %f)\n", nearPoint.n[0], nearPoint.n[1], nearPoint.n[2],
//			farPoint.n[0], farPoint.n[1], farPoint.n[2]);
//		CVector3 temp =  farPoint - nearPoint;
//
//		perspective[2] = abs(4-nearPoint.n[2]);//-0.1*temp.Magnitude();
//		perspective[3] = abs(4-farPoint.n[2]);//+0.1*temp.Magnitude();;
//		printf("Near=%f, Far=%f\n\n", perspective[2], perspective[3]);
//		
//		dynamicRange = (nearPoint.n[2] - farPoint.n[2]) / (far2Point.n[2] - farPoint.n[2]);
//
//		//scene = false;
//		delete [] depthmap;
//	}
//
//	return true;
//}

//Oringinal Scene
void display0(void)
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    if (trackballMove) {
		glPushMatrix();
			glLoadMatrixd(TRACKM);
			glRotatef(angle, axis[0], axis[1], axis[2]);
			glGetDoublev(GL_MODELVIEW_MATRIX, TRACKM);
		glPopMatrix();	    
	}

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glDisable(GL_LIGHT1);
	glDisable(GL_LIGHT2);
	glDisable(GL_LIGHT3);
	glDisable(GL_LIGHT4);
	
	//view transform
	glViewport(0, 0, winWidth0, winHeight0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
	glOrtho(0, winWidth, 0, winHeight, -2.0, 2.0);
    
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(winWidth0/2 , 0, winWidth0/2, winHeight0);
    glGetIntegerv( GL_VIEWPORT, viewport );
	
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	gluPerspective(perspective[0], perspective[1], perspective[2], perspective[3]); 
	glGetDoublev(GL_PROJECTION_MATRIX, projection);

    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(lookat[0], lookat[1], lookat[2], lookat[3], lookat[4], lookat[5], lookat[6], lookat[7], lookat[8]);
	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);

	lightPos0[0] = -LIGHTP;
	lightPos0[1] = LIGHTP;
	lightPos0[2] = LIGHTP;
	
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

	glTranslated(0, 0, 0);
	//world coordinate
	glColor3f(1, 0, 0);
	
	/*glPushMatrix();
		glTranslated(0,0,0);
		glutSolidSphere(1.5,9,9);
	glPopMatrix();*/
	/*OpenglLine(0, 0, 0, 3, 0, 0);
	glColor3f(0, 1, 0);
	OpenglLine(0, 0, 0, 0, 3, 0);
	glColor3f(0, 0, 1);
	OpenglLine(0, 0, 0, 0, 0, 3);*/

	
	glPushMatrix();
		//multiple trackball matrix
		model_transform();
		glColor3f(1.0, 1.0, 1.0);
		//glGetFloatv(GL_MODELVIEW_MATRIX, mat);
		glmDraw(MODEL, GLM_SMOOTH);//GLM_FLAT
		//glutSolidSphere(1, 20, 20);
	//glPopMatrix();

	//glPushMatrix();
	//	glTranslated(0, 2, 0);
	//	glMultMatrixd(TRACKM);

	//	/*glBegin(GL_TRIANGLES);
	//		glNormal3f(0, 0, 1);
	//		glColor3f(1, 0, 0);
	//		glVertex3f(-1, 0, 0);

	//		glColor3f(0, 1, 0);
	//		glVertex3f(1, 0, 0);

	//		glColor3f(0, 0, 1);
	//		glVertex3f(0, 1, 0);
	//	glEnd();		*/
	//glPopMatrix();

		//float maxDepthPoint[3], minDepthPoint[3], max2DepthPoint[3];
		//maxDepthPoint[0] = 0, maxDepthPoint[1] = 0, maxDepthPoint[2] = 0;
		//minDepthPoint[2] = 1;

		if( scene )
		{
			glGetFloatv(GL_MODELVIEW_MATRIX, m);

		//	float *depthmap = new float[winWidth0*winHeight0];
		//	glReadPixels(0, 0, winWidth0, winHeight0, GL_DEPTH_COMPONENT, GL_FLOAT, depthmap);
		//	
		//	for(int i=boundary; i<winWidth0 - boundary; i++)
		//	{
		//		for(int j=boundary; j<winHeight0 - boundary; j++)
		//		{
		//			GLfloat depth = depthmap[j*winWidth0 + i];
		//			//glReadPixels(i, j, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

		//			if( maxDepthPoint[2] < depth && depth !=1)
		//			{
		//				max2DepthPoint[0] = maxDepthPoint[0];
		//				max2DepthPoint[1] = maxDepthPoint[1];
		//				max2DepthPoint[2] = maxDepthPoint[2];
		//				
		//				maxDepthPoint[0] = i;
		//				maxDepthPoint[1] = j;
		//				maxDepthPoint[2] = depth;
		//			}
		//			if( minDepthPoint[2] > depth)
		//			{	
		//				minDepthPoint[0] = i;
		//				minDepthPoint[1] = j;
		//				minDepthPoint[2] = depth;
		//			}

		//		}
		//	}
		//	delete [] depthmap;
		//

		//	farPoint = GetOGLPos(maxDepthPoint);
		//	far2Point = GetOGLPos(max2DepthPoint);
		//	nearPoint = GetOGLPos(minDepthPoint);
		//	
		//	dynamicRange = (nearPoint.n[2] - farPoint.n[2]) / (far2Point.n[2] - farPoint.n[2]);

			scene = false;
		}
	glPopMatrix();

	//glViewport(0, 0, winWidth0, winHeight0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
	glOrtho(0, winWidth0, 0, winHeight0, -2.0, 2.0);
    
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(0 , 0, winWidth0/2, winHeight0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(60.0, (GLfloat)winWidth0/winHeight0, 1.0, 256.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -5.0);
    glRotatef(-45.0, 0.0, 1.0, 0.0);
    glClearColor(0.0, 0.0, 0.0, 0.0);

	world_display();
	
	glutSwapBuffers();
}

//void display0(void)
//{
//    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//
//    if (trackballMove) {
//		glPushMatrix();
//			glLoadMatrixd(TRACKM);
//			glRotatef(angle, axis[0], axis[1], axis[2]);
//			glGetDoublev(GL_MODELVIEW_MATRIX, TRACKM);
//		glPopMatrix();	    
//	}
//	
//	glEnable(GL_LIGHT0);
//	glDisable(GL_LIGHT1);
//	glDisable(GL_LIGHT2);
//	
//	//view transform
//	glViewport(0, 0, winWidth0, winHeight0);
//
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//	//glOrtho(-2.0, 2.0, -2.0, 2.0, -3.0, 25.0);
//	//glFrustum(-2.0, 2.0, -2.0, 2.0, -3.0, 3.0);
//	//gluPerspective(60, (GLfloat)winWidth0/winHeight0, 0.1, 25); 
//gluPerspective(60, (GLfloat)winWidth0/winHeight0, perspective[2], perspective[3]); 
//	
//	glGetDoublev(GL_PROJECTION_MATRIX, DEBUG_M);
//
//
//    glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//
//	gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0);
//	//glGetDoublev(GL_MODELVIEW_MATRIX, DEBUG_M);
//
//	lightPos0[0] = -LIGHTP;
//	lightPos0[1] = LIGHTP;
//	lightPos0[2] = LIGHTP;
//	
//	glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
//
//	glTranslated(0, 0, 0);
//	//world coordinate
//	glColor3f(1, 0, 0);
//
//	glPushMatrix();
//		//multiple trackball matrix
//		glRotated(angleX,1, 0, 0);
//		glRotated(angleY,0, 1, 0);
//		glRotated(angleZ,0, 0, 1);
//
//		glMultMatrixd(TRACKM);
//
//		glScaled(MODELSCALE, MODELSCALE, MODELSCALE);
//		glColor3f(1.0, 1.0, 1.0);
//		//glGetFloatv(GL_MODELVIEW_MATRIX, mat);
//		glmDraw(MODEL, GLM_SMOOTH);//GLM_FLAT
//	
//		if( scene )
//		{
//			glGetFloatv(GL_MODELVIEW_MATRIX, m);
//			scene = false;
//		}
//	glPopMatrix();
//	
//	glutSwapBuffers();
//}

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.size().area()/*I1.size().width*I1.size().height*//*I1.total()*/);
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

double getPSNR(const IplImage *I1, const IplImage *I2)
{
    IplImage *s1 = cvCreateImage( cvGetSize(I1),  IPL_DEPTH_8U, 3);
	IplImage *s2 = cvCreateImage( cvGetSize(I1),  IPL_DEPTH_32F, 3);
	cvAbsDiff(I1, I2, s1);
	cvConvertScale(s1, s2, 1, 0);
	cvMul(s2, s2, s2);

    Scalar s = sum(s2);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1->nChannels* /*I1->size().area()*/I1->width*I1->height/*I1.total()*/);
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

void qualityAssess()
{
	GLubyte *red1 = new GLubyte[winWidth/3*winHeight];
	GLubyte *green1 = new GLubyte[winWidth/3*winHeight];
	GLubyte *blue1 = new GLubyte[winWidth/3*winHeight];
	GLubyte *red2 = new GLubyte[winWidth/3*winHeight];
	GLubyte *green2 = new GLubyte[winWidth/3*winHeight];
	GLubyte *blue2 = new GLubyte[winWidth/3*winHeight];
	glReadPixels(winWidth/3, 0, winWidth/3, winHeight, GL_RED, GL_UNSIGNED_BYTE, red1);
	glReadPixels(winWidth/3, 0, winWidth/3, winHeight, GL_GREEN, GL_UNSIGNED_BYTE, green1);
	glReadPixels(winWidth/3, 0, winWidth/3, winHeight, GL_BLUE, GL_UNSIGNED_BYTE, blue1);

	glReadPixels(winWidth/3*2, 0, winWidth/3, winHeight, GL_RED, GL_UNSIGNED_BYTE, red2);
	glReadPixels(winWidth/3*2, 0, winWidth/3, winHeight, GL_GREEN, GL_UNSIGNED_BYTE, green2);
	glReadPixels(winWidth/3*2, 0, winWidth/3, winHeight, GL_BLUE, GL_UNSIGNED_BYTE, blue2);

	//vector <GLfloat> colorVector;
	IplImage *bImg1 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *gImg1 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *rImg1 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *bImg2 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *gImg2 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *rImg2 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	
	for(int i=boundary; i<winWidth/3 - boundary; i++)
	{
		for(int j=boundary; j<winHeight - boundary; j++)
		{		
			cvSetReal2D( bImg1, winHeight - 1 - j - boundary, i - boundary, blue1[ j*winWidth/3 + i ]);
			cvSetReal2D( gImg1, winHeight - 1 - j - boundary, i - boundary, green1[ j*winWidth/3 + i ]);
			cvSetReal2D( rImg1, winHeight - 1 - j - boundary, i - boundary, red1[ j*winWidth/3 + i ]);
			cvSetReal2D( bImg2, winHeight - 1 - j - boundary, i - boundary, blue2[ j*winWidth/3 + i ]);
			cvSetReal2D( gImg2, winHeight - 1 - j - boundary, i - boundary, green2[ j*winWidth/3 + i ]);
			cvSetReal2D( rImg2, winHeight - 1 - j - boundary, i - boundary, red2[ j*winWidth/3 + i ]);
		}
	}

	/*IplImage *img = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *img1 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	IplImage *img2 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	cvConvertScaleAbs(colorImg, img, 255, 0);
	cvConvertScaleAbs(colorImg1, img1, 255, 0);
	cvConvertScaleAbs(colorImg2, img2, 255, 0);*/

	IplImage *colorImg1 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 3);
	IplImage *colorImg2 = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 3);

	IplImage *bgImg = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
	Relief2Image(bgMask, bgImg, winHeight - 2*boundary);
	bgFilter(bImg1, bgImg);
	bgFilter(gImg1, bgImg);
	bgFilter(rImg1, bgImg);
	bgFilter(bImg2, bgImg);
	bgFilter(gImg2, bgImg);
	bgFilter(rImg2, bgImg);

	cvMerge(bImg1, gImg1, rImg1, 0, colorImg1);
	cvMerge(bImg2, gImg2, rImg2, 0, colorImg2);

	cvNamedWindow("colorImg", 1);
	cvShowImage("colorImg", colorImg);
	cvNamedWindow("colorImg1", 1);
	cvShowImage("colorImg1", colorImg1);
	cvNamedWindow("colorImg2", 1);
	cvShowImage("colorImg2", colorImg2);

	double psnr1, psnr2;
	Scalar ssim1, ssim2;
	psnr1 = getPSNR(colorImg, colorImg1);
	psnr2 = getPSNR(colorImg, colorImg2);
	ssim1 = getMSSIM(colorImg, colorImg1);
	ssim2 = getMSSIM(colorImg, colorImg2);

	cout << "PSNR1: " << psnr1 << "\t" << "PSNR2: " << psnr2 << endl;
	cout << "SSIM1: " << ssim1[0] << "\t" << ssim1[1] << "\t" << ssim1[2] << endl << "SSIM2: " << ssim2[0] << "\t" << ssim2[1] << "\t" << ssim2[2] << endl;
			
}

void setSkewness(const IplImage *src, IplImage *dst, int aperture=3)
{
	int width =src->width;
	int height = src->height;
	IplImage *bgImg = cvCreateImage( cvGetSize(img0),  IPL_DEPTH_32F, 1);
	Relief2Image(bgMask, bgImg, height);

	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
		
			if( cvGetReal2D(bgImg, j, i) )
			{
				cvSetReal2D(dst, j, i, 0);
			}

			else
			{
			
				int ext = (aperture-1) / 2;
				int extU,extD,extL,extR;
				
					extU = -ext;
					extD = ext;
					extL = -ext;
					extR = ext;
				
					if( ext > j)		extU = -j;
					if( ext > i)		extL = -i;
					if( j+ext >= height)		extD = height - 1 - j;
					if( i+ext >= width)		extR =  width - 1 - i;

				double mean=0, sum=0, count=0;
				for(int p=extL; p <= extR; p++)
				{
					for(int q=extU; q <= extD; q++)
					{
						if( !cvGetReal2D(bgImg, j+q, i+p) )
						{
							mean += cvGetReal2D(src, j+q, i+p);
							count++;
						}
					}
				}
				mean /= count;

				cvGetReal2D(src, j, i) - mean;
				for(int p=extL; p <= extR; p++)
				{
					for(int q=extU; q <= extD; q++)
					{
						if( !cvGetReal2D(bgImg, j+q, i+p) )
						{
							sum += pow(cvGetReal2D(src, j, i) - mean, 3);
						}
					}
				}			

				cvSetReal2D(dst, j, i, sum/count);
			}
		}
	}

	cvAbs(dst, dst);
	double min, max;
	cvMinMaxLoc(dst, &min, &max);
	cvConvertScale(dst, dst, 1.0/max);

}

void multiDensity(IplImage *src, IplImage *dst)
{	
	int width = src->width;
	int height = src->height;
	for(int k=0; k<n; k++)
				{
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			double value = cvGetReal2D(src, j, i);

			if(value)
			{
				/*for(int k=0; k<n; k++)
				{*/
					if( thetaS[k] >= value && value > thetaS[k+1] )
					{
						cvCircle(src, cvPoint(i, j), radius[k], cvScalarAll(0), -1);
						cvSetReal2D(dst, j, i, 255);
						break;
					}
				/*}*/
			}
		}
	}
				}
}

void multiDensity(IplImage *src, IplImage *dst, IplImage *gradientX, IplImage *gradientY)
{	
	IplImage *gradX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
	IplImage *gradY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
	cvConvertScale(gradientX, gradX, 1, 0);
	cvConvertScale(gradientY, gradY, 1, 0);
	cvPow(gradX, gradX, 2);
	cvPow(gradY, gradY, 2);

	IplImage *gradient = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
	cvAdd(gradX, gradY, gradient);
	cvPow(gradient, gradient, 0.5);
	
	
	int width = src->width;
	int height = src->height;
	for(int i=0; i < width; i++)
	{
		for(int j=0; j < height; j++)
		{
			
			double value = cvGetReal2D(src, j, i);

			if(value)
			{				
				//if( cvGetReal2D(gradient, j, i) < 40)
				if( cvGetReal2D(ImageL, j, i) )
				{
					
					for(int k=0; k<n; k++)
					{
						if( thetaS[k] >= value && value > thetaS[k+1] )
						{
							cvCircle(src, cvPoint(i, j), radius[k], cvScalarAll(0), -1);
							cvSetReal2D(dst, j, i, 255);
							break;
						}
					}

				}
			}

		}
	}
}

void extractLine(IplImage *src, IplImage *dst, int interval, int linewidth)
{
	int m = cvGetSize(src).width;
	int n = cvGetSize(src).height;
	
	cvSet( dst, cvScalar(255) );

	int value =255;
	
	for(int level=0; level<255; level++)
	{
		for(int i=0; i<n; i++)
		{
			for(int j=0; j<m; j++)
			{
				value = cvGetReal2D(src, i, j);
				if(level*interval <= value && level*interval+linewidth > value)
				{
					cvSetReal2D(dst, i, j, 0);
				}
			}
		}
	}
}



void windowShowImage(char *name, IplImage *image)
{
	double min, max;
	cvMinMaxLoc(image, &min, &max);
	double scale = 255.0 / (max - min);   
	double shift = min * scale;  
	IplImage *show =  cvCreateImage( cvGetSize(image),  IPL_DEPTH_8U, 1);
	cvConvertScaleAbs(show, show, scale, shift);

	cvNamedWindow(name, 1);
	cvShowImage(name, show);
}



void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	displayfont();
	//displayline();

	clock_t start_time1, end_time1, start_time2, end_time2;
	float total_time1 = 0, total_time2 = 0;

    /*****     Partition 1     *****/
	if (trackballMove) {
		glPushMatrix();
			glLoadMatrixd(TRACKM);
			glRotatef(angle, axis[0], axis[1], axis[2]);
			glGetDoublev(GL_MODELVIEW_MATRIX, TRACKM);
		glPopMatrix();	    
	}
	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glDisable(GL_LIGHT1);
	glDisable(GL_LIGHT2);
	glDisable(GL_LIGHT3);
	glDisable(GL_LIGHT4);

	IplImage *gradientX,*gradientY;

	if( !scene )
	{
		openglPath();	
		
	}
	/*****     Partition 1     *****/
	//we must disable the opengl's depth test, then the software depth test will work 
	//glDisable(GL_DEPTH_TEST); 
	//glDisable(GL_LIGHTING);
	
	/*****     Partition 2     *****/
	partition2();
	firstLaplace();
	
	glPushMatrix();
		glMultMatrixd(TRACKM);
		//glTranslated(-0.2, 0, 0);

		if( relief1 )
		{
			time1 = true;
			total_time1 = 0;
			//start_time1 = clock();

			vector<GLfloat> laplace[pyrLevel - 1];

			
			
			for(int i=1;i< pyrLevel - 1;i++)
			{
				laplacianFilter( heightPyr[i], laplace[i], alpha[i], beta[i], (winHeight - boundary*2)/pow(2.0, i) );
				laplaceList.push_back(laplace[i]);
				recordMax(laplace[i]);
			}
			
			

			if( method == 1 )		//F1
			{	
				BilateralDetailBase();
			}
			else if( method == 2 )		//F2
			{
				linearCompressedBase();
			}
			else if( method == 3 )		//F3
			{
				IplImage *Image =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(img0, Image, 255, 0);

				IplImage *gradientX =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_16S, 1);
				IplImage *gradientY =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_16S, 1);
				cvSobel( Image, gradientX, 1, 0);
				cvSobel( Image, gradientY, 0, 1);

				sample =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvSetZero(sample);

				IplImage *wX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
				IplImage *wY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
				cvConvertScale(gradientX, wX, 1, 0);
				cvConvertScale(gradientY, wY, 1, 0);
				cvPow(wX, wX, 2);
				cvPow(wY, wY, 2);

				IplImage *weight = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
				cvAdd(wX, wY, weight);
				cvPow(weight, weight, 0.5);
				compress(weight, 10);
		

				IplImage *laplace =  cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);
				cvLaplace(Image, laplace);
				//weight = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_8U, 1);
				cvAbs(laplace, laplace);
				compress(laplace, 2.24);

				/*IplImage *weight = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_16S, 1);
				cvCopy(laplace, weight);*/
				
				cvNamedWindow("Color Map", 1);
				cvShowImage("Color Map", colorImg);
				string infix = /*"Buddhist Temple"*//*"River Terrain2"*//*"valleyRiver"*//*"ramp2"*//*"vase-lion"*//*"temple"*//*"golf"*//*"armadillo"*/"asain dragon";
				string whole = "img/" + infix + "C.bmp";
				cvSaveImage(whole.c_str(), colorImg);
				cvNamedWindow("Depth Map", 1);
				cvShowImage("Depth Map", depthImg);
				whole = "img/" + infix + "D.bmp";
				cvSaveImage(whole.c_str(), depthImg);

				colorImg = cvCreateImage( cvSize(winWidth/3 - 2*boundary, winHeight - 2*boundary),  IPL_DEPTH_8U, 3);
				IplImage *bImg = cvCreateImage( cvGetSize(colorImg),  IPL_DEPTH_8U, 1);
				IplImage *gImg = cvCreateImage( cvGetSize(colorImg),  IPL_DEPTH_8U, 1);
				IplImage *rImg = cvCreateImage( cvGetSize(colorImg),  IPL_DEPTH_8U, 1);

				GLubyte *red = new GLubyte[winWidth/3*winHeight];
				GLubyte *green = new GLubyte[winWidth/3*winHeight];
				GLubyte *blue = new GLubyte[winWidth/3*winHeight];
				glReadPixels(0, 0, winWidth/3, winHeight, GL_RED, GL_UNSIGNED_BYTE, red);
				glReadPixels(0, 0, winWidth/3, winHeight, GL_GREEN, GL_UNSIGNED_BYTE, green);
				glReadPixels(0, 0, winWidth/3, winHeight, GL_BLUE, GL_UNSIGNED_BYTE, blue);
				
				for(int i=boundary; i<winWidth/3 - boundary; i++)
				{
					for(int j=boundary; j<winHeight - boundary; j++)
					{					
						cvSetReal2D( bImg, winHeight - 1 - j - boundary, i - boundary, blue[ j*winWidth/3 + i ]);
						cvSetReal2D( gImg, winHeight - 1 - j - boundary, i - boundary, green[ j*winWidth/3 + i ]);
						cvSetReal2D( rImg, winHeight - 1 - j - boundary, i - boundary, red[ j*winWidth/3 + i ]);
					}
				}

				cvMerge(bImg, gImg, rImg, 0, colorImg);
				
				IplImage *grayImg =  cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 1);
				cvCvtColor( colorImg, grayImg, CV_BGR2GRAY );
				

				IplImage *grayGradient =  cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);
				IplImage *grayLaplace =  cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);
				
				cvSobel( grayImg, grayGradient, 1, 1);
				//cvLaplace(grayImg, grayGradient);
				cvLaplace(grayImg, grayLaplace);

				cvAbs(grayGradient, grayGradient);
				cvAbs(grayLaplace, grayLaplace);
				//windowShowImage("Gray Laplace", rImg);
				double min, max;
				cvMinMaxLoc(grayGradient, &min, &max);
				double scale1 = 255.0 / (max - min);   
				double shift1 = -min * scale;  
				IplImage *show =  cvCreateImage( cvGetSize(grayGradient),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(grayGradient, show, scale1, shift1);
				cvConvertScale(show, show, -1, 255);
				//cvThreshold(show, show, 250, 255, CV_THRESH_BINARY);

				cvNamedWindow("Gray Gradient", 1);
				cvShowImage("Gray Gradient", show);

				

				double featureMin, featureMax;
				cvMinMaxLoc(weight, &featureMin, &featureMax);
				double scale = 255.0 / (featureMax - featureMin);   
				double shift = -featureMin * scale;  
				IplImage *featureGradient =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				IplImage *smoothedFeature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(weight, featureGradient, scale, shift);
				cvConvertScale(featureGradient, featureGradient, -1, 255);
				//cvThreshold(feature, feature, 252, 255, CV_THRESH_BINARY);

				//cvThreshold(feature, feature, 128, 255, CV_THRESH_BINARY);	
				cvNamedWindow("Gradient", 1);
				cvShowImage("Gradient", featureGradient);

				//cvMin(featureGradient, show, featureGradient);
				

				/*IplImage *featureX =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_16S, 1);
				IplImage *featureY =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_16S, 1);
				IplImage *featureXY =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_16S, 1);
				cvSobel( feature, featureX, 1, 0);
				cvSobel( feature, featureY, 0, 1);
				cvSobel( feature, featureXY, 1, 1);

				wX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
				wY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
				cvConvertScale(featureX, wX, 1, 0);
				cvConvertScale(featureY, wY, 1, 0);
				cvPow(wX, wX, 2);
				cvPow(wY, wY, 2);

				weight = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
				cvAdd(wX, wY, weight);
				cvPow(weight, weight, 0.5);

				cvMinMaxLoc(weight, &featureMin, &featureMax);
				scale = 255.0 / (featureMax - featureMin);   
				shift = featureMin * scale;  
				feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(weight, feature, scale, shift);
				cvConvertScale(feature, feature, -1, 255);*/

				/*cvMinMaxLoc(featureXY, &featureMin, &featureMax);
				scale = 255.0 / (featureMax - featureMin);   
				shift = -featureMin * scale;  
				feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(featureXY, feature, scale, shift);
				cvConvertScale(feature, feature, -1, 255);*/

				cvNamedWindow("Gradient Feature", 1);
				cvShowImage("Gradient Feature", featureGradient);

				cvMinMaxLoc(laplace, &featureMin, &featureMax);
				scale = 255.0 / (featureMax - featureMin);   
				shift = -featureMin * scale;  
				IplImage *featureLaplace =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(laplace, featureLaplace, scale, shift);
				cvConvertScale(featureLaplace, featureLaplace, -1, 255);
				//cvThreshold(feature, feature, 252, 255, CV_THRESH_BINARY);

				cvThreshold(featureLaplace, featureLaplace, 128, 255, CV_THRESH_BINARY);	
				cvNamedWindow("Laplace", 1);
				cvShowImage("Laplace", featureLaplace);

				cvMinMaxLoc(grayLaplace, &min, &max);
				scale1 = 255.0 / (max - min);   
				shift1 = -min * scale;  
				show =  cvCreateImage( cvGetSize(grayLaplace),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(grayLaplace, show, scale1, shift1);
				cvConvertScale(show, show, -1, 255);

				cvThreshold(show, show, 224, 255, CV_THRESH_BINARY);	
				cvNamedWindow("Gray Laplace", 1);
				cvShowImage("Gray Laplace", show);
				

				cvMin(featureLaplace, show, featureLaplace);

				cvNamedWindow("Laplace Feature", 1);
				cvShowImage("Laplace Feature", featureLaplace);
				whole = "img/" + infix + "E.bmp";
				cvSaveImage(whole.c_str(), featureLaplace);

				for(int i=0; i < featureGradient->width; i++)
				{
					for(int j=0; j < featureGradient->height; j++)
					{
						if( cvGetReal2D( featureLaplace, j, i) != 255 )
						{
							cvSetReal2D( featureGradient, j, i, 255);
						}
					}
				}

				cvNamedWindow("Feature", 1);
				cvShowImage("Feature", featureGradient);

				IplImage *outline =  cvCreateImage( cvGetSize(featureGradient),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(featureGradient, featureGradient, -1, 255);			
				extractOutline(featureGradient, outline, 1);
				bgFilter(featureGradient, outline);				
								
				
				cvSmooth(featureGradient, smoothedFeature, CV_BILATERAL, 0, 0, 25, 11);			
				cvConvertScaleAbs(smoothedFeature, smoothedFeature, -1, 255);
				cvNamedWindow("Smoothed Feature", 1);
				cvShowImage("Smoothed Feature", smoothedFeature);
				
				cvMinMaxLoc(smoothedFeature, &min, &max);
				scale = 255.0 / (max - min);   
				shift = -min * scale;  
				cvConvertScaleAbs(smoothedFeature, featureGradient, scale, shift);
				cvNamedWindow("Normalized Feature", 1);
				cvShowImage("Normalized Feature", featureGradient);



				sample =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvSetZero(sample);
				
				/*IplImage *img =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvSetZero(img);
				cvConvertScaleAbs(img0, img, 255);*/
				numNonZero = cvCountNonZero(Image);
				cvNamedWindow("Original Points", 1);
				cvShowImage("Original Points", Image);

				

				/*****	Harris Corner	*****/
				//IplImage *feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_32F, 1);
				//
				//cvCornerHarris(img, feature, 3);
				//
				//
				////cvAdaptiveThreshold(sample, sample, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 0);
				//double featureMin, featureMax;
				//cvMinMaxLoc(feature, &featureMin, &featureMax);
				//cout << "featureMax: " << featureMax << endl;
				//double scale = 255 / (featureMax - featureMin);   
				//double shift = -featureMin * scale;  
				//cvConvertScaleAbs(feature, sample, scale, 0);
				/*****	Harris Corner	*****/

				/*****	Canny Edge	*****/
				edge =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				IplImage *edgeInv =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				
				cvMinMaxLoc(img0, &featureMin, &featureMax);
				cout << "featureMax: " << featureMax << endl;
				scale = 255.0 / (featureMax - featureMin);   
				shift = -featureMin * scale;  
				cvConvertScaleAbs(img0, edge, scale, shift);

				//cvCanny(edge, edge, 5, 25);
				cvCanny(edge, edge, 0, 150);
				cvThreshold(edge, edgeInv, 250, 255, CV_THRESH_BINARY_INV);
				
				/*****	Canny Edge	*****/

				/***** Morphology Opening *****/
				int nSize=3;
				IplConvKernel* circularElem = cvCreateStructuringElementEx(	nSize, // columns
																														nSize, // rows
																														floor(nSize/2.0), // anchor_x
																														floor(nSize/2.0), // anchor_y
																														CV_SHAPE_ELLIPSE, // shape
																														NULL);
				cvNamedWindow("Edge Map before Opening", 1);
				cvShowImage("Edge Map before Opening", edgeInv);

				IplImage *DFPrime = cvCreateImage( cvGetSize(edge), IPL_DEPTH_32F, 1 );
				cvDistTransform(edgeInv, DFPrime);

				cvMorphologyEx( edgeInv, edgeInv, 0, circularElem, CV_MOP_OPEN, 3 );
				cvNamedWindow("Edge Map", 1);
				cvShowImage("Edge Map", edgeInv);
				/***** Morphology Opening *****/

				

				

				/***** Distance Field *****/
				IplImage *DF = cvCreateImage( cvGetSize(edge), IPL_DEPTH_32F, 1 );
				IplImage *ImageDF =  cvCreateImage( cvGetSize(edge),  IPL_DEPTH_8U, 1 );
				cvDistTransform(featureLaplace, DF);

				cvMinMaxLoc(DF, &featureMin, &featureMax);
				scale = 255.0 / (featureMax - featureMin);   
				shift = -featureMin * scale;  
				cvConvertScaleAbs(DF, ImageDF, scale, shift);
				

				IplImage *Gray = cvCreateImage( cvGetSize(edge), IPL_DEPTH_8U, 1 );
				cvConvertScaleAbs(ImageDF, Gray, 3, 0);
				/*cvSet( Gray, cvScalar(255) );
				cvSub( Gray, ImageDF, Gray );*/

				cvNamedWindow("Distance Field without Smooth", 1);
				cvShowImage("Distance Field without Smooth", Gray);

				/***** Offset Lane without Smooth*****/
				ImageL =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
				ImageLPrime =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );

				///*extractLine(ImageDFPrime, ImageLPrime, interval, linewidth);
				//cvNamedWindow("Offset Lane without Opening", 1);
				//cvShowImage("Offset Lane without Opening", ImageLPrime);*/

				//extractLine(ImageDF, ImageL, interval, linewidth);
				//cvNamedWindow("Offset Lane without Smooth", 1);
				//cvShowImage("Offset Lane without Smooth", ImageL);
				/***** Offset Lane without Smooth*****/

				cvSmooth(ImageDF, ImageDF, CV_GAUSSIAN, 5);
				cvSmooth(Gray, Gray, CV_GAUSSIAN, 5);

				cvNamedWindow("Distance Field", 1);
				cvShowImage("Distance Field", ImageDF);
				
				cvNamedWindow("Distance Field with Smooth", 1);
				cvShowImage("Distance Field with Smooth", Gray);
				/*IplImage *ImageDFPrime =  cvCreateImage( cvGetSize(edge),  IPL_DEPTH_8U, 1 );
				cvMinMaxLoc(DFPrime, &featureMin, &featureMax);
				scale = 255.0 / (featureMax - featureMin);   
				shift = -featureMin * scale;  
				cvConvertScaleAbs(DFPrime, ImageDFPrime, scale, shift);
				cvSmooth(ImageDFPrime, ImageDFPrime, CV_GAUSSIAN, 5);
				cvNamedWindow("Distance Field without Opening", 1);
				cvShowImage("Distance Field without Opening", ImageDFPrime);*/
				/***** Distance Field *****/

				/***** Offset Lane *****/
				/*ImageL =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
				ImageLPrime =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
				int interval = 12,linewidth = 6;*/

				/*extractLine(ImageDFPrime, ImageLPrime, interval, linewidth);
				cvNamedWindow("Offset Lane without Opening", 1);
				cvShowImage("Offset Lane without Opening", ImageLPrime);*/

				extractLine(ImageDF, ImageL, interval, linewidth);
				cvNamedWindow("Offset Lane", 1);
				cvShowImage("Offset Lane", ImageL);

				/*cvConvertScaleAbs(ImageL, ImageL, -1, 255);
				cvMax(ImageL, featureGradient, featureGradient);
				cvNamedWindow("Offset Feature", 1);
				cvShowImage("Offset Feature", featureGradient);*/
				/***** Offset Lane *****/
			
				/*****	Skewness	*****/	
				IplImage *skewness =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_32F, 1);
				setSkewness(Image, skewness, 5);

				multiDensity(skewness, sample, gradientX, gradientY);
				//
				//cvThreshold(skewness, skewness, 0.000001, 255, CV_THRESH_BINARY);

				//double featureMin, featureMax;
				//cvMinMaxLoc(skewness, &featureMin, &featureMax);
				//cout << "featureMax: " << featureMax << endl;
				//double scale = 255.0 / (featureMax - featureMin);   
				//double shift = -featureMin * scale;  
				//IplImage *feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				//cvConvertScaleAbs(skewness, feature, scale, shift);
				/*****	Skewness	*****/

				/*****	Poisson Disk Sampling	*****/

				vector<unsigned int> importance;
				Image2Relief(featureGradient, importance);

				/*IplImage *Image1 = cvLoadImage("gradation.bmp",0);
				Image2Relief(Image1, importance);*/
				PDSampler *sampler = new PenroseSamplerW(samplingRatio, img0->width, img0->height, importance);
				sampler->complete();

				
				IplImage *PDSample =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				cvSet(PDSample, cvScalar(0));
				int N = (int) sampler->points.size();
				for (int i=0; i<N; i++) {
					//cvCircle(PDSample, cvPoint( (1+sampler->points[i].y)*img0->width, (-sampler->points[i].x)*img0->height ), 0, cvScalarAll(0), -1);
					int x =	(1+sampler->points[i].y)*img0->width;
					int y = (-sampler->points[i].x)*img0->height;
					
					double brightness = 255 - cvGetReal2D(featureGradient, y, x);
					cvSetReal2D(PDSample, y, x, brightness);
				}

				
				//IplImage *featureOPEN =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_8U, 1);
				//IplImage *featureThreshold =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_8U, 1);
				//int nSize=3;
				//IplConvKernel* circularElem = cvCreateStructuringElementEx(	nSize, // columns
				//																										nSize, // rows
				//																										floor(nSize/2.0), // anchor_x
				//																										floor(nSize/2.0), // anchor_y
				//																										CV_SHAPE_ELLIPSE, // shape
				//																										NULL);

				//cvMorphologyEx( feature, featureOPEN, 0, circularElem, CV_MOP_OPEN, 1 );
				//cvNamedWindow("OPEN first", 1);
				//cvShowImage("OPEN first", featureOPEN);
				//cvThreshold( featureOPEN, featureThreshold, 5, 255, CV_THRESH_BINARY);			
				//cvNamedWindow("Threshold last", 1);
				//cvShowImage("Threshold last", featureThreshold);

				//cvThreshold(feature, featureThreshold, 5, 255, CV_THRESH_BINARY);			
				//cvNamedWindow("Threshold first", 1);
				//cvShowImage("Threshold first", featureThreshold);

				//cvErode(featureThreshold, feature, circularElem, 5);
				////cvDilate(feature, feature, circularElem, 1);		
				//cvNamedWindow("FeatureDilateErode", 1);
				//cvShowImage("FeatureDilateErode", feature);

				//cvMorphologyEx( featureThreshold, featureOPEN, 0, circularElem, CV_MOP_OPEN, 1 );
				//cvNamedWindow("OPEN last", 1);
				//cvShowImage("OPEN last", featureOPEN);	

				cvCopy( PDSample, sample );
				cvNamedWindow("Poisson Disk Sampling", 1);
				cvShowImage("Poisson Disk Sampling", PDSample);
				/*****	Poisson Disk Sampling	*****/

				/*****	Carving Path	*****/
				IplImage *img32f = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
				IplImage *img8u = cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 1);
				Relief2Image( compressedH, img32f, winHeight - boundary*2 );
				cvConvertScaleAbs( img32f, img8u, 255);
				IplImage *GradientX = cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);
				IplImage *GradientY = cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);
				GradientA = cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);

				cvSobel( img8u, GradientX, 1, 0);
				cvSobel( img8u, GradientY, 0, 1);

				for(int i=0; i<GradientX->width; i++)
				{
					for(int j=0; j<GradientY->height; j++)
					{
						if( cvGetReal2D(featureGradient, j, i) == 255 )
						{
							cvSetReal2D( GradientA, j, i, 100 );
						}
						else
						{
							double y = cvGetReal2D(GradientY, j, i);
							double x = cvGetReal2D(GradientX, j, i);
							cvSetReal2D( GradientA, j, i, atan2(y, x) );
						}
					}
				}
		
				IplImage *A = cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(GradientA, A, 1, 0);
				cvNamedWindow("Angle", 1);
				cvShowImage("Angle", A);
				generateCarve( GradientA, 0.5 );
				/*****	Carving Path	*****/

				start_time1 = clock();
 				histogramBase(heightPyr[0], gradientX, gradientY, 0/*pyrLevel-1*/, 0 );
				//histogramBase(heightPyr[0], gradientX, gradientY);
				
				vector< pair<int, int> > pathMask;
				setPathMask(pathMask, 5);
				cout << "Mask Size: " << pathMask.size() << endl;


			}
			else	//F4
			{
				relief1 = false;
			}
		}

		if(mesh1)
		{
			if(map1)
			{
				IplImage *heightMap = cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 1);
				IplImage *heightImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
				Relief2Image(compressedH, heightImg, winHeight - boundary*2);
				cvConvertScaleAbs(heightImg, heightMap, 255, 0);

				cvNamedWindow("Height Map1", 1);
				cvShowImage("Height Map1", heightMap);

				IplImage *heightImg1 = cvCreateImage( cvGetSize(depthImg),  IPL_DEPTH_8U, 1);
				IplImage *heightImg2 = cvCreateImage( cvGetSize(depthImg),  IPL_DEPTH_8U, 1);
				IplImage *heightImg3 = cvCreateImage( cvGetSize(depthImg),  IPL_DEPTH_8U, 1);

				GLfloat divisor = 1.0/256;
				for(int i=boundary; i<winWidth/3 - boundary; i++)
				{
					for(int j=boundary; j<winHeight - boundary; j++)
					{
						
						GLfloat depth = cvGetReal2D( heightImg, winHeight - 1 - j - boundary, i - boundary);
						
					
							int a = (int)(depth/divisor);
							int b = (int)(fmod(depth, divisor)/divisor/divisor);
							int c = (int)(fmod(fmod(depth, divisor), divisor*divisor)/divisor/divisor/divisor);
							cvSetReal2D( heightImg3, winHeight - 1 - j - boundary, i - boundary, (int)(depth/divisor) );
							cvSetReal2D( heightImg2, winHeight - 1 - j - boundary, i - boundary, (int)(fmod(depth, divisor)/divisor/divisor) );
							cvSetReal2D( heightImg1, winHeight - 1 - j - boundary, i - boundary, (int)(fmod(fmod(depth, divisor), divisor*divisor)/divisor/divisor/divisor) );					
						
					}
				}

				heightImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 3);
				cvMerge(heightImg1, heightImg2, heightImg3, 0, heightImg);

				string infix = /*"Buddhist Temple"*//*"River Terrain2"*//*"valleyRiver"*//*"ramp2"*//*"vase-lion"*//*"temple"*//*"golf"*//*"armadillo"*/"asain dragon";
				string whole = "img/" + infix + "H.bmp";
				cvSaveImage(whole.c_str(), heightImg);

				map1 = false;
			}
			
			DrawRelief(compressedH, pThreadRelief, pThreadNormal);

			if(time1)
			{
				end_time1 = clock();
				total_time1 = (float)(end_time1 - start_time1)/CLOCKS_PER_SEC;
				cout << "Time1: " <<  total_time1 << std::endl;
				time1 = false;
			}
		}

	glPopMatrix();
	/*****     Partition 2     *****/

	/*****     Partition 3     *****/
	partition3();

	glPushMatrix();
		glMultMatrixd(TRACKM);
		glTranslated(0, 0, 0);

		if(relief2)
		{
			int width = sqrt( (float) heightPyr[0].size() );
			/*IplImage *img= cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
			Relief2Image(heightPyr[0], img);*/

			IplImage *Image =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			cvConvertScaleAbs(img0, Image, 255, 0);

			IplImage *gradientX =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_16S, 1);
			IplImage *gradientY =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_16S, 1);
			cvSobel( Image, gradientX, 1, 0);
			cvSobel( Image, gradientY, 0, 1);

			IplImage *wX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
			IplImage *wY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
			cvConvertScale(gradientX, wX, 1, 0);
			cvConvertScale(gradientY, wY, 1, 0);
			cvPow(wX, wX, 2);
			cvPow(wY, wY, 2);

			IplImage *weight = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
			cvAdd(wX, wY, weight);
			cvPow(weight, weight, 0.5);
			compress(weight, 10);

			//IplImage *laplace =  cvCreateImage( cvGetSize(img0), IPL_DEPTH_16S, 1);
			//cvLaplace(Image, laplace);
			////cvSet(laplace, cvScalar(1));

			//cvAbs(laplace, laplace);
			//
			//compress(laplace, 0.5);
			//double featureMin, featureMax;
			//cvMinMaxLoc(weight, &featureMin, &featureMax);
			//double scale = 255.0 / (featureMax - featureMin);   
			//double shift = featureMin * scale;  
			//IplImage *feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvConvertScaleAbs(weight, feature, scale, shift);
			//cvConvertScale(feature, feature, -1, 255);
			////cvThreshold(feature, feature, 252, 255, CV_THRESH_BINARY);

			//cvThreshold(feature, feature, 128, 255, CV_THRESH_BINARY);	
			//cvNamedWindow("Gradient", 1);
			//cvShowImage("Gradient", feature);

			///*IplImage *featureX =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_16S, 1);
			//IplImage *featureY =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_16S, 1);
			//IplImage *featureXY =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_16S, 1);
			//cvSobel( feature, featureX, 1, 0);
			//cvSobel( feature, featureY, 0, 1);
			//cvSobel( feature, featureXY, 1, 1);

			//wX = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
			//wY = cvCreateImage( cvGetSize(gradientY), IPL_DEPTH_64F, 1);
			//cvConvertScale(featureX, wX, 1, 0);
			//cvConvertScale(featureY, wY, 1, 0);
			//cvPow(wX, wX, 2);
			//cvPow(wY, wY, 2);

			//weight = cvCreateImage( cvGetSize(gradientX), IPL_DEPTH_64F, 1);
			//cvAdd(wX, wY, weight);
			//cvPow(weight, weight, 0.5);

			//cvMinMaxLoc(weight, &featureMin, &featureMax);
			//scale = 255.0 / (featureMax - featureMin);   
			//shift = featureMin * scale;  
			//feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvConvertScaleAbs(weight, feature, scale, shift);
			//cvConvertScale(feature, feature, -1, 255);*/

			///*cvMinMaxLoc(featureXY, &featureMin, &featureMax);
			//scale = 255.0 / (featureMax - featureMin);   
			//shift = -featureMin * scale;  
			//feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvConvertScaleAbs(featureXY, feature, scale, shift);
			//cvConvertScale(feature, feature, -1, 255);*/

			//cvNamedWindow("Feature", 1);
			//cvShowImage("Feature", feature);

			//sample =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvSetZero(sample);
			//
			///*IplImage *img =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvSetZero(img);
			//cvConvertScaleAbs(img0, img, 255);*/
			//numNonZero = cvCountNonZero(Image);
			//cvNamedWindow("Original Points", 1);
			//cvShowImage("Original Points", Image);

			//

			///*****	Harris Corner	*****/
			////IplImage *feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_32F, 1);
			////
			////cvCornerHarris(img, feature, 3);
			////
			////
			//////cvAdaptiveThreshold(sample, sample, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 0);
			////double featureMin, featureMax;
			////cvMinMaxLoc(feature, &featureMin, &featureMax);
			////cout << "featureMax: " << featureMax << endl;
			////double scale = 255 / (featureMax - featureMin);   
			////double shift = -featureMin * scale;  
			////cvConvertScaleAbs(feature, sample, scale, 0);
			///*****	Harris Corner	*****/

			///*****	Canny Edge	*****/
			//edge =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//IplImage *edgeInv =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//
			//cvMinMaxLoc(img0, &featureMin, &featureMax);
			//cout << "featureMax: " << featureMax << endl;
			//scale = 255.0 / (featureMax - featureMin);   
			//shift = -featureMin * scale;  
			//cvConvertScaleAbs(img0, edge, scale, shift);

			////cvCanny(edge, edge, 5, 25);
			//cvCanny(edge, edge, 0, 150);
			//cvThreshold(edge, edgeInv, 250, 255, CV_THRESH_BINARY_INV);
			//
			///*****	Canny Edge	*****/

			///***** Morphology Opening *****/
			//int nSize=3;
			//IplConvKernel* circularElem = cvCreateStructuringElementEx(	nSize, // columns
			//																										nSize, // rows
			//																										floor(nSize/2.0), // anchor_x
			//																										floor(nSize/2.0), // anchor_y
			//																										CV_SHAPE_ELLIPSE, // shape
			//																										NULL);
			//cvNamedWindow("Edge Map before Opening", 1);
			//cvShowImage("Edge Map before Opening", edgeInv);

			//IplImage *DFPrime = cvCreateImage( cvGetSize(edge), IPL_DEPTH_32F, 1 );
			//cvDistTransform(edgeInv, DFPrime);

			//cvMorphologyEx( edgeInv, edgeInv, 0, circularElem, CV_MOP_OPEN, 3 );
			//cvNamedWindow("Edge Map", 1);
			//cvShowImage("Edge Map", edgeInv);
			///***** Morphology Opening *****/

			//

			//

			///***** Distance Field *****/
			//IplImage *DF = cvCreateImage( cvGetSize(edge), IPL_DEPTH_32F, 1 );
			//IplImage *ImageDF =  cvCreateImage( cvGetSize(edge),  IPL_DEPTH_8U, 1 );
			//cvDistTransform(feature, DF);

			//cvMinMaxLoc(DF, &featureMin, &featureMax);
			//scale = 255.0 / (featureMax - featureMin);   
			//shift = -featureMin * scale;  
			//cvConvertScaleAbs(DF, ImageDF, scale, shift);
			//

			//IplImage *Gray = cvCreateImage( cvGetSize(edge), IPL_DEPTH_8U, 1 );
			//cvConvertScaleAbs(ImageDF, Gray, 3, 0);
			///*cvSet( Gray, cvScalar(255) );
			//cvSub( Gray, ImageDF, Gray );*/

			//cvNamedWindow("Distance Field without Smooth", 1);
			//cvShowImage("Distance Field without Smooth", Gray);

			///***** Offset Lane without Smooth*****/
			//ImageL =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
			//ImageLPrime =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
			//int interval = 12,linewidth = 6;

			/////*extractLine(ImageDFPrime, ImageLPrime, interval, linewidth);
			////cvNamedWindow("Offset Lane without Opening", 1);
			////cvShowImage("Offset Lane without Opening", ImageLPrime);*/

			////extractLine(ImageDF, ImageL, interval, linewidth);
			////cvNamedWindow("Offset Lane without Smooth", 1);
			////cvShowImage("Offset Lane without Smooth", ImageL);
			///***** Offset Lane without Smooth*****/

			//cvSmooth(ImageDF, ImageDF, CV_GAUSSIAN, 5);
			//cvSmooth(Gray, Gray, CV_GAUSSIAN, 5);

			//cvNamedWindow("Distance Field", 1);
			//cvShowImage("Distance Field", ImageDF);
			//
			//cvNamedWindow("Distance Field with Smooth", 1);
			//cvShowImage("Distance Field with Smooth", Gray);
			///*IplImage *ImageDFPrime =  cvCreateImage( cvGetSize(edge),  IPL_DEPTH_8U, 1 );
			//cvMinMaxLoc(DFPrime, &featureMin, &featureMax);
			//scale = 255.0 / (featureMax - featureMin);   
			//shift = -featureMin * scale;  
			//cvConvertScaleAbs(DFPrime, ImageDFPrime, scale, shift);
			//cvSmooth(ImageDFPrime, ImageDFPrime, CV_GAUSSIAN, 5);
			//cvNamedWindow("Distance Field without Opening", 1);
			//cvShowImage("Distance Field without Opening", ImageDFPrime);*/
			///***** Distance Field *****/

			///***** Offset Lane *****/
			///*ImageL =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
			//ImageLPrime =  cvCreateImage( cvGetSize(ImageDF),  IPL_DEPTH_8U, 1 );
			//int interval = 12,linewidth = 6;*/

			///*extractLine(ImageDFPrime, ImageLPrime, interval, linewidth);
			//cvNamedWindow("Offset Lane without Opening", 1);
			//cvShowImage("Offset Lane without Opening", ImageLPrime);*/

			//extractLine(ImageDF, ImageL, interval, linewidth);
			//cvNamedWindow("Offset Lane", 1);
			//cvShowImage("Offset Lane", ImageL);
			/***** Offset Lane *****/
			

			/*****	Skewness	*****/	
			IplImage *skewness =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_32F, 1);
			setSkewness(Image, skewness, 5);

			//multiDensity(skewness, sample, gradientX, gradientY);
			//
			//cvThreshold(skewness, skewness, 0.000001, 255, CV_THRESH_BINARY);

			//double featureMin, featureMax;
			//cvMinMaxLoc(skewness, &featureMin, &featureMax);
			//cout << "featureMax: " << featureMax << endl;
			//double scale = 255.0 / (featureMax - featureMin);   
			//double shift = -featureMin * scale;  
			//IplImage *feature =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvConvertScaleAbs(skewness, feature, scale, shift);
			/*****	Skewness	*****/

			/*****	Poisson Disk Sampling	*****/

			//vector<unsigned int> importance;
			//Image2Relief(feature, importance);

			///*IplImage *Image1 = cvLoadImage("gradation.bmp",0);
			//Image2Relief(Image1, importance);*/
			//PDSampler *sampler = new PenroseSamplerW(0.02, img0->width, img0->height, importance);
			//sampler->complete();

			//
			//IplImage *PDSample =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
			//cvSet(PDSample, cvScalar(255));
			//int N = (int) sampler->points.size();
			//for (int i=0; i<N; i++) {
			//	cvCircle(PDSample, cvPoint( (1+sampler->points[i].y)*img0->width, (-sampler->points[i].x)*img0->height ), 0, cvScalarAll(0), -1);
			//}

			
			//IplImage *featureOPEN =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_8U, 1);
			//IplImage *featureThreshold =  cvCreateImage( cvGetSize(feature),  IPL_DEPTH_8U, 1);
			//int nSize=3;
			//IplConvKernel* circularElem = cvCreateStructuringElementEx(	nSize, // columns
			//																										nSize, // rows
			//																										floor(nSize/2.0), // anchor_x
			//																										floor(nSize/2.0), // anchor_y
			//																										CV_SHAPE_ELLIPSE, // shape
			//																										NULL);

			//cvMorphologyEx( feature, featureOPEN, 0, circularElem, CV_MOP_OPEN, 1 );
			//cvNamedWindow("OPEN first", 1);
			//cvShowImage("OPEN first", featureOPEN);
			//cvThreshold( featureOPEN, featureThreshold, 5, 255, CV_THRESH_BINARY);			
			//cvNamedWindow("Threshold last", 1);
			//cvShowImage("Threshold last", featureThreshold);

			//cvThreshold(feature, featureThreshold, 5, 255, CV_THRESH_BINARY);			
			//cvNamedWindow("Threshold first", 1);
			//cvShowImage("Threshold first", featureThreshold);

			//cvErode(featureThreshold, feature, circularElem, 5);
			////cvDilate(feature, feature, circularElem, 1);		
			//cvNamedWindow("FeatureDilateErode", 1);
			//cvShowImage("FeatureDilateErode", feature);

			//cvMorphologyEx( featureThreshold, featureOPEN, 0, circularElem, CV_MOP_OPEN, 1 );
			//cvNamedWindow("OPEN last", 1);
			//cvShowImage("OPEN last", featureOPEN);	

			/*cvNamedWindow("Poisson Disk Sampling", 1);
			cvShowImage("Poisson Disk Sampling", PDSample);*/
			/*****	Poisson Disk Sampling	*****/

			/*cout << "#NonZero of Original: " << numNonZero << endl;		

			numNonZero = cvCountNonZero(sample);
			cout << "#NonZero of Sample: " << numNonZero << endl;

			cvThreshold(sample, sample, 0, 255, CV_THRESH_BINARY);
			cvNamedWindow("Sample Points", 1);
			cvShowImage("Sample Points", sample);*/

			
			
			time2 = true;
			total_time2 = 0;
			start_time2 = clock();

			if( reference == 1 )		//F9
			{
				gradientCorrection(gradientX, gradientY);		
			}
			else if( reference == 2 )	//F10
			{
				cvConvertScale(gradientX,gradientX, 1, 1);
				cvConvertScale(gradientY, gradientY, 1, 1);
				reliefHistogram(heightPyr[0], gradientX, gradientY);
			}
			else if( reference == 3 )	//F11
			{
				int level = 2;
				//int side = sqrt( (float)heightPyr[2].size() );
				/*int height = (winHeight - boundary*2) / pow(2.0, level);
				int width = heightPyr[level].size() / height;
				IplImage *img = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
				Relief2Image(heightPyr[level], img, height);
				Image =  cvCreateImage( cvGetSize(img),  IPL_DEPTH_8U, 1);
				cvConvertScaleAbs(img, Image, 255, 0);

				gradientX =  cvCreateImage( cvGetSize(Image),  IPL_DEPTH_16S, 1);
				gradientY =  cvCreateImage( cvGetSize(Image),  IPL_DEPTH_16S, 1);
				cvSobel( Image, gradientX, 1, 0);
				cvSobel( Image, gradientY, 0, 1);*/
				
				reliefHistogram(heightPyr[0], gradientX, gradientY, level);

				//height = winHeight - boundary*2;
				//int size = referenceHeight.size();
				//vector<GLfloat> AHEHeight;
				//AHEHeight.resize(size);

				////cvMorphologyEx( edge, edge, 0, circularElem, CV_MOP_CLOSE, 3 );
				//cvDilate( edge, edge, circularElem, 1);
				//equalizeHist(heightPyr[0], AHEHeight, 1, edge, weight, pow(2.0, k-1) * 8*2 + 1, height);

				//IplImage *edgeHeight = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
				//Relief2Image(AHEHeight, edgeHeight);
				//IplImage *showedge =  cvCreateImage( cvGetSize(img0),  IPL_DEPTH_8U, 1);
				//cvConvertScaleAbs(edgeHeight, showedge, 255, 0);
				//cvNamedWindow("Show Edge", 1);
				//cvShowImage("Show Edge", showedge);
				//
				//vectorReplace(referenceHeight, AHEHeight, referenceHeight);
				//BuildRelief(referenceHeight, pThreadEqualizeRelief, pThreadEqualizeNormal);
			}
			else	//F12 linear compression
			{		
				/*referenceHeight.resize( heightPyr[pyrLevel-1].size() );

				
				int height = (winHeight - boundary*2) / pow(2.0, pyrLevel-1);
				int width = heightPyr[pyrLevel-1].size() / height;
				IplImage *srcImg = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1);
				Relief2Image(heightPyr[pyrLevel-1], srcImg, height);
				
				for(int i=pyrLevel-1; i>0; i--)
				{			
					IplImage *dstImg = cvCreateImage( cvSize(srcImg->width*2, srcImg->height*2), IPL_DEPTH_32F, 1);
					cvPyrUp(srcImg, dstImg);
					srcImg = cvCreateImage( cvSize(dstImg->width, dstImg->height), IPL_DEPTH_32F, 1);
					cvCopy(dstImg, srcImg);
				}
				Image2Relief(srcImg, referenceHeight);*/

				float	 maxHeight=0, minHeight=1;
				for(int i=0; i < heightPyr[0].size(); i++)
				{
					float height = heightPyr[0][i];
					if( maxHeight < height )
					{
						maxHeight = height;
					}
					if( minHeight > height && height !=0 )
					{
						minHeight = height;
					}
				}
				float range = maxHeight - minHeight;
				referenceHeight.resize( heightPyr[0].size() );
				for(int i=0; i < heightPyr[0].size(); i++)
				{
					referenceHeight[i] = heightPyr[0][i] / range;
				}
				/*for(int i=0; i<heightPyr[pyrLevel-1].size(); i++)
				{
					referenceHeight[i] = heightPyr[pyrLevel-1][i];
				}*/
				BuildRelief(referenceHeight, pThreadEqualizeRelief, pThreadEqualizeNormal);

				relief2 = false;
				mesh2 = true;
				profile2 = true;
			}
		}

		if(mesh2)
		{
			
		
			DrawRelief(referenceHeight, pThreadEqualizeRelief, pThreadEqualizeNormal);
			
			if(map2)
			{
				IplImage *heightMap = cvCreateImage( cvGetSize(img0), IPL_DEPTH_8U, 1);
				IplImage *heightImg = cvCreateImage( cvGetSize(img0), IPL_DEPTH_32F, 1);
				Relief2Image(referenceHeight, heightImg, winHeight - boundary*2);
				cvConvertScaleAbs(heightImg, heightMap, 255, 0);

				cvNamedWindow("Height Map2", 1);
				cvShowImage("Height Map2", heightMap);

				map2 = false;
			}

			if(time2)
			{
				end_time2 = clock();
				total_time2 = (float)(end_time2 - start_time2)/CLOCKS_PER_SEC;
				cout << "Time2: " <<  total_time2 << std::endl;
				time2 = false;
			}
		}

	glPopMatrix();
	/*****     Partition 3     *****/
	
	
	

    glutSwapBuffers();

}

/*----------------------------------------------------------------------*/

void mouseButton0(int button, int state, int x, int y)
{
	if(button==GLUT_RIGHT_BUTTON) {
		exit(0);
	}

	if(button==GLUT_LEFT_BUTTON) switch(state) 
	{
		case GLUT_DOWN:
			y=winHeight-y;
			startMotion0(x, y);
			break;
		case GLUT_UP:
			y=winHeight-y;
			stopMotion0(x, y);
			break;
    } 
}

void mouseButton(int button, int state, int x, int y)
{
	if(button==GLUT_RIGHT_BUTTON) {
		exit(0);
	}

	if(button==GLUT_LEFT_BUTTON) switch(state) 
	{
		case GLUT_DOWN:
			y=winHeight-y;
			//startMotion(x, y);
			break;
		case GLUT_UP:
			y=winHeight-y;
			//stopMotion(x, y);
			break;
    } 
}

void myReshape0(int w, int h)
{
    float multiple = h/4.0;

	winHeight0 = 4*round(multiple);
	winWidth0 =  winHeight0*2;
    
	perspective[1] = winWidth0/2.0/winHeight0;

	glutSetWindow(Window0);
	glutReshapeWindow(winHeight0*2, winHeight0);
	glutSetWindow(Window);
	glutReshapeWindow(winHeight0*horizontalSplit, winHeight0);


	mesh1=false;
	mesh2=false;

	swInitZbuffer(winWidth0/2, winHeight0);
}

void myReshape(int w, int h)
{
    winWidth = w;
    winHeight = h;

	vertCount = ( winWidth/2 - boundary*2 ) * ( winHeight - boundary*2 );
	pThreadRelief = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3);
	pThreadNormal = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3 *2);
	pThreadEqualizeRelief = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3);
	pThreadEqualizeNormal = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3 *2);

	swInitZbuffer(w/2, h);
}

void spinCube()
{
    if (redrawContinue) glutPostRedisplay();
}

void update(int i)
{
	TICK++;
	int temp=TICK%180;
	if(temp<90)
		Angle1++;
	else
		Angle1--;

	//int temp2=TICK%90;
	if(temp<90)
		Angle2+=0.5;
	else
		Angle2-=0.5;

	glutPostRedisplay();
	glutTimerFunc(33, update, ++i);
}

void setZeroAxis()
{
	axis[0] = 0;
	axis[1] = 0;
	axis[2] = 0;
}

void initTM()
{
	TRACKM[0] = 1;
	TRACKM[1] = 0;
	TRACKM[2] = 0;
	TRACKM[3] = 0;
	TRACKM[4] = 0;
	TRACKM[5] = 1;
	TRACKM[6] = 0;
	TRACKM[7] = 0;
	TRACKM[8] = 0;
	TRACKM[9] = 0;
	TRACKM[10] = 1;
	TRACKM[11] = 0;
	TRACKM[12] = 0;
	TRACKM[13] = 0;
	TRACKM[14] = 0;
	TRACKM[15] = 1;
}

void SpecialKeys(int key, int x, int y)
{
	switch(key)
	{
		case GLUT_KEY_LEFT:
			angleY--;
			setZeroAxis();
			break;
		case GLUT_KEY_UP:
			angleX--;
			setZeroAxis();
			break;
		case GLUT_KEY_PAGE_UP:
			angleZ--;
			setZeroAxis();
			break;
		case GLUT_KEY_RIGHT:
			angleY++;
			break;
		case GLUT_KEY_DOWN:
			angleX++;
			break;
		case GLUT_KEY_PAGE_DOWN:
			angleZ++;
			setZeroAxis();
			break;
		case GLUT_KEY_INSERT:
			reliefAngleZ--;
			setZeroAxis();
			break;
		case GLUT_KEY_F1:
			method = 1;
			break;
		case GLUT_KEY_F2:
			method = 2;
			break;
		case GLUT_KEY_F3:
			method = 3;
			break;
		case GLUT_KEY_F4:
			method = 0;
			break;
		case GLUT_KEY_F9:
			reference = 1;
			break;
		case GLUT_KEY_F10:
			reference = 2;
			break;
		case GLUT_KEY_F11:
			reference = 3;
			break;
		case GLUT_KEY_F12:
			reference = 0;
			break;
	}
}

void myKeys(unsigned char key, int x, int y)
{
	switch(key)
	{
		/*case ' ':
			glPushMatrix();
				glLoadIdentity();
				glGetDoublev(GL_MODELVIEW_MATRIX, TRACKM);
			glPopMatrix();	 
			break;*/
		case 'l':
			if( scalingFactor >= 2 )
			{
				scalingFactor /= 2;
			}
			break;
		case 'a':
			MODELSCALE += 0.5;
			break;
		case 's':
			if(MODELSCALE > 0.1)
				MODELSCALE -= 0.5;
			break;

		case 'z':
			LIGHTP += 1;
			std::cout<<"LIGHTP "<<LIGHTP<<'\n';
			break;
		case 'x':
			LIGHTP -= 1;
			std::cout<<"LIGHTP "<<LIGHTP<<'\n';
			break;

        case 'Q':
        case 'q': 
			qualityAssess();
			//exit(0); 
			break;
		/***** perspective setting *****/
        case '0':  
			//DRAWTYPE=0; 
			perspective[2] = 0.1;
			perspective[3] = 25;
			break;
        case '1':  
			//DRAWTYPE=1; 
			if( perspective[2] > 0.1 )
			{
				perspective[2] -= 0.1;
			}
			break;
		case '3':  
			perspective[2] += 0.1;
			break;
		case '7':  
			if( perspective[3] > 0.1 )
			{
				perspective[3] -= 0.1;
			}
			break;
		case '9':  
			perspective[3] += 0.1;
			break;
		case '[':  
			frame -= winWidth0/50;
			if(frame < 0)
			{
				frame = 0;
			}
			break;
		case ']':  
			frame += winWidth0/50;
			if(frame > winWidth0)
			{
				frame = winWidth0;
			}
			break;
		/***** perspective setting *****/
		/***** relief transformation *****/
        case '4':  
			reliefAngleY--;
			setZeroAxis();
			break;
        case '6':  
			reliefAngleY++;
			setZeroAxis();
			break;
		case '*':  
			 reliefAngleZ--;
			 setZeroAxis();
			break;
		case '/':  
			reliefAngleZ++;
			setZeroAxis();
			break;
		case '8':  
			 reliefAngleX--;
			 setZeroAxis();
			break;
		case '2':  
			reliefAngleX++;
			setZeroAxis();
			break;
		case '+':  
			scale += 0.125; 
			break;
        case '-': 
			if(scale > 0.1)
			scale -= 0.125; 
			break;
		/***** relief transformation *****/
		case 'r':
			perspective[2]=0.3; 
			perspective[3]=100;
			break;
		//Space
		case ' ':  
			scene = true;
			GetClipDistance();

			glutSetWindow(Window);
			glutReshapeWindow(winHeight0*horizontalSplit, winHeight0);
			
			break;
		//Enter
		case 13:  
			relief1 = true;
			relief2 = true;
			setZeroAxis();
			initTM();
			break;
		// Delete
		case 127 :
			reliefAngleZ++;
			setZeroAxis();
			break;
	}
	glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	
	glutInitWindowSize(windowHeight*2, windowHeight);
	glutInitWindowPosition(0, 20);
	Window0 = glutCreateWindow("Oringinal Scene");

    glutInitWindowSize(windowHeight*horizontalSplit, windowHeight);
	glutInitWindowPosition(0, 505);
    Window = glutCreateWindow("Digital Bas-Relief from 3D Scenes");


	vertCount = ( windowHeight - boundary*2 ) * ( windowHeight - boundary*2 );
	pThreadRelief = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3);
	pThreadNormal = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3 *2);
	pThreadEqualizeRelief = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3);
	pThreadEqualizeNormal = (GLdouble*) malloc ( sizeof(GLdouble) * vertCount *3 *2);

	GLfloat  ambientLight0[] = { 0.1f, 0.1f, 0.1f, 1.0f };
    GLfloat  diffuseLight0[] = { 0.7f, 0.7f, 0.7f, 1.0f };
    GLfloat  specular0[] = { 0.7f, 0.7f, 0.7f, 1.0f };

	GLfloat  ambientLight1[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	GLfloat  ambientLight2[] = { 0.05f, 0.05f, 0.05f, 1.0f };
    GLfloat  diffuseLight1[] = { 0.3f, 0.3f, 0.3f, 1.0f };
	GLfloat  diffuseLight2[] = { 0.05f, 0.05f, 0.05f, 1.0f };

    GLfloat  specular1[] = { 0.0f, 0.0f, 0.0f, 1.0f };

    GLfloat  specref[] = { 0.25f, 0.25f, 0.25f, 0.25f };
	GLfloat  shininess = 32.0f;


	glutSetWindow(Window0);
    glutReshapeFunc(myReshape0);

    glutDisplayFunc(display0);
    glutIdleFunc(spinCube);
    glutMouseFunc(mouseButton0);
	glutMotionFunc(mouseMotion0);
	glutKeyboardFunc(myKeys);
	glutSpecialFunc(SpecialKeys);
	glutTimerFunc(33, update, 0);

	glEnable(GL_DEPTH_TEST); 
	//glDepthRange(0.0, 5.0);
	//Font = FontCreate(wglGetCurrentDC(), "Times", 32, 0, 1);



	//Read model
	MODEL = glmReadOBJ("model/temple.obj");
	glmUnitize(MODEL);
	//glmFacetNormals(MODEL);
	//glmVertexNormals(MODEL, 90);

    // Light values and coordinates
   
    // Enable lighting
    glEnable(GL_LIGHTING);

    // Setup and enable light 0
    glLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight0);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight0);
    glLightfv(GL_LIGHT0,GL_SPECULAR, specular0);
    glEnable(GL_LIGHT0);

	/*glLightfv(GL_LIGHT1,GL_AMBIENT,ambientLight1);
    glLightfv(GL_LIGHT1,GL_DIFFUSE,diffuseLight1);
    glLightfv(GL_LIGHT1,GL_SPECULAR, specular1);

	glLightfv(GL_LIGHT2,GL_AMBIENT,ambientLight1);
    glLightfv(GL_LIGHT2,GL_DIFFUSE,diffuseLight1);
    glLightfv(GL_LIGHT2,GL_SPECULAR, specular1);*/
    

    // Enable color tracking
    glEnable(GL_COLOR_MATERIAL);
    // Set Material properties to follow glColor values
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	// All materials hereafter have full specular reflectivity with a high shine 
    glMaterialfv(GL_FRONT, GL_SPECULAR, specref);
    glMateriali(GL_FRONT, GL_SHININESS, shininess);

	//
	//hw3
	//
	swLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight0);
    swLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight0);
    swLightfv(GL_LIGHT0,GL_SPECULAR, specular0);

    swMaterialfv(GL_FRONT, GL_SPECULAR, specref);
    swMateriali(GL_FRONT, GL_SHININESS, shininess);


	
	glutSetWindow(Window);
    glutReshapeFunc(myReshape);

    glutDisplayFunc(display);
    glutIdleFunc(spinCube);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
	glutKeyboardFunc(myKeys);
	glutSpecialFunc(SpecialKeys);
	glutTimerFunc(33, update, 0);

	glEnable(GL_DEPTH_TEST); 
	//glDepthRange(0.0, 5.0);
	//Font = FontCreate(wglGetCurrentDC(), "Times", 32, 0, 1);



	//Read model
	/*MODEL = glmReadOBJ("asain dragon.obj");
	glmUnitize(MODEL);*/
	//glmFacetNormals(MODEL);
	//glmVertexNormals(MODEL, 90);

   
    // Enable lighting
    glEnable(GL_LIGHTING);

    // Setup and enable light 0
	glLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight0);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight0);
    glLightfv(GL_LIGHT0,GL_SPECULAR, specular0);
    glEnable(GL_LIGHT0);

	glLightfv(GL_LIGHT1,GL_AMBIENT,ambientLight1);
    glLightfv(GL_LIGHT1,GL_DIFFUSE,diffuseLight1);
    glLightfv(GL_LIGHT1,GL_SPECULAR, specular1);

	glLightfv(GL_LIGHT2,GL_AMBIENT,ambientLight1);
    glLightfv(GL_LIGHT2,GL_DIFFUSE,diffuseLight1);
    glLightfv(GL_LIGHT2,GL_SPECULAR, specular1);
    
	glLightfv(GL_LIGHT3,GL_DIFFUSE,diffuseLight2);
	glLightfv(GL_LIGHT4,GL_DIFFUSE,diffuseLight2);

    // Enable color tracking
    glEnable(GL_COLOR_MATERIAL);
    // Set Material properties to follow glColor values
    /*glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);*/

	// All materials hereafter have full specular reflectivity with a high shine 
    glMaterialfv(GL_FRONT, GL_SPECULAR, specref);
    glMateriali(GL_FRONT, GL_SHININESS, shininess);

	//
	//hw3
	//
	/*swLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight0);
    swLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight0);
    swLightfv(GL_LIGHT0,GL_SPECULAR, specular0);

    swMaterialfv(GL_FRONT, GL_SPECULAR, specref);
    swMateriali(GL_FRONT, GL_SHININESS, shininess);*/


    glutMainLoop();

	free(pThreadRelief);
	free(pThreadNormal);
	free(pThreadEqualizeRelief);
	free(pThreadEqualizeNormal);


	cvWaitKey(0);
}