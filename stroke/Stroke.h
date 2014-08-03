#include <GL/glut.h>
#include <math.h>
#include <vector>
#include "Point.h"

using namespace std;


#define NUM_SLICES 20

#ifndef M_PI
#define M_PI 3.14159
#endif

typedef enum { CUBIC_BSPLINE, FOUR_POINT } CurveType;



class Stroke
{
private:
  vector<StrokePoint> control;

  vector<StrokePoint> * limit;
  vector<StrokePoint> * temp;

  bool computed;

  int numLevels;

  static GLUquadricObj * qobj;
public:

  float z;

  float radius;

  bool useTexture;
  float ufreq;
  float vfreq;
  float ustart;
  float vstart;

  float maxX, minX, maxY, minY;

  CurveType curveType;

  vector<StrokePoint> points;
  vector<StrokePoint> fanPoints;
  int pointsWidth;
  int pointsHeight;

  Stroke();
  ~Stroke();
  void add(float x, float y);
  void clear();
  static void drawLines(vector<StrokePoint> * curve);
  void forceRecompute();
  void discPoint(float x,float y,float brushRadius);
  void drawCap(const StrokePoint & p0, float dx, float dy,float texU,float texV);
  void drawThickCurve(vector<StrokePoint> * curve, float radius,bool cap=true);
  void drawControl();
  void drawLineCurve();
  void render(int width=0, int height=0);
  void subdivideCubicBSpline(vector<StrokePoint> * inputCurve, 
			     vector<StrokePoint> * outputCurve);
  void subdivideFourPoint(vector<StrokePoint> * inputCurve, 
			  vector<StrokePoint> * outputCurve);
  void subdivide(vector<StrokePoint> * inputCurve, 
		 vector<StrokePoint> * outputCurve);
  void computeLimitCurve();
  void setPoint(float x, float y);
  void setFanPoint(float x, float y);
};

