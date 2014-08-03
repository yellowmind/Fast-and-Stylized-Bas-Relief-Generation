#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "Stroke.h"

//#define SCAN_CONVERT

/*inline float round( float d )
{
	return floor( d + 0.5 );
}*/

#ifdef SCAN_CONVERT
#include "polyfill.h"
#endif

#ifndef SCAN_CONVERT
GLUquadricObj * Stroke::qobj = NULL;
#endif

Stroke::Stroke()
{
  limit = temp = NULL;
  computed = GL_FALSE;
  curveType = CUBIC_BSPLINE;

  useTexture = true;

  numLevels = 3;

  z = 0;

#ifndef SCAN_CONVERT
  if (qobj == NULL)
    {
      qobj = gluNewQuadric();
    }
#endif

  ufreq = .005;
  vfreq = .1;
  ustart = .1;
  srand (time(NULL));
  vstart = rand()/*drand48()*/;

  pointsWidth = 0;
  pointsHeight = 0;

  maxX = -2048;
  minX = 2048;
  maxY =- 2048;
  minY=2048;
}


Stroke::~Stroke()
{
  if (limit != NULL)
    delete limit;

  if (temp != NULL)
    delete temp;
}

void Stroke::add(float x,float y)
{
  if (control.empty() || (control.back().x != x && control.back().y != y))
    {
      control.push_back(StrokePoint(x,y));
      computed = GL_FALSE;
    }
}

void Stroke::clear()
{
  computed = GL_FALSE;
  control.erase(control.begin(),control.end());
  limit->erase(control.begin(),control.end());
  ustart = .1;
  srand (time(NULL));
  vstart = rand()/*drand48()*/;
}

void Stroke::drawLines(vector<StrokePoint> *curve)
{
  glDisable(GL_TEXTURE_2D);
  glBegin(GL_LINE_STRIP);
  for(int i=0;i<curve->size();i++)
    {
      StrokePoint & p = (*curve)[i];
      glVertex2i(p.x,p.y);
    }
  glEnd();
}

void Stroke::forceRecompute()
{
  computed = false;

}

void Stroke::discPoint(float x,float y,float brushRadius)
{
#ifdef SCAN_CONVERT
  // really ought to implement scan-conversion of the circle here
  
  glColor3ub(255,0,0);
  drawCap(StrokePoint(x,y),1,0,0,0);
  drawCap(StrokePoint(x,y),-1,0,0,0);
#else
  glPushMatrix();
  glTranslatef(x,y,z);
  gluDisk(qobj,0,brushRadius,NUM_SLICES,1);
  glPopMatrix();
#endif
}

void Stroke::drawCap(const StrokePoint & p0, float dx, float dy,
		     float texU,float texV)
{
#ifdef SCAN_CONVERT
  glDisable(GL_TEXTURE_2D);

  float theta = atan2(-dy,-dx);

  StrokePoint p1(p0.x - radius * dx, p0.y - radius*dy);

  for(int i=1;i<NUM_SLICES-1;i++)
    {
      float dx1 = cos(theta + i * M_PI / NUM_SLICES);
      float dy1 = sin(theta + i * M_PI / NUM_SLICES);

      StrokePoint p2(p0.x + radius * dx1, p0.y + radius*dy1);

      drawTriangle(p0,p1,p2);

      p1 = p2;
    }

  drawTriangle(p0,p1,StrokePoint(p0.x+radius*dx,p0.y+radius*dy));
#else
  float theta = atan2(-dy,-dx);

  glPushMatrix();
  glTranslatef(p0.x,p0.y,z);

  glBegin(GL_TRIANGLE_FAN);
  //        glVertex3f(p0->x, p0->y,z);
  glTexCoord2f(texU,texV);
  glVertex3i(0,0,0);
  setFanPoint(p0.x, p0.y);

  glTexCoord2f(texU+ufreq*radius * dx,texV+vfreq*dy);
  glVertex3f(-radius*dx,-radius*dy,0);
  setFanPoint(p0.x-radius*dx, p0.y-radius*dy);

  for(int i=1;i<=NUM_SLICES-1;i++)
    {
      float dx1 = cos(theta + i * M_PI / NUM_SLICES);
      float dy1 = sin(theta + i * M_PI / NUM_SLICES);

      // these 
      glTexCoord2f(texU-ufreq*radius * dx1,texV-vfreq*dy1);

      glVertex3i(radius * dx1, radius * dy1, 0);
	  setFanPoint(p0.x+radius * dx1, p0.y+radius * dy1);
      //glVertex3f(p0->x + radius * dx1, p0->y + radius * dy1, z);
    }

  glTexCoord2f(texU-ufreq*radius * dx,texV-vfreq*dy);
  glVertex3f(radius*dx,radius*dy,0);
  setFanPoint(p0.x+radius*dx, p0.y+radius*dy);

  glEnd();

  glPopMatrix();
#endif
}

void Stroke::drawThickCurve(vector<StrokePoint> * curve, float radius,bool cap)
{

  if (useTexture)
    glEnable(GL_TEXTURE_2D);
  else
    glDisable(GL_TEXTURE_2D);

  int i=0;
  float dx,dy,mag;
  StrokePoint p0;
  StrokePoint p1;
  StrokePoint p2;

  if (curve->empty())
    return;

  p0 = (*curve)[0];

  if (curve->size() == 1)
    {
      if (cap)
	  {
			discPoint(p0.x,p0.y,radius);
			setPoint(p0.x, p0.y);
	  }

      return;
    }

  p1 = (*curve)[1];

  dx = p1.y - p0.y;
  dy = p0.x - p1.x;

  mag = sqrt(dx*dx + dy*dy);

  dx /= mag;
  dy /= mag;

  float textureU = ustart;
  float textureV = vstart;

  glColor3ub(255,255,0);

  if (cap)
    drawCap(p0, dx, dy,textureU,textureV);

#ifdef SCAN_CONVERT
  glDisable(GL_TEXTURE_2D);

  StrokePoint v0(p0.x + radius * dx, p0.y + radius * dy);
  StrokePoint v1(p0.x - radius * dx, p0.y - radius * dy);

#else
  glBegin(GL_TRIANGLE_STRIP);

  

  glTexCoord2f(textureU,textureV + vfreq);

  if (mag)
  {
	  glVertex3f(p0.x + radius * dx, p0.y + radius * dy,z);
	  //StrokePoint p = StrokePoint(p0.x + radius * dx, p0.y + radius * dy);
	  setPoint(p0.x + radius * dx, p0.y + radius * dy);
	  //points.push_back( StrokePoint(p0.x + radius * dx, p0.y + radius * dy) ); 
	  /*if (p0.x + radius * dx > maxX)
		  maxX = p0.x + radius * dx;
	  if (p0.y + radius * dy > maxY)
		  maxY = p0.y + radius * dy;

	   if (p0.x + radius * dx < minX)
		  minX = p0.x + radius * dx;
	  if (p0.y + radius * dy < minY)
		  minY = p0.y + radius * dy;*/

	  glTexCoord2f(textureU,textureV - vfreq);
	  glVertex3f(p0.x - radius * dx, p0.y - radius * dy,z);
	  setPoint( p0.x - radius * dx, p0.y - radius * dy);
	  //points.push_back(StrokePoint(p0.x - radius * dx, p0.y - radius * dy) );
	  /*if (p0.x - radius * dx > maxX)
		  maxX = p0.x - radius * dx;
	  if (p0.y - radius * dy > maxY)
		  maxY = p0.y - radius * dy;

	  if (p0.x - radius * dx < minX)
		  minX = p0.x - radius * dx;
	  if (p0.y - radius * dy < minY)
		  minY = p0.y - radius * dy;*/
  }

#endif

  textureU += ufreq;

#ifdef SCAN_CONVERT
  // draw the patch in between with extra subdivision to match the caps,
  // to prevent holes from appearing in the stroke
  if (curve->size() >= 2)
    {
      float dist = sqrt((p1.x-p0.x)*(p1.x-p0.x)+(p1.y-p0.y)*(p1.y-p0.y));
      textureU += ufreq * dist;

      v0 = StrokePoint(p0.x + radius * dx, p0.y + radius * dy);
      v1 = StrokePoint(p0.x - radius * dx, p0.y - radius * dy);
      StrokePoint v2(p1.x + radius * dx, p1.y + radius * dy);
      StrokePoint v3(p1.x - radius * dx, p1.y - radius * dy);
	  
      glColor3ub(255,0,0);
      //      drawTriangle(v2,v3,v0);
      drawTriangle(v2,p1,v0);
      glColor3ub(255,255,0);
      drawTriangle(p1,v3,v0);

      glColor3ub(255,0,0);
      //      drawTriangle(v0,v1,v3);
      drawTriangle(v0,p0,v3);
      glColor3ub(0,255,255);
      drawTriangle(p0,v1,v3);

      glColor3ub(0,0,255);

      if (curve->size() == 2)
	{
	  drawCap(p1,-dx,-dy,textureU,textureV);
	  return;
	}


      v0 = v2;
      v1 = v3;
    }
#endif

#ifdef SCAN_CONVERT
  for(i=2;i<curve->size()-1;i++)
#else
  for(i=1;i<curve->size()-1;i++)
#endif
    {
      p0 = (*curve)[i-1];
      p1 = (*curve)[i];
      p2 = (*curve)[i+1];

      dx = p2.y - p0.y;
      dy = p0.x - p2.x;

      mag = sqrt(dx*dx + dy*dy);

      dx /= mag;
      dy /= mag;

      float dist = sqrt(/*(float)*/(p1.x-p0.x)*(p1.x-p0.x)+(p1.y-p0.y)*(p1.y-p0.y));
      textureU += ufreq * dist;

#ifdef SCAN_CONVERT
      StrokePoint v2(p1.x + radius * dx, p1.y + radius * dy);
      StrokePoint v3(p1.x - radius * dx, p1.y - radius * dy);

      glColor3ub(255,0,0);
      drawTriangle(v2,v3,v0);

      glColor3ub(255,0,0);
      drawTriangle(v0,v1,v3);
	  
      v0 = v2;
      v1 = v3;
#else
      if (mag)
	  {
		  glTexCoord2f(textureU,textureV + vfreq);
		  glVertex3f(p1.x + radius * dx, p1.y + radius * dy,z);

		  setPoint(p1.x + radius * dx, p1.y + radius * dy);
		  //points.push_back(StrokePoint(p1.x + radius * dx, p1.y + radius * dy) );
		  
		  glTexCoord2f(textureU,textureV - vfreq);
		  glVertex3f(p1.x - radius * dx, p1.y - radius * dy,z);
		  setPoint(p1.x - radius * dx, p1.y - radius * dy);
		  //points.push_back(StrokePoint(p1.x - radius * dx, p1.y - radius * dy) );
	  }
#endif

    }

  p0 = (*curve)[curve->size()-2];
  p1 = (*curve)[curve->size()-1];
    
  dx = p1.y - p0.y;
  dy = p0.x - p1.x;
    
  mag = sqrt(dx*dx + dy*dy);

  dx /= mag;
  dy /= mag;
      
  textureU += ufreq *mag;
  
#ifdef SCAN_CONVERT
  StrokePoint v2(p1.x + radius * dx, p1.y + radius * dy);
  StrokePoint v3(p1.x - radius * dx, p1.y - radius * dy);
      
  glColor3ub(255,0,0);
  //  drawTriangle(v2,v3,v0);
  drawTriangle(v2,p1,v0);
  glColor3ub(255,255,0);
  drawTriangle(p1,v3,v0);

  glColor3ub(0,255,0);
    drawTriangle(v0,v3,v1);
    //  drawTriangle(v0,p0,v3);
    //  glColor3ub(0,255,255);
    //  drawTriangle(p0,v1,v3);

#else

  if (mag)
  {
	  glTexCoord2f(textureU,textureV + vfreq);
	  glVertex3f(p1.x + radius * dx, p1.y + radius * dy,z);
	  setPoint(p1.x + radius * dx, p1.y + radius * dy);
	  //points.push_back(StrokePoint(p1.x + radius * dx, p1.y + radius * dy) );
	  glTexCoord2f(textureU,textureV - vfreq);
	  glVertex3f(p1.x - radius * dx, p1.y - radius * dy,z);
	  setPoint(p1.x - radius * dx, p1.y - radius * dy);
	  //points.push_back(StrokePoint(p1.x - radius * dx, p1.y - radius * dy) );
  }
      
  glEnd();
#endif



  glColor3ub(0,0,255);

  if (cap)
    {
      drawCap(p1, -dx, -dy,textureU,textureV);
    }

}

void Stroke::drawControl()
{
  glColor3ub(255,0,0);
  drawLines(&control);
}

void Stroke::drawLineCurve()
{
  drawLines(limit);
}

void Stroke::render(int width, int height)
{
  if (!computed)
    {
		if (width * height)
		{
			/*points.resize(width * height);*/
			pointsWidth = width;
			pointsHeight = height;
		}
      //	printf("Computing curve \n");
      computeLimitCurve();
      computed = GL_TRUE;
    }

  /*
    printf("Control = ");
    control.print();
    printf("\nLimit = ");
    limit->print();
    printf("\n");
  */

  //    drawLines(limit);
  glPushMatrix();
	glTranslatef(0, 512, 0);
	glScalef(3.0, -1, 1.0);
	drawThickCurve(limit,radius);
  glPopMatrix();
}

void Stroke::subdivideCubicBSpline(vector<StrokePoint> * inputCurve, 
				   vector<StrokePoint> * outputCurve)
{
  outputCurve->erase(outputCurve->begin(),outputCurve->end());

  //    printf("ic-count=%d\n",inputCurve->count);

  if (inputCurve->size() < 1)
    return;

  StrokePoint pi0;
  StrokePoint pi1;
  StrokePoint pi2;

  pi0 = (*inputCurve)[0];

  outputCurve->push_back(StrokePoint(pi0.x,pi0.y));

  if (inputCurve->size() == 1)
    return;

  if (inputCurve->size() == 2)
    {
      pi1 = (*inputCurve)[1];

      outputCurve->push_back(StrokePoint(pi1.x,pi1.y));

      return;
    }

  pi1 = (*inputCurve)[1];

  outputCurve->push_back(StrokePoint((pi0.x + pi1.x)/2,(pi0.y + pi1.y)/2));

  for(int i=1;i<inputCurve->size()-1;i++)
    {
      pi0 = (*inputCurve)[i-1];
      pi1 = (*inputCurve)[i];
      pi2 = (*inputCurve)[i+1];

      outputCurve->push_back(StrokePoint( (pi0.x + 6*pi1.x + pi2.x)/8,
				    (pi0.y + 6*pi1.y + pi2.y)/8));

      outputCurve->push_back(StrokePoint( (pi1.x + pi2.x)/2,(pi1.y + pi2.y)/2));
    }
	
  outputCurve->push_back(StrokePoint(pi2.x,pi2.y));
}

void Stroke::subdivideFourPoint(vector<StrokePoint> * inputCurve, 
				vector<StrokePoint> * outputCurve)
{
  outputCurve->erase(outputCurve->begin(),outputCurve->end());

  if (inputCurve->size() < 1)
    return;

  StrokePoint pi0;
  StrokePoint pi1;
  StrokePoint pi2;
  StrokePoint pi3;

  if (inputCurve->size() == 1)
    {
      pi0 = (*inputCurve)[0];
      outputCurve->push_back(StrokePoint(pi0.x,pi0.y));

      return;
    }

  if (inputCurve->size() == 2)
    {
      pi0 = (*inputCurve)[0];
      pi1 = (*inputCurve)[1];
	
      outputCurve->push_back(StrokePoint(pi0.x,pi0.y));
      outputCurve->push_back(StrokePoint((pi0.x+pi1.x)/2,(pi0.y+pi1.y)/2));
      outputCurve->push_back(StrokePoint(pi1.x,pi1.y));
	
      return;
    }

  pi0 = (*inputCurve)[0];
  pi1 = (*inputCurve)[1];

  StrokePoint piminus1(2*pi0.x - pi1.x,2*pi0.y - pi1.y);

  pi0 = (*inputCurve)[inputCurve->size()-1];
  pi1 = (*inputCurve)[inputCurve->size()-2];

  StrokePoint piplus1(2*pi0.x - pi1.x,2*pi0.y - pi1.y);
    
  for(int i=0;i<inputCurve->size()-1;i++)
    {
      pi0 = (i==0 ? piminus1 : (*inputCurve)[i-1]);
      pi1 = (*inputCurve)[i];
      pi2 = (*inputCurve)[i+1];
      pi3 = (i==inputCurve->size()-2? piplus1:(*inputCurve)[i+2]);

      outputCurve->push_back(StrokePoint( pi1.x, pi1.y));

      outputCurve->push_back(StrokePoint( (-pi0.x + 9*pi1.x + 9*pi2.x - pi3.x)/16,
				    (-pi0.y + 9*pi1.y + 9*pi2.y - pi3.y)/16));
    }

  pi0 = inputCurve->back();

  outputCurve->push_back(StrokePoint(pi0.x,pi0.y));
}

void Stroke::subdivide(vector<StrokePoint> * inputCurve, vector<StrokePoint> * outputCurve)
{
  switch(curveType)
    {
    case CUBIC_BSPLINE:
      subdivideCubicBSpline(inputCurve,outputCurve);
      break;

    case FOUR_POINT:
      subdivideFourPoint(inputCurve,outputCurve);
      break;

    default:
      printf( "Illegal subdivision scheme selected\n");
      exit(-1);
    }
}

void Stroke::computeLimitCurve()
{
  //    printf("Computing limit curve.  Input length = %d\n",control.count);

  if (limit == NULL)
    limit = new vector<StrokePoint>();

  if (temp == NULL)
    temp = new vector<StrokePoint>();

  subdivide(&control,limit);

  //    limit->print();
  //    printf(" count = %d\n",limit->size());

  for(int i=0;i<numLevels/2;i++)
    {
      subdivide(limit,temp);
      subdivide(temp,limit);
    }
}

void Stroke::setPoint(float x, float y)
{
	if (x >= 0 && x < pointsWidth && y >= 0 && y <pointsHeight)
	{
		points.push_back( StrokePoint(x , y) );
	}
	else
	{
		int x0 = x;
		int y0 = y;
	}
	/*int address = x + y*pointsWidth;

	if ( address < points.size())
	{
		points[address] = 1;
	}*/
}

void Stroke::setFanPoint(float x, float y)
{
	if (x >= 0 && x < pointsWidth && y >= 0 && y <pointsHeight)
	{
		fanPoints.push_back( StrokePoint(x , y) );
	}
}