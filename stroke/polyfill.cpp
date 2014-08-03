#include <math.h>
#include <GL/glut.h>
#include <assert.h>
#include "polyfill.h"
#include "Point.h"

void ScanX(StrokePoint&l,StrokePoint & r, float y)
{
  

	int lx = /*l.x*/ceil(l.x), rx = /*r.x*/ceil(r.x);
  for(int x= lx;x < rx; x++)
  {
    glVertex2i(x,y);
	
  }

  
}

void DiffY(StrokePoint&b, StrokePoint&t,StrokePoint&m,StrokePoint&dy,int y)
{
  dy = (t-b)/(t.y-b.y);
  m = b+dy*(y-b.y);
}

int mod(int x,int y)
{
  if (x < 0)
    return mod(x+y,y);
  else
    return x%y;

  // or:  MOD(x,y) is x - y.*floor(x./y) if y ~= 0.  By convention, MOD(x,0) is x.

}


void ScanY(StrokePoint v[], int nv, int bot)
{
  // v: array of vertecies in counterclockwise order
  // nv: number of vertecies
  // bot: number of the StrokePoint with the min y coordinate

  int li = bot, ri = li;
  int y = /*v[bot].y*/ceil(v[bot].y), ly = y, ry = y;
  
  StrokePoint l, dl, r, dr;

  for(int rem = nv; rem > 0; )
    {
      // find left boundary edge of tile
      while (ly <= y && rem --  > 0)
	{
	  int i = li; 
	  li = mod(li-1,nv);  
	  assert(li >= 0);
	  ly = /*v[li].y*/ceil(v[li].y);
	  if (ly > y) 
	    DiffY(v[i],v[li],l,dl,y);
	}

      // find right boundary edge of  atile

      while (ry <= y && rem -- > 0)
	{
	  int i= ri; 
	  ri = mod(ri+1,nv); 
	  ry = /*v[ri].y*/ceil(v[ri].y);
	  if (ry> y)   
	    DiffY(v[i],v[ri],r,dr,y);
	}
      
      for(; y < ly && y < ry; y++)
	{	
	  ScanX(l,r,y); 
	  l+=dl; 
	  r+= dr;
	}
    }
}

void drawPoly(StrokePoint v[],int nv)
{
  assert(ccw(v[0],v[1],v[2]));

  int bot = 0;
  float miny = v[bot].y;

  for(int i=1;i<nv;i++)
    if (v[i].y < miny)
      {
	miny = v[i].y;
	bot = i;
      }

  //  printf("POLYGON\n");
  //  printf("v[%d] = (%f,%f)\n",bot,v[bot].x,v[bot].y);

  //  for(i=(bot+1)%nv;i!=bot;i=(i+1)%nv)
  //    printf("v[%d] = (%f,%f)\n",i,v[i].x,v[i].y);

    //  printf("det = %f\n",det2(v[1]-v[0],v[2]-v[0]));

  glPointSize(1);
  glBegin(GL_POINTS);

  ScanY(v,nv,bot);

  glEnd();
}
	
void drawTriangle(StrokePoint v1,StrokePoint v2,StrokePoint v3)
{
  if(ccw(v1,v2,v3))
    {
      //      assert(ccw(v1,v2,v3));
      //      printf("noflip\n");
      StrokePoint v[] = {v1,v2,v3};
      drawPoly(v,3);
    }
  else
    {
      //      assert(ccw(v1,v3,v2));
      //      printf("flip\n");
      StrokePoint v[] = {v1,v3,v2};
      drawPoly(v,3);
    }
}

void drawQuad(StrokePoint v1,StrokePoint v2,StrokePoint v3,StrokePoint v4)
{
  if (ccw(v1,v2,v3))
    {
      StrokePoint v[] = {v1,v2,v3,v4};
      drawPoly(v,4);
    }
  else
    {
      StrokePoint v[] = {v4,v3,v2,v1};
      drawPoly(v,4);
    }
}
