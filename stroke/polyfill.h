#include <stdio.h>
#include <float.h>
#include "Point.h"

void ScanY(StrokePoint v[], int nv, int bot);

void drawTriangle(StrokePoint v1,StrokePoint v2,StrokePoint v3);
void drawQuad(StrokePoint v1,StrokePoint v2,StrokePoint v3,StrokePoint v4);
void drawPoly(StrokePoint v[]);

inline float det2(StrokePoint v1,StrokePoint v2)
{
  //    printf("v1' = (%f,%f)\n",v1.x,v1.y);
  //    printf("v2' = (%f,%f)\n",v2.x,v2.y);

  return v1.x*v2.y - v1.y*v2.x;
}

inline bool ccw(StrokePoint v1,StrokePoint v2,StrokePoint v3)
{
  /*  printf("v1 = (%f,%f)\n",v1.x,v1.y);
  printf("v2 = (%f,%f)\n",v2.x,v2.y);
  printf("v3 = (%f,%f)\n",v3.x,v3.y);
  printf("%f\n",det2(v1-v2,v2-v3));
  */
  return det2(v2-v1,v3-v1) >= 0;
}

