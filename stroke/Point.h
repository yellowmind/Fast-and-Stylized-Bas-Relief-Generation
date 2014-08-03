#ifndef __POINT_HH__
#define __POINT_HH__

struct StrokePoint
{
  float x,y;

  StrokePoint() { };
  StrokePoint(float x,float y) { this->x = x; this->y = y; }
  StrokePoint operator+(const StrokePoint & v) const { return StrokePoint(x+v.x,y+v.y); }
  StrokePoint operator-(const StrokePoint & v) const { return StrokePoint(x-v.x,y-v.y); }
  StrokePoint operator*(float v) const { return StrokePoint(x*v,y*v); }
  StrokePoint operator/(float v) const { return StrokePoint(x/v,y/v); }

  StrokePoint & operator=(const StrokePoint & v) { x = v.x;y=v.y; return *this; }
  StrokePoint & operator+=(const StrokePoint & v) { x += v.x;y+=v.y; return *this; }
};

#endif
