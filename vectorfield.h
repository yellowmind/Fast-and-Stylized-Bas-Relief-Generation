#ifndef _MY_VECTORFIELD_H_
#define _MY_VECTORFIELD_H_

#include <cmath>
#include <vector>

class Direction2D {
public:	
	float dx, dy;
	
	Direction2D(float idx = 0, float idy = 1) //:dx=idx, dy=idy
	{
		dx=idx;
		dy=idy;
	}

    inline Direction2D operator+(const Direction2D &p) const
    {
        return Direction2D(dx+p.dx, dy+p.dy);
    }

    inline Direction2D operator*(const float c) const
    {
        return Direction2D(dx*c, dy*c);
    }

	float length() {
		return std::sqrt(dx*dx+dy*dy);
	}

	inline void Normalize()
	{
		float d = sqrt(dx*dx+dy*dy);
		if (d == 0.0f) {
			dx = dy = 0.0f;
			return;
		}

		d = 1.0f / d;
		dx *= d; dy *= d;
	}

	void Rotate(float theta) {
		float tx=dx, ty=dy;
		float angle = theta * 3.14159265f / 180;

		dx = cos(angle)*tx - sin(angle)*ty;
		dy = sin(angle)*tx + cos(angle)*ty;
	}
};


class VectorField {
public:
	int _w;
	int _h;
	std::vector<Direction2D> _data;

	VectorField(int w, int h) {
		_w = w;
		_h = h;
		_data.resize(_w*_h);
	}

	Direction2D get(int x, int y) {
		int index = x + y*_w;
		return _data[index];
	}

	bool Readvectorfield_binary(const char *filename);

	bool Savevectorfield_binary(const char *filename);
};



#endif // _MY_VECTORFIELD_H_
