#include "vectorfield.h"
#include "point.h"
#include "vector.h"
#include <GL/glut.h>

//#define L 40
#define Ht 0.25

extern VectorField *vectorfield;
extern VectorField *vectorfield2;

class StreamLine{

public:
    void GenStreamLine(int, int, int);
	void GenStreamLine2(int, int, int);
	//void Draw(void);
    inline Point &operator[](int m){
        if (m == 0)
            return origin;
        else if (m>0)
            return fwd[m-1];
        else
            return bwd[-m-1];
    }

	
	Point fwd[100];
    Point bwd[100];
    Point origin;
};

Point transferINT(Point);

Point Center(int, int);

Vector getVF(Point, int);
Vector getVF2(Point, int);

Point RK(Point pt, double h, int d);
Point RK2(Point pt, double h, int d);
