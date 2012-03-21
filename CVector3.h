// ************************************************************* //
// ****** CVector3.h				 		   Orzech (c) 2005 * //
// ************************************************************* //

#ifndef _CVECTOR3_H
#define _CVECTOR3_H

#include <GL\gl.h>
#include <math.h>

class CVector3
{
	public:

	float	n[3];

	// Default constructor
	CVector3()	{}

	// Constructor with parameters
	CVector3(float x, float y, float z) 
	{
		n[0] = x; n[1] = y; n[2] = z;
	}

	// Destructor
	~CVector3()	{}

	// Assigns (x,y) coordinates of a vector
	void		Set(float x, float y, float z) { n[0] = x; n[1] = y; n[2] = z; }

	// Vector assignment
	CVector3&	operator=(const CVector3 &vec)
	{
		n[0] = vec.n[0];
		n[1] = vec.n[1];
		n[2] = vec.n[2];
		return *this;
	}

	// Vector addition
	CVector3	operator+(const CVector3 &vec)
	{
		return CVector3(n[0] + vec.n[0], n[1] + vec.n[1], n[2] + vec.n[2]);
	}

	// Vector subtraction
	CVector3	operator-(const CVector3 &vec)
	{
		return CVector3(n[0] - vec.n[0], n[1] - vec.n[1], n[2] - vec.n[2]);
	}

	// Vector negation
	CVector3 operator- ()
	{
		return CVector3(-n[0], -n[1], -n[2]);
	}

	// Vector increment
	const CVector3&	operator+=(const CVector3 &vec)
	{
		n[0] += vec.n[0];
		n[1] += vec.n[1];
		n[2] += vec.n[2];
		return *this;
	}

	// Vector decrement
	const CVector3&	operator-=(const CVector3 &vec)
	{
		n[0] -= vec.n[0];
		n[1] -= vec.n[1];
		n[2] -= vec.n[2];
		return *this;
	}

	// Vector multiplication by scalar
	CVector3	operator*(float k)
	{
		return CVector3(n[0] * k, n[1] * k, n[2] * k);
	}

	// Vector division by scalar
	CVector3	operator/(float k)
	{
		float d = 1.0/k;
		return CVector3(n[0] * d, n[1] * d, n[2] * d);
	}

	// Vector dot product
	float		DotProduct(const CVector3 &vec)
	{
		return (n[0] * vec.n[0] + n[1] * vec.n[1], n[2] * vec.n[2]);
	}
	
	// Vector cross product
	CVector3&	CrossProduct(const CVector3 &vec)
	{
		return CVector3( (n[1] * vec.n[2]) - (n[2] * vec.n[1]),
						 (n[2] * vec.n[0]) - (n[0] * vec.n[2]),
						 (n[0] * vec.n[1]) - (n[1] * vec.n[0]) );
	}

	// Returns vector's magnitude
	float		Magnitude()
	{
		return sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
	}

	// Normalizes a vector
	CVector3	Normalize()
	{
		float	d = Magnitude();
		return CVector3(n[0]/d, n[1]/d, n[2]/d);
	}

	// Displays a vector
	void	DrawGL(const CVector3 &pos)
	{	
		
		glPushMatrix();
		glTranslatef(pos.n[0], pos.n[1], pos.n[2]);

		glBegin(GL_LINES);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(n[0], n[1], n[2]);
		glEnd();

		glTranslatef(n[0], n[1], n[2]);

		glRotatef(160, 0.0, 0.0, 1.0);
		glBegin(GL_LINES);		
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(0.25 * n[0], 0.25 * n[1], 0.25 * n[2]);
		glEnd();

		glRotatef(-320, 0, 0, 1);
		glBegin(GL_LINES);		
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(0.25 * n[0], 0.25 * n[1], 0.25 * n[2]);
		glEnd();


		glPopMatrix();
	}

};

#endif
