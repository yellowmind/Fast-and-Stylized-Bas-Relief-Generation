#include "StreamLine.h"

void StreamLine::GenStreamLine(int i, int j){
	origin = Center(i, j);
	//getVF(b);
	fwd[0] = RK(origin,Ht, 0);
	if(fwd[0].coord[0] == origin.coord[0] && fwd[0].coord[1] == origin.coord[1]){
		for(int k = 0; k<L; k++){
			fwd[k].coord[0] = -1;
			fwd[k].coord[1] = -1;
		}
	}else{
		for(int k=1; k<L; k++){
			fwd[k] = RK(fwd[k-1], Ht, 0);
			if(fwd[k].coord[0] == fwd[k-1].coord[0] && fwd[k].coord[1] == fwd[k-1].coord[1]){
				for(int i = k; k<L; k++){
					fwd[k].coord[0]=-1;
					fwd[k].coord[1]=-1;
				}
				break;
			}
		}
	}

    bwd[0] = RK(origin,-Ht, 0);
	if(bwd[0].coord[0] == origin.coord[0] && bwd[0].coord[1] == origin.coord[1]){
		for(int k = 0; k<L; k++){
			bwd[k].coord[0] = -1;
			bwd[k].coord[1] = -1;
		}
	}else{
		for (int k=1; k<L; k++){
			bwd[k] = RK(bwd[k-1], -Ht, 0);
			if(bwd[k].coord[0] == bwd[k-1].coord[0] && bwd[k].coord[1] == bwd[k-1].coord[1]){
				for(int i = k; k<L; k++){
					bwd[k].coord[0]=-1;
					bwd[k].coord[1]=-1;
				}
				break;
			}
		}
	}
}

void StreamLine::Draw(){           
    glMatrixMode(GL_PROJECTION);
	glPushMatrix();
    glLoadIdentity();
	glOrtho( 0, vectorfield->_w, 0, vectorfield->_h, 1.0, -1.0 );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity( );
	glLineWidth(2);
	glBegin(GL_LINES);
		glColor3f(0,0,1);
		glVertex2f(origin.coord[0], origin.coord[1]);
		glVertex2f(fwd[0].coord[0], fwd[0].coord[1]);
	glEnd();
	glLineWidth(2);
	glBegin(GL_LINES);
		glColor3f(0,0,1);
		glVertex2f(origin.coord[0], origin.coord[1]);
		glVertex2f(bwd[0].coord[0], bwd[0].coord[1]);
	glEnd();
	for(int i = 1; i < L; i++){
		if(fwd[i].coord[0] >= 0){
			glLineWidth(2);
			glBegin(GL_LINES);
				glColor3f(0,0,1);
				glVertex2f(fwd[i].coord[0], fwd[i].coord[1]);
				glVertex2f(fwd[i-1].coord[0], fwd[i-1].coord[1]);
			glEnd();
		}else break;
	}
	for(int i = 1; i < L; i++){
		if(bwd[i].coord[0] >= 0){
			glLineWidth(2);
			glBegin(GL_LINES);
				glColor3f(0,0,1);
				glVertex2f(bwd[i].coord[0], bwd[i].coord[1]);
				glVertex2f(bwd[i-1].coord[0], bwd[i-1].coord[1]);
			glEnd();	
		}else break;
	}

	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity();
}

Point transferINT(Point pt){
	Point point;
	point.coord[0] = int(point.coord[0]);
	point.coord[1] = int(point.coord[1]);
	return point;
}
Point Center(int i, int j){
	Point point;
	point.coord[0] = i + 0.5;
	point.coord[1] = j + 0.5;
	return point;
}

Vector getVF(Point pt, int d){
	Vector vec;
	Direction2D vf;
	if(pt.coord[0]>=0 && pt.coord[1]>=0 && pt.coord[0]<vectorfield->_w && pt.coord[1]<vectorfield->_h){
		//if(!t1){
			vf =  vectorfield->get(pt.coord[0], pt.coord[1]);
		//}else  vf =  vectorfield2->get(pt.coord[0], pt.coord[1]);
		if(d==0){
			vec.coord[0] = vf.dx;
			vec.coord[1] = vf.dy;
		}else if(d==1){
			vec.coord[0] = vf.dy;
			vec.coord[1] = -vf.dx;
		}else if(d==2){
			vec.coord[0] = -vf.dy;
			vec.coord[1] = vf.dx;
		}
	}else vec = NULL;
	return vec;
}

Point RK(Point pt, double h, int d){

	Vector v = getVF(pt, d);
	Vector k1, k2, k3, k4;

	k1 = v * h;
    v = (getVF(pt + k1*.5, d));
    if (!v.iszero())
        v = v.unit();
    //v.Print();
    k2 = v * h;
    v = (getVF(pt + k2*.5, d));
    if (!v.iszero())
        v = v.unit();
    //v.Print();
    k3 = v * h;
    v = (getVF(pt + k3, d));
    if (!v.iszero())
        v = v.unit();
    //v.Print();
    k4 = v * h;
    pt += k1/6 + k2/3 + k3/3 + k4/6;
	return pt;
}
