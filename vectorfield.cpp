#include "vectorfield.h"
#include <fstream>

//
bool VectorField::Readvectorfield_binary(const char *filename)
{
	using namespace std;

	FILE *stream = fopen( filename, "rb" );
	if ( !stream ) {
		//std::cout<<"ERROR!! Can't read "<<featurefile<<'\n';
		return false;
	}

	int vf_w, vf_h;
	int sint = sizeof(vf_w);
	
	fread((void *)&(vf_w), sint, 1, stream);
	fread((void *)&(vf_h), sint, 1, stream);
	//vf_w-=2; vf_h-=2;
	//vectorfield.resize(vf_w*vf_h);

	int sfloat = sizeof(_data[0].dx);
	float *data = new float[vf_w*vf_h*2];
	fread((void *)(data), sfloat, vf_w*vf_h*2, stream);

	//float *data = new float[(vf_w+2)*(vf_h+2)*2];
	//fread((void *)(data), sfloat, (vf_w+2)*(vf_h+2)*2, stream);

	int w = _w;
	int h = _h;
	if(w==vf_w && h==vf_h) {
		for(int i=0; i<vf_w*vf_h; i++) {
			_data[i].dx = data[i*2];
			_data[i].dy = data[i*2+1];
		}
	} else {
		//bilinear interpolation
		float scale_w = (float)(vf_w-1)/(w-1);
		float scale_h = (float)(vf_h-1)/(h-1);
		float temp;

		for(int j=0; j<h; j++) {
			for(int i=0; i<w; i++) {
				float x = i *scale_w;
				float y = j *scale_h;				
				int si = int(x);
				int sj = int(y);
				float u_ratio = x - si;
				float v_ratio = y - sj;
				float u_opposite = 1 - u_ratio;
				float v_opposite = 1 - v_ratio;

				if(si+1 < vf_w && sj+1 < vf_h) { 
					int index1 = si+sj*vf_w;
					int index2 = (si+1)+sj*vf_w;
					int index3 = si+(sj+1)*vf_w;
					int index4 = (si+1)+(sj+1)*vf_w;
					
					temp = (data[index1*2] * u_opposite  + data[index2*2] * u_ratio) * v_opposite + 
							(data[index3*2]* u_opposite  + data[index4*2] * u_ratio) * v_ratio;
					_data[i+j*w].dx = temp;

					temp = (data[index1*2+1] * u_opposite + data[index2*2+1] * u_ratio) * v_opposite + 
							(data[index3*2+1]* u_opposite + data[index4*2+1] * u_ratio) * v_ratio;
					_data[i+j*w].dy = temp;
				} else {
					_data[i+j*w].dx = data[(si+sj*vf_w)*2];
					_data[i+j*w].dy = data[(si+sj*vf_w)*2+1];
				}	
			}
		}

	}

	delete [] data;

	fclose( stream );
	
	return true;
}


bool VectorField::Savevectorfield_binary(const char *filename)
{
	using namespace std;
	//fwrite((void *)buf, padding, 1, f);

	FILE *stream = fopen( filename, "wb" );
	if ( !stream ) {
		//std::cout<<"ERROR!! Can't read "<<featurefile<<'\n';
		return false;
	}

	int vf_w = _w, vf_h = _h;
	int sint = sizeof(vf_w);
	
	fwrite((void *)&(vf_w), sint, 1, stream);
	fwrite((void *)&(vf_h), sint, 1, stream);

	int sfloat = sizeof(_data[0].dx);
	float *data = new float[vf_w*vf_h*2];
	for(int i=0; i<vf_w*vf_h; i++) {
		//fwrite((void *)&(vectorfield[i].dx), sfloat, 1, stream);
		//fwrite((void *)&(vectorfield[i].dy), sfloat, 1, stream);
		data[2*i] = _data[i].dx;
		data[2*i+1] = _data[i].dy;
	}

	fwrite((void *)(data), sfloat, vf_w*vf_h*2, stream);
	delete [] data;

	fclose( stream );

	return true;
}