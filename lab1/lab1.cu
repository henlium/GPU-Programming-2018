#include "lab1.h"
#include <cmath>
#include <cstdio>
#include <vector>
static const unsigned W = 1280;
static const unsigned H = 720;
static const unsigned NFRAME = 500;

//Shape sizes.
//Should fit into a 2x2x2 box.
const float CUBE_LENGTH = 0.25f;
const float SPHERE_RADIUS = 0.2f;
const float TORUS_RADII[2] = {0.4, 0.2}; //inner and outer radii

//Camera.
const float CAMERA_DISTANCE = 2.0f;
const float CAMERA_ROTATION_TIMESCALE = 0.25f; //2pi = one second per revolution.
const float CAMERA_PAN_TIMESCALE[3] = {0.0f, 1.0f, 1.0f};
	
//Raymarch.
const float RAY_FUZZ_TIMESCALE = 1.0f; //distorts the ray based on time.
const float RAY_FUZZ_MIN = 0.2f, RAY_FUZZ_MAX = 1.0f; //1.0 = exact.
const int RAYMARCH_STEPS = 32; //raymarch steps. fewer = blurrier.

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

std::vector<uint8_t> RGB2YUV(float r, float g, float b) {
	std::vector<uint8_t> res = {(uint8_t)((0.299*r+0.587*g+0.114*b)*255), (uint8_t)((-0.169*r-0.331*g+b/2+128)*255), (uint8_t)((r/2-0.419*g-0.081*b+128)*255)};
	return res;
}

/// Time ///

/** Sine oscillation.
    x : Input value, in radians.
    ymin, ymax : Minimum and maximum output values.
*/
float oscillate(float x, float ymin, float ymax) {
	float range = ymax - ymin;
	float x1 = (sin(x)+1.0f)/2.0f; 
	return ymin + range*x1;
}
 
 
/// Geometry ///

float sdTorus(float p0, float p1, float p2, float r0, float r1) {
    float c = (r0 + r1) / 2.0f; //ring radius
    float a = abs(r0 - r1) / 2.0f; //thickness; tube radius
    
    float d0 = abs(c - sqrt(p0*p0+p2*p2)); // Horizontal distance from ring.
    return sqrt(d0*d0+p1*p1) - a; //Distance from ring.
}

//Signed distance to a box.
float sdBox(float p0, float p1, float p2, float b0, float b1, float b2) {
	float d[3] = {abs(p0) - b0, abs(p1) - b1, abs(p2) - b2}, dd[3];
	for(int i = 0; i < 3; i++) {
		if(d[i] < 0.0f) dd[i] = 0.0f;
		else dd[i] = d[i];
	}
	float l = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
	return min(max(d[0],max(d[1],d[2])),0.0f) + l;
}

//Sphere
float sdSphere(float p0, float p1, float p2, float radius) {
	return sqrt(p0*p0+p1*p1+p2*p2) - radius; //sphere
}


/// Marching ///

/** Wraps 3-space "into" a 1x1x1 box, and draws a shape inside it.
	This gives the repetition effect. It's not that there are infinitely many cubes, but that it's the same cube being seen infinitely many times.
	
	Returns distance from p to the shape, measured within the 1x1x1 box.
*/
float map(float p0, float p1, float p2) {
	//Take the fractional parts of the coordinates.
		//[math] This defines a quotient of R^3 onto T^3.
	//Then send [0,1) to [-1,1), so the shape is centered at 0.
		//Note: This doubles distances and lengths, because [-1,1) is twice as big as [0,1).
	double trash;
	float q[3] = {(float)(modf(p0, &trash)*2.0f-1.0f), (float)(modf(p1, &trash)*2.0f-1.0f), (float)(modf(p2, &trash)*2.0f-1.0f)};
	
	return 0.5f*sdTorus(q[0], q[1], q[2], TORUS_RADII[0], TORUS_RADII[1]);
	// return 0.5*sdSphere(q[0], q[1], q[2], SPHERE_RADIUS);
	// return 0.5f*sdBox(q[0], q[1], q[2], CUBE_LENGTH, CUBE_LENGTH, CUBE_LENGTH); //rectangle
}

/** March from `origin` in the direction `ray`.
	
	`ray` is possibly not length 1, which would have these effects:
	1. The apparent distance will be wrong, making the object brighter or darker due to distance fog. For example, if the ray is length 2, the returned distance will be half the real distance.
	2. The algorithm may overshoot or undershoot at each step, causing it to not reach the target or overshoot the target.
*/
float trace(float o0, float o1, float o2, float r0, float r1, float r2) {
	float t = 0.0f;  //Estimated distance to object.
	for (int i = 0; i < RAYMARCH_STEPS; ++i) {
		float p[3] = {o0+r0*t, o1+r1*t, o2+r2*t};
		float d = map(p[0], p[1], p[2]);
		t += d;
	}
	return t;
}

// u act as real part, v as imaginary part
void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	float iTime = (float)(impl->t) / 24.0f;
	for(int i = 0; i < H/2; i++) {
		for(int j = 0; j < W/2; j++) {
			// uint8_t r, g, b;
			std::vector<uint8_t> u, v;
			for(int k = 0; k < 2; k++) {
				for(int l = 0; l < 2; l++) {
					// initialize coordinates
					int x = 2*j+l, y = 2*i+k;
					
					// computing RGB
					float uv[2] = {(float)(x)/(float)(W)*2.0f-1.0f, (float)(y)/(float)(H)*2.0f-1.0f};

					// Scale x to aspect ratio.
					uv[0] *= (float)W/(float)H;
					
					//Camera:
					//Puts screen at distance from camera, and r points to current pixel.
					float len = sqrt(uv[0]*uv[0]+uv[1]*uv[1]+CAMERA_DISTANCE*CAMERA_DISTANCE);
					float r[3] = {uv[0]/len, uv[1]/len, CAMERA_DISTANCE/len};
					//Rotate camera around y-axis over time.
					float the = iTime * CAMERA_ROTATION_TIMESCALE;
					float rx = r[0]*cos(the)+r[2]*sin(the), rz = -r[0]*sin(the)+r[2]*cos(the);
					r[0] = rx;
					r[2] = rz;
					//Pan camera over time.
					float o[3] = {iTime*CAMERA_PAN_TIMESCALE[0], iTime*CAMERA_PAN_TIMESCALE[1], iTime*CAMERA_PAN_TIMESCALE[2]};
					
					//Distortion factor for the ray.
					float st = oscillate(iTime*RAY_FUZZ_TIMESCALE, RAY_FUZZ_MIN, RAY_FUZZ_MAX);
					
					//Distance to a visible object from this ray.
					float tt = trace(o[0], o[1], o[2], r[0]*st, r[1]*st, r[2]*st);
					
					//Distance fog.
					float fc = 1.0f / (1.0f + tt * tt * 0.1f) * 2.0f;
					
					//Tint based on distortion factor.
					float tint[3] = {st+0.5f,st,st+0.3f};
					
					float C = fc * tint[0];
					float M = fc * tint[1];
					float Y = fc * tint[2];
					if(C > 1) C = 1;
					if(M > 1) M = 1;
					if(Y > 1) Y = 1;

					// RGB post-processing
					std::vector<uint8_t> temp;
					temp = RGB2YUV( C, M, Y);
					cudaMemset(yuv+x+y*W, temp[0], 1);
					u.push_back(temp[1]);
					v.push_back(temp[2]);
				}
			}
			uint8_t u_channel = (u[0]+u[1]+u[2]+u[3])/4;
			cudaMemset(yuv+W*H+j+i*W/2, u_channel, 1);
			uint8_t v_channel = (v[0]+v[1]+v[2]+v[3])/4;
			cudaMemset(yuv+5*W*H/4+j+i*W/2, v_channel, 1);
		}
	}
	++(impl->t);
}