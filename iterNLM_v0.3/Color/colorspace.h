#ifndef COLORSPACE_H
#define COLORSPACE_H

#include <iostream>

namespace color
{
	void RGB2HSV(float *R, float *G, float *B, int shape[3])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float r = R[idx]/255.f;
			float g = G[idx]/255.f;
			float b = B[idx]/255.f;

			float Cmax = std::max(std::max(r,g), b);
			float Cmin = std::min(std::min(r,g), b);
			float delta = Cmax-Cmin;

			float h = 0.0f;
			float s = 0.0f;
			float v = 0.0f;

			if (Cmax == Cmin) h = 0.0f;
			else if (Cmax == r) h = 60.f*(  (g-b)/delta);
			else if (Cmax == g) h = 60.f*(2+(b-r)/delta);
			else if (Cmax == b) h = 60.f*(4+(r-g)/delta);

			if(h < 0.0f) h+= 360.f;

			if (Cmax != 0.0f) s = (Cmax-Cmin)/Cmax;
			v = Cmax;

			R[idx] = h/360.f;
			G[idx] = s;
			B[idx] = v;
		}

		return;
	}

	void HSV2RGB(float *H, float *S, float *V, int shape[3])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float h = H[idx]*360.f;
			float s = S[idx];
			float v = V[idx];

			if (s == 0)
			{
				H[idx] = S[idx] = V[idx] = v*255.f;
				continue;
			}

			int hi = floor(h/60.f);
			float fi = (h/60.f-hi);

			float p = v*(1.f-s);
			float q = v*(1.f-s*fi);
			float t = v*(1.f-s*(1.f-fi));

			p *= 255.f;
			q *= 255.f;
			t *= 255.f;
			v *= 255.f;

			if (hi == 0 || hi == 6) {H[idx] = v; S[idx] = t; V[idx] = p;}
			else if (hi == 1) {H[idx] = q; S[idx] = v; V[idx] = p;}
			else if (hi == 2) {H[idx] = p; S[idx] = v; V[idx] = t;}
			else if (hi == 3) {H[idx] = p; S[idx] = q; V[idx] = v;}
			else if (hi == 4) {H[idx] = t; S[idx] = p; V[idx] = v;}
			else if (hi == 5) {H[idx] = v; S[idx] = p; V[idx] = q;}
		}

		return;
	}
	void anscombe_transform(float *R, float *G, float *B, int shape[3])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float r = R[idx];
			float g = G[idx];
			float b = B[idx];

			R[idx] = 2.f*sqrtf(r+3.f/8.f);
			G[idx] = 2.f*sqrtf(g+3.f/8.f);
			B[idx] = 2.f*sqrtf(b+3.f/8.f);
		}

		return;
	}
	void inverse_anscombe_transform(float *R, float *G, float *B, int shape[3])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float r = R[idx];
			float g = G[idx];
			float b = B[idx];

			R[idx] = (r/2.f)*(r/2.f)-3.f/8.f;
			G[idx] = (g/2.f)*(g/2.f)-3.f/8.f;
			B[idx] = (b/2.f)*(b/2.f)-3.f/8.f;
		}

		return;
	}

}

#endif //COLORSPACE_H
