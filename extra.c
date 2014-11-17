#include <emmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <stdio.h>
#include <omp.h>


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel, int kernel_x, int kernel_y)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (kernel_x - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (kernel_y - 1)/2;

    int newX = ((data_size_X + (2*kern_cent_X))*4)/4;
    int newY = ((data_size_Y + (2*kern_cent_Y))*4)/4;
    __m128 v0, v1, v2, v3, v4, v5, v6, v7, kernV;
    int curr_kern, area, y, x, i, j, yD, kernY, iCoord, halfX, halfY, kernArea, iX, centX, centY, aligned;
    int headX = (data_size_X/32) * 32; 
    int head8X = (headX/8) * 8;
    area = newX * newY;
    kernArea = kernel_y * kernel_x;
    halfX = data_size_X + kern_cent_X;
    halfY = data_size_Y + kern_cent_Y;
    float *curr_pad, *out1, *padY;
    float *pad = _mm_malloc(area*sizeof(float), 16); 
    float *kern = _mm_malloc((kernArea)*sizeof(float), 16);

    for(i = 0; i < kernArea; i++) {
      kern[i]  = kernel[i];
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);    

    int area8 = (area / 8) * 8;
    #pragma omp parallel
    {
    #pragma omp for firstprivate(area, newX, newY, pad, area8) private(i)
      for (i = 0; i < area8; i+= 8) {
        *(pad+i) = 0.0F;
        *(pad+i+1) = 0.0F;
        *(pad+i+2) = 0.0F;
        *(pad+i+3) = 0.0F;
        *(pad+i+4) = 0.0F;
        *(pad+i+5) = 0.0F;
        *(pad+i+6) = 0.0F;
        *(pad+i+7) = 0.0F;
      }
    
      #pragma omp for firstprivate(area, newX, newY, pad, area8) private(i)
      for (i = area8; i < area; i++) {
        *(pad + i) = 0.0F;
      }
    }
    
    #pragma omp parallel for firstprivate(kern_cent_Y,kern_cent_X,data_size_X,data_size_Y,newX,newY,pad,in) private(i,j,iCoord,iX)
    for (i = kern_cent_Y;  i < halfY; i++) {
      iCoord = (i - kern_cent_Y) * data_size_X;
      iX = i * newX;
      for (j = kern_cent_X; j < halfX; j++) {
        *(pad+j+iX) = *(in+(j-kern_cent_X)+iCoord);
      } 
    }

    #pragma omp parallel for schedule(dynamic) firstprivate(kernel_x,kernel_y,newX,newY,headX,head8X,kern_cent_X,kern_cent_Y,kern,pad,out,data_size_X,data_size_Y) private(centX,centY,v0,v1,v2,v3,v4,v5,v6,v7,curr_kern,curr_pad,y,x,j,i,kernV,yD,out1,padY,kernY)
    for(y = 0; y < data_size_Y; y++) {
    yD = y * data_size_X;
    centY = y + kern_cent_Y;
    for(x = 0; x < headX; x+=32) {  
      out1 = out + x + yD;
      centX = kern_cent_X + x;
      v0 = _mm_setzero_ps();
      v1 = _mm_setzero_ps();
      v2 = _mm_setzero_ps();
      v3 = _mm_setzero_ps();
      v4 = _mm_setzero_ps();
      v5 = _mm_setzero_ps();
      v6 = _mm_setzero_ps();
      v7 = _mm_setzero_ps();
      for (j = -kern_cent_Y; j <= kern_cent_Y; j++) {
        padY = (((centY + j) * newX) + pad) + centX;
        kernY = kern_cent_X + ((kern_cent_Y - j)*kernel_x);
        for (i = -kern_cent_X; i <= kern_cent_X; i++) {   
          curr_kern = kernY - i;
          curr_pad = i + padY;
          kernV = _mm_set1_ps(*(kern+curr_kern));
          v0 = _mm_add_ps(v0, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad)));
          v1 = _mm_add_ps(v1, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+4)));
          v2 = _mm_add_ps(v2, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+8)));
          v3 = _mm_add_ps(v3, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+12)));
          v4 = _mm_add_ps(v4, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+16)));
          v5 = _mm_add_ps(v5, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+20)));
          v6 = _mm_add_ps(v6, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+24)));
          v7 = _mm_add_ps(v7, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+28)));
        }
      }
      _mm_storeu_ps(out1, v0);
      _mm_storeu_ps(out1+4, v1);
      _mm_storeu_ps(out1+8, v2);
      _mm_storeu_ps(out1+12, v3);
      _mm_storeu_ps(out1+16, v4);
      _mm_storeu_ps(out1+20, v5);
      _mm_storeu_ps(out1+24, v6);
      _mm_storeu_ps(out1+28, v7);
    }

    for (x = headX; x < head8X; x+=8) {
      out1 = out + x + yD;
      centX = kern_cent_X + x;
      v0 = _mm_setzero_ps();
      v1 = _mm_setzero_ps();
      for (j = -kern_cent_Y; j <= kern_cent_Y; j++) {
        padY = centX + (((centY + j) * newX) + pad);
        kernY = kern_cent_X + ((kern_cent_Y - j) * kernel_x);
        for (i = -kern_cent_X; i <= kern_cent_X; i++) {
          curr_kern = kernY - i;
          curr_pad = i + padY;
          kernV = _mm_set1_ps(*(kern+curr_kern)); 
          v0 = _mm_add_ps(v0, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad)));
          v1 = _mm_add_ps(v1, _mm_mul_ps(kernV, _mm_loadu_ps(curr_pad+4)));
        }
      }
      _mm_storeu_ps(out1, v0);  
      _mm_storeu_ps(out1+4, v1);
    }

    for (x = head8X; x < data_size_X; x++) {
      out1 = out + x + yD;
      *out1 = 0;
      centX = kern_cent_X + x;
      for (j = -kern_cent_Y; j <= kern_cent_Y; j++) {
        padY = ((centY + j) * newX) + pad + centX;
        kernY = kern_cent_X + ((kern_cent_Y - j) * kernel_x);
        for (i = -kern_cent_X; i <= kern_cent_X; i++) {
          curr_kern = kernY - i;
          curr_pad = i + padY;
          *out1 += *curr_pad * kern[curr_kern];       
        }
      } 
    }
  }

  _mm_free(pad);
  _mm_free(kern);
  return 1;
}









