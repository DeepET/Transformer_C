#include <stdio.h>
#include <stdlib.h>

float Tensor_Uniform(float array[][64])
{
	int i;
	int j;

	for (i = 0; i < 64; i++) {
		for (j = 0; j < 64; j++) {
			array[i][j] = (((float)rand() / (float)(RAND_MAX)) * 2);
			array[i][j] = array[i][j] - 1;
			array[i][j] = array[i][j] / 64;
			//printf("%f ", array[i][j]);
		}
		//printf("\n");
	}

	return array[64][64];
}