#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "extra.h"

int main()
{
	
	float query_dense[64][64];
	query_dense[64][64] = Tensor_Uniform(query_dense);

	float key_dense[64][64];
	key_dense[64][64] = Tensor_Uniform(key_dense);
	
	float value_dense[64][64];
	value_dense[64][64] = Tensor_Uniform(value_dense);
	
	float final[64][64];
	final[64][64] = Tensor_Uniform(final);
	
	float ff1[64][64];
	ff1[64][64] = Tensor_Uniform(ff1);

	float ff2[64][64];
	ff2[64][64] = Tensor_Uniform(ff2);


	TransformerBlock(query_dense, key_dense, value_dense, final, ff1, ff2);

	return 0;
	
}
