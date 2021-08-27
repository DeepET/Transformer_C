#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void TransformerBlock(float query_dense[][64], float key_dense[][64], float value_dense[][64], float final[][64], float ff1[][64], float ff2[][64])
{

	//embed_dim = 64;
	//num_heads = 4;

	int head_size = 16;
	//head_size = embed_dim / num_heads;
	//embed_dim = head_size * num_heads;

	//int bs = embed_dim;
	//float x[128][6][128];
	float input[384][64];

	int i, j, k;

	for (k = 0; k < 6; k++) {
		if (k == 0) {
			for (j = 0; j < 64; j++) {
				for (i = 0; i < 64; i++) {
					//x[j][0][i] = query_dense[j][i];
					//input[j][i] = x[j][0][i];
					input[j][i] = query_dense[j][i];
				}
			}
		}
		if (k == 1) {
			for (j = 0; j < 64; j++) {
				for (i = 0; i < 64; i++) {
					//x[j][1][i] = key_dense[j][i];
					//input[j + 128][i] = x[j][1][i];
					input[j+64][i] = key_dense[j][i];
				}
			}
		}
		if (k == 2) {
			for (j = 0; j < 64; j++) {
				for (i = 0; i < 64; i++) {
					//x[j][2][i] = value_dense[j][i];
					//input[j + 256][i] = x[j][2][i];
					input[j + 128][i] = value_dense[j][i];
				}
			}
		}
		if (k == 3) {
			for (j = 0; j < 64; j++) {
				for (i = 0; i < 64; i++) {
					//x[j][3][i] = final[j][i];
					//input[j + 384][i] = x[j][3][i];
					input[j + 192][i] = final[j][i];
				}
			}
		}
		if (k == 4) {
			for (j = 0; j < 64; j++) {
				for (i = 0; i < 64; i++) {
					//x[j][4][i] = ff1[j][i];
					//input[j + 512][i] = x[j][4][i];
					input[j + 256][i] = ff1[j][i];
				}
			}
		}
		if (k == 5) {
			for (j = 0; j < 64; j++) {
				for (i = 0; i < 64; i++) {
					//x[j][5][i] = ff2[j][i];
					//input[j + 640][i] = x[j][5][i];
					input[j + 320][i] = ff2[j][i];
				}
			}
		}
	}

	//query, key, value = [inputs.dot(y).reshape(shape=(bs, -1, self.num_heads, self.head_size)) for y in [self.query_dense, self.key_dense, self.value_dense]]
	float query_dot[384][64] = { 0 };
	float key_dot[384][64] = { 0 };
	float value_dot[384][64] = { 0 };
	
	for (j = 0; j < 384; j++) {
		for (i = 0; i < 64; i++) {
			for (k = 0; k < 64; k++) {
				query_dot[j][i] += input[j][k]*query_dense[k][i];
				key_dot[j][i] += input[j][k] * key_dense[k][i];
				value_dot[j][i] += input[j][k] * value_dense[k][i];
			}
		}
	}

	int l, m;
	float query_res[64][6][4][16];
	float key_res[64][6][4][16];
	float value_res[64][6][4][16];

	for (m = 0; m < 64; m++) {
		for (l = 0; l < 6; l++) {
			for (j = 0; j < 4; j++) {
				for (i = 0; i < 16; i++) {
					query_res[m][l][j][i] = query_dot[m * 5 + l][j * 15 + i];
					key_res[m][l][j][i] = key_dot[m * 5 + l][j * 15 + i];
					value_res[m][l][j][i] = value_dot[m * 5 + l][j * 15 + i];
				}
			}
		}
	}

	//query = query.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)
	//key = key.transpose(order=(0,2,3,1))      # (bs, num_heads, head_size, T)
	//value = value.transpose(order=(0,2,1,3))  # (bs, num_heads, T, head_size)
	float query_trans[64][4][6][16];
	float key_trans[64][4][16][6];
	float value_trans[64][4][6][16];

	for (m = 0; m < 64; m++) {
		for (l = 0; l < 6; l++) {
			for (j = 0; j < 4; j++) {
				for (i = 0; i < 16; i++) {
					query_trans[m][j][l][i] = query_res[m][l][j][i];
					key_trans[m][j][i][l] = key_res[m][l][j][i];
					value_trans[m][j][l][i] = value_res[m][l][j][i];
				}
			}
		}
	}

	//score = query.dot(key) * (1 / np.sqrt(self.head_size))
	float score[64][4][6][6] = { 0 };
	for (m = 0; m < 64; m++) {
		for (l = 0; l < 4; l++) {
			for (j = 0; j < 6; j++) {
				for (i = 0; i < 6; i++) {
					for (k = 0; k < 16; k++) {
						score[m][l][j][i] += query_trans[m][l][j][k] * key_trans[m][l][k][i];
						score[m][l][j][i] *= 1 / sqrt(head_size);
					}
				}
			}
		}
	}

	//weights = score.softmax()
	float weights[64][4][6][6];
	int o;
	o = -INFINITY;
	for (m = 0; m < 64; m++) {
		for (l = 0; l < 4; l++) {
			for (j = 0; j < 6; j++) {
				for (i = 0; i < 6; i++) {
					if (o < score[m][l][j][i]) {
						o = score[m][l][j][i];
					}
				}
			}
		}
	}

	float sum = 0;
	for (m = 0; m < 64; m++) {
		for (l = 0; l < 4; l++) {
			for (j = 0; j < 6; j++) {
				for (i = 0; i < 6; i++) {
					sum += exp(score[m][l][j][i] - o);
				}
			}
		}
	}
	
	float constant = 0;
	constant = o + log(sum);
	for (m = 0; m < 64; m++) {
		for (l = 0; l < 4; l++) {
			for (j = 0; j < 6; j++) {
				for (i = 0; i < 6; i++) {
					weights[m][l][j][i] = exp(score[m][l][j][i] - constant);
				}
			}
		}
	}

	//attention = weights.dot(value).transpose(order=(0,2,1,3))   # (bs, T, num_heads, head_size)
	float weighths_dot[64][4][6][16] = { 0 };
	float attention[64][6][4][16];

	for (m = 0; m < 64; m++) {
		for (l = 0; l < 4; l++) {
			for (j = 0; j < 6; j++) {
				for (i = 0; i < 16; i++) {
					for (k = 0; k < 6; k++) {
						weighths_dot[m][l][j][i] += weights[m][l][j][k] * value_trans[m][l][k][i];
					}
				}
			}
		}
	}
	for (m = 0; m < 64; m++) {
		for (l = 0; l < 4; l++) {
			for (j = 0; j < 6; j++) {
				for (i = 0; i < 16; i++) {
					attention[m][j][l][i] = weighths_dot[m][l][j][i];
				}
			}
		}
	}
    //x = inputs + attention.reshape(shape=(-1, embed_dim)).dot(self.final).dropout(0.1)
    //x = layernorm(x, embed_dim)
    //x = x + x.dot(self.ff1).relu().dot(self.ff2).dropout(0.1)
    //x = layernorm(x, embed_dim)
}
