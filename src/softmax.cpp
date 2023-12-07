#include <stdio.h>
#include <math.h>
#include <assert.h>

void softmax(float* input, size_t size) {

	assert(0 <= size <= sizeof(input) / sizeof(float));

	int i;
	float m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - constant);
	}

}