#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 512
#define NUM_HEADS 8

typedef struct {
    double *W_q, *W_k, *W_v, *W_o;
    double *b_q, *b_k, *b_v, *b_o;
} MultiheadAttention;

void initialize_attention(MultiheadAttention *attention) {
    attention->W_q = (double *)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(double));
    attention->W_k = (double *)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(double));
    attention->W_v = (double *)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(double));
    attention->W_o = (double *)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(double));

    attention->b_q = (double *)malloc(INPUT_SIZE * sizeof(double));
    attention->b_k = (double *)malloc(INPUT_SIZE * sizeof(double));
    attention->b_v = (double *)malloc(INPUT_SIZE * sizeof(double));
    attention->b_o = (double *)malloc(INPUT_SIZE * sizeof(double));

    // Initialize weights and biases randomly (you can replace this with your initialization logic)
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; ++i) {
        attention->W_q[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        attention->W_k[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        attention->W_v[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        attention->W_o[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    for (int i = 0; i < INPUT_SIZE; ++i) {
        attention->b_q[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        attention->b_k[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        attention->b_v[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        attention->b_o[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void free_attention(MultiheadAttention *attention) {
    free(attention->W_q);
    free(attention->W_k);
    free(attention->W_v);
    free(attention->W_o);

    free(attention->b_q);
    free(attention->b_k);
    free(attention->b_v);
    free(attention->b_o);
}

// Implement matrix multiplication: C = A * B
void matrix_multiply(const double *A, const double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i * p + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Implement vector addition: C = A + B
void vector_addition(const double *A, const double *B, double *C, int size) {
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Implement the softmax function
void softmax(double *x, int size) {
    double max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    double exp_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        x[i] = exp(x[i] - max_val);
        exp_sum += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= exp_sum;
    }
}

void multihead_attention(const double *input, int batch_size, int sequence_length, double *output, MultiheadAttention *attention) {
    int head_size = INPUT_SIZE / NUM_HEADS;
    int i, j;

    // Linear transformations for query, key, and value
    double *Q = (double *)malloc(batch_size * sequence_length * INPUT_SIZE * sizeof(double));
    double *K = (double *)malloc(batch_size * sequence_length * INPUT_SIZE * sizeof(double));
    double *V = (double *)malloc(batch_size * sequence_length * INPUT_SIZE * sizeof(double));

    matrix_multiply(input, attention->W_q, Q, batch_size * sequence_length, INPUT_SIZE, INPUT_SIZE);
    matrix_multiply(input, attention->W_k, K, batch_size * sequence_length, INPUT_SIZE, INPUT_SIZE);
    matrix_multiply(input, attention->W_v, V, batch_size * sequence_length, INPUT_SIZE, INPUT_SIZE);

    for (i = 0; i < batch_size * sequence_length; ++i) {
        vector_addition(Q + i * INPUT_SIZE, attention->b_q, Q + i * INPUT_SIZE, INPUT_SIZE);
        vector_addition(K + i * INPUT_SIZE, attention->b_k, K + i * INPUT_SIZE, INPUT_SIZE);
        vector_addition(V + i * INPUT_SIZE, attention->b_v, V + i * INPUT_SIZE, INPUT_SIZE);
    }

    // Split into multiple heads
    double **Q_heads = (double **)malloc(NUM_HEADS * sizeof(double *));
    double **K_heads = (double **)malloc(NUM_HEADS * sizeof(double *));
    double **V_heads = (double **)malloc(NUM_HEADS * sizeof(double *));

    for (i = 0; i < NUM_HEADS; ++i) {
        Q_heads[i] = (double *)malloc(batch_size * sequence_length * head_size * sizeof(double));
        K_heads[i] = (double *)malloc(batch_size * sequence_length * head_size * sizeof(double));
        V_heads[i] = (double *)malloc(batch_size * sequence_length * head_size * sizeof(double));

        for (j = 0; j < batch_size * sequence_length; ++j) {
            Q_heads[i][j * head_size] = Q[j * INPUT_SIZE + i * head_size];
            K_heads[i][j * head_size] = K[j * INPUT_SIZE + i * head_size];
            V_heads[i][j * head_size] = V[j * INPUT_SIZE + i * head_size];

            for (int k = 1; k < head_size; ++k) {
                Q_heads[i][j * head_size + k] = Q[j * INPUT_SIZE + i * head_size + k];
                K_heads[i][j * head_size + k] = K[j * INPUT_SIZE + i * head_size + k];
                V_heads[i][j * head_size + k] = V[j * INPUT_SIZE + i * head_size + k];
            }
        }
    }

    // Apply attention for each head
    double **attention_outputs = (double **)malloc(NUM_HEADS * sizeof(double *));
    for (i = 0; i < NUM_HEADS; ++i) {
        attention_outputs[i] = (double *)malloc(batch_size * sequence_length * head_size * sizeof(double));
        for (j = 0; j < batch_size * sequence_length; ++j) {
            // Implement attention mechanism (this is a basic dot-product attention)
            double *attention_scores = (double *)malloc(head_size * sizeof(double));
            matrix_multiply(Q_heads[i] + j * head_size, K_heads[i] + j * head_size, attention_scores, 1, head_size, head_size);

            // Apply scaling factor
            for (int k = 0; k < head_size; ++k) {
                attention_scores[k] /= sqrt((double)head_size);
            }

            // Apply softmax activation
            softmax(attention_scores, head_size);

            // Weighted sum
            matrix_multiply(attention_scores, V_heads[i] + j * head_size, attention_outputs[i] + j * head_size, 1, head_size, head_size);

            free(attention_scores);
        }
    }

    // Concatenate the attention outputs
    double *concatenated_output = (double *)malloc(batch_size * sequence_length * INPUT_SIZE * sizeof(double));
    for (i = 0; i < NUM_HEADS; ++i) {
        for (j = 0; j < batch_size * sequence_length; ++j) {
            for (int k = 0; k < head_size; ++k) {
                concatenated_output[j * INPUT_SIZE + i * head_size + k] = attention_outputs[i][j * head_size + k];
            }
        }
        free(attention_outputs[i]);
    }
    free(attention_outputs);

    // Apply final linear transformation
    matrix_multiply(concatenated_output, attention->W_o, output, batch_size * sequence_length, INPUT_SIZE, INPUT_SIZE);
    for (i = 0; i < batch_size * sequence_length; ++i) {
        vector_addition(output + i * INPUT_SIZE, attention->b_o, output + i * INPUT_SIZE, INPUT_SIZE);
    }

    free(Q);
    free(K);
    free(V);
    free(concatenated_output);

    for (i = 0; i < NUM_HEADS; ++i) {
        free(Q_heads[i]);
        free(K_heads[i]);
        free(V_heads[i]);
    }
    free(Q_heads);
    free(K_heads);
    free(V_heads);
}

int main() {
    srand(42); // Set a seed for reproducibility

    int batch_size = 16;
    int sequence_length = 10;
    double *input = (double *)malloc(batch_size * sequence_length * INPUT_SIZE * sizeof(double));
    for (int i = 0; i < batch_size * sequence_length * INPUT_SIZE; ++i) {
        input[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    MultiheadAttention attention;
    initialize_attention(&attention);

    double *output = (double *)malloc(batch_size * sequence_length * INPUT_SIZE * sizeof(double));

    multihead_attention(input, batch_size, sequence_length, output, &attention);

    // Print input and output shapes (for demonstration purposes)
    printf("Input shape: %d x %d\n", batch_size * sequence_length, INPUT_SIZE);
    printf("Output shape: %d x %d\n", batch_size * sequence_length, INPUT_SIZE);

    // Clean up
    free(input);
    free(output);
    free_attention(&attention);

    return 0;
}
