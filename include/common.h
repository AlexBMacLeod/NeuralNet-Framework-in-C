#ifndef _COMMON_H
#define _COMMON_H

#define Fn_apply(type, fn, ...) {                                   \
void *stopper_for_apply = (int[]){0};                               \
type **list_for_apply = (type*[]){__VA_ARGS__, stopper_for_apply};  \
for (int i=0; list_for_apply[i] != stopper_for_apply; i++)          \
fn(list_for_apply[i]);                                              \
}

#define free_all(...) Fn_apply(void, free, __VA_ARGS__);

struct mnist{
    float *test_labels;
    float *test_data;
    float *train_labels;
    float *train_data;
    int len_test;
    int len_train;
};

void load_data(char[], float*, int*);

void checkLen(char[], int*);

void one_hot_encoder(int*, float*, int);

void splitLabels(float*, float*, int*, int);

int argmax(float*, float*, int);

int argmax_batch(float*, float*, int, int);

void test_train_split(float*, float*, float*, float*, float*, float*, int, int, int);

float calc_batch_error(float*, float*, int, int);

struct mnist load_mnist(char[], float);

#endif