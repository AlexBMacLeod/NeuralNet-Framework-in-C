#ifndef _COMMON_H
#define _COMMON_H

#define Fn_apply(type, fn, ...) {                                   \
void *stopper_for_apply = (int[]){0};                               \
type **list_for_apply = (type*[]){__VA_ARGS__, stopper_for_apply};  \
for (int i=0; list_for_apply[i] != stopper_for_apply; i++)          \
fn(list_for_apply[i]);                                              \
}

#define free_all(...) Fn_apply(void, free, __VA_ARGS__);

void load_data(char[], float*, int*);

void checkLen(char[], int*);

void one_hot_encoder(int*, float*, int);

void splitLabels(float*, float*, int*, int);

#endif