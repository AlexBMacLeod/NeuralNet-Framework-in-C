#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "../include/common.h"

void checkLen(char file[], int *len)
{
    int rows=0;
    FILE *in = fopen(file, "r");
    if (!in) {
        fprintf(stderr, "Could not open file: %s\n", file);
        exit(1);
    }else{

        char line[8192];
        while (!feof(in) && fgets(line, 8192, in)) {
            ++rows;
        }
        printf("Loading %d rows from %s\n", rows, file);
        *len = rows-1;
        fclose(in);
    }
}

void load_data(char file[], float* data, int* labels)
{
    FILE *in = fopen(file, "r");
    if (!in) {
        fprintf(stderr, "Could not open file: %s\n", file);
        exit(1);
    }else{
        char line[8192];
        char* token = NULL;
        long index = 0;
    
        int row = 0;
        int column = 0;
    
        while (!feof(in) && fgets(line, 8192, in)) {
            row++;
            if (row == 1)
                continue;
            token = strtok(line, ",");
            column = 0;
            while (token) {
                if(column==0){ 
                    labels[row-2]=atoi(token);
                }else{
                    index = (row-2)*784+(column-1);
                    data[index] = atof(token)/255.0f;
                }
                token = strtok(NULL, ",");
                column++;
            }
        }
    fclose(in);
    }
}

void splitLabels(float *data, float *training_data, int *labels, int len)
{
    int y = 0;
    long index;
    long indexOne;
    long indexTwo;
    for(int i=0; i<len; i++)
    {
        y=0;
        for(int j=0; j<785; j++)
            {
            if(j==0){
                index = 785*i;
                labels[i] = (int)data[index];
            }else{
                indexOne = 785*i+j;
                indexTwo = 784*i+y;
                training_data[indexTwo] = data[indexOne]/255.0f;
                y++;
            }
        }
    }
}

void one_hot_encoder(int *data, float *one_hot_encoded, int len)
{
    for(int i=0; i<len; i++)
    {
        one_hot_encoded[i*10+data[i]] = 1.0f;
    }
}