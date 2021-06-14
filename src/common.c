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

void load_data(char file[], float* data)
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
            column = 0;
            if (row == 0)
                continue;

            token = strtok(line, ",");
    
            while (token) {
                index = (row-1)*785+column;
                data[index] = atof(token);
                token = strtok(NULL, ",");
                column++;
            }
            row++;
        }
    fclose(in);
    }
}

void splitLabels(float *data, float *training_data, int *labels, int len)
{
    int y = 0;
    long indexOne;
    long indexTwo;
    for(int i=0; i<len; i++)
    {
        y=0;
        for(int j=0; j<785; j++)
            {
            if(j==0) labels[i] = (int)data[785*i];
            else{
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
    for(int i=0; i<(len); i++)
    {
        for(int j=0; j<10; j++){
            if(j==data[i]){ one_hot_encoded[i*10+j]=1.0f;
        }else{
            one_hot_encoded[i*10+j]=0.0f;
        }
    }
    }
}