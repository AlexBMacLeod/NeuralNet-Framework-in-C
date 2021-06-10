#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/common.h"


void load_data(char file[], float *data, float *labels)
{
    int rows;
    FILE *in = fopen(file, "r");
    if (!in) {
        printf("Could not open file: %s\n", file);
        exit(1);
    }


    char line[2560];
    while (!feof(in) && fgets(line, 2560, in)) {
        ++rows;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d rows from %s\n", rows, file);


    data = malloc(sizeof(float) * rows * 28 * 28 - 1);
    labels = malloc(sizeof(float) * rows - 1);

    int row_count = 0;
    int field_count = 0;
    int i = 0;
    while(fgets(line, 2560, in))
    {
        field_count = 0;
        row_count ++;
        if(row_count==1) continue;

        char *field = strtok(line, ",");
        while(field)
        {
            if(field_count == 0){
             labels[i]=atof(field);
            }else{
                data[i*28*28+field_count] = atof(field)/256.0f;
            }
            field = strtok(NULL, ",");
            field_count++;
        }
        i++;
    }

    fclose(in);
}

void one_hot_encoder(int *data, float *one_hot_encoded)
{
    int len;
    int tmp;
    len = sizeof(data)/sizeof(int);
    one_hot_encoded = (float*)malloc(sizeof(float)*len*10);
    for(int i=0; i<len; i++)
    {
        tmp = data[i];
        one_hot_encoded[i*10+tmp] = 1.0f;
    }
}