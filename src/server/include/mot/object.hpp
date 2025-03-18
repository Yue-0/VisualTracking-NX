/* @Author: YueLin */

#pragma once

#include <vector>
#include "opencv2/opencv.hpp"

struct Object
{
    int label;
    float prob;
    cv::Rect_<float> rect;
};

typedef std::vector<Object> Objects;

/* Quick sort algorithm for Objects */
void sorted(Objects& objects, int left, int right)
{
    int i = left, j = right;
    float p = objects[(left + right) / 2].prob;
    while (i <= j)
    {
        while(objects[i].prob > p) i++;
        while(objects[j].prob < p) j--;
        if(i <= j) std::swap(objects[i++], objects[j--]);
    }

    /* Segmented parallelism */
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) sorted(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) sorted(objects, i, right);
        }
    }
}
