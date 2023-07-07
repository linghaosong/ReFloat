#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	double ms = (double)tp.tv_sec * 1000 + (double)tp.tv_usec / 1000; //milliseconds
	return ms / 1000;
}