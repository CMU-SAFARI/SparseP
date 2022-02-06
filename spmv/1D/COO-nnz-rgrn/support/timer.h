#ifndef _TIMER_H_
#define _TIMER_H_

#include <sys/time.h>

typedef struct Timer{
    struct timeval startTime[6];
    struct timeval stopTime[6];
    double         time[6];

} Timer;

static void startTimer(Timer *timer, int i) {
    timer->time[i] = 0.0;
    gettimeofday(&timer->startTime[i], NULL);
}

static void stopTimer(Timer *timer, int i) {
    gettimeofday(&timer->stopTime[i], NULL);
    timer->time[i] += (timer->stopTime[i].tv_sec - timer->startTime[i].tv_sec) * 1000000.0 +
        (timer->stopTime[i].tv_usec - timer->startTime[i].tv_usec);
}

static void printTimer(Timer *timer, int i) { 
    if (i == 0 || i == 2)
        printf("Time (ms): \t\t\t%f\n", timer->time[i] / (1000)); 
    else if (i == 1)
        printf("Time (ms): \t\t%f\n", timer->time[i] / (1000)); 
    else
        printf("Time (ms): \t%f\n", timer->time[i] / (1000)); 
}

#endif
