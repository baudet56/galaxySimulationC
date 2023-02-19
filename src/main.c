#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#include "imageGenerator.h"
#include "star.h"
#include "saveData.h"

#define PI 3.1415

#define galaxy 0
int cptCapt;
double softening;
double pourcentageStar = 0.8;
double T;
double dt;
int nb_star;
int reload;

void initStar(star *tab_star);
void loadSetting();
void load(star *tab_star);
double invsqrtQuake(double);
int main()
{
    loadSetting();

    star *tab_star;
    int cpt = 0;
    clock_t begin, end;
    double minute = 0;

    double ax, ay;
    star ns, s;
    double dt2 = dt / 2.;
    double softening2 = softening * softening;
    double expensionRate = 0.0384; // real 0.0384
    int nb_star_reduce = (int)(pourcentageStar * nb_star);

    tab_star = (star *)malloc(sizeof(star) * nb_star);

    if (reload)
    {
        load(tab_star);
    }
    else
    {
        initStar(tab_star);
        saveStar(tab_star, nb_star);
    }

    begin = time(NULL);
    printf("start\n");

#pragma omp parallel
    {
        for (double t = 0; t < T; t += dt)
        {
#pragma omp barrier
#pragma omp single
            {
                if (cpt % cptCapt == 0)
                {
                    end = time(NULL);
                    if (t)
                        minute = (double)(end - begin) * (T / t - 1);
                    printf("%.2f%% : time left: %i h %i m %i sec\n",
                           t / T * 100, (int)(minute / 60 / 60), (int)(minute / 60) % 60, (int)(minute) % 60);
                    createImg(tab_star, nb_star, 3);
                }

                if (cpt % 800 == 1)
                    saveStar(tab_star, nb_star);
            }
            for (int i = 0; i < nb_star; ++i)
            {
#pragma omp single
                {
                    s = tab_star[i];
                    s.vx += s.ax * dt2;
                    s.vy += s.ay * dt2;

                    s.x += s.vx * dt + s.x * expensionRate * dt;
                    s.y += s.vy * dt + s.y * expensionRate * dt;

                    ax = 0;
                    ay = 0;
                }
#pragma omp for reduction(+ \
                          : ax, ay) schedule(static) firstprivate(s)
                for (int j = 0; j < nb_star_reduce; ++j)
                {
                    const double dx = tab_star[j].x - s.x;
                    const double dy = tab_star[j].y - s.y;
                    double F = dx * dx + dy * dy + softening2;
                    if (F < 20)
                    {
                        F = tab_star[j].mass * invsqrtQuake(F * F * F);
                        ax += F * dx;
                        ay += F * dy;
                    }
                }
#pragma omp single
                {
                    s.ax = ax;
                    s.ay = ay;

                    s.vx += ax * dt2;
                    s.vy += ay * dt2;
                    tab_star[i] = s;
                }
#pragma omp barrier
            }
#pragma omp single
            {
                ++cpt;
            }
        } // time loop
    }     // end parallel
    printf("Simulation finished\n");
    saveStar(tab_star, nb_star);
    free(tab_star);
    return 0;
}

double invsqrtQuake(double number)
{
    double y = number;
    double x2 = y * 0.5;
    int64_t i = *(int64_t *)&y;
    // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    y = *(double *)&i;
    y = y * (1.5 - (x2 * y * y)); // 1st iteration
    //      y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
    return y;
}

double getRandomValue()
{
    return (double)rand() / RAND_MAX;
}

void initStar(star *tab_star)
{
    double u, phi;
    int i;

    srand(time(NULL));
    for (i = 0; i < nb_star; ++i)
    {
#if galaxy
        u = getRandomValue();
        phi = 2 * PI * u;
        tab_star[i].x = pow(2 * getRandomValue(), (double)1. / 3) * cos(phi);
        tab_star[i].y = pow(2 * getRandomValue(), 1. / 3) * sin(phi);
        tab_star[i].mass = (double)1. / nb_star * 1. / pourcentageStar;
        tab_star[i].vx = 1000 * pow(2 * getRandomValue(), (double)1. / 3) * (-sin(phi)) * 2 * PI * dt;
        tab_star[i].vy = 1000 * pow(2 * getRandomValue(), (double)1. / 3) * cos(phi) * 2 * PI * dt;
        tab_star[i].ax = 0;
        tab_star[i].ay = 0;

        if (i % 2)
        {
            tab_star[i].x += 1.5;
            tab_star[i].y += 1.5;
        }
        else
        {
            tab_star[i].x -= 1.5;
            tab_star[i].y -= 1.5;
        }
#else
        tab_star[i].x = 15 - 30 * getRandomValue();
        tab_star[i].y = 15 - 30 * getRandomValue();
        tab_star[i].mass = (double)1. / nb_star * 1. / pourcentageStar;
        tab_star[i].vx = 0.08 - 0.16 * getRandomValue();
        tab_star[i].vy = 0.08 - 0.16 * getRandomValue();
        tab_star[i].ax = 0;
        tab_star[i].ay = 0;
#endif
    }
    /* Black hole ! */
    /*
    tab_star[0].x = 2;
    tab_star[0].y = 2;

    tab_star[0].vx = 0;
    tab_star[0].vy = 0;
    tab_star[0].mass = 100 * 1. / nb_star * 1. / pourcentageStar;

    tab_star[1].x = -2;
    tab_star[1].y = -2;
    tab_star[1].vx = 0;
    tab_star[1].vy = 0;
    tab_star[1].mass = 100 * 1. / nb_star * 1. / pourcentageStar;
    */
}

void loadSetting()
{
    FILE *f = fopen("./save/setting.txt", "r");

    fscanf(f, "simulationDuration: %lf\n", &T);
    fscanf(f, "dt: %lf\n", &dt);
    fscanf(f, "numberOfStar: %i\n", &nb_star);
    fscanf(f, "softenningCoef: %lf\n", &softening);
    fscanf(f, "counterCapture: %i\n", &cptCapt);
    fscanf(f, "reloadSimulation(0=no/1=yes): %i\n", &reload);
    fclose(f);
    return;
}

void load(star *tab_star)
{
    FILE *f = fopen("./save/star", "r");
    int i;
    star *s;
    fscanf(f, "x;y;Vx;Vy;Ax;Ay;m\n");

    for (i = 0; i < nb_star; ++i)
    {
        s = &tab_star[i];
        fscanf(f, "%lf;%lf;%lf;%lf;%lf;%lf;%lf\n",
               &s->x, &s->y, &s->vx, &s->vy, &s->ax, &s->ay, &s->mass);
    }
    fclose(f);
}
