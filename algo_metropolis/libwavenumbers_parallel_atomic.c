#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

void wavenumbers_stein_squire_c(
    double rho_ice, double h, double H,
    double E, double nu,
    double *freq, int m,
    double c_w, double rho_w,
    double *k_QS
) {
    const int n = 200000;
    double g = 9.81;
    double D = E * h*h*h / (12.0 * (1 - nu*nu));
    double inv_cw2 = 1.0/(c_w*c_w);

    // Allocation et initialisation de k (identique pour tous les threads)
    double *k = malloc(n * sizeof(double));
    for(int j = 0; j < n; ++j)
        k[j] = 1e-12 + j*(8.0 - 1e-12)/(n-1);

    // Parallélisation de la boucle externe i
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < m; ++i) {
        double omeg = 2*M_PI*freq[i];
        double rho_w_div_D = rho_w/D;
        double h_omeg2 = h*omeg*omeg*rho_w_div_D;

        // Recherche séquentielle du premier changement de signe dans j
        int idx0 = 0;
        int prev_sign = 0;
        for(int j = 0; j < n; ++j) {
            double cph = omeg/k[j];
            double expr = (1.0/(cph*cph)) - inv_cw2; 
            double func = (expr <= 0)
                ? -1.0
                : rho_w_div_D*(g - omeg/sqrt(expr)) - h_omeg2 + pow(omeg/cph,4);
            
            int sign = func >= 0.0;
            if(sign != prev_sign) {
                idx0 = j;
                break;
            }
            prev_sign = sign;
        }
        k_QS[i] = k[idx0];
    }

    free(k);
}
