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

        // On veut le plus petit j où le signe change
        int idx0 = n;  // valeur maxi
        int global_prev_sign = 0;

        // Boucle j en parallèle, avec réduction min sur idx0
        #pragma omp parallel for reduction(min:idx0) schedule(static)
        for(int j = 0; j < n; ++j) {
            // ATTENTION : prev_sign doit être local à chaque thread!
            // On le réinitialise à 0 pour chaque thread.
            int prev_sign = 0;
            double cph = omeg/k[j];
            double expr = (1.0/(cph*cph)) - inv_cw2; 
            double func = (expr <= 0)
                ? -1.0
                : rho_w_div_D*(g - omeg/sqrt(expr)) - h_omeg2 + pow(omeg/cph,4);
            
            int sign = func >= 0.0;
            if (sign != prev_sign) {
                // on a trouvé un changement : on demande la réduction minima
                idx0 = j < idx0 ? j : idx0;
            }
            prev_sign = sign;
        }

        // Si on n’a rien trouvé (idx0 == n), on peut choisir 0 ou un flag d’erreur
        k_QS[i] = (idx0 < n) ? k[idx0] : k[0];
    }

    free(k);
}
