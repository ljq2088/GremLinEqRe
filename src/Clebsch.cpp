#include "Clebsch.h"
#include <iostream>
#include <vector>

double Clebsch::ln_factorial(int n) {
    if (n < 0) return 0.0;
    static std::vector<double> cache = {0.0};
    if (n < (int)cache.size()) return cache[n];
    
    double val = cache.back();
    for (int i = cache.size(); i <= n; ++i) {
        val += std::log((double)i);
        cache.push_back(val);
    }
    return cache[n];
}

double Clebsch::cgcof(int j1, int j2, int m1, int m2, int J, int M) {
    if (m1 + m2 != M) return 0.0;
    if (std::abs(j1 - j2) > J || (j1 + j2) < J) return 0.0;
    if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(M) > J) return 0.0;

    double ln_pre = 0.5 * (ln_factorial(J + M) + ln_factorial(J - M) +
                           ln_factorial(J - j2 + j1) + ln_factorial(J - j1 + j2) +
                           ln_factorial(j1 + j2 - J) - ln_factorial(j1 + j2 + J + 1));
    
    ln_pre += 0.5 * (ln_factorial(j1 + m1) + ln_factorial(j1 - m1) +
                     ln_factorial(j2 + m2) + ln_factorial(j2 - m2));

    double sum = 0.0;
    int min_lam = 0;
    int max_lam = j1 + j2 - J;
    
    for (int lam = min_lam; lam <= max_lam; ++lam) {
        int term1 = j1 + j2 - J - lam;
        int term2 = j1 - m1 - lam;
        int term3 = j2 + m2 - lam;
        int term4 = J - j2 + m1 + lam;
        int term5 = J - j1 - m2 + lam;
        
        if (term1 < 0 || term2 < 0 || term3 < 0 || term4 < 0 || term5 < 0) continue;
        
        double ln_term = - (ln_factorial(lam) + ln_factorial(term1) + ln_factorial(term2) + 
                            ln_factorial(term3) + ln_factorial(term4) + ln_factorial(term5));
        
        double term = std::exp(ln_term);
        if (lam % 2 != 0) term = -term;
        
        sum += term;
    }
    
    return std::sqrt(2.0 * J + 1.0) * std::exp(ln_pre) * sum;
}

double Clebsch::integral_x(int s, int p, int q, int m) {
    double term1 = cgcof(q, 1, m, 0, p, m);
    double term2 = cgcof(q, 1, -s, 0, p, -s);
    double factor = std::sqrt((2.0 * q + 1.0) / (2.0 * p + 1.0));
    return term1 * term2 * factor;
}

double Clebsch::integral_x_sqr(int s, int p, int q, int m) {
    double term1 = cgcof(q, 2, m, 0, p, m);
    double term2 = cgcof(q, 2, -s, 0, p, -s);
    double factor = (2.0 / 3.0) * std::sqrt((2.0 * q + 1.0) / (2.0 * p + 1.0));
    
    double ans = term1 * term2 * factor;
    if (q == p) ans += 1.0 / 3.0;
    return ans;
}