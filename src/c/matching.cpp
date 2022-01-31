
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <functional>

extern "C" {
    
    void match_template_mad(
        int nax, int nay, int naz, float *array,
        int ntx, int nty, int ntz, float *temp,
        float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0 || ntz % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;
        int tk_middle = (ntz - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;
                for (int k = 0; k < naz; k++) {
                    int ak_start = std::max(0, k - tk_middle);
                    int tk_start = std::max(0, tk_middle - k);
                    int kk_end = std::min(k + tk_middle + 1, naz) - ak_start;

                    float sum_array = 0;
                    for (int ii = 0; ii < ii_end; ii++) {
                        int ti = ti_start + ii;
                        int ai = ai_start + ii;
                        for (int jj = 0; jj < jj_end; jj++) {
                            int tj = tj_start + jj;
                            int aj = aj_start + jj;
                            for (int kk = 0; kk < kk_end; kk++) {
                                int tk = tk_start + kk;
                                int ak = ak_start + kk;
                                int a_ind = ai*nay*naz + aj*naz + ak;
                                int t_ind = ti*nty*ntz + tj*ntz + tk;
                                float diff = std::abs(array[a_ind] - temp[t_ind]);
                                sum_array += diff;
                            }
                        }
                    }

                    dist_array[i*nay*naz + j*naz + k] = sum_array / (ii_end * jj_end * kk_end);
                    
                }
            }
        }

    }

    void match_template_mad_norm(
        int nax, int nay, int naz, float *array,
        int ntx, int nty, int ntz, float *temp,
        float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0 || ntz % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;
        int tk_middle = (ntz - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;
                for (int k = 0; k < naz; k++) {
                    int ak_start = std::max(0, k - tk_middle);
                    int tk_start = std::max(0, tk_middle - k);
                    int kk_end = std::min(k + tk_middle + 1, naz) - ak_start;

                    float sum_diff = 0;
                    float sum_temp = 0;
                    for (int ii = 0; ii < ii_end; ii++) {
                        int ti = ti_start + ii;
                        int ai = ai_start + ii;
                        for (int jj = 0; jj < jj_end; jj++) {
                            int tj = tj_start + jj;
                            int aj = aj_start + jj;
                            for (int kk = 0; kk < kk_end; kk++) {
                                int tk = tk_start + kk;
                                int ak = ak_start + kk;
                                int a_ind = ai*nay*naz + aj*naz + ak;
                                int t_ind = ti*nty*ntz + tj*ntz + tk;
                                float diff = std::abs(array[a_ind] - temp[t_ind]);
                                sum_diff += diff;
                                sum_temp += temp[t_ind];
                            }
                        }
                    }

                    float mean_diff = sum_diff / (ii_end * jj_end * kk_end);
                    float mean_temp = sum_temp / (ii_end * jj_end * kk_end);
                    dist_array[i*nay*naz + j*naz + k] = mean_diff / mean_temp;
                    
                }
            }
        }

    }

    void match_template_msd(
            int nax, int nay, int naz, float *array,
            int ntx, int nty, int ntz, float *temp,
            float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0 || ntz % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;
        int tk_middle = (ntz - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;
                for (int k = 0; k < naz; k++) {
                    int ak_start = std::max(0, k - tk_middle);
                    int tk_start = std::max(0, tk_middle - k);
                    int kk_end = std::min(k + tk_middle + 1, naz) - ak_start;

                    float sum_array = 0;
                    for (int ii = 0; ii < ii_end; ii++) {
                        int ti = ti_start + ii;
                        int ai = ai_start + ii;
                        for (int jj = 0; jj < jj_end; jj++) {
                            int tj = tj_start + jj;
                            int aj = aj_start + jj;
                            for (int kk = 0; kk < kk_end; kk++) {
                                int tk = tk_start + kk;
                                int ak = ak_start + kk;
                                int a_ind = ai*nay*naz + aj*naz + ak;
                                int t_ind = ti*nty*ntz + tj*ntz + tk;
                                float diff = array[a_ind] - temp[t_ind];
                                sum_array += diff * diff;
                            }
                        }
                    }
                    
                    dist_array[i*nay*naz + j*naz + k] = sum_array / (ii_end * jj_end * kk_end);
                    
                }
            }
        }

    }

    void match_template_msd_norm(
        int nax, int nay, int naz, float *array,
        int ntx, int nty, int ntz, float *temp,
        float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0 || ntz % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;
        int tk_middle = (ntz - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;
                for (int k = 0; k < naz; k++) {
                    int ak_start = std::max(0, k - tk_middle);
                    int tk_start = std::max(0, tk_middle - k);
                    int kk_end = std::min(k + tk_middle + 1, naz) - ak_start;

                    float sum_diff = 0;
                    float sum_temp = 0;
                    for (int ii = 0; ii < ii_end; ii++) {
                        int ti = ti_start + ii;
                        int ai = ai_start + ii;
                        for (int jj = 0; jj < jj_end; jj++) {
                            int tj = tj_start + jj;
                            int aj = aj_start + jj;
                            for (int kk = 0; kk < kk_end; kk++) {
                                int tk = tk_start + kk;
                                int ak = ak_start + kk;
                                int a_ind = ai*nay*naz + aj*naz + ak;
                                int t_ind = ti*nty*ntz + tj*ntz + tk;
                                float diff = array[a_ind] - temp[t_ind];
                                sum_diff += diff * diff;
                                sum_temp += temp[t_ind] * temp[t_ind];
                            }
                        }
                    }

                    float mean_diff = sum_diff / (ii_end * jj_end * kk_end);
                    float mean_temp = sum_temp / (ii_end * jj_end * kk_end);
                    dist_array[i*nay*naz + j*naz + k] = mean_diff / mean_temp;
                    
                }
            }
        }

    }

    void match_template_mad_2d(
        int nax, int nay, float *array,
        int ntx, int nty, float *temp,
        float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;

                float sum_array = 0;
                for (int ii = 0; ii < ii_end; ii++) {
                    int ti = ti_start + ii;
                    int ai = ai_start + ii;
                    for (int jj = 0; jj < jj_end; jj++) {
                        int tj = tj_start + jj;
                        int aj = aj_start + jj;
                        int a_ind = ai*nay + aj;
                        int t_ind = ti*nty + tj;
                        float diff = std::abs(array[a_ind] - temp[t_ind]);
                        sum_array += diff;
                    }
                }

                dist_array[i*nay + j] = sum_array / (ii_end * jj_end);
                    
            }
        }

    }

    void match_template_mad_norm_2d(
        int nax, int nay, float *array,
        int ntx, int nty, float *temp,
        float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;

                float sum_diff = 0;
                float sum_temp = 0;
                for (int ii = 0; ii < ii_end; ii++) {
                    int ti = ti_start + ii;
                    int ai = ai_start + ii;
                    for (int jj = 0; jj < jj_end; jj++) {
                        int tj = tj_start + jj;
                        int aj = aj_start + jj;
                        int a_ind = ai*nay + aj;
                        int t_ind = ti*nty + tj;
                        float diff = std::abs(array[a_ind] - temp[t_ind]);
                        sum_diff += diff;
                        sum_temp += temp[t_ind];
                    }
                }

                float mean_diff = sum_diff / (ii_end * jj_end);
                float mean_temp = sum_temp / (ii_end * jj_end);
                dist_array[i*nay + j] = mean_diff / mean_temp;
                    
            }
        }

    }

    void match_template_msd_2d(
            int nax, int nay, float *array,
            int ntx, int nty, float *temp,
            float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;

                float sum_array = 0;
                for (int ii = 0; ii < ii_end; ii++) {
                    int ti = ti_start + ii;
                    int ai = ai_start + ii;
                    for (int jj = 0; jj < jj_end; jj++) {
                        int tj = tj_start + jj;
                        int aj = aj_start + jj;
                        int a_ind = ai*nay + aj;
                        int t_ind = ti*nty + tj;
                        float diff = array[a_ind] - temp[t_ind];
                        sum_array += diff * diff;
                    }
                }
                
                dist_array[i*nay + j] = sum_array / (ii_end * jj_end);
                    
            }
        }

    }

    void match_template_msd_norm_2d(
        int nax, int nay, float *array,
        int ntx, int nty, float *temp,
        float *dist_array
    ) {

        if (ntx % 2 == 0 || nty % 2 == 0) {
            throw std::invalid_argument("Template dimensions must all be odd size.");
        }

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;

        for (int i = 0; i < nax; i++) {
            int ai_start = std::max(0, i - ti_middle);
            int ti_start = std::max(0, ti_middle - i);
            int ii_end = std::min(i + ti_middle + 1, nax) - ai_start;
            for (int j = 0; j < nay; j++) {
                int aj_start = std::max(0, j - tj_middle);
                int tj_start = std::max(0, tj_middle - j);
                int jj_end = std::min(j + tj_middle + 1, nay) - aj_start;

                float sum_diff = 0;
                float sum_temp = 0;
                for (int ii = 0; ii < ii_end; ii++) {
                    int ti = ti_start + ii;
                    int ai = ai_start + ii;
                    for (int jj = 0; jj < jj_end; jj++) {
                        int tj = tj_start + jj;
                        int aj = aj_start + jj;
                        int a_ind = ai*nay + aj;
                        int t_ind = ti*nty + tj;
                        float diff = array[a_ind] - temp[t_ind];
                        sum_diff += diff * diff;
                        sum_temp += temp[t_ind] * temp[t_ind];
                    }
                }

                float mean_diff = sum_diff / (ii_end * jj_end);
                float mean_temp = sum_temp / (ii_end * jj_end);
                dist_array[i*nay + j] = mean_diff / mean_temp;
                
            }
        }

    }

}