
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <defs.h>

__device__ float sum_mad(
    int ti_start, int tj_start, int tk_start,
    int ai_start, int aj_start, int ak_start,
    int i_range, int j_range, int k_range,
    int aj_size, int ak_size, float *array,
    int tj_size, int tk_size, float *temp
) {

    float sum_array = 0;
    for (int ii = 0; ii < i_range; ii++) {
        int ti = ti_start + ii;
        int ai = ai_start + ii;
        for (int jj = 0; jj < j_range; jj++) {
            int tj = tj_start + jj;
            int aj = aj_start + jj;
            for (int kk = 0; kk < k_range; kk++) {
                int tk = tk_start + kk;
                int ak = ak_start + kk;
                sum_array += abs(
                    array[ai*ak_size*aj_size + aj*ak_size + ak] 
                    - temp[ti*tj_size*tk_size + tj*tk_size + tk]);
            }
        }
    }

    return sum_array / (i_range * j_range * k_range);
        
}

__device__ float sum_msd(
    int ti_start, int tj_start, int tk_start,
    int ai_start, int aj_start, int ak_start,
    int i_range, int j_range, int k_range,
    int aj_size, int ak_size, float *array,
    int tj_size, int tk_size, float *temp
) {

    float sum_array = 0;
    for (int ii = 0; ii < i_range; ii++) {
        int ti = ti_start + ii;
        int ai = ai_start + ii;
        for (int jj = 0; jj < j_range; jj++) {
            int tj = tj_start + jj;
            int aj = aj_start + jj;
            for (int kk = 0; kk < k_range; kk++) {
                int tk = tk_start + kk;
                int ak = ak_start + kk;
                float diff = array[ai*ak_size*aj_size + aj*ak_size + ak]
                    - temp[ti*tj_size*tk_size + tj*tk_size + tk];
                sum_array += diff*diff;
            }
        }
    }

    return sum_array / (i_range * j_range * k_range);
        
}

__device__ float sum_mad_norm(
    int ti_start, int tj_start, int tk_start,
    int ai_start, int aj_start, int ak_start,
    int i_range, int j_range, int k_range,
    int aj_size, int ak_size, float *array,
    int tj_size, int tk_size, float *temp
) {

    float sum_diff = 0;
    float sum_temp = 0;
    for (int ii = 0; ii < i_range; ii++) {
        int ti = ti_start + ii;
        int ai = ai_start + ii;
        for (int jj = 0; jj < j_range; jj++) {
            int tj = tj_start + jj;
            int aj = aj_start + jj;
            for (int kk = 0; kk < k_range; kk++) {
                int tk = tk_start + kk;
                int ak = ak_start + kk;
                int a_ind = ai*ak_size*aj_size + aj*ak_size + ak;
                int t_ind = ti*tj_size*tk_size + tj*tk_size + tk;
                sum_diff += abs(array[a_ind] - temp[t_ind]);
                sum_temp += temp[t_ind];
            }
        }
    }
    
    float mean_diff = sum_diff / (i_range * j_range * k_range);
    float mean_temp = sum_temp / (i_range * j_range * k_range);

    return mean_diff / mean_temp;
        
}

__device__ float sum_msd_norm(
    int ti_start, int tj_start, int tk_start,
    int ai_start, int aj_start, int ak_start,
    int i_range, int j_range, int k_range,
    int aj_size, int ak_size, float *array,
    int tj_size, int tk_size, float *temp
) {

    float sum_diff = 0;
    float sum_temp = 0;
    for (int ii = 0; ii < i_range; ii++) {
        int ti = ti_start + ii;
        int ai = ai_start + ii;
        for (int jj = 0; jj < j_range; jj++) {
            int tj = tj_start + jj;
            int aj = aj_start + jj;
            for (int kk = 0; kk < k_range; kk++) {
                int tk = tk_start + kk;
                int ak = ak_start + kk;
                int a_ind = ai*ak_size*aj_size + aj*ak_size + ak;
                int t_ind = ti*tj_size*tk_size + tj*tk_size + tk;
                float diff = array[a_ind] - temp[t_ind];
                sum_diff += diff * diff;
                sum_temp += temp[t_ind] * temp[t_ind];
            }
        }
    }
    
    float mean_diff = sum_diff / (i_range * j_range * k_range);
    float mean_temp = sum_temp / (i_range * j_range * k_range);

    return mean_diff / mean_temp;
        
}

template <typename scalar_t>
__global__ void match_template_mad(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> array,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> temp,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dist_array
) {

    int nax = array.size(1);
    int nay = array.size(2);
    int naz = array.size(3);

    int ib = blockIdx.x * blockDim.x + threadIdx.x;
    int xthreads_per_batch = ceil(float(nax) / blockDim.x) * blockDim.x;

    int b = ib / xthreads_per_batch;
    int i = ib % xthreads_per_batch;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nax && j < nay && k < naz) {

        int ntx = temp.size(0);
        int nty = temp.size(1);
        int ntz = temp.size(2);

        int ti_middle = (ntx - 1) / 2;
        int tj_middle = (nty - 1) / 2;
        int tk_middle = (ntz - 1) / 2;

        int ai_start = max(0, i - ti_middle);
        int aj_start = max(0, j - tj_middle);
        int ak_start = max(0, k - tk_middle);

        int ti_start = max(0, ti_middle - i);
        int tj_start = max(0, tj_middle - j);
        int tk_start = max(0, tk_middle - k);

        int ii_end = min(i + ti_middle + 1, nax) - ai_start;
        int jj_end = min(j + tj_middle + 1, nay) - aj_start;
        int kk_end = min(k + tk_middle + 1, naz) - ak_start;
            
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
                    float diff = abs(array[b][ai][aj][ak] - temp[ti][tj][tk]);
                    sum_array += diff;
                }
            }
        }

        float d = sum_array / (ii_end * jj_end * kk_end);
        dist_array[b][i][j][k] = d;

    }

}

template <typename scalar_t>
__global__ void match_template_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> array,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> temp,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dist_array,
    int b, TMMethod method
) {

    int nax = array.size(1);
    int nay = array.size(2);
    int naz = array.size(3);

    int ntx = temp.size(0);
    int nty = temp.size(1);
    int ntz = temp.size(2);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int ti_middle = (ntx - 1) / 2;
    int tj_middle = (nty - 1) / 2;
    int tk_middle = (ntz - 1) / 2;

    // Shared memory
    extern __shared__ float shmem[];
    float *temp_s = (float*) shmem;
    float *array_s = (float*) &shmem[ntx*nty*ntz];

    // Copy template to shared memory
    for (int ti = threadIdx.x; ti < ntx; ti += blockDim.x) {
        for (int tj = threadIdx.y; tj < nty; tj += blockDim.y) {
            for (int tk = threadIdx.z; tk < ntz; tk += blockDim.z) {
                temp_s[ti*nty*ntz + tj*ntz + tk] = temp[ti][tj][tk];
            }
        }
    }

    // Copy subarray to shared memory
    int i_start = blockIdx.x * blockDim.x;
    int j_start = blockIdx.y * blockDim.y;
    int k_start = blockIdx.z * blockDim.z;
    int ai_start = max(0, i_start - ti_middle);
    int aj_start = max(0, j_start - tj_middle);
    int ak_start = max(0, k_start - tk_middle);
    int i_range = min(i_start + ti_middle + int(blockDim.x), nax) - ai_start;
    int j_range = min(j_start + tj_middle + int(blockDim.y), nay) - aj_start;
    int k_range = min(k_start + tk_middle + int(blockDim.z), naz) - ak_start;
    for (int ii = threadIdx.x; ii < i_range; ii += blockDim.x) {
        int ai = ai_start + ii;
        for (int jj = threadIdx.y; jj < j_range; jj += blockDim.y) {
            int aj = aj_start + jj;
            for (int kk = threadIdx.z; kk < k_range; kk += blockDim.z) {
                int ak = ak_start + kk;
                array_s[ii*k_range*j_range + jj*k_range + kk] = array[b][ai][aj][ak];
            }
        }
    }

    __syncthreads();

    if (i < nax && j < nay && k < naz) {

        int tsi_start = max(0, ti_middle - i);
        int tsj_start = max(0, tj_middle - j);
        int tsk_start = max(0, tk_middle - k);

        int asi_start = max(0, int(threadIdx.x) + min(0, i_start - ti_middle));
        int asj_start = max(0, int(threadIdx.y) + min(0, j_start - tj_middle));
        int ask_start = max(0, int(threadIdx.z) + min(0, k_start - tk_middle));

        int si_range = min(i + ti_middle + 1, nax) - max(0, i - ti_middle);
        int sj_range = min(j + tj_middle + 1, nay) - max(0, j - tj_middle);
        int sk_range = min(k + tk_middle + 1, naz) - max(0, k - tk_middle);

        if (method == TM_MAD) {
            dist_array[b][i][j][k] = sum_mad(
                tsi_start, tsj_start, tsk_start,
                asi_start, asj_start, ask_start,
                si_range, sj_range, sk_range,
                j_range, k_range, array_s,
                nty, ntz, temp_s
            );
        } else if (method == TM_MSD) {
            dist_array[b][i][j][k] = sum_msd(
                tsi_start, tsj_start, tsk_start,
                asi_start, asj_start, ask_start,
                si_range, sj_range, sk_range,
                j_range, k_range, array_s,
                nty, ntz, temp_s
            );
        } else if (method == TM_MAD_NORM) {
            dist_array[b][i][j][k] = sum_mad_norm(
                tsi_start, tsj_start, tsk_start,
                asi_start, asj_start, ask_start,
                si_range, sj_range, sk_range,
                j_range, k_range, array_s,
                nty, ntz, temp_s
            );
        } else if (method == TM_MSD_NORM) {
            dist_array[b][i][j][k] = sum_msd_norm(
                tsi_start, tsj_start, tsk_start,
                asi_start, asj_start, ask_start,
                si_range, sj_range, sk_range,
                j_range, k_range, array_s,
                nty, ntz, temp_s
            );
        }
        

    }

}

torch::Tensor match_template_cuda(torch::Tensor array, torch::Tensor temp, TMMethod method) {

    cudaSetDevice(array.device().index());

    if (temp.size(0) % 2 == 0 || temp.size(1) % 2 == 0 || temp.size(2) % 2 == 0) {
        throw std::invalid_argument("Template dimensions must all be odd size.");
    }

    auto dist_array = torch::zeros_like(array);

    dim3 threads(10, 10, 2);
    dim3 blocks(
        ceil(float(array.size(1)) / threads.x),
        ceil(float(array.size(2)) / threads.y),
        ceil(float(array.size(3)) / threads.z)
    );
    int batch_size = array.size(0);

    int temp_size = temp.size(0) * temp.size(1) * temp.size(2);
    int subarray_size = (temp.size(0)+threads.x-1) * (temp.size(1)+threads.y-1) * (temp.size(2)+threads.z-1);

    for (int batch_ind = 0; batch_ind < batch_size; batch_ind++) {
        AT_DISPATCH_FLOATING_TYPES(array.type(), "match_template_kernel", ([&] {
            match_template_kernel<scalar_t><<<blocks, threads, (temp_size+subarray_size)*sizeof(float)>>>(
                array.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                temp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                dist_array.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                batch_ind, method
            );
        }));
    }

    return dist_array;

}
