
#include <vector>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <defs.h>

__device__ int find_root(const int* __restrict__ labels, int a) {
    int l = labels[a];
    while (l != a) {
        a = l;
        l = labels[a];
    }
    return a;
}

__device__ void tree_union(int* __restrict__ labels, int a, int b) {
    bool done = false;
    int old;
    while (!done) {
        a = find_root(labels, a);
        b = find_root(labels, b);
        if (a < b) {
            old = atomicMin(&labels[b], a);
            done = (old == b);
            b = old;
        } else if (b < a){
            old = atomicMin(&labels[a], b);
            done = (old == a);
            a = old;
        } else {
            break;
        }
    }
}

__global__ void init_labels(
    const int* __restrict__ array,
    int* __restrict__ labels,
    int d, int w, int h
) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < d && j < w && k < h) {
        int a = i*w*h + j*h + k;
        labels[a] = (array[a] == 1) ? a : -1;
    }


}

__global__ void merge(
    const int* __restrict__ array,
    int* __restrict__ labels,
    int d, int w, int h
) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < d && j < w && k < h) {

        int wh = w*h;
        int a = i*wh + j*h + k;

        bool js = (j > 0);
        bool ks = (k > 0);
        bool je = (j < (w - 1));
        bool ke = (k < (h - 1));
        
        int n;
        if (array[a] == 1) {
            if (i > 0) {
                n = a - wh; if (array[n] == 1) {tree_union(labels, a, n);}
                n = a - wh - h; if (js && array[n] == 1) {tree_union(labels, a, n);}
                n = a - wh + h; if (je && array[n] == 1) {tree_union(labels, a, n);}
                if (ks) {
                    n = a - wh - 1; if (array[n] == 1) {tree_union(labels, a, n);}
                    n = a - wh - h - 1; if (js && array[n] == 1) {tree_union(labels, a, n);}
                    n = a - wh + h - 1; if (je && array[n] == 1) {tree_union(labels, a, n);}
                }
                if (ke) {
                    n = a - wh + 1; if (array[n] == 1) {tree_union(labels, a, n);}
                    n = a - wh - h + 1; if (js && array[n] == 1) {tree_union(labels, a, n);}
                    n = a - wh + h + 1; if (je && array[n] == 1) {tree_union(labels, a, n);}
                }
            }
            if (js) {
                n = a - h; if (array[n] == 1) {tree_union(labels, a, n);}
                n = a - h - 1; if (ks && array[n] == 1) {tree_union(labels, a, n);}
                n = a - h + 1; if (ke && array[n] == 1) {tree_union(labels, a, n);}
            }
            n = a - 1; if (ks && array[n] == 1) {tree_union(labels, a, n);}
        }

    }

}

__global__ void compress(
    const int* __restrict__ array,
    int* __restrict__ labels,
    int d, int w, int h
) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < d && j < w && k < h) {
        int a = i*w*h + j*h + k;
        if (array[a] == 1) {
            int b = a;
            int l = labels[b];
            while (l != b) {
                b = l;
                l = labels[b];
                labels[a] = b;
            }
        }
    }

}

__global__ void find_min_inds(
    const float* __restrict__ array,
    const int* __restrict__ labels,
    int* __restrict__ max_ind_array,
    int d, int w, int h,
    int* __restrict__ counter, int* __restrict__ unique_labels
) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i < d && j < w && k < h) {

        int a = i*w*h + j*h + k;
        int root = labels[a];

        if (root >= 0) {

            int *address = &max_ind_array[root];
            int old_ind = max_ind_array[root];
            float new_val = array[a];
            int assumed;
            float old_val;
            do {
                if (old_ind < 0 || new_val < array[old_ind]) {
                    assumed = old_ind;
                    old_ind = atomicCAS(address, assumed, a);
                } else {
                    break;
                }
            } while (assumed != old_ind);

            if (a == root) {
                int label_ind = atomicAdd(counter, 1);
                unique_labels[label_ind] = a;
            }

        }
    }

}

__global__ void cpy_max_inds(
    int* __restrict__ max_inds,
    int* __restrict__ max_ind_array,
    int* __restrict__ unique_labels,
    int* __restrict__ counter,
    int d, int w, int h
) {

    for (int i = threadIdx.x; i < *counter; i += blockDim.x) {
        int ind = max_ind_array[unique_labels[i]];
        max_inds[3*i  ] = ind / (w*h);
        max_inds[3*i+1] = (ind / h) % w;
        max_inds[3*i+2] = ind % h;
    }

}

torch::Tensor ccl_cuda(torch::Tensor array) {

    torch::Tensor labels = torch::empty_like(array);

    dim3 threads(8, 8, 8);
    dim3 blocks(
        ceil(float(array.size(1)) / threads.x),
        ceil(float(array.size(2)) / threads.y),
        ceil(float(array.size(3)) / threads.z)
    );

    int batch_size = array.size(0);
    int d = array.size(1);
    int w = array.size(2);
    int h = array.size(3);

    for (int b = 0; b < batch_size; b++) {

        int* array_ptr = (int*) &array.data<int>()[b*d*w*h];
        int* label_ptr = (int*) &labels.data<int>()[b*d*w*h];

        init_labels<<<blocks, threads>>>(
            array_ptr, label_ptr,
            d, w, h
        );

        merge<<<blocks, threads>>>(
            array_ptr, label_ptr,
            d, w, h
        );

        compress<<<blocks, threads>>>(
            array_ptr, label_ptr,
            d, w, h
        );

    }

    return labels;

};

std::vector<torch::Tensor> find_label_min_cuda(torch::Tensor array, torch::Tensor labels) {

    cudaSetDevice(array.device().index());

    dim3 threads(8, 8, 8);
    dim3 blocks(
        ceil(float(array.size(1)) / threads.x),
        ceil(float(array.size(2)) / threads.y),
        ceil(float(array.size(3)) / threads.z)
    );

    int batch_size = labels.size(0);
    int d = labels.size(1);
    int w = labels.size(2);
    int h = labels.size(3);

    std::vector<torch::Tensor> max_inds;
    torch::Tensor max_ind_array = torch::full_like(labels, -1);
    
    for (int b = 0; b < batch_size; b++) {
        
        int counter = 0;
        int* counter_d;
        cudaMalloc(&counter_d, sizeof(int));
        cudaMemcpy(counter_d, &counter, sizeof(int), cudaMemcpyHostToDevice);

        torch::Tensor unique_labels = torch::zeros(MAX_LABELS,
            torch::TensorOptions().dtype(torch::kInt32).device(array.device()));

        float* array_ptr = (float*) &array.data<float>()[b*d*w*h];
        int* label_ptr = (int*) &labels.data<int>()[b*d*w*h];
        int* max_ind_array_ptr = (int*) &max_ind_array.data<int>()[b*d*w*h];

        find_min_inds<<<blocks, threads>>>(
            array_ptr, label_ptr,max_ind_array_ptr,
            d, w, h, counter_d, unique_labels.data<int>()
        );

        cudaMemcpy(&counter, counter_d, sizeof(int), cudaMemcpyDeviceToHost);
        assert (counter <= MAX_LABELS);

        torch::Tensor max_inds_ = torch::zeros({counter, 3},
            torch::TensorOptions().dtype(torch::kInt32).device(array.device()));

        cpy_max_inds<<<1, 32>>>(
            max_inds_.data<int>(), max_ind_array_ptr,
            unique_labels.data<int>(), counter_d, 
            d, w, h
        );

        max_inds.push_back(max_inds_);

    }

    return max_inds;

}