#include <omp.h>
#include <bits/stdc++.h>
#include <stdio.h>      
#include <stdlib.h>  
#include <math.h>
#include <time.h>   

using namespace std;

#define MAX_MATRIX_SIZE 65536
#define DEBUG false
#define BLOCK_SIZE 16

using namespace std;

int matrix_size, terminal_matrix_size; 



void print(int n, int** mat)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


int** getSlice(int n, int** mat, int offseti, int offsetj)
{
    int m = n / 2;
    
    int** slice = (int**)malloc(n * sizeof(int*));
    int* data_slice = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        slice[i] = &(data_slice[n * i]);
    }
    
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}

int** addMatrices(int n, int** mat1, int** mat2, bool add)
{
    int** result = (int**)malloc(n * sizeof(int*));
    int* data_result = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        result[i] = &(data_result[n * i]);
    }
    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (add)
                result[i][j] = mat1[i][j] + mat2[i][j];
            else
                result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return result;
}

int** combineMatrices(int m, int** c11, int** c12, int** c21, int** c22)
{
    int n = 2 * m;
    
    int** result = (int**)malloc(n * sizeof(int*));
    int* data_result = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        result[i] = &(data_result[n * i]);
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j];
            else if (i < m)
                result[i][j] = c12[i][j - m];
            else if (j < m)
                result[i][j] = c21[i - m][j];
            else
                result[i][j] = c22[i - m][j - m];
        }
    }
    
    return result;
}


int** naive(int n, int** mat1, int** mat2)
{
    int** prod = (int**)malloc(n * sizeof(int*));
    int* data_prod = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        prod[i] = &(data_prod[n * i]);
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return prod;
}

__global__ void multiply(int *left, int *right, int *res, int dim) {

    int i,j;
    int temp = 0;

    __shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        // Column j of matrix left
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;
        // Load left[i][j] to shared mem

        Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
        // Load right[i][j] to shared mem

        Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of res from tiles of left and right in shared mem
        for (int k = 0; k < BLOCK_SIZE; k++) {

            temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    res[row * dim + col] = temp;
}

int** cudaNaive(int n, int** mat1, int** mat2)
{
    size_t bytes;
    bytes = n*n * sizeof(int);
    
    int* h_mat1 = (int*)malloc(bytes);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            h_mat1[i*n + j] = mat1[i][j];
            //cout<< h_mat1[i*n +j]<<" ";
        }
        //cout<<endl;
    }

    int* h_mat2 = (int*)malloc(bytes);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            h_mat2[i*n + j] = mat2[i][j];
           //cout<< h_mat2[i*n +j]<<" ";
        }
        //cout<<endl;
    }
    
    

    int* h_product = (int*)malloc(bytes);
    int *d_mat1, *d_mat2, *d_product;

    cudaMalloc((void**)&d_mat1, bytes);
    cudaMalloc((void**)&d_mat2, bytes);
    cudaMalloc((void**)&d_product, bytes);

    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_product, h_product, bytes, cudaMemcpyHostToDevice);
    
    int block_size = min(n, 16);
    //dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Block_dim(block_size, block_size);
    //Grid dimension is found by dividing matrix dimension to block_size
    dim3 Grid_dim(n / block_size, n / block_size);

    multiply<<<Grid_dim, Block_dim>>>(d_mat1, d_mat2, d_product, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_product, d_product, bytes, cudaMemcpyDeviceToHost);

    int** product = (int**)malloc(n * sizeof(int*));
    int* data_product = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        product[i] = &(data_product[n * i]);
    }
    
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            // cout<< h_product[i*n+j]<<" ";
            product[i][j] = h_product[i*n + j];
        }
    }
    
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_product);

    free(h_mat1);
    free(h_mat2);
    
    return product;
}


int** strassen(int n, int** mat1, int** mat2)
{

    if (n <= terminal_matrix_size)
    {
        return cudaNaive(n, mat1, mat2);
    }

    int m = n / 2;

    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    int** bds = addMatrices(m, b, d, false);
    int** gha = addMatrices(m, g, h, true);
    int** s1 = strassen(m, bds, gha);

    free(bds[0]); free(bds);

    free(gha[0]); free(gha);

    int** ada = addMatrices(m, a, d, true);
    int** eha = addMatrices(m, e, h, true);
    int** s2 = strassen(m, ada, eha);
    
    free(ada[0]); free(ada);

    free(eha[0]); free(eha);

    int** acs = addMatrices(m, a, c, false);
    int** efa = addMatrices(m, e, f, true);
    int** s3 = strassen(m, acs, efa);

    free(acs[0]); free(acs);

    free(efa[0]); free(efa);
    
    int** aba = addMatrices(m, a, b, true);
    int** s4 = strassen(m, aba, h);

    free(aba[0]); free(aba);
    free(b[0]); free(b);
    
    int** fhs = addMatrices(m, f, h, false);
    int** s5 = strassen(m, a, fhs);

    free(fhs[0]); free(fhs);
    free(a[0]); free(a);
    free(f[0]); free(f);
    free(h[0]); free(h);

    int** ges = addMatrices(m, g, e, false);
    int** s6 = strassen(m, d, ges);

    free(ges[0]); free(ges);
    free(g[0]); free(g);

    int** cda = addMatrices(m, c, d, true);
    int** s7 = strassen(m, cda, e);

    free(cda[0]); free(cda);
    free(c[0]); free(c);
    free(d[0]); free(d);
    free(e[0]); free(e);

    int** s1s2a = addMatrices(m, s1, s2, true);
    int** s6s4s = addMatrices(m, s6, s4, false);
    int** c11 = addMatrices(m, s1s2a, s6s4s, true);
    
    free(s1s2a[0]); free(s1s2a);
    free(s6s4s[0]); free(s6s4s);
    free(s1[0]); free(s1);

    int** c12 = addMatrices(m, s4, s5, true);
    free(s4[0]); free(s4);

    int** c21 = addMatrices(m, s6, s7, true);
    free(s6[0]); free(s6);

    int** s2s3s = addMatrices(m, s2, s3, false);
    int** s5s7s = addMatrices(m, s5, s7, false);
    int** c22 = addMatrices(m, s2s3s, s5s7s, true);
    
    free(s2s3s[0]); free(s2s3s);
    free(s5s7s[0]); free(s5s7s);
    free(s2[0]); free(s2);
    free(s3[0]); free(s3);
    free(s5[0]); free(s5);
    free(s7[0]); free(s7);

    int** prod = combineMatrices(m, c11, c12, c21, c22);
    
    free(c11[0]); free(c11);
    free(c12[0]); free(c12);
    free(c21[0]); free(c21);
    free(c22[0]); free(c22);

    return prod;
}

int main(int argc, char *argv[]){
    // variables declaration
    int k, k_bar, n; //
    struct timespec start, stop, stop_naive;
    double total_time, total_time_naive;  
    int error = 0;
    
    if (argc != 3){
        printf("Please Enter the Size of Matrix and Terminal Matrix Size!\n");
        exit(0);
    }
    else{
        k = atoi(argv[argc-2]);
        if ((n = 1 << k) > MAX_MATRIX_SIZE){
            printf("Exceed Maximum Matrix Size: %d!\n", MAX_MATRIX_SIZE);
            exit(0);
        }
        k_bar = atoi(argv[argc-1]);
        if ((terminal_matrix_size = 1 << (k-k_bar)) < 1){
            printf("Terminal matrix size cannot be bigger than the actual matrix size!!\n");
            exit(0);
        }
    } 
    // cout<<terminal_matrix_size<<endl;
    matrix_size = n;
    
    // malloc matrices
    int** a = (int**)malloc(n * sizeof(int*));    
    int* data_a = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        a[i] = &(data_a[n * i]);
    }
    
    
    int** b = (int**)malloc(n * sizeof(int*));    
    int* data_b = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        b[i] = &(data_b[n * i]);
    }
    
    int** prod = (int**)malloc(n * sizeof(int*));    
    int* data_prod = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        prod[i] = &(data_prod[n * i]);
    }
    
    int** prod_naive = (int**)malloc(n * sizeof(int*));    
    int* data_prod_n = (int*)malloc(n * n * sizeof(int));
    
    for (int i = 0; i < n; i++)
    {
        prod_naive[i] = &(data_prod_n[n * i]);
    }

    // put random numbers as element in the intial matrices a & b
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = rand() % 5;
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[i][j] = rand() % 5;
        }
    }

    //
    //printf("Start Parallel Matrix Multiplication Execution!\n");

    // set up starting time
    clock_gettime(CLOCK_REALTIME, &start);

    prod = strassen(n, a, b);
    if(DEBUG) print(n, prod);

    // calculate the execution time for the strassen algorithm
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000000001*(stop.tv_nsec-start.tv_nsec);
    
    // calculate the execution time for the naive algorithm
    //printf("Start Naive Matrix Multiplication!\n");
    prod_naive = naive(n, a, b);
    if(DEBUG) print(n, prod_naive);
    clock_gettime(CLOCK_REALTIME, &stop_naive);
    total_time_naive = (stop_naive.tv_sec-stop.tv_sec)
	+0.000000001*(stop_naive.tv_nsec-stop.tv_nsec);

    // check answer here
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (prod_naive[i][j] != prod[i][j]) error = 1;
        }
    }

    if (error != 0) printf("Incorrect Answer!!\n");

    printf("Matrix Size = %d, Terminal Matrix Size = %d, error = %d, time_strassen (sec) = %8.5f, time_naive = %8.5f\n", 
	    n, terminal_matrix_size, error, total_time, total_time_naive);

    free(a); free(b); free(prod); free(prod_naive);

    return 0;
}