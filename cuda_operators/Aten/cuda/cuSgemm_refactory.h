#ifndef __CUSGEMM_REFACTORY_H
#define __CUSGEMM_REFACTORY_H

int cuSgemm_refactory(int m, int n, int k,
                const float           *alpha,
                const float           *A, int lda,
                const float           *B, int ldb,
                const float           *beta,
                float           *C, int ldc);

#endif

