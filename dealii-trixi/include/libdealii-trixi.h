#ifndef LIBDEAL_H_
#define LIBDEAL_H_

#include "intercomm.h"

#ifdef __cplusplus
extern "C" {
#endif

void * init_libs(int argc, char **argv);
void * init_2d(int polydeg, int MeshSettings2D, int maxiters, double abstol);
void * init_3d(int polydeg, int MeshSettings3D, int maxiters, double abstol);
unsigned long get_problem_size_on_rank(void *wrapped);
void repartition(void *wrapped);
void set_maxiters(void *wrapped, int maxiters);
void set_abstol(void *wrapped, double abstol);
void set_reltol(void *wrapped, double reltol);
double * get_pointer_f(void *wrapped);
double * get_pointer_u(void *wrapped);
double ** get_pointer_grad_u(void *wrapped);
void solve(void *wrapped);
int * get_pointer_amr_indicator(void *wrapped);
void adapt_grid(void *wrapped);
unsigned int get_mesh_checksum(void *wrapped);
void finalize(void *wrapped);
void finalize_libs(void *mpi);

#ifdef __cplusplus
}
#endif

#endif /* ! LIBDEAL_H_ */
