#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void preencher_matriz(int *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = rand() % 10;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int n; 
    int *matriz_a = NULL, *matriz_b = NULL, *matriz_c = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Uso: mpirun -np <procs> %s <tamanho_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    n = atoi(argv[1]);

    if (n % size != 0) {
        if (rank == 0) {
            printf("O tamanho da matriz (n) deve ser divisível pelo número de processos.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int linhas_por_proc = n / size;
    int elementos_por_proc = linhas_por_proc * n;

    int *sub_a = (int *)malloc(elementos_por_proc * sizeof(int));
    int *sub_b = (int *)malloc(elementos_por_proc * sizeof(int));
    int *sub_c = (int *)malloc(elementos_por_proc * sizeof(int));

    if (rank == 0) {
        matriz_a = (int *)malloc(n * n * sizeof(int));
        matriz_b = (int *)malloc(n * n * sizeof(int));
        matriz_c = (int *)malloc(n * n * sizeof(int));
        srand(time(NULL));
        preencher_matriz(matriz_a, n);
        preencher_matriz(matriz_b, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();


    MPI_Scatter(matriz_a, elementos_por_proc, MPI_INT, sub_a, elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(matriz_b, elementos_por_proc, MPI_INT, sub_b, elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < elementos_por_proc; i++) {
        sub_c[i] = sub_a[i] + sub_b[i];
    }

    MPI_Gather(sub_c, elementos_por_proc, MPI_INT, matriz_c, elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Tempo de execução da soma para n=%d com %d processos: %f segundos\n", n, size, end_time - start_time);
        free(matriz_a);
        free(matriz_b);
        free(matriz_c);
    }

    free(sub_a);
    free(sub_b);
    free(sub_c);

    MPI_Finalize();
    return 0;
}
