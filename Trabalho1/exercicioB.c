#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <limits.h>

void preencher_matriz(int *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = rand() % (n * n);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int n;
    int *matriz_a = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("[Processo %d de %d]: Olá!\n", rank, size);

    if (argc < 2) {
        if (rank == 0) printf("Uso: mpirun -np <procs> %s <tamanho_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    n = atoi(argv[1]);

    if (n % size != 0) {
        if (rank == 0) printf("O tamanho da matriz (n) deve ser divisível pelo número de processos.\n");
        MPI_Finalize();
        return 1;
    }

    int linhas_por_proc = n / size;
    int elementos_por_proc = linhas_por_proc * n;

    int *sub_a = (int *)malloc(elementos_por_proc * sizeof(int));

    if (rank == 0) {
        printf("[Processo %d]: Vou inicializar a matriz de tamanho %dx%d.\n", rank, n, n);
        matriz_a = (int *)malloc(n * n * sizeof(int));
        srand(time(NULL));
        preencher_matriz(matriz_a, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    if (rank == 0) {
        printf("[Processo %d]: Distribuindo as linhas da matriz para todos...\n", rank);
    }
    MPI_Scatter(matriz_a, elementos_por_proc, MPI_INT, sub_a, elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[Processo %d]: Recebi minhas linhas. Procurando pelo maior valor local.\n", rank);
    
    int max_local = INT_MIN;
    for (int i = 0; i < elementos_por_proc; i++) {
        if (sub_a[i] > max_local) {
            max_local = sub_a[i];
        }
    }
    
    printf("[Processo %d]: Meu maior valor local é %d. Enviando para a redução...\n", rank, max_local);

    int max_global;
    MPI_Reduce(&max_local, &max_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("[Processo %d]: Operação de redução concluída.\n", rank);
        printf("Tempo de execução para encontrar o maior valor: %f segundos\n", end_time - start_time);
        printf("O maior valor global encontrado foi: %d\n", max_global);
        free(matriz_a);
    }

    free(sub_a);
    printf("[Processo %d]: Finalizando.\n", rank);
    MPI_Finalize();
    return 0;
}
