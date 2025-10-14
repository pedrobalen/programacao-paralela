
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void imprimir_vetor(int *vetor, int n, char *label) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%d%s", vetor[i], (i == n - 1) ? "" : ", ");
    }
    printf("]\n");
}

void preencher_matriz(int *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = rand() % 10;
    }
}

void preencher_vetor(int *vetor, int n) {
    for (int i = 0; i < n; i++) {
        vetor[i] = rand() % 10;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int n;
    int *matriz_a = NULL, *vetor_b = NULL, *vetor_c = NULL;
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
        if (rank == 0) printf("O tamanho (n) deve ser divisível pelo número de processos.\n");
        MPI_Finalize();
        return 1;
    }

    int linhas_por_proc = n / size;
    int elementos_mat_por_proc = linhas_por_proc * n;

    int *sub_a = (int *)malloc(elementos_mat_por_proc * sizeof(int));
    vetor_b = (int *)malloc(n * sizeof(int)); 
    int *sub_c = (int *)malloc(linhas_por_proc * sizeof(int));

    if (rank == 0) {
        printf("[Processo %d]: Vou inicializar a matriz A (%dx%d) e o vetor B (tamanho %d).\n", rank, n, n, n);
        matriz_a = (int *)malloc(n * n * sizeof(int));
        vetor_c = (int *)malloc(n * sizeof(int)); 
        srand(time(NULL));
        preencher_matriz(matriz_a, n);
        preencher_vetor(vetor_b, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    if(rank == 0){
        printf("[Processo %d]: Distribuindo linhas de A (Scatter) e enviando vetor B para todos (Broadcast).\n", rank);
    }

    MPI_Scatter(matriz_a, elementos_mat_por_proc, MPI_INT, sub_a, elementos_mat_por_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vetor_b, n, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[Processo %d]: Recebi minhas %d linhas e o vetor B completo. Calculando minha parte do vetor resultante.\n", rank, linhas_por_proc);
    for (int i = 0; i < linhas_por_proc; i++) {
        sub_c[i] = 0;
        for (int j = 0; j < n; j++) {
            sub_c[i] += sub_a[i * n + j] * vetor_b[j];
        }
    }

    printf("[Processo %d]: Cálculo local finalizado. ", rank);
    imprimir_vetor(sub_c, linhas_por_proc, "Meus elementos do vetor C são");

    MPI_Gather(sub_c, linhas_por_proc, MPI_INT, vetor_c, linhas_por_proc, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("[Processo %d]: Recebi os resultados de todos e montei o vetor final.\n", rank);
        printf("Tempo de execução da multiplicação: %f segundos\n", end_time - start_time);
        
        if (n <= 20) { 
            imprimir_vetor(vetor_c, n, "Vetor Resultante Final");
        } else {
            printf("O vetor resultante é muito grande (tamanho %d) para ser impresso.\n", n);
        }

        free(matriz_a);
        free(vetor_c);
    }

    free(sub_a);
    free(vetor_b);
    free(sub_c);

    printf("[Processo %d]: Finalizando.\n", rank);
    MPI_Finalize();
    return 0;
}
