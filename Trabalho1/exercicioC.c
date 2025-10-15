// Incluímos as bibliotecas de sempre: I/O, alocação de memória, MPI e tempo.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Uma função auxiliar para imprimir vetores de forma legível.
// O 'label' nos ajuda a identificar o que está sendo impresso.
void imprimir_vetor(int *vetor, int n, char *label) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        // Um truque para não imprimir a vírgula depois do último elemento.
        printf("%d%s", vetor[i], (i == n - 1) ? "" : ", ");
    }
    printf("]\n");
}

// Funções auxiliares para preencher a matriz e o vetor com dados aleatórios.
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

// Função principal.
int main(int argc, char **argv) {
    int rank, size;
    int n;
    int *matriz_a = NULL, *vetor_b = NULL, *vetor_c = NULL;
    double start_time, end_time;

    // Inicialização padrão do ambiente MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Mensagem de "Olá" de cada processo.
    printf("[Processo %d de %d]: Olá!\n", rank, size);

    // Validação de entrada (verificando se 'n' foi fornecido e é divisível).
    if (argc < 2) {
        if (rank == 0)
            printf("Uso: mpirun -np <procs> %s <tamanho_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    n = atoi(argv[1]);

    if (n % size != 0) {
        if (rank == 0)
            printf(
                "O tamanho (n) deve ser divisível pelo número de processos.\n");
        MPI_Finalize();
        return 1;
    }

    // Calculando a carga de trabalho.
    int linhas_por_proc = n / size;
    int elementos_mat_por_proc = linhas_por_proc * n;

    // Alocação de memória local em CADA processo.
    // 'sub_a' para a fatia da matriz.
    int *sub_a = (int *)malloc(elementos_mat_por_proc * sizeof(int));
    // 'vetor_b' para a CÓPIA COMPLETA do vetor. Todos precisam dele.
    vetor_b = (int *)malloc(n * sizeof(int));
    // 'sub_c' para a fatia do vetor resultante que cada processo irá calcular.
    int *sub_c = (int *)malloc(linhas_por_proc * sizeof(int));

    // APENAS o processo 0 (mestre) prepara os dados iniciais.
    if (rank == 0) {
        printf(
            "[Processo %d]: Vou inicializar a matriz A (%dx%d) e o vetor B "
            "(tamanho %d).\n",
            rank, n, n, n);
        // Aloca a matriz A completa e o vetor de resultado C.
        matriz_a = (int *)malloc(n * n * sizeof(int));
        vetor_c = (int *)malloc(n * sizeof(int));
        srand(time(NULL));
        // Preenche a matriz A e o vetor B com dados.
        preencher_matriz(matriz_a, n);
        preencher_vetor(vetor_b, n);
    }

    // Sincroniza e inicia o cronômetro.
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    if (rank == 0) {
        printf(
            "[Processo %d]: Distribuindo linhas de A (Scatter) e enviando "
            "vetor B para todos (Broadcast).\n",
            rank);
    }

    // Fase de DISTRIBUIÇÃO - aqui temos duas operações:
    // 1. Scatter: A matriz A é fatiada e distribuída. Cada processo recebe suas
    // próprias linhas.
    MPI_Scatter(matriz_a, elementos_mat_por_proc, MPI_INT, sub_a,
                elementos_mat_por_proc, MPI_INT, 0, MPI_COMM_WORLD);
    // 2. Broadcast: O vetor B é "transmitido" do processo 0 para TODOS os
    // outros. Agora, cada processo tem uma cópia idêntica do vetor B.
    MPI_Bcast(vetor_b, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Fase de COMPUTAÇÃO LOCAL.
    printf(
        "[Processo %d]: Recebi minhas %d linhas e o vetor B completo. "
        "Calculando minha parte do vetor resultante.\n",
        rank, linhas_por_proc);
    // Cada processo agora tem os ingredientes que precisa: suas linhas da
    // matriz A ('sub_a') e o vetor B completo. O loop aninhado calcula o
    // produto escalar de cada linha pela vetor, gerando os elementos parciais
    // do vetor de resultado 'sub_c'.
    for (int i = 0; i < linhas_por_proc; i++) {
        sub_c[i] = 0;
        for (int j = 0; j < n; j++) {
            sub_c[i] += sub_a[i * n + j] * vetor_b[j];
        }
    }

    // Cada processo mostra o resultado parcial que calculou.
    printf("[Processo %d]: Cálculo local finalizado. ", rank);
    imprimir_vetor(sub_c, linhas_por_proc, "Meus elementos do vetor C são");

    // Fase de CONSOLIDAÇÃO (Gather).
    // Funciona como no primeiro exemplo: todos enviam seus resultados parciais
    // ('sub_c') para o processo 0, que os junta em ordem para formar o
    // 'vetor_c' final.
    MPI_Gather(sub_c, linhas_por_proc, MPI_INT, vetor_c, linhas_por_proc,
               MPI_INT, 0, MPI_COMM_WORLD);

    // Sincroniza e para o cronômetro.
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // APENAS o processo 0 tem o resultado final.
    if (rank == 0) {
        printf(
            "[Processo %d]: Recebi os resultados de todos e montei o vetor "
            "final.\n",
            rank);
        printf("Tempo de execução da multiplicação: %f segundos\n",
               end_time - start_time);

        // Uma verificação para não imprimir vetores enormes no console.
        if (n <= 20) {
            imprimir_vetor(vetor_c, n, "Vetor Resultante Final");
        } else {
            printf(
                "O vetor resultante é muito grande (tamanho %d) para ser "
                "impresso.\n",
                n);
        }

        // Libera a memória dos vetores/matrizes grandes.
        free(matriz_a);
        free(vetor_c);
    }

    // TODOS os processos liberam a memória que alocaram localmente.
    free(sub_a);
    free(vetor_b);
    free(sub_c);

    // Mensagem de despedida e finalização do ambiente MPI.
    printf("[Processo %d]: Finalizando.\n", rank);
    MPI_Finalize();
    return 0;
}