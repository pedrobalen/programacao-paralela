// Começamos incluindo as bibliotecas necessárias.
// As três primeiras (stdio, stdlib, mpi) são para entrada/saída,
// alocação de memória e, claro, as funções de paralelismo do MPI.
// A 'time.h' é usada para gerar números aleatórios diferentes a cada execução.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Uma função simples para preencher uma matriz com números aleatórios entre 0
// e 9. É uma função auxiliar para não poluir o código principal.
void preencher_matriz(int *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = rand() % 10;
    }
}

// A função principal, onde toda a lógica acontece.
int main(int argc, char **argv) {
    // Variáveis do MPI: 'rank' é o ID único de um processo (ex: 0, 1, 2...),
    // e 'size' é o número total de processos que foram iniciados.
    int rank, size;
    int n;  // Tamanho da matriz (n x n)
    // Ponteiros para as matrizes. As matrizes completas só existirão no
    // processo "mestre".
    int *matriz_a = NULL, *matriz_b = NULL, *matriz_c = NULL;
    // Variáveis para medir o tempo de execução.
    double start_time, end_time;

    // A primeira coisa a se fazer em qualquer programa MPI: inicializar o
    // ambiente. Pense nisso como "ligar" o sistema de comunicação entre os
    // processos.
    MPI_Init(&argc, &argv);
    // Cada processo pergunta: "Qual é o meu ID (rank)?"
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // E também: "Quantos processos somos no total (size)?"
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Validação de entrada: verificamos se o usuário passou o tamanho 'n' da
    // matriz como um argumento ao executar o programa. Apenas o processo 0
    // avisa o usuário.
    if (argc < 2) {
        if (rank == 0)
            printf("Uso: mpirun -np <procs> %s <tamanho_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    // Convertemos o argumento de texto para um número inteiro.
    n = atoi(argv[1]);

    // Uma verificação crucial para este algoritmo simples: o número de linhas
    // (n) precisa ser perfeitamente divisível pelo número de processos. Isso
    // garante que todo mundo receba a mesma quantidade de trabalho.
    if (n % size != 0) {
        if (rank == 0) {
            printf(
                "O tamanho da matriz (n) deve ser divisível pelo número de "
                "processos.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculando a divisão do trabalho.
    int linhas_por_proc =
        n / size;  // Quantas linhas da matriz cada processo vai cuidar.
    int elementos_por_proc =
        linhas_por_proc * n;  // Total de números (células) por processo.

    // Alocação de memória. IMPORTANTE: esta parte é executada por TODOS os
    // processos. Cada processo prepara três "caixas" (buffers) de memória para
    // receber sua fatia da matriz A, sua fatia da matriz B e para guardar sua
    // fatia do resultado C.
    int *sub_a = (int *)malloc(elementos_por_proc * sizeof(int));
    int *sub_b = (int *)malloc(elementos_por_proc * sizeof(int));
    int *sub_c = (int *)malloc(elementos_por_proc * sizeof(int));

    // Agora, apenas o processo "mestre" (rank 0) faz o trabalho de preparação.
    if (rank == 0) {
        // Ele aloca memória para as matrizes completas.
        matriz_a = (int *)malloc(n * n * sizeof(int));
        matriz_b = (int *)malloc(n * n * sizeof(int));
        matriz_c = (int *)malloc(n * n * sizeof(int));
        // E preenche as matrizes A e B com dados aleatórios.
        srand(time(NULL));
        preencher_matriz(matriz_a, n);
        preencher_matriz(matriz_b, n);
    }

    // A partir daqui, medimos o tempo do trabalho pesado.
    // A barreira garante que todos os processos cheguem a este ponto antes de
    // começar.
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Esta é a fase de DISTRIBUIÇÃO.
    // O processo 0 "espalha" (Scatter) as fatias das matrizes A e B.
    // Cada processo, incluindo o 0, recebe sua parte nos seus buffers 'sub_a' e
    // 'sub_b'.
    MPI_Scatter(matriz_a, elementos_por_proc, MPI_INT, sub_a,
                elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(matriz_b, elementos_por_proc, MPI_INT, sub_b,
                elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Esta é a fase de COMPUTAÇÃO PARALELA.
    // Cada processo executa este loop de forma independente e simultânea,
    // somando os números da sua fatia e guardando no seu buffer de resultado
    // 'sub_c'.
    for (int i = 0; i < elementos_por_proc; i++) {
        sub_c[i] = sub_a[i] + sub_b[i];
    }

    // Esta é a fase de CONSOLIDAÇÃO.
    // Todos os processos enviam seus resultados parciais ('sub_c') de volta
    // para o processo 0, que "junta" (Gather) tudo na matriz final 'matriz_c'.
    MPI_Gather(sub_c, elementos_por_proc, MPI_INT, matriz_c, elementos_por_proc,
               MPI_INT, 0, MPI_COMM_WORLD);

    // Paramos o cronômetro aqui, depois que todo o trabalho e comunicação foram
    // concluídos.
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Apenas o processo 0 tem o resultado final e sabe o tempo total.
    if (rank == 0) {
        // Ele imprime o tempo de execução.
        printf(
            "Tempo de execução da soma para n=%d com %d processos: %f "
            "segundos\n",
            n, size, end_time - start_time);
        // E libera a memória das matrizes grandes que só ele usou.
        free(matriz_a);
        free(matriz_b);
        free(matriz_c);
    }

    // TODOS os processos liberam a memória dos seus buffers locais.
    free(sub_a);
    free(sub_b);
    free(sub_c);

    // Finaliza o ambiente MPI. É a última coisa a se fazer.
    MPI_Finalize();
    return 0;
}