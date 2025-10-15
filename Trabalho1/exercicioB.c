// Incluímos as bibliotecas padrão e a do MPI.
// A novidade aqui é a <limits.h>, que nos dá acesso a constantes
// como INT_MIN (o menor número inteiro possível), muito útil para
// iniciar nossa busca pelo maior valor.
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Função auxiliar para preencher a matriz com números aleatórios.
// O valor rand() % (n*n) gera números entre 0 e (n*n - 1),
// garantindo uma boa dispersão de valores.
void preencher_matriz(int *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = rand() % (n * n);
    }
}

// Função principal do programa.
int main(int argc, char **argv) {
    // 'rank' é o ID do processo atual, 'size' é o total de processos.
    int rank, size;
    int n;  // Tamanho da matriz.
    // Só precisamos de um ponteiro para a matriz completa, que existirá apenas
    // no processo 0.
    int *matriz_a = NULL;
    // Variáveis para cronometrar a execução.
    double start_time, end_time;

    // Inicializamos o ambiente MPI.
    MPI_Init(&argc, &argv);
    // Cada processo pega seu ID (rank) e o total de processos (size).
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Uma mensagem de "Olá" de cada processo para confirmar que todos
    // começaram.
    printf("[Processo %d de %d]: Olá!\n", rank, size);

    // Validação de entrada, garantindo que o tamanho 'n' foi fornecido.
    if (argc < 2) {
        if (rank == 0)
            printf("Uso: mpirun -np <procs> %s <tamanho_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    n = atoi(argv[1]);

    // Verificamos se a matriz pode ser dividida igualmente entre os processos.
    if (n % size != 0) {
        if (rank == 0)
            printf(
                "O tamanho da matriz (n) deve ser divisível pelo número de "
                "processos.\n");
        MPI_Finalize();
        return 1;
    }

    // Calculamos a carga de trabalho de cada processo.
    int linhas_por_proc = n / size;
    int elementos_por_proc = linhas_por_proc * n;

    // TODOS os processos alocam um buffer ('sub_a') para receber sua fatia da
    // matriz.
    int *sub_a = (int *)malloc(elementos_por_proc * sizeof(int));

    // APENAS o processo 0 (o "mestre") cria e preenche a matriz completa.
    if (rank == 0) {
        printf("[Processo %d]: Vou inicializar a matriz de tamanho %dx%d.\n",
               rank, n, n);
        matriz_a = (int *)malloc(n * n * sizeof(int));
        srand(time(NULL));
        preencher_matriz(matriz_a, n);
    }

    // Sincronizamos todos os processos e iniciamos o cronômetro.
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Fase de DISTRIBUIÇÃO (Scatter).
    // O processo 0 distribui as fatias da 'matriz_a' para todos os processos.
    if (rank == 0) {
        printf(
            "[Processo %d]: Distribuindo as linhas da matriz para todos...\n",
            rank);
    }
    MPI_Scatter(matriz_a, elementos_por_proc, MPI_INT, sub_a,
                elementos_por_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Fase de COMPUTAÇÃO LOCAL (Map).
    // Cada processo agora trabalha de forma independente em sua própria fatia
    // ('sub_a').
    printf(
        "[Processo %d]: Recebi minhas linhas. Procurando pelo maior valor "
        "local.\n",
        rank);

    // Inicializamos 'max_local' com o menor valor inteiro possível.
    // Isso garante que qualquer número da matriz será maior que ele.
    int max_local = INT_MIN;
    // Iteramos sobre os elementos recebidos para encontrar o maior local.
    for (int i = 0; i < elementos_por_proc; i++) {
        if (sub_a[i] > max_local) {
            max_local = sub_a[i];
        }
    }

    // Cada processo anuncia o resultado parcial que encontrou.
    printf(
        "[Processo %d]: Meu maior valor local é %d. Enviando para a "
        "redução...\n",
        rank, max_local);

    // Fase de CONSOLIDAÇÃO (Reduce).
    int max_global;
    // Esta é a operação chave: cada processo envia seu 'max_local'.
    // O MPI compara todos os valores recebidos usando a operação MPI_MAX
    // e entrega o resultado final ('max_global') apenas ao processo 0.
    // É muito mais eficiente que juntar todos os dados (Gather) e depois
    // processar.
    MPI_Reduce(&max_local, &max_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Sincronizamos e paramos o cronômetro.
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // APENAS o processo 0 tem o resultado final e o imprime.
    if (rank == 0) {
        printf("[Processo %d]: Operação de redução concluída.\n", rank);
        printf("Tempo de execução para encontrar o maior valor: %f segundos\n",
               end_time - start_time);
        printf("O maior valor global encontrado foi: %d\n", max_global);
        // O processo 0 também libera a memória da matriz completa.
        free(matriz_a);
    }

    // TODOS os processos liberam a memória de seus buffers locais.
    free(sub_a);
    printf("[Processo %d]: Finalizando.\n", rank);
    // Finalizamos o ambiente MPI.
    MPI_Finalize();
    return 0;
}