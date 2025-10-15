# Começamos importando as bibliotecas necessárias.
# 'os' para interagir com o sistema (como verificar se um arquivo existe).
# 'subprocess' para executar comandos do terminal (como mpicc e mpirun).
# 're' para expressões regulares, que usaremos para "ler" o tempo na saída do programa.
# 'pandas' para criar e manipular a tabela de resultados de forma eficiente.
# 'time' não é usado aqui, mas seria útil para outras medições de tempo.
import os
import subprocess
import re
import pandas as pd
import time

# esse script compila e roda os exercicios, tambem imprime a tabela de resultados no terminal
# importante o nome dos arquivos estarem de acordo com o dicionario definido em setup_dict
# esse trabalho foi feito utilizando o google colab

# --- ETAPA 1: CONFIGURAÇÃO DO AMBIENTE E DO EXPERIMENTO ---
# Esta seção funciona como o "painel de controle" do seu experimento.
# É aqui que você define tudo que o script vai fazer.

# 'setup_dict' mapeia o nome do programa executável ao seu arquivo de código-fonte.
# Ex: 'exercicioA' será compilado a partir de 'exercicioA.c'.
setup_dict = {
    'exercicioA': 'exercicioA.c',
    'exercicioB': 'exercicioB.c',
    'exercicioC': 'exercicioC.c'
}

# 'params' define a série de testes que serão executados.
# Você pode facilmente adicionar mais valores a estas listas para expandir os testes.
params = {
    "executables": ['exercicioA', 'exercicioB', 'exercicioC'], # Quais programas testar.
    "problem_sizes": [1024, 2048],                            # Quais tamanhos de 'n' usar.
    "process_counts": [1, 2, 4, 8]                             # Com quantos processos rodar.
}

# --- ETAPA 2: COMPILAÇÃO DOS CÓDIGOS C ---
# Antes de rodar, o script garante que os programas estão compilados e atualizados.
print("--- Iniciando Compilação ---")
# O loop itera sobre o dicionário 'setup_dict'.
for exe, source in setup_dict.items():
    # Verifica se o arquivo .c realmente existe no ambiente.
    if os.path.exists(source):
        print(f"Compilando {source} -> {exe}...")
        # Monta o comando de compilação.
        compile_command = f"mpicc {source} -o {exe}"
        # Executa o comando. 'check=True' é uma segurança: se a compilação falhar
        # (por um erro no código C), o script para aqui, evitando problemas futuros.
        subprocess.run(compile_command, shell=True, check=True)
    else:
        print(f"AVISO: Arquivo-fonte {source} não encontrado. Pulando compilação.")
print("--- Compilação Finalizada ---\n")


# --- ETAPA 3: EXECUÇÃO AUTOMATIZADA E COLETA DE DADOS ---
# Esta é a parte principal, onde os experimentos são de fato executados.
all_results = [] # Uma lista vazia para guardar os resultados de cada teste.
print("--- Iniciando Execução dos Testes (isso pode levar um tempo) ---")

# Três loops aninhados garantem que testaremos todas as combinações possíveis
# definidas no dicionário 'params'.
for exe in params["executables"]:
    # Medida de segurança: verifica se o executável existe antes de tentar rodá-lo.
    if not os.path.exists(exe):
        print(f"ERRO: Executável {exe} não encontrado. Pulando para o próximo.")
        continue
    for n in params["problem_sizes"]:
        for p in params["process_counts"]:
            # Monta o comando de execução para a combinação atual.
            run_command = f"mpirun --oversubscribe --allow-run-as-root -np {p} ./{exe} {n}"

            # Executa o comando e captura a saída de texto.
            result = subprocess.run(run_command, shell=True, capture_output=True, text=True)

            execution_time = None
            # Verifica se o programa C rodou sem erros.
            if result.returncode == 0:
                # Aqui está a "mágica": usamos uma expressão regular (re.search) para
                # encontrar um padrão de número com ponto decimal (ex: "0.12345")
                # na saída de texto que foi capturada.
                match = re.search(r"(\d+\.\d+)", result.stdout)
                if match:
                    # Se encontrou, converte o texto para um número.
                    execution_time = float(match.group(1))
            else:
                # Se o programa C deu erro, avisa no terminal.
                print(f"ERRO ao executar {exe} com n={n}, p={p}. Erro: {result.stderr}")

            # Imprime um status para sabermos o que está acontecendo.
            status_msg = f"OK ({execution_time:.4f}s)" if execution_time is not None else "FALHOU"
            print(f"Teste: {exe} (n={n}, p={p}) -> {status_msg}")

            # Adiciona os dados coletados (um dicionário) à nossa lista de resultados.
            all_results.append({
                'algoritmo': exe,
                'n': n,
                'processos': p,
                'tempo_s': execution_time
            })

print("--- Execução Finalizada ---\n")

# --- ETAPA 4: PROCESSAMENTO E APRESENTAÇÃO DOS RESULTADOS ---
# Agora que temos os dados brutos, vamos analisá-los e formatá-los.
print("--- Tabela Consolidada de Resultados ---")

# Converte a lista de resultados em uma tabela do Pandas, que é muito poderosa.
df = pd.DataFrame(all_results)
# Limpa a tabela, removendo quaisquer testes que falharam.
df.dropna(inplace=True)

if not df.empty:
    # Esta é uma etapa crucial para o cálculo. Agrupamos os dados por algoritmo e por 'n'.
    # Em cada grupo, pegamos o primeiro valor de tempo (que é o teste com p=1)
    # e o usamos como o tempo sequencial de referência para todo aquele grupo.
    df['tempo_seq_s'] = df.groupby(['algoritmo', 'n'])['tempo_s'].transform('first')

    # Com o tempo sequencial em mãos, calculamos o Speedup e a Eficiência
    # para cada linha da tabela.
    df['speedup'] = df.apply(
        lambda row: row['tempo_seq_s'] / row['tempo_s'] if row['tempo_s'] > 0 else 0,
        axis=1
    )
    df['eficiencia_%'] = df.apply(
        lambda row: (row['speedup'] / row['processos']) * 100 if row['processos'] > 0 else 0,
        axis=1
    )

    # Configuração para garantir que a tabela inteira seja impressa, sem cortar linhas.
    pd.set_option('display.max_rows', None)
    # Imprime a tabela final, com os números arredondados para 4 casas decimais.
    print(df.round(4).to_string(index=False))
else:
    print("Nenhum resultado foi coletado com sucesso. Verifique os erros de execução acima.")