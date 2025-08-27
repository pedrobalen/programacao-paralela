#include <stdio.h>   
#include <string.h> 
#include <mpi.h>
#include <stdlib.h>
int main(int argc, char* argv[]){
	int p, id, source, dest, tag=1,n,a;
	char message[100];
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	if (id == 0){
		n = rand()%100;
		for(int destino = 1; destino < p; destino++){
			MPI_Send(&n, 1, MPI_INT, destino, tag, MPI_COMM_WORLD);
			printf("processo 0 enviu para processo %d\n",destino);
		}		
		
	} else{
		MPI_Recv(&a, 1, MPI_INT, 0, tag,MPI_COMM_WORLD, &status);
		printf("processo %d recebeu %d de proceso 0\n",id, a);
	}
	MPI_Finalize ();
}	
