#include <stdio.h>
#include <mpi.h>
#define N 12
int main(int argc, char* argv[]){
		int p, id;		
		MPI_Init (&argc, &argv);
		MPI_Status status;
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		MPI_Comm_size(MPI_COMM_WORLD,&p);
		int bloco;
		bloco = N / p;
			
		if(id == 0){
			int v[N],i,destino;
			for(i=0; i<N; i++){
				v[i] = i;
			}
			for(destino=1; destino<p; destino++){
				MPI_Send(&v[(N/p)*destino],bloco,MPI_INT,destino,1,MPI_COMM_WORLD);	
				printf("P0 enviou bloco %d elementos a partir de v[%d]\n",bloco, bloco*destino);
			}
		} else {
			int v[bloco], i;
			MPI_Recv(v, bloco, MPI_INT, 0, 1, MPI_COMM_WORLD, &status );
		}
	     
		MPI_Finalize ();
		return 0;
}
