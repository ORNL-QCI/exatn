#include "MPIClient.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

  std::cout << "Entering client code\n";
  MPI_Init(&argc, &argv);
  std::cout << "Instantiating client code\n";

  MPIClient client;

  std::cout << "Starting to send taprol\n";
  auto jobId = client.sendTaProl("hello world");

  std::cout << "JOB ID: " << jobId << "\n";

//   auto value = client.retrieveResult(jobId);

  client.shutdown();

  MPI_Finalize();

}
// #include <stdio.h>
// #include <mpi.h>

// #define MAX_DATA    100

// int main (int argc, char *argv[]){

//   int rank, size;
//   char port_name[MPI_MAX_PORT_NAME];

//   MPI_Init (&argc, &argv);  /* starts MPI */
//   MPI_Comm_rank (MPI_COMM_WORLD, &rank);    /* get current process id */
//   MPI_Comm_size (MPI_COMM_WORLD, &size);    /* get number of processes */

//   //Server rank
//   if (rank == 0){
//     double buf[MAX_DATA];
//     MPI_Open_port(MPI_INFO_NULL, port_name);
//     printf("server available at %s\n", port_name);

//     // server tells the client about the port_name
//     MPI_Send(port_name, MPI_MAX_PORT_NAME, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
//     MPI_Comm client;
//     MPI_Status status;
//     while (1) {
//         MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &client);
//         int again = 1;
//         while (again) {
//             printf("server starting to receive\n");
//             MPI_Recv(buf, MAX_DATA, MPI_DOUBLE, 0, MPI_ANY_TAG, client, &status);
//             printf("server received sth!\n");
//             switch (status.MPI_TAG) {
//             case 0:
//                 MPI_Comm_free(&client);
//                 MPI_Close_port(port_name);
//                 MPI_Finalize();
//                 return 0;
//             case 1:
//                 MPI_Comm_disconnect(&client);
//                 again = 0;
//                 break;
//             case 2: /* do something */
//                 printf("case 2\n");
//                 break;
//             default:
//                 /* Unexpected message type */
//                 MPI_Abort(MPI_COMM_WORLD, 1);
//             }
//         }
//     }
//   } else{
//     MPI_Status status;
//     // client receives the port information from server
//     MPI_Recv(port_name, MPI_MAX_PORT_NAME, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

//     std::cout << "client got port_name " << std::string(port_name) << "\n";
//     MPI_Comm server;
//     double buf[MAX_DATA];

//     MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &server);
//     int i = 0;
//     while (1) {
//         int tag = 2; /* Action to perform */
//         MPI_Send(buf, MAX_DATA, MPI_DOUBLE, 0, tag, server);
//         printf("client send somthing!\n");
//         // usleep(1);
//         break;
//     }
//     MPI_Send(buf, 0, MPI_DOUBLE, 0, 1, server);
//     MPI_Comm_disconnect(&server);
//   }

//   MPI_Finalize();
// }