#include "MPIServer.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

    MPI_Init(&argc,&argv);

    MPIServer server;
    server.start();

}