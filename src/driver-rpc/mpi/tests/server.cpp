#include "MPIServer.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

    MPI_Init(&argc,&argv);

    // Create the server and
    // start it up, this kicks off the
    // event loop
    MPIServer server;
    server.start();

    // If we make it here, then the
    // client has shutdown the server
    
    std::cout << "Exiting server.cpp\n";

    MPI_Finalize();

    return 0;

}