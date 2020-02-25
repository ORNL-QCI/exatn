#include "DriverServer.hpp"
#include "exatn.hpp"
#include <gtest/gtest.h>
#include "mpi.h"

using namespace exatn::rpc;

int main(int argc, char** argv) {

    MPI_Init(&argc,&argv);
    exatn::initialize();

    // Create the server and start it up,
    // this kicks off the event loop
    auto server = exatn::getService<DriverServer>("mpi");
    server->start();

    // If we make it here, then the client has shutdown the server
    std::cout << "[server.cpp] exited server\n";

    exatn::finalize();
    MPI_Finalize();

    return 0;
}
