#include "exatensor_backend.hpp"
#include <iostream>

namespace exatn {
namespace numerics {
namespace exatensor {

void ExatensorBackend::execute(const std::string& taProl) {

    auto commands = split(taProl, "\n");
    std::cout << "[exatensor] Executing taprol strings\n";

}

}
}
}
