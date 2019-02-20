#include "exatensor_backend.hpp"
#include <iostream>

namespace exatn {
namespace numerics {
namespace exatensor {

std::vector<std::string> ExatensorBackend::translate(const std::string taProl) {
    return split(taProl, "\n");
}

void ExatensorBackend::execute(std::vector<std::string>& simpleTaPrlList) {

    std::cout << "[exatensor] Executing taprol strings\n";

}

}
}
}
