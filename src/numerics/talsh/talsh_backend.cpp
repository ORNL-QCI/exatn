#include "talsh_backend.hpp"
#include <iostream>

namespace exatn {
namespace numerics {
namespace talsh {

std::vector<std::string> TalshBackend::translate(const std::string taProl) {
    // Just splitting on all new lines
    auto commands = split(taProl, "\n");

    // Do something with these commands

    return commands;
}

void TalshBackend::execute(std::vector<std::string>& simpleTaPrlList) {

    std::cout << "[talsh] Executing taprol strings\n";

}

}
}
}
