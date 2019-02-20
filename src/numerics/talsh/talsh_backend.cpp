#include "talsh_backend.hpp"
#include <iostream>

namespace exatn {
namespace numerics {
namespace talsh {

std::vector<std::string> TalshBackend::translate(const std::string taProl) {
    return split(taProl, "\n");
}

void TalshBackend::execute(std::vector<std::string>& simpleTaPrlList) {

    std::cout << "[talsh] Executing taprol strings\n";

}

}
}
}
