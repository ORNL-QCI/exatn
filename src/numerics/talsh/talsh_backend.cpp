#include "talsh_backend.hpp"
#include <iostream>
#include "talshxx.hpp"

namespace exatn {
namespace numerics {
namespace talsh {


void TalshBackend::execute(const std::string& taProl) {
    // Just splitting on all new lines
    auto commands = split(taProl, "\n");
    std::cout << "[talsh] Executing taprol strings\n";

    // ::talsh::initialize();

}

}
}
}
