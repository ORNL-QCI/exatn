#include "ServiceRegistry.hpp"
#include <dirent.h>
#include "exatn_config.hpp"

namespace exatn {

void ServiceRegistry::initialize(const std::string pluginPath) {

  if (!initialized) {
    framework = FrameworkFactory().NewFramework();

    // Initialize the framework
    framework.Init();
    context = framework.GetBundleContext();
    if (!context) {
      std::cerr << "Invalid ExaTN Framework plugin context.\n";
    }

    // Get the paths/files we'll be searching
    std::string exatnPluginPath = pluginPath;
    if (exatnPluginPath.empty()) {
       exatnPluginPath = std::string(EXATN_INSTALL_DIR) + std::string("/plugins");
    }

    std::string parentPath = "";

    // std::cout << "[service-registry] ExaTN Plugin Path: " << exatnPluginPath << "\n";;

    auto has_suffix = [](const std::string &str, const std::string &suffix) {
        return str.size() >= suffix.size() &&
               str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(exatnPluginPath.c_str())) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir(dir)) != NULL) {
        auto fileName = std::string(ent->d_name);
        if (fileName.find("lib") != std::string::npos && (has_suffix(fileName, ".so") || has_suffix(fileName, ".dylib"))) {
        //   printf("[service-registry] Installing Plugin: %s\n", ent->d_name);
          context.InstallBundles(exatnPluginPath + "/" + fileName);
        }
      }
      closedir(dir);
    } else {
      /* could not open directory */
      std::cerr << "[service-registry] Could not open plugin directory.\n";
    }

    // Start the framework itself.
    framework.Start();
    auto bundles = context.GetBundles();
    for (auto b : bundles) {
      b.Start();
    }

    initialized = true;
  }
}

} // namespace exatn
