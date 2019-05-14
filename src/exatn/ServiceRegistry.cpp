#include "ServiceRegistry.hpp"
#include <dirent.h>

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
       exatnPluginPath = std::getenv("HOME") + std::string("/.exatn/plugins");
    }

    std::string parentPath = "";

    // std::cout << "[service-registry] ExaTN Plugin Path: " << exatnPluginPath << "\n";;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(exatnPluginPath.c_str())) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir(dir)) != NULL) {
        if (std::string(ent->d_name).find("lib") != std::string::npos) {
        //   printf("[service-registry] Installing Plugin: %s\n", ent->d_name);
          context.InstallBundles(exatnPluginPath + "/" + std::string(ent->d_name));
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
        std::cout << "HOWDY " << b.GetSymbolicName() << "\n";
      b.Start();
    }

    initialized = true;
  }
}

} // namespace exatn
