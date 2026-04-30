// filename: dispatcher.cpp
// purpose: Implements named backend registration and dispatch routing.
// phase: Phase 3 - Runtime Abstraction
// last modified: 2026-04-29

#include "runtime/dispatcher.h"

namespace opengpu::runtime {

bool Dispatcher::register_backend(const std::string& backend_name,
                                  std::shared_ptr<Backend> backend) {
  if (backend_name.empty() || !backend) {
    return false;
  }
  backends_[backend_name] = std::move(backend);
  return true;
}

bool Dispatcher::dispatch(const std::string& backend_name, const Kernel& kernel,
                         const LaunchConfig& launch) const {
  const auto it = backends_.find(backend_name);
  if (it == backends_.end() || !it->second) {
    return false;
  }
  return it->second->launch(kernel, launch);
}

}  // namespace opengpu::runtime
