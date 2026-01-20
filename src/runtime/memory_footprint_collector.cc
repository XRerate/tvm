#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/memory_footprint_collector.h>

#include <string>
#include <unordered_map>

#include "tvm/runtime/profiling.h"

namespace tvm {
namespace runtime {

void MemoryFootprintCollectorNode::Init(ffi::Array<profiling::DeviceWrapper> devices) {
#ifdef BUILD_FOR_ANDROID
  if (!profiler_.Initialize()) {
    LOG(ERROR) << "Failed to initialize memory footprint profiler";
  }
#endif  // BUILD_FOR_ANDROID
}

ObjectRef MemoryFootprintCollectorNode::Start(Device dev) {
#ifdef BUILD_FOR_ANDROID
  profiler_.Start();
#endif  // BUILD_FOR_ANDROID
  return ObjectRef(nullptr);
}

ffi::Map<ffi::String, ffi::Any> MemoryFootprintCollectorNode::Stop(ObjectRef obj) {
  std::unordered_map<ffi::String, ffi::Any> metrics;
#ifdef BUILD_FOR_ANDROID
  profiler_.Stop();

  metrics["total_memory_footprint_bytes"] =
      ObjectRef(ffi::make_object<profiling::CountNode>(profiler_.GetTotalMemoryFootprint()));
#else
  metrics["total_memory_footprint_bytes"] = ObjectRef(ffi::make_object<profiling::CountNode>(1234));
#endif  // BUILD_FOR_ANDROID
  return metrics;
}

profiling::MetricCollector& MemoryFootprintCollector::GetMemoryFootprintCollector() {
  static profiling::MetricCollector collector =
      profiling::MetricCollector(ffi::make_object<MemoryFootprintCollectorNode>());
  return collector;
}

ObjectRef MemoryFootprintCollector::Start(profiling::DeviceWrapper dev_wrapper) {
  profiling::MetricCollector& collector = GetMemoryFootprintCollector();
  // Call once for init
  static std::once_flag init_once;
  std::call_once(init_once, [&collector, &dev_wrapper]() {
    collector->Init({dev_wrapper});
  });
  return collector->Start(dev_wrapper->device);
}

ffi::Map<ffi::String, ffi::Any> MemoryFootprintCollector::Stop(ObjectRef obj) {
  static profiling::MetricCollector& collector = GetMemoryFootprintCollector();
  return collector->Stop(obj);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.profiling.MemoryFootprintCollector",
                        []() { return MemoryFootprintCollector::GetMemoryFootprintCollector(); });
}

}  // namespace runtime
}  // namespace tvm