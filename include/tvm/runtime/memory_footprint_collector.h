#ifndef TVM_RUNTIME_MEMORY_FOOTPRINT_COLLECTOR_H_
#define TVM_RUNTIME_MEMORY_FOOTPRINT_COLLECTOR_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/profiling.h>

#ifdef BUILD_FOR_ANDROID
#include "MemoryFootprintProfiler.h"
#endif  // BUILD_FOR_ANDROID

namespace tvm {
namespace runtime {

struct MemoryFootprintCollectorNode final : public profiling::MetricCollectorNode {
  explicit MemoryFootprintCollectorNode() {}

  void Init(ffi::Array<profiling::DeviceWrapper> devices);

  ObjectRef Start(Device dev);

  ffi::Map<ffi::String, ffi::Any> Stop(ObjectRef obj) final;

  ~MemoryFootprintCollectorNode() {}

 private:
#ifdef BUILD_FOR_ANDROID
  MemoryFootprintProfiler::MemoryFootprintProfiler profiler_;
#endif  // BUILD_FOR_ANDROID
};

class MemoryFootprintCollector : public profiling::MetricCollector {
 public:
  static TVM_DLL profiling::MetricCollector& GetMemoryFootprintCollector();
  static TVM_DLL ObjectRef Start(profiling::DeviceWrapper dev_wrapper);
  static TVM_DLL ffi::Map<ffi::String, ffi::Any> Stop(ObjectRef obj);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MemoryFootprintCollector, profiling::MetricCollector,
                                             MemoryFootprintCollectorNode);
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MEMORY_FOOTPRINT_COLLECTOR_H_