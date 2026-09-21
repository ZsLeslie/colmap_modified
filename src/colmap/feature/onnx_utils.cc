// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/feature/onnx_utils.h"

#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace colmap {

#ifdef COLMAP_ONNX_ENABLED

namespace {
constexpr size_t kSharedCudaArenaLimit = 4ULL * 1024 * 1024 * 1024;

Ort::Env& GetSharedONNXEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "colmap");
  return env;
}

#ifdef COLMAP_CUDA_ENABLED
void RegisterSharedCUDAAllocator(const int device_id) {
  static std::mutex mutex;
  static std::unordered_set<int> registered_devices;

  const std::lock_guard<std::mutex> lock(mutex);
  if (registered_devices.count(device_id) > 0) {
    return;
  }

  Ort::MemoryInfo memory_info(
      "Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
  // ALIKED and LightGlue execute sequentially in COLMAP. Sharing one arena
  // prevents their per-session high-water marks from accumulating on devices
  // with unified CPU/GPU memory, while retaining enough memory for either
  // model's largest individual inference.
  Ort::ArenaCfg arena_config(kSharedCudaArenaLimit,
                             /*arena_extend_strategy=*/1,
                             /*initial_chunk_size_bytes=*/-1,
                             /*max_dead_bytes_per_chunk=*/-1);
  GetSharedONNXEnv().CreateAndRegisterAllocatorV2(
      "CUDAExecutionProvider", memory_info, {}, arena_config);
  registered_devices.insert(device_id);
}
#endif

[[noreturn]] void RethrowONNXException() {
  try {
    std::rethrow_exception(std::current_exception());
  } catch (const Ort::Exception& e) {
    // ONNX Runtime may write to stderr without a trailing newline.
    // Insert a newline here to avoid mixing with COLMAP log output.
    std::cerr << '\n';
    LOG(ERROR) << "ONNX Runtime error: " << e.what();
    throw;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Unexpected exception during ONNX execution: " << e.what();
    throw;
  } catch (...) {
    LOG(ERROR) << "Unknown exception during ONNX execution";
    throw;
  }
}
}  // namespace

std::string FormatONNXTensorShape(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

void ThrowCheckONNXNode(const std::string_view name,
                        const std::string_view expected_name,
                        const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& expected_shape) {
  THROW_CHECK_EQ(name, expected_name);
  THROW_CHECK_EQ(shape.size(), expected_shape.size())
      << "Invalid shape for " << name << ": " << FormatONNXTensorShape(shape)
      << " != " << FormatONNXTensorShape(expected_shape);
  for (size_t i = 0; i < shape.size(); ++i) {
    // -1 is treated as a wildcard for dynamic dimensions.
    if (expected_shape[i] != -1) {
      THROW_CHECK_EQ(shape[i], expected_shape[i])
          << "Invalid shape for " << name << ": "
          << FormatONNXTensorShape(shape)
          << " != " << FormatONNXTensorShape(expected_shape);
    }
  }
}

ONNXModel::ONNXModel(std::string model_path,
                     int num_threads,
                     bool use_gpu,
                     const std::string& gpu_index) {
  {
    static std::mutex download_mutex;
    const std::lock_guard<std::mutex> lock(download_mutex);
    model_path = MaybeDownloadAndCacheFile(model_path).string();
  }

  const int num_eff_threads = GetEffectiveNumThreads(num_threads);

  try {
    InitializeSession(model_path, num_eff_threads, use_gpu, gpu_index);
  } catch (...) {
    RethrowONNXException();
  }
}

void ONNXModel::InitializeSession(const std::string& model_path,
                                  int num_threads,
                                  bool use_gpu,
                                  const std::string& gpu_index) {
  // Use sequential execution mode with a single inter-op thread, since our
  // models (ALIKED, LightGlue) are sequential CNNs/Transformers without
  // independent graph branches. Inter-op parallelism would only cause thread
  // contention. Intra-op threads parallelize within individual operators
  // (convolutions, matrix multiplications) and are managed at the caller level.
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetExecutionMode(ORT_SEQUENTIAL);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);
  // Both ALIKED and LightGlue have dynamic input shapes. A cached memory
  // pattern from a large input can otherwise keep a correspondingly large
  // allocation alive for the lifetime of the session.
  session_options_.DisableMemPattern();

#ifdef COLMAP_CUDA_ENABLED
  if (use_gpu) {
    const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1)
        << "ONNX model can only run on one GPU";
    const int device_id = gpu_indices[0] >= 0 ? gpu_indices[0] : 0;

    // Avoid retaining fallback CPU allocations between runs. This is
    // particularly important on devices with unified CPU/GPU memory.
    session_options_.DisableCpuMemArena();

    // Use a single CUDA arena shared by all ONNX sessions on this device.
    // Without this, ALIKED and LightGlue can each retain up to gpu_mem_limit,
    // doubling the process-wide high-water mark even though inference is
    // serialized by the caller.
    RegisterSharedCUDAAllocator(device_id);
    session_options_.AddConfigEntry("session.use_env_allocators", "1");

    // Use CUDA EP V2 so that cuDNN workspace usage can be constrained.
    Ort::CUDAProviderOptions cuda_options;
    const std::unordered_map<std::string, std::string> cuda_provider_options = {
        {"device_id", std::to_string(device_id)},
        // Grow by exactly the requested amount instead of rounding arena
        // extensions to powers of two.
        {"arena_extend_strategy", "kSameAsRequested"},
        // This limits the CUDA EP arena, not total CUDA memory consumption.
        {"gpu_mem_limit", std::to_string(kSharedCudaArenaLimit)},
        // Avoid the temporary allocations made by exhaustive cuDNN search.
        {"cudnn_conv_algo_search", "HEURISTIC"},
        {"cudnn_conv_use_max_workspace", "0"},
        {"do_copy_in_default_stream", "1"}};
    cuda_options.Update(cuda_provider_options);
    session_options_.AppendExecutionProvider_CUDA_V2(*cuda_options);

    arena_shrinkage_devices_ = "gpu:" + std::to_string(device_id);
    VLOG(2) << "ONNX CUDA EP enabled: device=" << device_id
            << ", arena=shared, arena_strategy=kSameAsRequested"
            << ", shared_gpu_mem_limit=4096MB"
            << ", cudnn_search=HEURISTIC, max_workspace=0";
  }
#endif

  VLOG(2) << "Loading ONNX model from " << model_path;
#ifdef _WIN32
  const unsigned int code_page = GetACP();
  const int wide_len =
      MultiByteToWideChar(code_page, 0, model_path.c_str(), -1, nullptr, 0);
  std::wstring model_path_wide(wide_len, L'\0');
  MultiByteToWideChar(
      code_page, 0, model_path.c_str(), -1, &model_path_wide[0], wide_len);
  const wchar_t* model_path_cstr = model_path_wide.c_str();
#else
  const char* model_path_cstr = model_path.c_str();
#endif
  session_ = std::make_unique<Ort::Session>(
      GetSharedONNXEnv(), model_path_cstr, session_options_);

  VLOG(2) << "Parsing the inputs";
  const int num_inputs = session_->GetInputCount();
  input_name_strs_.reserve(num_inputs);
  input_names_.reserve(num_inputs);
  input_shapes_.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_name_strs_.emplace_back(
        session_->GetInputNameAllocated(i, allocator_));
    input_names_.emplace_back(input_name_strs_[i].get());
    input_shapes_.emplace_back(
        session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  VLOG(2) << "Parsing the outputs";
  const int num_outputs = session_->GetOutputCount();
  output_name_strs_.reserve(num_outputs);
  output_names_.reserve(num_outputs);
  output_shapes_.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_name_strs_.emplace_back(
        session_->GetOutputNameAllocated(i, allocator_));
    output_names_.emplace_back(output_name_strs_[i].get());
    output_shapes_.emplace_back(
        session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
}

std::vector<Ort::Value> ONNXModel::Run(
    const std::vector<Ort::Value>& input_tensors) const {
  try {
    Ort::RunOptions run_options;
    if (!arena_shrinkage_devices_.empty()) {
      run_options.AddConfigEntry("memory.enable_memory_arena_shrinkage",
                                 arena_shrinkage_devices_.c_str());
    }
    return session_->Run(run_options,
                         input_names_.data(),
                         input_tensors.data(),
                         input_tensors.size(),
                         output_names_.data(),
                         output_names_.size());
  } catch (...) {
    RethrowONNXException();
  }
}

#endif  // COLMAP_ONNX_ENABLED

}  // namespace colmap
