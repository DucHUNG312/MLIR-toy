#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace toy {
std::unique_ptr<mlir::Pass> createShapeInferencePass();
} // namespace toy