#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

#include "ShapeInferenceInterface.cpp.inc"

namespace {

struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass,
                               mlir::OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  void runOnOperation() override {
    auto F = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> OpWorklist;
    F.walk([&](mlir::Operation *Op) {
      if (returnsDynamicShape(Op))
        OpWorklist.insert(Op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!OpWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto NextOp = llvm::find_if(OpWorklist, allOperandsInferred);
      if (NextOp == OpWorklist.end())
        break;

      Operation *Op = *NextOp;
      OpWorklist.erase(Op);

      // Ask the operation to infer its output shapes.
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *Op << "\n");
      if (auto ShapeOp = dyn_cast<ShapeInference>(Op)) {
        ShapeOp.inferShapes();
      } else {
        Op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!OpWorklist.empty()) {
      F.emitError("Shape inference failed, ")
          << OpWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(Operation *Op) {
    return llvm::all_of(Op->getOperandTypes(), [](Type OperandType) {
      return llvm::isa<RankedTensorType>(OperandType);
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returnsDynamicShape(Operation *Op) {
    return llvm::any_of(Op->getResultTypes(), [](Type ResultType) {
      return !llvm::isa<RankedTensorType>(ResultType);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}