#pragma once

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace toy {
class ModuleAST;

/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          ModuleAST &ModuleAst);

} // namespace toy