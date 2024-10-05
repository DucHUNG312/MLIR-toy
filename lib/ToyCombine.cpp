#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"
#include <llvm/Support/LogicalResult.h>

using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *Context)
      : OpRewritePattern<TransposeOp>(Context, /*benefit=*/1) {}

  llvm::LogicalResult
  matchAndRewrite(TransposeOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Value TransposeInput = Op.getOperand();
    TransposeOp TransposeInputOp = TransposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!TransposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    Rewriter.replaceOp(Op, {TransposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &Results,
                                              MLIRContext *Context) {
  Results.add<SimplifyRedundantTranspose>(Context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &Results,
                                            MLIRContext *Context) {
  Results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(Context);
}