/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Check if the input and output types are compatible for a cast operation.
bool CastInference::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
      return getImpl()->areCastCompatible(getImpl(), getOperation(), inputs, outputs);
  }
/// Infer the output shape of the cast operation from the input
void CastInference::inferShapes() {
      return getImpl()->inferShapes(getImpl(), getOperation());
  }
