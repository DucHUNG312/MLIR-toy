include "mlir/IR/OpBase.td"

def CastOpInterface : OpInterface<"CastInference"> {
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let methods = [
    InterfaceMethod<"Check if the input and output types are compatible for a cast operation.",
    "bool", "areCastCompatible", (ins "mlir::TypeRange":$inputs, "mlir::TypeRange":$outputs)>,
    
    InterfaceMethod<"Infer the output shape of the cast operation from the input", 
    "void", "inferShapes">
  ];
}