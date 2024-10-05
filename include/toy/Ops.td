include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/Dialect.td"

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Toy_Op<"constant"> {
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
        %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                            : tensor<2x3xf64>
        ```
    }];

    let arguments = (ins F64ElementsAttr:$value);

    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,
        OpBuilder<(ins "double":$value)>
    ];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Toy_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    let arguments = (ins F64Tensor:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Toy_Op<"add"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shapes of the tensor operands are expected to match.
    }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}

def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
]> {
    let summary = "user defined function operation";
    let description = [{
        The "toy.func" operation represents a user defined function. These are
        callable SSA-region operations that contain toy computations.

        Example:

        ```mlir
        toy.func @main() {
        %0 = toy.constant dense<5.500000e+00> : tensor<f64>
        %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
        toy.print %1 : tensor<2x2xf64>
        toy.return
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);
    let builders = [OpBuilder<(ins
        "llvm::StringRef":$name, "mlir::FunctionType":$type,
        CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

        mlir::Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : Toy_Op<"generic_call"> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to
        be specialized for the shape of its arguments. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
        %4 = toy.generic_call @my_func(%1, %3)
            : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    // The generic call operation takes a symbol reference attribute as the
    // callee, and inputs for the call.
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

    // The generic call operation returns a single value of TensorType.
    let results = (outs F64Tensor);

    // Specialize assembly printing and parsing using a declarative format.
    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    // Add custom build methods for the generic call operation.
    let builders = [
        OpBuilder<(ins "llvm::StringRef":$callee, "llvm::ArrayRef<mlir::Value>":$arguments)>
    ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Toy_Op<"mul"> {
    let summary = "element-wise multiplication operation";
    let description = [{
        The "mul" operation performs element-wise multiplication between two
        tensors. The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}


//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : Toy_Op<"reshape"> {
    let summary = "tensor reshape operation";
    let description = [{
        Reshape operation is transforming its input tensor into a new tensor with
        the same number of elements but different shapes. For example:

        ```mlir
        %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);

    // We expect that the reshape operation returns a statically shaped tensor.
    let results = (outs StaticShapeTensorOf<[F64]>);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
        toy.func @foo() -> tensor<2xf64> {
            ...
            toy.return %0 : tensor<2xf64>
        }
        ```
    }];

    // The return operation takes an optional input operand to return. This
    // value must match the return type of the enclosing function.
    let arguments = (ins Variadic<F64Tensor>:$input);

    // The return operation only emits the input in the format if it is present.
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    // Allow building a ReturnOp with no return operand.
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    // Invoke a static verify method to verify this return operation.
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def TransposeOp : Toy_Op<"transpose"> {
    let summary = "transpose operation";

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    // Allow building a TransposeOp with from the input operand.
    let builders = [
        OpBuilder<(ins "mlir::Value":$input)>
    ];

    // Invoke a static verify method to verify this transpose operation.
    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}