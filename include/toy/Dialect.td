include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Toy_Dialect : Dialect {
    let name = "toy";
    let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";
    let description = [{
        The Toy language is a tensor-based language that allows you to define
        functions, perform some math computation, and print results. This dialect
        provides a representation of the language that is amenable to analysis and
        optimization.
    }];
    let cppNamespace = "::toy";
}