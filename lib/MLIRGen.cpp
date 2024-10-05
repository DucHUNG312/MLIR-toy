#include "toy/MLIRGen.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &Context) : Builder(&Context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &ModuleAst) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    TheModule = mlir::ModuleOp::create(Builder.getUnknownLoc());

    for (FunctionAST &F : ModuleAst)
      mlirGen(F);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(TheModule))) {
      TheModule.emitError("module verification error");
      return nullptr;
    }

    return TheModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp TheModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder Builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> SymbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(const Location &Loc) {
    return mlir::FileLineColLoc::get(Builder.getStringAttr(*Loc.File), Loc.Line,
                                     Loc.Col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  llvm::LogicalResult declare(llvm::StringRef Var, mlir::Value Value) {
    if (SymbolTable.count(Var))
      return mlir::failure();
    SymbolTable.insert(Var, Value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  FuncOp mlirGen(PrototypeAST &Proto) {
    auto Location = loc(Proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> ArgTypes(Proto.getArgs().size(),
                                              getType(VarType{}));
    auto FuncType = Builder.getFunctionType(ArgTypes, std::nullopt);
    return Builder.create<FuncOp>(Location, Proto.getName(), FuncType);
  }

  /// Emit a new function and add it to the MLIR module.
  FuncOp mlirGen(FunctionAST &FuncAst) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> VarScope(SymbolTable);

    // Create an MLIR function for the given prototype.
    Builder.setInsertionPointToEnd(TheModule.getBody());
    FuncOp Function = mlirGen(*FuncAst.getProto());
    if (!Function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &EntryBlock = Function.front();
    auto ProtoArgs = FuncAst.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto NameValue :
         llvm::zip(ProtoArgs, EntryBlock.getArguments())) {
      if (failed(declare(std::get<0>(NameValue)->getName(),
                         std::get<1>(NameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    Builder.setInsertionPointToStart(&EntryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*FuncAst.getBody()))) {
      Function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp RetOp;
    if (!EntryBlock.empty())
      RetOp = dyn_cast<ReturnOp>(EntryBlock.back());
    if (!RetOp) {
      Builder.create<ReturnOp>(loc(FuncAst.getProto()->loc()));
    } else if (RetOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      Function.setType(Builder.getFunctionType(
          Function.getFunctionType().getInputs(), getType(VarType{})));
    }

    return Function;
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &Binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value Lhs = mlirGen(*Binop.getLHS());
    if (!Lhs)
      return nullptr;
    mlir::Value Rhs = mlirGen(*Binop.getRHS());
    if (!Rhs)
      return nullptr;
    auto Location = loc(Binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (Binop.getOp()) {
    case '+':
      return Builder.create<AddOp>(Location, Lhs, Rhs);
    case '*':
      return Builder.create<MulOp>(Location, Lhs, Rhs);
    }

    emitError(Location, "invalid binary operator '") << Binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &Expr) {
    if (auto Variable = SymbolTable.lookup(Expr.getName()))
      return Variable;

    emitError(loc(Expr.loc()), "error: unknown variable '")
        << Expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  llvm::LogicalResult mlirGen(ReturnExprAST &Ret) {
    auto Location = loc(Ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value Expr = nullptr;
    if (Ret.getExpr().has_value()) {
      if (!(Expr = mlirGen(**Ret.getExpr())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    Builder.create<ReturnOp>(Location,
                             Expr ? ArrayRef(Expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen(LiteralExprAST &Lit) {
    auto Type = getType(Lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> Data;
    Data.reserve(std::accumulate(Lit.getDims().begin(), Lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(Lit, Data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type ElementType = Builder.getF64Type();
    auto DataType = mlir::RankedTensorType::get(Lit.getDims(), ElementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto DataAttribute =
        mlir::DenseElementsAttr::get(DataType, llvm::ArrayRef(Data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return Builder.create<ConstantOp>(loc(Lit.loc()), Type, DataAttribute);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &Expr, std::vector<double> &Data) {
    if (auto *Lit = dyn_cast<LiteralExprAST>(&Expr)) {
      for (auto &Value : Lit->getValues())
        collectData(*Value, Data);
      return;
    }

    assert(isa<NumberExprAST>(Expr) && "expected literal or number expr");
    Data.push_back(cast<NumberExprAST>(Expr).getValue());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen(CallExprAST &Call) {
    llvm::StringRef Callee = Call.getCallee();
    auto Location = loc(Call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> Operands;
    for (auto &Expr : Call.getArgs()) {
      auto Arg = mlirGen(*Expr);
      if (!Arg)
        return nullptr;
      Operands.push_back(Arg);
    }

    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.
    if (Callee == "transpose") {
      if (Call.getArgs().size() != 1) {
        emitError(Location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return Builder.create<TransposeOp>(Location, Operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return Builder.create<GenericCallOp>(Location, Callee, Operands);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  llvm::LogicalResult mlirGen(PrintExprAST &Call) {
    auto Arg = mlirGen(*Call.getArg());
    if (!Arg)
      return mlir::failure();

    Builder.create<PrintOp>(loc(Call.loc()), Arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &Num) {
    return Builder.create<ConstantOp>(loc(Num.loc()), Num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &Expr) {
    switch (Expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(Expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(Expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(Expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(Expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(Expr));
    default:
      emitError(loc(Expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(Expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &Vardecl) {
    auto *Init = Vardecl.getInitVal();
    if (!Init) {
      emitError(loc(Vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value Value = mlirGen(*Init);
    if (!Value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!Vardecl.getType().Shape.empty()) {
      Value = Builder.create<ReshapeOp>(loc(Vardecl.loc()),
                                        getType(Vardecl.getType()), Value);
    }

    // Register the value in the symbol table.
    if (failed(declare(Vardecl.getName(), Value)))
      return nullptr;
    return Value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  llvm::LogicalResult mlirGen(ExprASTList &BlockAst) {
    ScopedHashTableScope<StringRef, mlir::Value> VarScope(SymbolTable);
    for (auto &Expr : BlockAst) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *Vardecl = dyn_cast<VarDeclExprAST>(Expr.get())) {
        if (!mlirGen(*Vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *Ret = dyn_cast<ReturnExprAST>(Expr.get()))
        return mlirGen(*Ret);
      if (auto *Print = dyn_cast<PrintExprAST>(Expr.get())) {
        if (mlir::failed(mlirGen(*Print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*Expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> Shape) {
    // If the shape is empty, then this type is unranked.
    if (Shape.empty())
      return mlir::UnrankedTensorType::get(Builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(Shape, Builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const VarType &Type) { return getType(Type.Shape); }
};

} // namespace

namespace toy {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          ModuleAST &ModuleAst) {
  return MLIRGenImpl(Context).mlirGen(ModuleAst);
}

} // namespace toy
