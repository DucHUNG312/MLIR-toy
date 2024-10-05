#include "toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <mlir/IR/BuiltinAttributes.h>

#include "Dialect.cpp.inc"

namespace toy {

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &Parser,
                                       mlir::OperationState &Result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> Operands;
  llvm::SMLoc OperandsLoc = Parser.getCurrentLocation();
  mlir::Type Type;
  if (Parser.parseOperandList(Operands, /*requiredOperandCount=*/2) ||
      Parser.parseOptionalAttrDict(Result.attributes) ||
      Parser.parseColonType(Type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (mlir::FunctionType FuncType = llvm::dyn_cast<mlir::FunctionType>(Type)) {
    if (Parser.resolveOperands(Operands, FuncType.getInputs(), OperandsLoc,
                               Result.operands))
      return mlir::failure();
    Result.addTypes(FuncType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (Parser.resolveOperands(Operands, Type, Result.operands))
    return mlir::failure();
  Result.addTypes(Type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &Printer, mlir::Operation *Op) {
  Printer << " " << Op->getOperands();
  Printer.printOptionalAttrDict(Op->getAttrs());
  Printer << " : ";

  // If all of the types are the same, print the type directly.
  mlir::Type ResultType = *Op->result_type_begin();
  if (llvm::all_of(Op->getOperandTypes(),
                   [=](mlir::Type Type) { return Type == ResultType; })) {
    Printer << ResultType;
    return;
  }

  // Otherwise, print a functional type.
  Printer.printFunctionalType(Op->getOperandTypes(), Op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                       double Value) {
  auto DataType = mlir::RankedTensorType::get({}, Builder.getF64Type());
  auto DataAttr = mlir::DenseElementsAttr::get(DataType, Value);
  ConstantOp::build(Builder, State, DataType, DataAttr);
}

mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &Parser,
                                    ::mlir::OperationState &Result) {
  mlir::DenseElementsAttr Value;
  if (Parser.parseOptionalAttrDict(Result.attributes) ||
      Parser.parseAttribute(Value, "value", Result.attributes)) {
    return mlir::failure();
  }

  Result.addTypes(Value.getType());
  return mlir::success();
}

void ConstantOp::print(::mlir::OpAsmPrinter &Printer) {
  Printer << " ";
  Printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  Printer << getValue();
}

llvm::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto ResultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!ResultType)
    return mlir::success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto AttrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (AttrType.getRank() != ResultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << AttrType.getRank() << " != " << ResultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int Dim = 0, DimE = AttrType.getRank(); Dim < DimE; ++Dim) {
    if (AttrType.getShape()[Dim] != ResultType.getShape()[Dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << Dim << ": " << AttrType.getShape()[Dim]
             << " != " << ResultType.getShape()[Dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                  mlir::Value Lhs, mlir::Value Rhs) {
  State.addTypes(mlir::UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands({Lhs, Rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &Parser,
                               mlir::OperationState &Result) {
  return parseBinaryOp(Parser, Result);
}

void AddOp::print(mlir::OpAsmPrinter &P) { printBinaryOp(P, *this); }

llvm::LogicalResult AddOp::verify() {

  auto LhsType = llvm::cast<mlir::RankedTensorType>(getLhs().getType());
  auto RhsType = llvm::cast<mlir::RankedTensorType>(getRhs().getType());

  if (LhsType.getRank() != RhsType.getRank()) {
    return emitOpError("LHS type must match the one of the RHS "
                       "attribute: ")
           << LhsType.getRank() << " != " << RhsType.getRank();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                          llvm::StringRef Callee,
                          llvm::ArrayRef<mlir::Value> Args) {
  State.addTypes(mlir::UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands(Args);
  State.addAttribute("callee",
                     mlir::SymbolRefAttr::get(Builder.getContext(), Callee));
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                   llvm::StringRef Name, mlir::FunctionType Type,
                   llvm::ArrayRef<mlir::NamedAttribute> Attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(Builder, State, Name, Type, Attrs, Type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &Parser,
                                mlir::OperationState &Result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto BuildFuncType =
      [](mlir::Builder &Builder, llvm::ArrayRef<mlir::Type> ArgTypes,
         llvm::ArrayRef<mlir::Type> Results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return Builder.getFunctionType(ArgTypes, Results); };

  return mlir::function_interface_impl::parseFunctionOp(
      Parser, Result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(Result.name), BuildFuncType,
      getArgAttrsAttrName(Result.name), getResAttrsAttrName(Result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &P) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      P, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                  mlir::Value Lhs, mlir::Value Rhs) {
  State.addTypes(mlir::UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands({Lhs, Rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &Parser,
                               mlir::OperationState &Result) {
  return parseBinaryOp(Parser, Result);
}

void MulOp::print(mlir::OpAsmPrinter &P) { printBinaryOp(P, *this); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto Function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &Results = Function.getFunctionType().getResults();
  if (getNumOperands() != Results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << Results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto InputType = *operand_type_begin();
  auto ResultType = Results.front();

  // Check that the result type of the function matches the operand type.
  if (InputType == ResultType ||
      llvm::isa<mlir::UnrankedTensorType>(InputType) ||
      llvm::isa<mlir::UnrankedTensorType>(ResultType))
    return mlir::success();

  return emitError() << "type of return operand (" << InputType
                     << ") doesn't match function result type (" << ResultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                        mlir::Value Value) {
  State.addTypes(mlir::UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands(Value);
}

llvm::LogicalResult TransposeOp::verify() {
  auto InputType =
      llvm::dyn_cast<mlir::RankedTensorType>(getOperand().getType());
  auto ResultType = llvm::dyn_cast<mlir::RankedTensorType>(getType());
  if (!InputType || !ResultType)
    return mlir::success();

  auto InputShape = InputType.getShape();
  if (!std::equal(InputShape.begin(), InputShape.end(),
                  ResultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}

} // namespace toy

#define GET_OP_CLASSES
#include "Ops.cpp.inc"