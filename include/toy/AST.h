#pragma once

#include "toy/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace toy {

struct VarType {
  std::vector<int64_t> Shape;
};

class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind K, Location Loc) : Kind(K), Location(std::move(Loc)) {}

  ExprASTKind getKind() const { return Kind; }

  const Location &loc() { return Location; }

private:
  const ExprASTKind Kind;
  Location Location;
};

using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class NumberExprAST : public ExprAST {
public:
  NumberExprAST(struct Location Loc, double Val)
      : ExprAST(Expr_Num, std::move(Loc)), Value(Val) {}

  double getValue() { return Value; }

  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Num; }

private:
  double Value;
};

class LiteralExprAST : public ExprAST {
public:
  LiteralExprAST(struct Location Loc,
                 std::vector<std::unique_ptr<ExprAST>> Vals,
                 std::vector<int64_t> Ds)
      : ExprAST(Expr_Literal, std::move(Loc)), Values(std::move(Vals)),
        Dims(std::move(Ds)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return Values; }
  llvm::ArrayRef<int64_t> getDims() { return Dims; }

  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Literal; }

private:
  std::vector<std::unique_ptr<ExprAST>> Values;
  std::vector<int64_t> Dims;
};

class VariableExprAST : public ExprAST {
public:
  VariableExprAST(struct Location Loc, llvm::StringRef Name)
      : ExprAST(Expr_Var, std::move(Loc)), Name(Name) {}

  llvm::StringRef getName() { return Name; }

  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Var; }

private:
  std::string Name;
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
public:
  VarDeclExprAST(struct Location Loc, llvm::StringRef Name, VarType Type,
                 std::unique_ptr<ExprAST> InitVal)
      : ExprAST(Expr_VarDecl, std::move(Loc)), Name(Name),
        Type(std::move(Type)), InitVal(std::move(InitVal)) {}

  llvm::StringRef getName() { return Name; }
  ExprAST *getInitVal() { return InitVal.get(); }
  const VarType &getType() { return Type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_VarDecl; }

private:
  std::string Name;
  VarType Type;
  std::unique_ptr<ExprAST> InitVal;
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
public:
  ReturnExprAST(struct Location Loc, std::optional<std::unique_ptr<ExprAST>> E)
      : ExprAST(Expr_Return, std::move(Loc)), Expr(std::move(E)) {}

  std::optional<ExprAST *> getExpr() {
    if (Expr.has_value())
      return Expr->get();
    return std::nullopt;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Return; }

private:
  std::optional<std::unique_ptr<ExprAST>> Expr;
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
public:
  char getOp() { return Op; }
  ExprAST *getLHS() { return Lhs.get(); }
  ExprAST *getRHS() { return Rhs.get(); }

  BinaryExprAST(struct Location Loc, char O, std::unique_ptr<ExprAST> L,
                std::unique_ptr<ExprAST> R)
      : ExprAST(Expr_BinOp, std::move(Loc)), Op(O), Lhs(std::move(L)),
        Rhs(std::move(R)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_BinOp; }

private:
  char Op;
  std::unique_ptr<ExprAST> Lhs, Rhs;
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
public:
  CallExprAST(struct Location Loc, const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> A)
      : ExprAST(Expr_Call, std::move(Loc)), Callee(Callee), Args(std::move(A)) {
  }

  llvm::StringRef getCallee() { return Callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return Args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Call; }

private:
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
public:
  PrintExprAST(struct Location Loc, std::unique_ptr<ExprAST> A)
      : ExprAST(Expr_Print, std::move(Loc)), Arg(std::move(A)) {}

  ExprAST *getArg() { return Arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) { return C->getKind() == Expr_Print; }

private:
  std::unique_ptr<ExprAST> Arg;
};

class PrototypeAST {
public:
  PrototypeAST(Location Loc, const std::string &Name,
               std::vector<std::unique_ptr<VariableExprAST>> Args)
      : Location(std::move(Loc)), Name(Name), Args(std::move(Args)) {}

  const Location &loc() { return Location; }
  llvm::StringRef getName() const { return Name; }
  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return Args; }

private:
  Location Location;
  std::string Name;
  std::vector<std::unique_ptr<VariableExprAST>> Args;
};

class FunctionAST {
public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprASTList> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}

  PrototypeAST *getProto() { return Proto.get(); }
  ExprASTList *getBody() { return Body.get(); }

private:
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprASTList> Body;
};

class ModuleAST {
public:
  ModuleAST(std::vector<FunctionAST> Functions)
      : Functions(std::move(Functions)) {}

  auto begin() { return Functions.begin(); }
  auto end() { return Functions.end(); }

private:
  std::vector<FunctionAST> Functions;
};

void dump(ModuleAST &);

} // namespace toy