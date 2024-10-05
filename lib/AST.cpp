#include "toy/AST.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &Level) : Level(Level) { ++Level; }
  ~Indent() { --Level; }
  int &Level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *Node);

private:
  void dump(const VarType &Type);
  void dump(VarDeclExprAST *VarDecl);
  void dump(ExprAST *Expr);
  void dump(ExprASTList *ExprList);
  void dump(NumberExprAST *Num);
  void dump(LiteralExprAST *Node);
  void dump(VariableExprAST *Node);
  void dump(ReturnExprAST *Node);
  void dump(BinaryExprAST *Node);
  void dump(CallExprAST *Node);
  void dump(PrintExprAST *Node);
  void dump(PrototypeAST *Node);
  void dump(FunctionAST *Node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int Idx = 0; Idx < curIndent; Idx++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *Node) {
  const auto &Loc = Node->loc();
  return (llvm::Twine("@") + *Loc.File + ":" + llvm::Twine(Loc.Line) + ":" +
          llvm::Twine(Loc.Col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *Expr) {
  llvm::TypeSwitch<ExprAST *>(Expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
          [&](auto *Node) { this->dump(Node); })
      .Default([&](ExprAST *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Expr, kind " << Expr->getKind() << ">\n";
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *VarDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << VarDecl->getName();
  dump(VarDecl->getType());
  llvm::errs() << " " << loc(VarDecl) << "\n";
  dump(VarDecl->getInitVal());
}

/// A "block", or a list of expression
void ASTDumper::dump(ExprASTList *ExprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &Expr : *ExprList)
    dump(Expr.get());
  indent();
  llvm::errs() << "} // Block\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *Num) {
  INDENT();
  llvm::errs() << Num->getValue() << " " << loc(Num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
///    [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array with the dimensions spelled out at every level:
///    <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
void printLitHelper(ExprAST *LitOrNum) {
  // Inside a literal expression we can have either a number or another literal
  if (auto *Num = llvm::dyn_cast<NumberExprAST>(LitOrNum)) {
    llvm::errs() << Num->getValue();
    return;
  }
  auto *Literal = llvm::cast<LiteralExprAST>(LitOrNum);

  // Print the dimension for this literal first
  llvm::errs() << "<";
  llvm::interleaveComma(Literal->getDims(), llvm::errs());
  llvm::errs() << ">";

  // Now print the content, recursing on every element of the list
  llvm::errs() << "[ ";
  llvm::interleaveComma(Literal->getValues(), llvm::errs(),
                        [&](auto &Elt) { printLitHelper(Elt.get()); });
  llvm::errs() << "]";
}

/// Print a literal, see the recursive helper above for the implementation.
void ASTDumper::dump(LiteralExprAST *Node) {
  INDENT();
  llvm::errs() << "Literal: ";
  printLitHelper(Node);
  llvm::errs() << " " << loc(Node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableExprAST *Node) {
  INDENT();
  llvm::errs() << "var: " << Node->getName() << " " << loc(Node) << "\n";
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(ReturnExprAST *Node) {
  INDENT();
  llvm::errs() << "Return\n";
  if (Node->getExpr().has_value())
    return dump(*Node->getExpr());
  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(BinaryExprAST *Node) {
  INDENT();
  llvm::errs() << "BinOp: " << Node->getOp() << " " << loc(Node) << "\n";
  dump(Node->getLHS());
  dump(Node->getRHS());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *Node) {
  INDENT();
  llvm::errs() << "Call '" << Node->getCallee() << "' [ " << loc(Node) << "\n";
  for (auto &Arg : Node->getArgs())
    dump(Arg.get());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *Node) {
  INDENT();
  llvm::errs() << "Print [ " << loc(Node) << "\n";
  dump(Node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &Type) {
  llvm::errs() << "<";
  llvm::interleaveComma(Type.Shape, llvm::errs());
  llvm::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *Node) {
  INDENT();
  llvm::errs() << "Proto '" << Node->getName() << "' " << loc(Node) << "\n";
  indent();
  llvm::errs() << "Params: [";
  llvm::interleaveComma(Node->getArgs(), llvm::errs(),
                        [](auto &Arg) { llvm::errs() << Arg->getName(); });
  llvm::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *Node) {
  INDENT();
  llvm::errs() << "Function \n";
  dump(Node->getProto());
  dump(Node->getBody());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *Node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &F : *Node)
    dump(&F);
}

namespace toy {

// Public API
void dump(ModuleAST &Module) { ASTDumper().dump(&Module); }

} // namespace toy
