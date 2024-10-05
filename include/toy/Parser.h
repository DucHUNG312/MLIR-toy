#pragma once

#include "toy/AST.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>
#include <vector>

namespace toy {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &Lexer) : Lexer(Lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule() {
    Lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time and accumulate in this vector.
    std::vector<FunctionAST> Functions;
    while (auto F = parseDefinition()) {
      Functions.push_back(std::move(*F));
      if (Lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (Lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(Functions));
  }

private:
  Lexer &Lexer;

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> parseReturn() {
    auto Loc = Lexer.getLastLocation();
    Lexer.consume(tok_return);

    // return takes an optional argument
    std::optional<std::unique_ptr<ExprAST>> Expr;
    if (Lexer.getCurToken() != ';') {
      Expr = parseExpression();
      if (!Expr)
        return nullptr;
    }
    return std::make_unique<ReturnExprAST>(std::move(Loc), std::move(Expr));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto Loc = Lexer.getLastLocation();
    auto Result =
        std::make_unique<NumberExprAST>(std::move(Loc), Lexer.getValue());
    Lexer.consume(tok_number);
    return std::move(Result);
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
    auto Loc = Lexer.getLastLocation();
    Lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<std::unique_ptr<ExprAST>> Values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> Dims;
    do {
      // We can have either another nested array or a number literal.
      if (Lexer.getCurToken() == '[') {
        Values.push_back(parseTensorLiteralExpr());
        if (!Values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (Lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        Values.push_back(parseNumberExpr());
      }

      // End of this list on ']'
      if (Lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (Lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      Lexer.getNextToken(); // eat ,
    } while (true);
    if (Values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    Lexer.getNextToken(); // eat ]

    /// Fill in the dimensions now. First the current nesting level:
    Dims.push_back(Values.size());

    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(Values, [](std::unique_ptr<ExprAST> &Expr) {
          return llvm::isa<LiteralExprAST>(Expr.get());
        })) {
      auto *FirstLiteral = llvm::dyn_cast<LiteralExprAST>(Values.front().get());
      if (!FirstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // Append the nested dimensions to the current level
      auto FirstDims = FirstLiteral->getDims();
      Dims.insert(Dims.end(), FirstDims.begin(), FirstDims.end());

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &Expr : Values) {
        auto *ExprLiteral = llvm::cast<LiteralExprAST>(Expr.get());
        if (!ExprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        if (ExprLiteral->getDims() != FirstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }
    return std::make_unique<LiteralExprAST>(std::move(Loc), std::move(Values),
                                            std::move(Dims));
  }

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr() {
    Lexer.getNextToken(); // eat (.
    auto V = parseExpression();
    if (!V)
      return nullptr;

    if (Lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    Lexer.consume(Token(')'));
    return V;
  }

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string Name(Lexer.getId());

    auto Loc = Lexer.getLastLocation();
    Lexer.getNextToken(); // eat identifier.

    if (Lexer.getCurToken() != '(') // Simple variable ref.
      return std::make_unique<VariableExprAST>(std::move(Loc), Name);

    // This is a function call.
    Lexer.consume(Token('('));
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (Lexer.getCurToken() != ')') {
      while (true) {
        if (auto Arg = parseExpression())
          Args.push_back(std::move(Arg));
        else
          return nullptr;

        if (Lexer.getCurToken() == ')')
          break;

        if (Lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        Lexer.getNextToken();
      }
    }
    Lexer.consume(Token(')'));

    // It can be a builtin call to print
    if (Name == "print") {
      if (Args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return std::make_unique<PrintExprAST>(std::move(Loc), std::move(Args[0]));
    }

    // Call to a user-defined function
    return std::make_unique<CallExprAST>(std::move(Loc), Name, std::move(Args));
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (Lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << Lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case '(':
      return parseParenExpr();
    case '[':
      return parseTensorLiteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int ExprPrec,
                                         std::unique_ptr<ExprAST> Lhs) {
    // If this is a binop, find its precedence.
    while (true) {
      int TokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (TokPrec < ExprPrec)
        return Lhs;

      // Okay, we know this is a binop.
      int BinOp = Lexer.getCurToken();
      Lexer.consume(Token(BinOp));
      auto Loc = Lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto Rhs = parsePrimary();
      if (!Rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with rhs than the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int NextPrec = getTokPrecedence();
      if (TokPrec < NextPrec) {
        Rhs = parseBinOpRHS(TokPrec + 1, std::move(Rhs));
        if (!Rhs)
          return nullptr;
      }

      // Merge lhs/RHS.
      Lhs = std::make_unique<BinaryExprAST>(std::move(Loc), BinOp,
                                            std::move(Lhs), std::move(Rhs));
    }
  }

  /// expression::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression() {
    auto Lhs = parsePrimary();
    if (!Lhs)
      return nullptr;

    return parseBinOpRHS(0, std::move(Lhs));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  std::unique_ptr<VarType> parseType() {
    if (Lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");
    Lexer.getNextToken(); // eat <

    auto Type = std::make_unique<VarType>();

    while (Lexer.getCurToken() == tok_number) {
      Type->Shape.push_back(Lexer.getValue());
      Lexer.getNextToken();
      if (Lexer.getCurToken() == ',')
        Lexer.getNextToken();
    }

    if (Lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");
    Lexer.getNextToken(); // eat >
    return Type;
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl ::= var identifier [ type ] = expr
  std::unique_ptr<VarDeclExprAST> parseDeclaration() {
    if (Lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    auto Loc = Lexer.getLastLocation();
    Lexer.getNextToken(); // eat var

    if (Lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identified",
                                        "after 'var' declaration");
    std::string Id(Lexer.getId());
    Lexer.getNextToken(); // eat id

    std::unique_ptr<VarType> Type; // Type is optional, it can be inferred
    if (Lexer.getCurToken() == '<') {
      Type = parseType();
      if (!Type)
        return nullptr;
    }

    if (!Type)
      Type = std::make_unique<VarType>();
    Lexer.consume(Token('='));
    auto Expr = parseExpression();
    return std::make_unique<VarDeclExprAST>(std::move(Loc), std::move(Id),
                                            std::move(*Type), std::move(Expr));
  }

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> parseBlock() {
    if (Lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    Lexer.consume(Token('{'));

    auto ExprList = std::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (Lexer.getCurToken() == ';')
      Lexer.consume(Token(';'));

    while (Lexer.getCurToken() != '}' && Lexer.getCurToken() != tok_eof) {
      if (Lexer.getCurToken() == tok_var) {
        // Variable declaration
        auto VarDecl = parseDeclaration();
        if (!VarDecl)
          return nullptr;
        ExprList->push_back(std::move(VarDecl));
      } else if (Lexer.getCurToken() == tok_return) {
        // Return statement
        auto Ret = parseReturn();
        if (!Ret)
          return nullptr;
        ExprList->push_back(std::move(Ret));
      } else {
        // General expression
        auto Expr = parseExpression();
        if (!Expr)
          return nullptr;
        ExprList->push_back(std::move(Expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (Lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // Ignore empty expressions: swallow sequences of semicolons.
      while (Lexer.getCurToken() == ';')
        Lexer.consume(Token(';'));
    }

    if (Lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    Lexer.consume(Token('}'));
    return ExprList;
  }

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> parsePrototype() {
    auto Loc = Lexer.getLastLocation();

    if (Lexer.getCurToken() != tok_def)
      return parseError<PrototypeAST>("def", "in prototype");
    Lexer.consume(tok_def);

    if (Lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    std::string FnName(Lexer.getId());
    Lexer.consume(tok_identifier);

    if (Lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    Lexer.consume(Token('('));

    std::vector<std::unique_ptr<VariableExprAST>> Args;
    if (Lexer.getCurToken() != ')') {
      do {
        std::string Name(Lexer.getId());
        auto Loc = Lexer.getLastLocation();
        Lexer.consume(tok_identifier);
        auto Decl = std::make_unique<VariableExprAST>(std::move(Loc), Name);
        Args.push_back(std::move(Decl));
        if (Lexer.getCurToken() != ',')
          break;
        Lexer.consume(Token(','));
        if (Lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }
    if (Lexer.getCurToken() != ')')
      return parseError<PrototypeAST>(")", "to end function prototype");

    // success.
    Lexer.consume(Token(')'));
    return std::make_unique<PrototypeAST>(std::move(Loc), FnName,
                                          std::move(Args));
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition() {
    auto Proto = parsePrototype();
    if (!Proto)
      return nullptr;

    if (auto Block = parseBlock())
      return std::make_unique<FunctionAST>(std::move(Proto), std::move(Block));
    return nullptr;
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(Lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(Lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&Expected, U &&Context = "") {
    auto CurToken = Lexer.getCurToken();
    llvm::errs() << "Parse error (" << Lexer.getLastLocation().Line << ", "
                 << Lexer.getLastLocation().Col << "): expected '" << Expected
                 << "' " << Context << " but has Token " << CurToken;
    if (isprint(CurToken))
      llvm::errs() << " '" << (char)CurToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace toy
