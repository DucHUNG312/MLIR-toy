#pragma once

#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cctype>
#include <cstdio>
#include <memory>
#include <string>

namespace toy {

struct Location {
  std::shared_ptr<std::string> File;
  int Line;
  int Col;
};

enum Token : int8_t {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_number = -6,
};

class Lexer {
public:
  Lexer(std::string Filename)
      : LastLocation(
            {std::make_shared<std::string>(std::move(Filename)), 0, 0}) {}
  virtual ~Lexer() = default;

  Token getCurToken() { return CurTok; }

  Token getNextToken() { return CurTok = getTok(); }

  void consume(Token Tok) {
    assert(Tok == CurTok && "consume Token mismatch expectation");
    getNextToken();
  }

  llvm::StringRef getId() {
    assert(CurTok == tok_identifier);
    return IdentifierStr;
  }

  double getValue() {
    assert(CurTok == tok_number);
    return NumVal;
  }

  Location getLastLocation() { return LastLocation; }

  int getLine() { return CurLineNum; }

  int getCol() { return CurCol; }

private:
  virtual llvm::StringRef readNextLine() = 0;

  int getNextChar() {
    if (CurLineBuffer.empty()) {
      return EOF;
    }
    ++CurCol;
    auto NextChar = CurLineBuffer.front();
    CurLineBuffer = CurLineBuffer.drop_front();
    if (CurLineBuffer.empty()) {
      CurLineBuffer = readNextLine();
    }
    if (NextChar == '\n') {
      ++CurLineNum;
      CurCol = 0;
    }
    return NextChar;
  }

  Token getTok() {
    while (isspace(LastChar)) {
      LastChar = Token(getNextChar());
    }

    LastLocation.Line = CurLineNum;
    LastLocation.Col = CurCol;

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(LastChar)) {
      IdentifierStr = (char)LastChar;
      while (isalnum((LastChar = Token(getNextChar()))) || LastChar == '_')
        IdentifierStr += (char)LastChar;

      if (IdentifierStr == "return")
        return tok_return;
      if (IdentifierStr == "def")
        return tok_def;
      if (IdentifierStr == "var")
        return tok_var;
      return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(LastChar) || LastChar == '.') {
      std::string NumStr;
      do {
        NumStr += LastChar;
        LastChar = Token(getNextChar());
      } while (isdigit(LastChar) || LastChar == '.');

      NumVal = strtod(NumStr.c_str(), nullptr);
      return tok_number;
    }

    if (LastChar == '#') {
      // Comment until end of line.
      do {
        LastChar = Token(getNextChar());
      } while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

      if (LastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (LastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token ThisChar = Token(LastChar);
    LastChar = Token(getNextChar());
    return ThisChar;
  }

private:
  Token CurTok = tok_eof;

  Location LastLocation;

  std::string IdentifierStr;

  double NumVal = 0;

  Token LastChar = Token(' ');

  int CurLineNum = 0;

  int CurCol = 0;

  llvm::StringRef CurLineBuffer = "\n";
};

class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *Begin, const char *End, std::string Filename)
      : Lexer(std::move(Filename)), Current(Begin), End(End) {}

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override {
    auto *Begin = Current;
    while (Current <= End && *Current && *Current != '\n') {
      ++Current;
    }
    if (Current <= End && *Current) {
      ++Current;
    }
    llvm::StringRef Result{Begin, static_cast<size_t>(Current - Begin)};
    return Result;
  }

private:
  const char *Current;
  const char *End;
};

} // namespace toy