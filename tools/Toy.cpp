#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> InType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> EmitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef Filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code Ec = FileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << Ec.message() << "\n";
    return nullptr;
  }
  auto Buffer = FileOrErr.get()->getBuffer();
  LexerBuffer Lexer(Buffer.begin(), Buffer.end(), std::string(Filename));
  Parser Parser(Lexer);
  return Parser.parseModule();
}

int dumpMLIR() {
  mlir::MLIRContext Context;
  // Load our Dialect in this MLIR Context.
  Context.getOrLoadDialect<toy::ToyDialect>();

  // Handle '.toy' input to the compiler.
  if (InType != InputType::MLIR &&
      !llvm::StringRef(InputFilename).ends_with(".mlir")) {
    auto ModuleAst = parseInputFile(InputFilename);
    if (!ModuleAst)
      return 6;
    mlir::OwningOpRef<mlir::ModuleOp> Module = mlirGen(Context, *ModuleAst);
    if (!Module)
      return 1;

    Module->dump();
    return 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code Ec = FileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << Ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr SourceMgr;
  SourceMgr.AddNewSourceBuffer(std::move(*FileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> Module =
      mlir::parseSourceFile<mlir::ModuleOp>(SourceMgr, &Context);
  if (!Module) {
    llvm::errs() << "Error can't load file " << InputFilename << "\n";
    return 3;
  }

  Module->dump();
  return 0;
}

int dumpAST() {
  if (InType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto ModuleAst = parseInputFile(InputFilename);
  if (!ModuleAst)
    return 1;

  dump(*ModuleAst);
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (EmitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
