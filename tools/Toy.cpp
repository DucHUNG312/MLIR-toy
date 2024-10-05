//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>

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

static cl::opt<bool> EnableOpt("opt", cl::desc("Enable optimizations"));

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

int loadMLIR(llvm::SourceMgr &SourceMgr, mlir::MLIRContext &Context,
             mlir::OwningOpRef<mlir::ModuleOp> &Modeule) {
  // Handle '.toy' input to the compiler.
  if (InType != InputType::MLIR &&
      !llvm::StringRef(InputFilename).ends_with(".mlir")) {
    auto ModuleAst = parseInputFile(InputFilename);
    if (!ModuleAst)
      return 6;
    Modeule = mlirGen(Context, *ModuleAst);
    return !Modeule ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code Ec = FileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << Ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  SourceMgr.AddNewSourceBuffer(std::move(*FileOrErr), llvm::SMLoc());
  Modeule = mlir::parseSourceFile<mlir::ModuleOp>(SourceMgr, &Context);
  if (!Modeule) {
    llvm::errs() << "Error can't load file " << InputFilename << "\n";
    return 3;
  }
  return 0;
}

int dumpMLIR() {
  mlir::MLIRContext Context;
  // Load our Dialect in this MLIR Context.
  Context.getOrLoadDialect<ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> Module;
  llvm::SourceMgr SourceMgr;
  mlir::SourceMgrDiagnosticHandler SourceMgrHandler(SourceMgr, &Context);
  if (int Error = loadMLIR(SourceMgr, Context, Module))
    return Error;

  if (EnableOpt) {
    mlir::PassManager Pm(Module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(Pm)))
      return 4;

    // Add a run of the canonicalizer to optimize the mlir module.
    Pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(Pm.run(*Module)))
      return 4;
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
  mlir::registerPassManagerCLOptions();

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
