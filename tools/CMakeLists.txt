create_subdirectory_options(TOY TOOL)

set(LLVM_LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    Analysis
    CodeGen
    Core
    IPO
    AggressiveInstCombine
    InstCombine
    Instrumentation
    MC
    MCParser
    ObjCARCOpts
    Option
    ScalarOpts
    Support
    TransformUtils
    Vectorize
    )

add_toy_tool(toy
    Toy.cpp
    )

target_link_libraries(toy PRIVATE
    toyMLIR
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
    )