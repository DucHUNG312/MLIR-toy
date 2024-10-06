/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class CastInference;
namespace detail {
struct CastInferenceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*areCastCompatible)(const Concept *impl, ::mlir::Operation *, mlir::TypeRange, mlir::TypeRange);
    void (*inferShapes)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = CastInference;
    Model() : Concept{areCastCompatible, inferShapes} {}

    static inline bool areCastCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::TypeRange inputs, mlir::TypeRange outputs);
    static inline void inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = CastInference;
    FallbackModel() : Concept{areCastCompatible, inferShapes} {}

    static inline bool areCastCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::TypeRange inputs, mlir::TypeRange outputs);
    static inline void inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct CastInferenceTrait;

} // namespace detail
class CastInference : public ::mlir::OpInterface<CastInference, detail::CastInferenceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CastInference, detail::CastInferenceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CastInferenceTrait<ConcreteOp> {};
  /// Check if the input and output types are compatible for a cast operation.
  bool areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs);
  /// Infer the output shape of the cast operation from the input
  void inferShapes();
};
namespace detail {
  template <typename ConcreteOp>
  struct CastInferenceTrait : public ::mlir::OpInterface<CastInference, detail::CastInferenceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
bool detail::CastInferenceInterfaceTraits::Model<ConcreteOp>::areCastCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::TypeRange inputs, mlir::TypeRange outputs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areCastCompatible(inputs, outputs);
}
template<typename ConcreteOp>
void detail::CastInferenceInterfaceTraits::Model<ConcreteOp>::inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferShapes();
}
template<typename ConcreteOp>
bool detail::CastInferenceInterfaceTraits::FallbackModel<ConcreteOp>::areCastCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::TypeRange inputs, mlir::TypeRange outputs) {
  return static_cast<const ConcreteOp *>(impl)->areCastCompatible(tablegen_opaque_val, inputs, outputs);
}
template<typename ConcreteOp>
void detail::CastInferenceInterfaceTraits::FallbackModel<ConcreteOp>::inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->inferShapes(tablegen_opaque_val);
}
