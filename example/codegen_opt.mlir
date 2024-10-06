module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.cast %0 : tensor<2x3xf64> to tensor<2x3xf64>
    %2 = toy.cast %0 : tensor<2x3xf64> to tensor<2x3xf64>
    %3 = toy.cast %0 : tensor<2x3xf64> to tensor<2x3xf64>
    %4 = toy.cast %0 : tensor<2x3xf64> to tensor<2x3xf64>
    %5 = toy.transpose(%3 : tensor<2x3xf64>) to tensor<3x2xf64>
    %6 = toy.transpose(%4 : tensor<2x3xf64>) to tensor<3x2xf64>
    %7 = toy.mul %5, %6 : tensor<3x2xf64>
    toy.print %7 : tensor<3x2xf64>
    toy.return
  }
}
