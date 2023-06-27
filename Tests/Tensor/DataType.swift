//
//  DataType.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

// Used for setting function constants.
protocol MTLConvertible {
  static var mtlDataType: MTLDataType { get }
}

extension Bool: MTLConvertible {
  static var mtlDataType: MTLDataType { .bool }
}

extension UInt16: MTLConvertible {
  static var mtlDataType: MTLDataType { .ushort }
}

extension UInt32: MTLConvertible {
  static var mtlDataType: MTLDataType { .uint }
}

// MARK: - TensorElement

// Uses for declaring types of tensors.
protocol TensorElement: MTLConvertible { }

extension Float16: TensorElement {
  static var mtlDataType: MTLDataType { .half }
}

extension Float: TensorElement {
  static var mtlDataType: MTLDataType { .float }
}

// MARK: - MTLDataType Extensions

extension MTLDataType {
  private func unrecognizedError() -> Never {
    fatalError("MTLDataType with code \(self.rawValue) not recognized.")
  }
  
  var numpy: PythonObject {
    let ctx = PythonContext.global
    switch self {
    case .half:
      return ctx.np.float16
    case .float:
      return ctx.np.float32
    default:
      unrecognizedError()
    }
  }
  
  var size: Int {
    switch self {
    case .half:
      return 2
    case .float:
      return 4
    default:
      unrecognizedError()
    }
  }
  
  /*
   let input1 = bufferC_mfa.contents()
   let input2 = bufferC_mps.contents()
   var tolerance = Float(B * M * N) * sqrt(Float(K))
   if precision == .f32 {
     tolerance = max(0.001, 3e-7 * tolerance)
   } else {
     // Up the tolerance a little for FP16
     tolerance = max(0.01, 1e-2 * tolerance)
     //        tolerance = max(0.01, 5e-3 * tolerance)
   }
   */
}