//
//  CorrectnessTests.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/29/23.
//

import Metal
import QuartzCore

class CorrectnessTests: MFATestCase {
  override class func typeDescription() -> String {
    "CorrectnessTests"
  }
  
  override func runQuickTests() {
    testRandomMatrices(logProgress: true)
  }
  
  func testRandomMatrices(logProgress: Bool) {
    let start = CACurrentMediaTime()
    
    //  0 -  25: batch 1, NN
    // 25 -  75: batch 1, NN/NT/TN/TT
    // 75 - 150: batch 2-16 for A/C
    let numNonTransposedTrials = 25
    let numNonBatchedTrials = 75
    let nonBroadcastedCutoff = 100
    let numTrials = numNonBatchedTrials + 75
    
    // Create a biased random distribution that favors smaller numbers. Take the
    // uniform distribution, then cube the results.
    let allRandomInts: [SIMD3<Int>] = (0..<numTrials).map { i in
      let maxMatrixDimension = (i < numNonBatchedTrials) ? 1000 : 500
      
      var randomVecFloat = SIMD3<Float>.random(in: 0..<1)
      randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
      var randomInts = SIMD3<Int>(randomVecFloat * Float(maxMatrixDimension))
      randomInts.replace(with: .one, where: randomInts .== .zero)
      return randomInts
    }
    
    let allRandomTransposes: [(Bool, Bool)] = (0..<numTrials).map { i in
      if i < numNonTransposedTrials {
        return (false, false)
      } else {
        return (Bool.random(), Bool.random())
      }
    }
    
    let allRandomB: [Int?] = (0..<numTrials).map { i in
      if i < numNonBatchedTrials {
        return nil
      } else {
        return Int.random(in: 2...16)
      }
    }
    
    func testRandomSize(index: Int, ghost: Bool) {
      let randomInts = allRandomInts[index]
      let randomTransposes = allRandomTransposes[index]
      let batchSize = allRandomB[index]
      
      let M = randomInts[0]
      let N = randomInts[1]
      let K = randomInts[2]
      let A_trans = randomTransposes.0
      let B_trans = randomTransposes.1
      let DTypeRepr = (Real.self == Float.self) ? "f32" : "f16"
      let transRepr = (A_trans ? "T" : "N") + (B_trans ? "T" : "N")
      
      var shapeA = A_trans ? [K, M] : [M, K]
      var shapeB = B_trans ? [N, K] : [K, N]
      var shapeC = [M, N]
      if let batchSize {
        shapeA = [batchSize] + shapeA
        if index < nonBroadcastedCutoff {
          if index % 2 == 0 {
            shapeB = [1] + shapeB
          }
        } else {
          shapeB = [batchSize] + shapeB
        }
        shapeC = [batchSize] + shapeC
      }
      
      let mps_A = Tensor<Real>(
        shape: shapeA, randomUniform: 0..<1, backend: .mps)
      let mps_B = Tensor<Real>(
        shape: shapeB, randomUniform: 0..<1, backend: .mps)
      var mps_C = Tensor<Real>(
        zerosLike: shapeC, backend: .mps)
      
      let mfa_A = Tensor(copying: mps_A, backend: .mfa)
      let mfa_B = Tensor(copying: mps_B, backend: .mfa)
      var mfa_C = Tensor(copying: mps_C, backend: .mfa)
      
      func act(A: Tensor<Real>, B: Tensor<Real>, C: inout Tensor<Real>) {
        C.matmul(A, B, transposeA: A_trans, transposeB: B_trans)
      }
      
      if ghost {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(A: mps_A, B: mps_B, C: &mps_C)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.withGhostExecution {
            TensorBackend.default.markFirstCommand()
            act(A: mfa_A, B: mfa_B, C: &mfa_C)
            TensorBackend.default.markLastCommand()
            _ = TensorBackend.default.synchronize()
          }
        }
      } else {
        _ExecutionContext.withDefaultBackend(.mps) {
          TensorBackend.default.markFirstCommand()
          act(A: mps_A, B: mps_B, C: &mps_C)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        _ExecutionContext.withDefaultBackend(.mfa) {
          TensorBackend.default.markFirstCommand()
          act(A: mfa_A, B: mfa_B, C: &mfa_C)
          TensorBackend.default.markLastCommand()
          _ = TensorBackend.default.synchronize()
        }
        
        let params = EuclideanDistanceParameters(
          matrixK: K, batchSize: batchSize)
        if !mfa_C.isApproximatelyEqual(to: mps_C, parameters: params) {
          MPL_showComparison(
            actual: mfa_C, actualName: "MFA",
            expected: mps_C, expectedName: "MPS", parameters: params)
          fatalError("Tensors did not match.")
        }
        if logProgress {
          var shapeRepr: String
          if let batchSize {
            shapeRepr = "\(batchSize)x\(M)x\(N)x\(K)"
          } else {
            shapeRepr = "\(M)x\(N)x\(K)"
          }
          print("Passed test: \(shapeRepr)x\(DTypeRepr) (\(transRepr))")
        }
      }
    }
    
    for i in 0..<numTrials {
      testRandomSize(index: i, ghost: true)
    }
    for i in 0..<numTrials {
      testRandomSize(index: i, ghost: false)
    }
    
    let end = CACurrentMediaTime()
    let repr = String(format: "%.3f", end - start)
    print("Finished 'testRandomMatrices' in \(repr) seconds.")
  }
  
//  let M = 100
//  let N = 50
//  let K = 25
//
//  typealias Real = Float32
//
//  let py_A = Tensor<Real>(shape: [K, M], randomUniform: 0..<1, backend: .numpy)
//  let py_B = Tensor<Real>(shape: [K, N], randomUniform: 0..<1, backend: .numpy)
//  var py_C = Tensor<Real>(zerosLike: [M, N], backend: .numpy)
//  _ExecutionContext.withDefaultBackend(.numpy) {
//    _ExecutionContext.profileCommands {
//      py_C.matmul(py_A, py_B, transposeA: true)
//    }
//  }
//
//  let mps_A = Tensor(copying: py_A, backend: .mps)
//  let mps_B = Tensor(copying: py_B, backend: .mps)
//  var mps_C = Tensor<Real>(zerosLike: [M, N], backend: .mps)
//  _ExecutionContext.withDefaultBackend(.mps) {
//    _ExecutionContext.profileCommands {
//      mps_C.matmul(mps_A, mps_B, transposeA: true)
//    }
//  }
//
//  let mfa_A = Tensor(copying: py_A, backend: .mfa)
//  let mfa_B = Tensor(copying: py_B, backend: .mfa)
//  var mfa_C = Tensor<Real>(zerosLike: [M, N], backend: .mfa)
//  _ExecutionContext.withDefaultBackend(.mfa) {
//    _ExecutionContext.profileCommands {
//      mfa_C.matmul(mfa_A, mfa_B, transposeA: true)
//    }
//  }
//
//  MPL_showBackends(
//    mfa: mfa_C, mps: mps_C, numpy: py_C,
//    parameters: .init(matrixK: K))
}