//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore
import FlashAttention

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.
@main
struct Script {
  static func main() throws {
    // Tasks:
    // - Support the attention variant that stores dS^T to memory.
    // - Compare performance with FP32 and BF16 storage.
    
    // Currently, this is scratch space for encoding the "store dS" variant of
    // backward attention. The author was running the first performance tests
    // ever done, for backward FlashAttention on Apple silicon. Some data is
    // recorded, but not yet in a legible state for publication.
    
    // Define the problem dimensions.
    testAsyncCopies()
  }

  static func testForward() {
    for N in [128, 256, 512, 1024, 2048, 4096, 8192] {
      var samples: [Int] = []
      for D in [32, 48, 64, 80, 96, 128, 160, 192, 256] {
        let performance = profileProblemSize(sequenceDimension: N, headDimension: D)
        samples.append(performance)
      }
      for sample in samples {
        print(sample, terminator: ", ")
      }
      print()
    }
  }

  static func testForwardAnddQdKdV() {
    let N = 7680
    for D in [8, 16, 32, 64, 96, 128, 160, 256, 384] {
      let fwd = profileProblemSize(sequenceDimension: N, headDimension: D)
      print(fwd, terminator: ", ")
      let dQ = profileProblemSize(sequenceDimension: N, headDimension: D, benchmarkedKernel: .backwardQuery)
      print(dQ, terminator: ", ")
      let dKdV = profileProblemSize(sequenceDimension: N, headDimension: D, benchmarkedKernel: .backwardKeyValue)
      print(dKdV)
    }
  }

  static func testAsyncCopies() {
    print("Forward")
    for N in [128, 256, 512, 1024, 2048, 4096, 8192] {
      var samples: [Int] = []
      for D in [32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 512] {
        let performance = profileProblemSize(sequenceDimension: N, headDimension: D)
        samples.append(performance)
      }
      for sample in samples {
        print(sample, terminator: ", ")
      }
      print()
    }
    print("BackwardQuery")
    for N in [128, 256, 512, 1024, 2048, 4096, 8192] {
      var samples: [Int] = []
      for D in [32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 512] {
        let performance = profileProblemSize(sequenceDimension: N, headDimension: D, benchmarkedKernel: .backwardQuery)
        samples.append(performance)
      }
      for sample in samples {
        print(sample, terminator: ", ")
      }
      print()
    }
    print("BackwardKeyValue")
    for N in [128, 256, 512, 1024, 2048, 4096, 8192] {
      var samples: [Int] = []
      for D in [32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 512] {
        let performance = profileProblemSize(sequenceDimension: N, headDimension: D, benchmarkedKernel: .backwardKeyValue)
        samples.append(performance)
      }
      for sample in samples {
        print(sample, terminator: ", ")
      }
      print()
    }
  }
}

// Returns throughput in gigainstructions per second.
private func profileProblemSize(
  sequenceDimension: Int,
  headDimension: Int,
  benchmarkedKernel: AttentionKernelType = .forward
) -> Int {
  autoreleasepool {
    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = sequenceDimension
    networkDesc.columnDimension = sequenceDimension
    networkDesc.headDimension = headDimension
    let network = Network(descriptor: networkDesc)
    
    // MARK: - Kernels
    
    var attentionDesc = AttentionDescriptor()
    attentionDesc.lowPrecisionInputs = false
    attentionDesc.lowPrecisionIntermediates = false
    attentionDesc.matrixDimensions = (
      row: UInt32(sequenceDimension),
      column: UInt32(sequenceDimension),
      head: UInt16(headDimension))
    attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
    
    func createKernel(type: AttentionKernelType) -> AttentionKernel {
      let attentionKernelDesc = attentionDesc.kernelDescriptor(type: type)
      let attentionKernel = AttentionKernel(descriptor: attentionKernelDesc)
      return attentionKernel
    }
    let kernelForward = createKernel(type: .forward)
    let kernelBackwardQuery = createKernel(type: .backwardQuery)
    let kernelBackwardKeyValue = createKernel(type: .backwardKeyValue)
    
    func createPipeline(kernel: AttentionKernel) -> MTLComputePipelineState {
      let device = MTLContext.global.device
      let source = kernel.createSource()
      let library = try! device.makeLibrary(source: source, options: nil)
      
      let functionConstants = MTLFunctionConstantValues()
      attentionDesc.setFunctionConstants(functionConstants)
      let function = try! library.makeFunction(
        name: "attention", constantValues: functionConstants)
      
      // A critical part of the heuristic: force the occupancy to 1024 on M1.
      let pipelineDesc = MTLComputePipelineDescriptor()
      pipelineDesc.computeFunction = function
      pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
      return try! device.makeComputePipelineState(
        descriptor: pipelineDesc, options: [], reflection: nil)
    }
    let pipelineForward = createPipeline(kernel: kernelForward)
    let pipelineBackwardQuery = createPipeline(kernel: kernelBackwardQuery)
    let pipelineBackwardKeyValue = createPipeline(kernel: kernelBackwardKeyValue)
    
    // MARK: - Buffers
    
    // Utility function to make buffer initialization more concise.
    func createBuffer(
      _ array: [Float],
      _ operand: AttentionOperand
    ) -> MTLBuffer {
      let memoryPrecisions = attentionDesc.memoryPrecisions
      guard let precision = memoryPrecisions[operand] else {
        fatalError("Precision of operand \(operand) was not specified.")
      }
      return MTLContext.global.createBuffer(array, precision)
    }
    
    let operandSize = sequenceDimension * headDimension
    var resultO = [Float](repeating: .zero, count: operandSize)
    let resultL = [Float](repeating: .zero, count: sequenceDimension)
    let resultD = [Float](repeating: .zero, count: sequenceDimension)
    let resultDerivativeV = [Float](repeating: .zero, count: operandSize)
    let resultDerivativeK = [Float](repeating: .zero, count: operandSize)
    let resultDerivativeQ = [Float](repeating: .zero, count: operandSize)
    resultO[0] = .nan
    
    let bufferQ = createBuffer(network.Q, .Q)
    let bufferK = createBuffer(network.K, .K)
    let bufferV = createBuffer(network.V, .V)
    let bufferDerivativeO = createBuffer(network.dO, .dO)
    
    let bufferL = createBuffer(resultL, .L)
    let bufferD = createBuffer(resultD, .D)
    
    let bufferO = createBuffer(resultO, .O)
    let bufferDerivativeV = createBuffer(resultDerivativeV, .dV)
    let bufferDerivativeK = createBuffer(resultDerivativeK, .dK)
    let bufferDerivativeQ = createBuffer(resultDerivativeQ, .dQ)
    
    // MARK: - GPU Commands
    
    // - Parameter dispatchCount: Number of times to duplicate the FWD / BWD
    //                            combined pass.
    // - Returns: Latency of the entire command buffer, in seconds.
    @discardableResult
    func executeCommandBuffer(
      dispatchCount: Int
    ) -> Double {
      let commandQueue = MTLContext.global.commandQueue
      let commandBuffer = commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      
      func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
        (target + Int(granularity) - 1) / Int(granularity)
      }
      
      // Bind all necessary MTLBuffer arguments before calling this function.
      func dispatch(
        kernel: AttentionKernel,
        pipeline: MTLComputePipelineState,
        along parallelizationDimension: Int
      ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setThreadgroupMemoryLength(
          Int(kernel.threadgroupMemoryAllocation), index: 0)
        
        let blockCount = ceilDivide(
          parallelizationDimension, kernel.blockDimensions.parallelization)
        let gridSize = MTLSize(
          width: blockCount,
          height: 1,
          depth: 1)
        let groupSize = MTLSize(
          width: Int(kernel.threadgroupSize),
          height: 1,
          depth: 1)
        encoder.dispatchThreadgroups(
          gridSize, threadsPerThreadgroup: groupSize)
      }
      
      encoder.setBuffer(bufferQ, offset: 0, index: 0)
      encoder.setBuffer(bufferK, offset: 0, index: 1)
      encoder.setBuffer(bufferV, offset: 0, index: 2)
      encoder.setBuffer(bufferO, offset: 0, index: 3)
      
      encoder.setBuffer(bufferL, offset: 0, index: 4)
      encoder.setBuffer(bufferD, offset: 0, index: 5)
      
      encoder.setBuffer(bufferDerivativeO, offset: 0, index: 6)
      encoder.setBuffer(bufferDerivativeV, offset: 0, index: 7)
      encoder.setBuffer(bufferDerivativeK, offset: 0, index: 8)
      encoder.setBuffer(bufferDerivativeQ, offset: 0, index: 9)
      
      for _ in 0..<dispatchCount {
        switch benchmarkedKernel {
        case .forward:
          dispatch(
            kernel: kernelForward,
            pipeline: pipelineForward,
            along: sequenceDimension)
        case .backwardQuery:
          dispatch(
            kernel: kernelBackwardQuery,
            pipeline: pipelineBackwardQuery,
            along: sequenceDimension)
        case .backwardKeyValue:
          dispatch(
            kernel: kernelBackwardKeyValue,
            pipeline: pipelineBackwardKeyValue,
            along: sequenceDimension)
        }
      }
      
      encoder.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      
      // Determine the time taken.
      let start = commandBuffer.gpuStartTime
      let end = commandBuffer.gpuEndTime
      let latency = end - start
      return latency
    }
    
    // MARK: - Profiling
    
    // Benchmark performance.
    var maxGINSTRS: Int = .zero
    for _ in 0..<5 {
      let dispatchCount: Int = 5
      let latencySeconds = executeCommandBuffer(dispatchCount: dispatchCount)
      
      // Determine the amount of work done.
      //
      // WARNING: Change this code to match the kernel you're profiling.
      var operations: Int
      switch benchmarkedKernel {
      case .forward:
        operations = 2 * headDimension + 5
      case .backwardQuery:
        operations = 3 * headDimension + 5
      case .backwardKeyValue:
        operations = 4 * headDimension + 5
      }
      operations *= (sequenceDimension * sequenceDimension)
      operations *= dispatchCount
      
      // Divide the work by the latency, resulting in throughput.
      let instrs = Double(operations) / Double(latencySeconds)
      let ginstrs = Int(instrs / 1e9)
      
      // Accumulate the sample from this trial.
      maxGINSTRS = max(maxGINSTRS, ginstrs)
    }
    return maxGINSTRS
  }
}
