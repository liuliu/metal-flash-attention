//
//  Attention.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import MetalPerformanceShadersGraph
import PythonKit

protocol Attention: Operation {
  typealias Tensors = Attention_Tensors
  var parameters: Attention_Parameters { get set }
  
  init(parameters: Attention_Parameters)
}

extension Attention {
  func equals(_ other: Attention) -> Bool {
    (type(of: self) == type(of: other)) && (parameters == other.parameters)
  }
}

// Broadcasting only supported along the mask.
struct Attention_Parameters: Hashable, Equatable {
  var dataType: MTLDataType
  var R: Int
  var C: Int
  var H: Int
  var D: Int
  var Q_trans: Bool
  var K_trans: Bool
  var V_trans: Bool
  var O_trans: Bool
  var batched: Bool
  var masked: Bool
  var blockSparse: Bool
  var accumulateInFloat: Bool
  
  // These are only needed by MPSGraph; MFA supports dynamic batch size.
  var batchDimensionsQ: [Int]?
  var batchDimensionsMask: [Int]?
}

struct Attention_Tensors {
  var q: TensorBuffer
  var k: TensorBuffer
  var v: TensorBuffer
  var o: TensorBuffer
  var mask: TensorBuffer?
}

struct MFA_Attention: Attention, MFA_Operation {
  var parameters: Attention_Parameters
  
  static var functionConstants: [String: MTLConvertible] = [
    :
  ]
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncResource() -> AsyncPipeline {
    let dataType = parameters.dataType
    precondition(dataType == .float || dataType == .half)
    
    let constants = MTLFunctionConstantValues()
    var pcopy = self.parameters
    constants.setConstantValue(&pcopy.R, type: .uint, index: 0)
    constants.setConstantValue(&pcopy.C, type: .uint, index: 1)
    constants.setConstantValue(&pcopy.H, type: .uint, index: 2)
    constants.setConstantValue(&pcopy.D, type: .uint, index: 3)
    constants.setConstantValue(&pcopy.Q_trans, type: .bool, index: 10)
    constants.setConstantValue(&pcopy.K_trans, type: .bool, index: 11)
    constants.setConstantValue(&pcopy.V_trans, type: .bool, index: 12)
    constants.setConstantValue(&pcopy.O_trans, type: .bool, index: 13)
    
    var alpha = rsqrt(Float(pcopy.D))
    constants.setConstantValue(&alpha, type: .float, index: 20)
    
    var dataTypeRawValue = dataType.rawValue
    constants.setConstantValue(&dataTypeRawValue, type: .uint, index: 30)
    constants.setConstantValue(&pcopy.batched, type: .bool, index: 100)
    constants.setConstantValue(&pcopy.masked, type: .bool, index: 50000)
    constants.setConstantValue(&pcopy.blockSparse, type: .bool, index: 102)
    constants.setConstantValue(&pcopy.accumulateInFloat, type: .bool, index: 114)
    
    var triangular = false
    #if false
    if parameters.masked && parameters.blockSparse {
      // If masked and R ≈ C, is it very likely this is causal self-attention.
      let ratio = Float(parameters.R) / Float(parameters.C)
      let ratioCutoff: Float = 1.1
      let additiveCutoff: Int = 128
      if abs(parameters.R - parameters.C) < additiveCutoff {
        triangular = true
      } else if 1 / ratioCutoff < ratio, ratio < ratioCutoff {
        triangular = true
      }
    }
    #endif
    constants.setConstantValue(&triangular, type: .bool, index: 103)
    
    var forward = true
    var backward = false
    var generateBlockMask = false
    var groupedQuery = false
    constants.setConstantValue(&forward, type: .bool, index: 110)
    constants.setConstantValue(&backward, type: .bool, index: 111)
    constants.setConstantValue(&generateBlockMask, type: .bool, index: 112)
    constants.setConstantValue(&groupedQuery, type: .bool, index: 113)
    
    var R_simd: UInt16
    var C_simd: UInt16
    var R_splits: UInt16
    var fuseAsyncLoads = false
    if dataType == .float {
      R_simd = 8
      C_simd = 32
      R_splits = 4
    } else {
      let D = pcopy.D
      if pcopy.masked {
        if D <= 16 {
          R_simd = 16
          C_simd = 64
          R_splits = 4
        } else if D <= 24 {
          R_simd = 8
          C_simd = 64
          R_splits = 8
        } else if D <= 80 {
          R_simd = 8
          C_simd = 64
          R_splits = 4
        } else {
          R_simd = 8
          C_simd = 32
          R_splits = 4
        }
      } else {
        R_simd = 8
        R_splits = 8
        
        if D <= 8 {
          R_simd = 16
          C_simd = 64
        } else if D <= 16 {
          C_simd = 72
          fuseAsyncLoads = true
        } else if D <= 24 {
          C_simd = 56
          fuseAsyncLoads = true
        } else if D <= 56 {
          C_simd = 64
        } else if D <= 64 {
          C_simd = 40
          fuseAsyncLoads = true
        } else if D <= 96 {
          C_simd = 64
        } else if D <= 304 {
          C_simd = 32
          R_splits = 4
        } else {
          C_simd = 40
          R_splits = 8
        }
      }
    }
    
    constants.setConstantValue(&R_simd, type: .ushort, index: 200)
    constants.setConstantValue(&C_simd, type: .ushort, index: 201)
    constants.setConstantValue(&R_splits, type: .ushort, index: 210)
    if fuseAsyncLoads {
      constants.setConstantValue(&fuseAsyncLoads, type: .bool, index: 213)
    }
    
    let D_simd = UInt16(pcopy.D + 7) / 8 * 8
    let R_group = R_simd * R_splits
    var K_block_length: UInt16
    var V_block_length: UInt16
    var Q_block_length: UInt16
    var O_block_length: UInt16
    
    var R_block_dim = R_group
    var C_block_dim = C_simd
    var D_block_dim = D_simd
    func setBankOffset(_ dim: inout UInt16, index: Int) {
      precondition(dim % 8 == 0, "Original dimension must be divisible by 8.")
      let dimBytes = dim * UInt16(dataType.size)
      
      // How the heuristic works:
      //
      // FP16:
      // Pad 8 -> 8       (16B -> 16B)
      // Pad 16 -> 24     (32B -> 48B)
      // Pad 24 -> 24     (48B -> 48B)
      // Pad 32 -> 36, 40 (64B -> 72B, 80B)
      // Pad 40 -> 40     (80B -> 80B)
      // Pad 48 -> 52, 56 (96B -> 104B, 112B)
      // Pad 56 -> 56     (112B -> 112B)
      // Pad 64 -> 72     (128B -> 144B)
      // Pad 80 -> 88     (160B -> 176B)
      // Pad 96 -> 104    (192B -> 208B)
      let dimBytesModulo = dimBytes % 64
      if dimBytesModulo == 16 || dimBytesModulo == 48 {
        return
      } else if dimBytesModulo == 0 || dimBytesModulo == 32 {
        let bankOffsetBytes: UInt16 = 16
        var bankOffset = bankOffsetBytes / UInt16(dataType.size)
        dim += bankOffset
        constants.setConstantValue(&bankOffset, type: .ushort, index: index)
      } else {
        fatalError("This should never happen.")
      }
    }
    setBankOffset(&R_block_dim, index: 220)
    setBankOffset(&C_block_dim, index: 221)
    setBankOffset(&D_block_dim, index: 222)
    
    let library = MetalContext.global.library
    var functions = [
      try! library.makeFunction(
        name: "attention", constantValues: constants)
    ]
    if parameters.blockSparse {
      var generateBlockMask = true
      constants.setConstantValue(&generateBlockMask, type: .bool, index: 112)
      functions.append(try! library.makeFunction(
        name: "attention", constantValues: constants))
    }
    
    if parameters.Q_trans {
      Q_block_length = D_simd * R_block_dim
    } else {
      Q_block_length = R_group * D_block_dim
    }
    if parameters.K_trans {
      K_block_length = C_simd * D_block_dim
    } else {
      K_block_length = D_simd * C_block_dim
    }
    if parameters.V_trans {
      V_block_length = D_simd * C_block_dim
    } else {
      V_block_length = C_simd * D_block_dim
    }
    if parameters.O_trans {
      O_block_length = D_simd * R_block_dim
    } else {
      O_block_length = R_group * D_block_dim
    }
    
    var deviceElements: UInt64
    if triangular {
      let floats_per_cacheline = 128 / 4
      var num_lm_elements = Int(2 * R_group)
      num_lm_elements += floats_per_cacheline - 1
      num_lm_elements /= floats_per_cacheline
      num_lm_elements *= floats_per_cacheline
      
      var num_O_elements = Int(R_group * D_simd)
      num_O_elements += floats_per_cacheline - 1
      num_O_elements /= floats_per_cacheline
      num_O_elements *= floats_per_cacheline
      
      let head_O_blocks = (parameters.R + Int(R_group) - 1) / Int(R_group)
      deviceElements = UInt64(num_lm_elements + num_O_elements)
      deviceElements *= UInt64(parameters.H * head_O_blocks)
    } else {
      deviceElements = 0
    }
    var deviceBytes = [
      deviceElements * 4
    ]
    
    var blockElements: UInt16
    if fuseAsyncLoads {
      blockElements = K_block_length + V_block_length
    } else {
      blockElements = max(K_block_length, V_block_length)
    }
    blockElements = max(blockElements, Q_block_length)
    blockElements = max(blockElements, O_block_length)
    var blockBytes = [
      blockElements * UInt16(dataType.size)
    ]
    
    func ceilDivide(target: Int, granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    var gridX = ceilDivide(target: parameters.R, granularity: R_group)
    if triangular {
      let completeBlocks = parameters.R / Int(R_group)
      let upperBlocks = completeBlocks / 2
      let lowerBlocks = completeBlocks - upperBlocks
      let edgeBlocks = gridX - completeBlocks
      precondition(lowerBlocks >= upperBlocks)
      
      gridX = (lowerBlocks + edgeBlocks) * 2
    }
    
    var gridSizes = [
      MTLSize(width: gridX, height: parameters.H, depth: 1)
    ]
    var groupSizes = [
      MTLSize(width: 32 * Int(R_splits), height: 1, depth: 1)
    ]
    
    if parameters.blockSparse {
      blockBytes.append(4 * R_splits)
      deviceBytes.append(0)
      gridSizes.append(MTLSize(
        width: ceilDivide(target: parameters.R, granularity: R_group),
        height: ceilDivide(target: parameters.C, granularity: C_simd),
        depth: 1))
      groupSizes.append(MTLSize(
        width: 32 * Int(R_splits),
        height: 1,
        depth: 1))
      
      // Reduce memory allocation overhead during benchmarks.
      let scratchBufferSize = 8 * gridSizes[1].width * gridSizes[1].height
      _ = MFA_Backend.global.cache
        .requestScratchBuffer(size: scratchBufferSize)
    }
    
    var flags: UInt32 = 0
    if parameters.batched {
      flags |= 0x1
    }
    if parameters.masked {
      flags |= 0x2
    }
    if parameters.blockSparse {
      flags |= 0x4
    }
    if triangular {
      flags |= 0x8
    }
    return AsyncPipeline(
      functions: functions,
      flags: flags,
      deviceMemoryLengths: deviceBytes,
      threadgroupMemoryLengths: blockBytes,
      gridSizes: gridSizes,
      groupSizes: groupSizes)
  }
  
  func encode(
    encoder: MTLComputeCommandEncoder,
    tensors: Attention_Tensors,
    resource: AsyncPipeline
  ) {
    let tensorQ = tensors.q as! MFA_TensorBuffer
    let tensorK = tensors.k as! MFA_TensorBuffer
    let tensorV = tensors.v as! MFA_TensorBuffer
    let tensorO = tensors.o as! MFA_TensorBuffer
    encoder.setBuffer(tensorQ.buffer, offset: 0, index: 0)
    encoder.setBuffer(tensorK.buffer, offset: 0, index: 1)
    encoder.setBuffer(tensorV.buffer, offset: 0, index: 2)
    encoder.setBuffer(tensorO.buffer, offset: 0, index: 3)
    
    var gridZ: Int
    var scratchBufferSize: Int = -1
    var partialsBufferSize: Int = -1
    var locksBufferSize: Int = -1
    
    if resource.flags & 0x4 > 0 {
      let gridSize = resource.gridSizes[1]
      scratchBufferSize = gridSize.height * gridSize.width
    }
    if resource.flags & 0x8 > 0 {
      let gridSize = resource.gridSizes[0]
      partialsBufferSize = Int(resource.deviceMemoryLengths[0])
      locksBufferSize = gridSize.height * gridSize.width
    }
    if resource.flags & 0x1 > 0 {
      let batchDimensionsQ = tensors.q.shape.dropLast(3)
      let batchDimensionsK = tensors.k.shape.dropLast(3)
      let batchDimensionsV = tensors.v.shape.dropLast(3)
      let batchDimensionsO = tensors.o.shape.dropLast(3)
      assert(batchDimensionsQ.reduce(1, *) > 0)
      assert(batchDimensionsQ == batchDimensionsK)
      assert(batchDimensionsQ == batchDimensionsV)
      assert(batchDimensionsQ == batchDimensionsO)
      
      gridZ = batchDimensionsQ.reduce(1, *)
      partialsBufferSize *= gridZ
      locksBufferSize *= gridZ
      
      let elementSize = tensors.q.dataType.size
      var byteStrideMask = 0
      var byteStrideBlockMask = 0
      
      func setMaskStride(shape: [Int]) {
        var output = elementSize
        output *= shape[shape.count - 1]
        output *= shape[shape.count - 2]
        output *= shape[shape.count - 3]
        
        if shape.dropLast(3).reduce(1, *) > 1 {
          byteStrideMask = output
          byteStrideBlockMask = max(0, scratchBufferSize)
          scratchBufferSize *= gridZ
        }
      }
      
      if resource.flags & 0x2 > 0 {
        let batchDimensionsMask = tensors.mask!.shape.dropLast(3)
        assert(
          batchDimensionsMask.reduce(1, *) == 1 ||
          batchDimensionsMask == batchDimensionsQ)
        setMaskStride(shape: tensors.mask!.shape)
      }
      
      withUnsafeTemporaryAllocation(
        of: SIMD4<UInt64>.self, capacity: gridZ
      ) { buffer in
        for i in 0..<buffer.count {
          buffer[i] = SIMD4(
            UInt64(truncatingIfNeeded: i * byteStrideMask),
            UInt64(truncatingIfNeeded: i * byteStrideBlockMask),
            UInt64(0),
            UInt64(0))
        }
        
        let bufferLength = buffer.count * MemoryLayout<SIMD4<UInt64>>.stride
        assert(MemoryLayout<SIMD4<UInt64>>.stride == 8 * 4)
        encoder.setBytes(buffer.baseAddress!, length: bufferLength, index: 10)
      }
    } else {
      assert(tensors.q.shape.count == 3)
      assert(tensors.k.shape.count == 3)
      assert(tensors.v.shape.count == 3)
      assert(tensors.o.shape.count == 3)
      if let tensorMask = tensors.mask {
        assert(tensorMask.shape.count == 3)
      }
      gridZ = 1
    }
    
    if scratchBufferSize > 0 {
      let scratchBuffer = MFA_Backend.global.cache
        .requestScratchBuffer(size: scratchBufferSize)
      encoder.setBuffer(scratchBuffer, offset: 0, index: 13)
    }
    if locksBufferSize > 0 {
      let locksBuffer = MFA_Backend.global.cache
        .requestLocksBuffer(size: locksBufferSize)
      encoder.setBuffer(locksBuffer, offset: 0, index: 14)
    }
    if partialsBufferSize > 0 {
      let partialsBuffer = MFA_Backend.global.cache
        .requestPartialsBuffer(size: partialsBufferSize)
      encoder.setBuffer(partialsBuffer, offset: 0, index: 15)
    }
    
    if resource.flags & 0x2 > 0 {
      let tensorMask = tensors.mask! as! MFA_TensorBuffer
      let maskShape = tensors.mask!.shape
      assert(maskShape[maskShape.count - 3] == 1)
      encoder.setBuffer(tensorMask.buffer, offset: 0, index: 12)
    }
    if resource.flags & 0x4 > 0 {
      scratchBufferSize *= gridZ
      precondition(resource.flags & 0x2 > 0)
      precondition(scratchBufferSize > 0)
      
      encoder.setComputePipelineState(resource.resource(index: 1))
      encoder.setThreadgroupMemoryLength(
        Int(resource.threadgroupMemoryLengths[1]), index: 0)
      
      var gridSize = resource.gridSizes[1]
      gridSize.depth = gridZ
      encoder.dispatchThreadgroups(
        gridSize, threadsPerThreadgroup: resource.groupSizes[1])
    }
    
    encoder.setComputePipelineState(resource.resource(index: 0))
    encoder.setThreadgroupMemoryLength(
      Int(resource.threadgroupMemoryLengths[0]), index: 0)
    
    var gridSize = resource.gridSizes[0]
    gridSize.depth = gridZ
    encoder.dispatchThreadgroups(
      gridSize, threadsPerThreadgroup: resource.groupSizes[0])
  }
}

struct MPS_Attention: Attention, MPS_Operation {
  var parameters: Attention_Parameters
  
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncResource() -> AsyncGraph {
    let dataType = parameters.dataType
    precondition(dataType == .float || dataType == .half)
    if parameters.batched {
      precondition(parameters.batchDimensionsQ!.count > 0)
      if parameters.masked {
        precondition(parameters.batchDimensionsMask!.count > 0)
        if let batchDimensionsMask = parameters.batchDimensionsMask {
          precondition(
            batchDimensionsMask.reduce(1, *) == 1 ||
            batchDimensionsMask == parameters.batchDimensionsQ!)
        }
      }
    } else {
      precondition(parameters.batchDimensionsQ! == [])
      if parameters.masked {
        precondition(parameters.batchDimensionsMask! == [])
      }
    }
    if !parameters.masked {
      precondition(parameters.batchDimensionsMask == nil)
    }
    
    let qBatch: [Int] = parameters.batchDimensionsQ!
    let maskBatch: [Int]? = parameters.batchDimensionsMask
    
    let qShape: [Int] = qBatch + [parameters.R, parameters.H, parameters.D]
    let kShape: [Int] = qBatch + [parameters.H, parameters.D, parameters.C]
    let vShape: [Int] = qBatch + [parameters.C, parameters.H, parameters.D]
    let oShape: [Int] = qBatch + [parameters.R, parameters.H, parameters.D]
    var maskShape: [Int]?
    if let maskBatch {
      maskShape = maskBatch + [1, parameters.R, parameters.C]
    }
    
    var qShapeTranspose: [Int]?
    var kShapeTranspose: [Int]?
    var vShapeTranspose: [Int]?
    var oShapeTranspose: [Int]?
    if parameters.Q_trans {
      qShapeTranspose = qBatch + [parameters.H, parameters.D, parameters.R]
    }
    if parameters.K_trans {
      kShapeTranspose = qBatch + [parameters.C, parameters.H, parameters.D]
    }
    if parameters.V_trans {
      vShapeTranspose = qBatch + [parameters.H, parameters.D, parameters.C]
    }
    if parameters.O_trans {
      oShapeTranspose = qBatch + [parameters.H, parameters.D, parameters.R]
    }
    _ = oShape
    _ = oShapeTranspose
    
    let graph = MPSGraph()
    func shape(_ shape: [Int]?) -> [Int] {
      shape!
    }
    func nsShape(_ shape: [Int]?) -> [NSNumber] {
      shape!.map(NSNumber.init)
    }
    func shapedType(_ shape: [Int]?) -> MPSGraphShapedType {
      MPSGraphShapedType(shape: nsShape(shape), dataType: dataType.mps)
    }
    func placeholder(_ shape: [Int]?, _ name: String) -> MPSGraphTensor {
      graph.placeholder(
        shape: nsShape(shape), dataType: dataType.mps, name: name)
    }
    func transpose(
      _ tensor: MPSGraphTensor, _ name: String,
      batchDims: [Int], permutation: [Int]
    ) -> MPSGraphTensor {
      let batchPart = Array<Int>(batchDims.indices.map { $0 })
      let permPart = permutation.map { $0 + batchDims.count }
      let _permutation = nsShape(batchPart + permPart)
      return graph.transpose(tensor, permutation: _permutation, name: name)
    }
    
    var originalQ: MPSGraphTensor
    var shapedTypeQ: MPSGraphShapedType
    var postTransposeQ: MPSGraphTensor
    if parameters.Q_trans {
      originalQ = placeholder(qShapeTranspose, "Q_trans")
      shapedTypeQ = shapedType(qShapeTranspose)
      postTransposeQ = transpose(
        originalQ, "Q", batchDims: qBatch, permutation: [2, 0, 1])
    } else {
      originalQ = placeholder(qShape, "Q")
      shapedTypeQ = shapedType(qShape)
      postTransposeQ = originalQ
    }
    
    var originalK: MPSGraphTensor
    var shapedTypeK: MPSGraphShapedType
    var postTransposeK: MPSGraphTensor
    if parameters.K_trans {
      originalK = placeholder(kShapeTranspose, "K_trans")
      shapedTypeK = shapedType(kShapeTranspose)
      postTransposeK = transpose(
        originalK, "K", batchDims: qBatch, permutation: [1, 2, 0])
    } else {
      originalK = placeholder(kShape, "K")
      shapedTypeK = shapedType(kShape)
      postTransposeK = originalK
    }
    
    var originalV: MPSGraphTensor
    var shapedTypeV: MPSGraphShapedType
    var postTransposeV: MPSGraphTensor
    if parameters.V_trans {
      originalV = placeholder(vShapeTranspose, "V_trans")
      shapedTypeV = shapedType(vShapeTranspose)
      postTransposeV = transpose(
        originalV, "V", batchDims: qBatch, permutation: [2, 0, 1])
    } else {
      originalV = placeholder(vShape, "V")
      shapedTypeV = shapedType(vShape)
      postTransposeV = originalV
    }
    
    var originalMask: MPSGraphTensor?
    var shapedTypeMask: MPSGraphShapedType?
    if let maskShape {
      originalMask = placeholder(maskShape, "mask")
      shapedTypeMask = shapedType(maskShape)
    }
    
    let contiguousQ = transpose(
      postTransposeQ, "Q_contiguous", batchDims: qBatch, permutation: [1, 0, 2])
    var attentionMatrix = graph.matrixMultiplication(
      primary: contiguousQ,
      secondary: postTransposeK,
      name: "QK")
    if dataType == .half {
      attentionMatrix = graph.cast(
        attentionMatrix, to: .float32, name: "QK_f32")
    }
    let alpha = graph.constant(
      rsqrt(Double(parameters.D)),
      dataType: .float32)
    attentionMatrix = graph.multiplication(
      attentionMatrix,
      alpha,
      name: "QK/sqrt(D)")
    
    var zeroMask: MPSGraphTensor?
    if var originalMask {
      if dataType == .half {
        originalMask = graph.cast(
          originalMask, to: .float32, name: "mask_f32")
      }
      attentionMatrix = graph.addition(
        attentionMatrix,
        originalMask,
        name: "mask(QK/sqrt(D))")
      
      var value: Double
      if dataType == .float {
        value = Double(-Float.greatestFiniteMagnitude / 2)
      } else {
#if arch(arm64)
        value = Double(-Float16.greatestFiniteMagnitude / 2)
#else
        value = Double(-Float.greatestFiniteMagnitude / 2)
#endif
      }
      let scalar = graph.constant(
        value,
        dataType: .float32)
      zeroMask = graph.lessThanOrEqualTo(
        attentionMatrix,
        scalar,
        name: "zero_mask")
    }
    if let zeroMask {
      let lastAxes = [NSNumber(value: qShape.count - 1)]
      var summary = graph.reductionMaximum(
        with: attentionMatrix,
        axes: lastAxes,
        name: "m")
      attentionMatrix = graph.subtraction(
        attentionMatrix,
        summary,
        name: "QK - m")
      attentionMatrix = graph.exponent(
        with: attentionMatrix,
        name: "exp(QK - m)")
      
      let zero = graph.constant(
        0,
        dataType: .float32)
      attentionMatrix = graph.select(
        predicate: zeroMask,
        trueTensor: zero,
        falseTensor: attentionMatrix,
        name: "exp(QK - m)")
      
      summary = graph.reductionSum(
        with: attentionMatrix,
        axes: lastAxes,
        name: "l")
      attentionMatrix = graph.divisionNoNaN(
        attentionMatrix,
        summary, name:
          "sm(QK)")
    } else {
      attentionMatrix = graph.softMax(
        with: attentionMatrix,
        axis: qShape.count - 1,
        name: "sm(QK)")
    }
    
    let contiguousV = transpose(
      postTransposeV, "V_contiguous", batchDims: qBatch, permutation: [1, 0, 2])
    var contiguousO = graph.matrixMultiplication(
      primary: attentionMatrix,
      secondary: contiguousV,
      name: "O_contiguous")
    if dataType == .half {
      contiguousO = graph.cast(
        contiguousO,
        to: .float16,
        name: "O_contiguous_f16")
    }
    
    var originalO: MPSGraphTensor
    var postTransposeO: MPSGraphTensor
    if parameters.O_trans {
      originalO = transpose(
        contiguousO, "O_trans", batchDims: qBatch, permutation: [1, 0, 2])
      postTransposeO = transpose(
        originalO, "O", batchDims: qBatch, permutation: [1, 2, 0])
    } else {
      originalO = transpose(
        contiguousO, "O", batchDims: qBatch, permutation: [1, 0, 2])
      postTransposeO = originalO
    }
    var feeds: [MPSGraphTensor: MPSGraphShapedType] = [
      originalQ: shapedTypeQ,
      originalK: shapedTypeK,
      originalV: shapedTypeV,
    ]
    if let originalMask, let shapedTypeMask {
      feeds[originalMask] = shapedTypeMask
    }
    return AsyncGraph(
      graph: graph, feeds: feeds, targetTensors: [postTransposeO])
  }
  
  func encode(
    encoder: MPSCommandBuffer,
    tensors: Attention_Tensors,
    resource: AsyncGraph
  ) {
    let tensorQ = tensors.q as! MPS_TensorBuffer
    let tensorK = tensors.k as! MPS_TensorBuffer
    let tensorV = tensors.v as! MPS_TensorBuffer
    let tensorO = tensors.o as! MPS_TensorBuffer
    var inputs = [tensorQ.tensorData, tensorK.tensorData, tensorV.tensorData]
    
    if let mask = tensors.mask {
      let tensorMask = mask as! MPS_TensorBuffer
      inputs.append(tensorMask.tensorData)
    }
    let results = [tensorO.tensorData]
    resource.resource(index: 0).encode(
      to: encoder,
      inputs: inputs,
      results: results,
      executionDescriptor: nil)
  }
}

struct Py_Attention: Attention, Py_Operation {
  var parameters: Attention_Parameters
  
  init(parameters: Attention_Parameters) {
    self.parameters = parameters
  }
  
  func execute(tensors: Attention_Tensors) {
    let tensorQ = tensors.q as! Py_TensorBuffer
    let tensorK = tensors.k as! Py_TensorBuffer
    let tensorV = tensors.v as! Py_TensorBuffer
    let tensorO = tensors.o as! Py_TensorBuffer
    var tensorMask: Py_TensorBuffer?
    if parameters.masked {
      // WARNING: Mask dimensions need to be [B, 1, R, C].
      tensorMask = (tensors.mask! as! Py_TensorBuffer)
    }
    
    let np = PythonContext.global.np
    
    var postTransposeQ = tensorQ.ndarray
    if parameters.Q_trans {
      postTransposeQ = np.einsum("...ijk->...kij", tensorQ.ndarray)
    }
    
    var postTransposeK = tensorK.ndarray
    if parameters.K_trans {
      postTransposeK = np.einsum("...ijk->...jki", tensorK.ndarray)
    }
    
    var postTransposeV = tensorV.ndarray
    if parameters.V_trans {
      postTransposeV = np.einsum("...ijk->...kij", tensorV.ndarray)
    }
    
    // Multiply Q * K.
    // [R, H, D] * [H, D, C] -> [H, R, C]
    var attentionMatrix = np.einsum(
      "...ijk,...jkl->...jil", postTransposeQ, postTransposeK)
    if tensors.q.dataType == .half {
      attentionMatrix = attentionMatrix.astype(np.float32)
    }
    attentionMatrix *= PythonObject(rsqrt(Double(parameters.D)))
    
    // Apply explicit mask.
    var zeroMask: PythonObject?
    if let tensorMask {
      np.add(attentionMatrix, tensorMask.ndarray, out: attentionMatrix)
      if tensors.q.dataType == .float {
        zeroMask = np.less_equal(
          attentionMatrix, -Float.greatestFiniteMagnitude / 2)
      } else {
#if arch(arm64)
        zeroMask = np.less_equal(
          attentionMatrix, Float(-Float16.greatestFiniteMagnitude / 2))
#else
        fatalError()
#endif
      }
    }
    
    // Perform softmax.
#if true
    let lastAxis = PythonObject(tensorQ.shape.count - 1)
    var summary = np[dynamicMember: "max"](
      attentionMatrix, axis: lastAxis, keepdims: true)
    np.subtract(attentionMatrix, summary, out: attentionMatrix)
    np.exp(attentionMatrix, out: attentionMatrix)
    if let zeroMask {
      attentionMatrix[zeroMask] = 0
    }
    
    np.sum(
      attentionMatrix, axis: lastAxis, keepdims: true, out: summary)
    if parameters.masked {
      let zeroMask = np.equal(summary, 0)
      summary[zeroMask] = PythonObject(Float.leastNormalMagnitude)
      summary = 1 / summary
      np.multiply(attentionMatrix, summary, out: attentionMatrix)
    } else {
      np.divide(attentionMatrix, summary, out: attentionMatrix)
    }
#endif
    
    // Multiply P * V.
    // [H, R, C] * [C, H, D] -> [R, H, D]
    if tensors.q.dataType == .half {
      var O_f32: PythonObject
      if parameters.O_trans {
        O_f32 = np.einsum(
          "...ijk,...kil->...ilj",
          attentionMatrix, postTransposeV)
      } else {
        O_f32 = np.einsum(
          "...ijk,...kil->...jil",
          attentionMatrix, postTransposeV)
      }
      let O_f16 = O_f32.astype(np.float16)
      np.add(O_f16, 0, out: tensorO.ndarray)
    } else {
      if parameters.O_trans {
        np.einsum(
          "...ijk,...kil->...ilj",
          attentionMatrix, postTransposeV, out: tensorO.ndarray)
      } else {
        np.einsum(
          "...ijk,...kil->...jil",
          attentionMatrix, postTransposeV, out: tensorO.ndarray)
      }
    }
  }
}
