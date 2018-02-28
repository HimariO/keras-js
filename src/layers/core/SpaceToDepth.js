import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import _ from 'lodash'
import ops from 'ndarray-ops'
import mapInputProgramSource from '../../webgl/mapInput.glsl'

/**
 * SpaceToDepth layer class
 * Note there is no concept of batch size in these layers (single-batch).
 */
export default class SpaceToDepth extends Layer {
  /**
   * Creates a SpaceToDepth layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.block_size]
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'SpaceToDepth'

    const { block_size = [] } = attrs
    this.block_size = block_size
    console.log(block_size)
    // console.log(this.inputShape)


    this.description = `block_size ${this.block_size}`

    // GPU setup
    if (this.gpu) {
      this.mapInputProgram = webgl2.compileProgram(mapInputProgramSource)
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    this.throwError('CPU version of SpaceToDepth Layer haven\'t been implemented.')
    // if (this.targetShape.reduce((a, b) => a * b, 1) !== x.tensor.size) {
    //   this.throwError('The total size of new array must be unchanged in reshape layer.')
    // }
    // this.output = new Tensor([], this.targetShape)
    // this.output.replaceTensorData(x.tensor.data)
  }

  /**
   * Creates row/col index mappings to map input texture to output texture
   */
  _createIndexMap() {
    if (this.indexMap) {
      return
    }
    console.log(this.inputShape, this.targetShape)

    // const indices = new Tensor([], this.inputShape, { type: Int32Array })
    this.indexMap = new Tensor([], this.targetShape, { type: Int32Array })
    const block_height = this.inputShape[0] / this.block_size
    const block_width = this.inputShape[1] / this.block_size
    const block_row_surface = block_height * this.inputShape[1]

    var subIndexMap = new Tensor([], [this.block_size * this.block_size * this.inputShape[2]], { type: Int32Array })

    for (let k = 0; k < this.inputShape[2]; k++) {
      for (let i = 0; i < this.block_size; i++) {
        for (let j = 0; j < this.block_size; j++) {
          subIndexMap.tensor.set(
            i * this.block_size + j + k * Math.pow(this.block_size, 2),
            k + j * this.inputShape[2] + i * this.inputShape[1] * this.inputShape[2]
          )
        }
      }
    }
    console.log('subIndexMap: ', subIndexMap)

    for (let j = 0; j < block_height; j++) {
      for (let k = 0; k < block_width; k++) {
        ops.assign(
          this.indexMap.tensor.pick(j, k, null),
          subIndexMap.tensor
        )
        ops.addseq(
          this.indexMap.tensor.pick(j, k, null),
          j * block_row_surface * this.inputShape[2] + k * this.block_size * this.inputShape[2]
        )
      }
    }

    if (this.targetShape.length > 2) {
      this.indexMap.reshapeTo2D()
    }

    this.indexMap.createGLTexture({ type: '2d', format: 'int' })
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {

    if (!x.glTexture) {
      this.inputShape = x.tensor.shape
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture({ type: '2d', format: 'float' })
      } else if (x.tensor.shape.length > 2 && !x.is2DReshaped) {
        x.reshapeTo2D()
        x.createGLTexture({ type: '2d', format: 'float' })
      }
    } else if (x.is2DReshaped || x.is2DSquareReshaped) {
      this.inputShape = x.originalShape
    } else {
      this.inputShape = x.tensor.shape
    }

    this.targetShape = [
      this.inputShape[0] /  this.block_size,
      this.inputShape[1] /  this.block_size,
      this.inputShape[2] * Math.pow(this.block_size, 2)
    ]

    this._createIndexMap()
    // console.log(this.indexMap)

    if (!this.output) {
      this.output = new Tensor([], this.targetShape)
      if (this.targetShape.length > 2) {
        this.output.reshapeTo2D()
      }
      this.output.createGLTexture({ type: '2d', format: 'float' })
    }

    webgl2.runProgram({
      program: this.mapInputProgram,
      output: this.output,
      inputs: [{ input: x, name: 'x' }, { input: this.indexMap, name: 'indexMap' }],
      uniforms: [{ value: x.glTextureShape[1], type: 'int', name: 'inputCols' }]
    })

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D()
      } else if (this.output.is2DSquareReshaped) {
        this.output.reshapeFrom2DSquare()
      }
    }
  }
}
