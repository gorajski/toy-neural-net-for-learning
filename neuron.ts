import { ActFunc } from './types'

export default class Neuron {
    actFunc: ActFunc
    weights: number[]
    bias: number

    constructor(actFunc: ActFunc, initialWeights: number[], bias: number) {
        this.actFunc = actFunc
        this.weights = initialWeights
        this.bias = bias
    }

    eval(inputs: number[]) {
        // TODO Validate that inputs.length is same as weights.length
        const weightedSumOfInputs: number = this.weights.reduce(
            (acc: number, weight: number, index: number) => (
                acc + (weight * inputs[index])
            ),
        0)

        return this.actFunc(weightedSumOfInputs + this.bias)
    }
}
