export default class Neuron {
    actFunc: Function;
    weights: number[];
    bias: number;

    constructor(actFunc: Function, initialWeights: number[], bias: number) {
        this.actFunc = actFunc
        this.weights = initialWeights
        this.bias = bias
    }

    eval(inputs: number[]) {
        const weightedSumOfInputs: number = this.weights.reduce((acc: number, weight: number, index: number) => (
            acc + weight * inputs[index]
        ), 0)

        return this.actFunc(weightedSumOfInputs + this.bias)
    }
}
