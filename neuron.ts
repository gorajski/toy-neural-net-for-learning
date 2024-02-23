import { ActFunc } from "./types";

export default class Neuron {
  actFunc: ActFunc;
  weights: number[];
  bias: number;

  constructor(actFunc: ActFunc, initialWeights: number[], bias: number) {
    this.actFunc = actFunc;
    this.weights = initialWeights;
    this.bias = bias;
  }

  /** The value our neuron gives when firing, given these inputs */
  eval(inputs: number[]) {
    // TODO Validate that inputs.length is same as weights.length
    const weightedSumOfInputs = this.weightedSumOfInputs(inputs);
    return this.actFunc(weightedSumOfInputs + this.bias);
  }

  weightedSumOfInputs(inputs: number[]): number {
    return this.weights.reduce(
      (acc: number, weight: number, index: number) => (
        acc + (weight * inputs[index])
      ),
      0,
    );
  }

  /** See img/math.jpg for the math of what's going on here */
  backprop(inputs: number[], desiredOutput: number): number[] {
    // (dCost/dWeight)[], slope of our function, where we want it to be 0
    const gradientArray: number[] = [];

    const actualOutput = this.eval(inputs);
    const weightedSum = this.weightedSumOfInputs(inputs);
    const actualOutputPerWeightedSum = Math.exp(weightedSum) /
      Math.pow(1 + Math.exp(weightedSum), 2);
    const costPerActualOutput = 2 * (actualOutput - desiredOutput);

    const weightedSumPerBias = 1;
    const costPerBias = costPerActualOutput * actualOutputPerWeightedSum *
      weightedSumPerBias;
    gradientArray.push(costPerBias);

    inputs.forEach((input) => {
      const weightedSumPerWeight = input;
      const costPerWeight = costPerActualOutput * actualOutputPerWeightedSum *
        weightedSumPerWeight;
      gradientArray.push(costPerWeight);
    });

    return gradientArray;
  }

  update(inputs: number[], desiredOutput: number) {
    const gradientArray = this.backprop(inputs, desiredOutput);
    const [biasGradient, ...weightGradientArray] = gradientArray;
    this.bias -= biasGradient * 0.1;
    weightGradientArray.forEach((gradient, index) => {
      this.weights[index] -= gradient * 0.1;
    });
  }
}
